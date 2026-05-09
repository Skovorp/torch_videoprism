"""JAX-side fixture for PE-interpolation parity tests.

Saves a `.npz` containing reference outputs for every interpolation case the
PyTorch port should reproduce:

  source_2d         : (256, 768) flattened spatial pos emb (loaded from videoprism_v1_base)
  target_2d_<NxN>   : 2D-bilinear interpolation of source_2d to (N, N), flattened
  source_1d         : (16, 768) temporal pos emb (loaded from videoprism_v1_base)
  target_1d_<L>     : 1D-linear interpolation of source_1d to L
  full_<H>          : (1, T*N, D) output of the full v1_base model run at H×H input
                      (and T derived from the original 16-frame native; we keep T=16).
  full_<H>_input    : (1, 16, H, H, 3) input

Run inside the JAX venv (`/workspace/envs/vp_jax/bin/python`).
"""
from __future__ import annotations

import argparse
import numpy as np
import jax, jax.numpy as jnp
from videoprism import models as vp
from videoprism import encoders


def _interp_2d_jax(emb: np.ndarray, src_hw, tgt_hw):
    return np.asarray(encoders._interpolate_emb_2d(
        jnp.asarray(emb)[None],   # (1, N, D)
        src_hw, tgt_hw,
    )[0])                          # drop leading 1


def _interp_1d_jax(emb: np.ndarray, target_len: int):
    return np.asarray(encoders._interpolate_emb_1d(
        jnp.asarray(emb)[None],   # (1, T, D)
        target_len,
    )[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    flax_model = vp.get_model("videoprism_public_v1_base")
    state = vp.load_pretrained_weights("videoprism_public_v1_base")
    params = state["params"]

    source_2d = np.asarray(params["spatial_pos_emb"]["emb_var"])  # (256, 768)
    source_1d = np.asarray(params["temporal_pos_emb"]["emb_var"]) # (16, 768)
    print(f"source 2d: {source_2d.shape} | source 1d: {source_1d.shape}")

    save = {"source_2d": source_2d, "source_1d": source_1d}

    # --- Direct interpolation cases.
    for hp, wp in [(8, 8), (12, 12), (16, 16), (20, 20), (24, 24), (32, 32)]:
        save[f"target_2d_{hp}x{wp}"] = _interp_2d_jax(source_2d, (16, 16), (hp, wp))
    for tlen in [4, 8, 12, 16, 24, 32]:
        save[f"target_1d_{tlen}"] = _interp_1d_jax(source_1d, tlen)

    # --- Full-model parity at non-native spatial resolutions.
    # We pick image sizes that are exact multiples of patch_size=18 and produce
    # an integer h_p other than the native 16 (which corresponds to 288 px).
    # Stick with T=16 (native) so we isolate the 2D-PE-interpolation effect.
    @jax.jit
    def fwd(x):
        return flax_model.apply(state, x, train=False)

    rng = np.random.default_rng(args.seed)
    for img_size in [144, 216, 288, 432, 576]:   # 8/12/16/24/32 patches per side
        h_p = img_size // 18
        video = rng.uniform(0.0, 1.0, size=(1, 16, img_size, img_size, 3)).astype(np.float32)
        out, _ = fwd(video)
        save[f"full_{img_size}_input"] = video
        save[f"full_{img_size}"] = np.asarray(out)
        print(f"img_size={img_size} h_p={h_p}: out shape={out.shape}")

    # --- Full-model parity at non-native temporal length (T != 16).
    # Keep image size native (288, h_p=16) so we isolate the temporal-PE path.
    for n_frames in [8, 12, 24]:
        video = rng.uniform(0.0, 1.0, size=(1, n_frames, 288, 288, 3)).astype(np.float32)
        out, _ = fwd(video)
        save[f"full_T{n_frames}_input"] = video
        save[f"full_T{n_frames}"] = np.asarray(out)
        print(f"n_frames={n_frames}: out shape={out.shape}")

    np.savez(args.out, **save)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
