"""Generates a JAX-side fixture for parity testing.

Saves to `<out>` an .npz with:
  input        : (1, 16, 288, 288, 3) float32, uniform in [0, 1]
  output       : (1, 4096, 768) float32 — full model output
  spatial_out  : (16, 256, 768) float32 — features after spatial encoder + spatial_ln
  pre_temporal : (256, 16, 768) float32 — features after spatial_ln, reshaped + temporal pos added
  temporal_out : (256, 16, 768) float32 — features after temporal encoder + temporal_ln (pre final reshape)

Run inside the JAX venv (`/workspace/envs/vp_jax`) on the pod.
"""
from __future__ import annotations

import argparse
import numpy as np
import jax, jax.numpy as jnp
from flax.traverse_util import flatten_dict
from videoprism import models as vp
from videoprism import encoders, layers as vp_layers
import einshape


def _load_state(model_name: str):
    flax_model = vp.get_model(model_name)
    state = vp.load_pretrained_weights(model_name)
    return flax_model, state


def _split_params(state):
    """Split full state into per-submodule param trees."""
    p = state["params"]
    return p


def _build_video_batch(seeds: list[int], video_path: str | None = None) -> np.ndarray:
    """Builds a (B, 16, 288, 288, 3) float32 batch in [0,1].

    For each seed, generates one independent random sample. If `video_path`
    is provided, replaces the first sample with frames decoded from that video.
    """
    samples = []
    for s in seeds:
        rng = np.random.default_rng(s)
        samples.append(rng.uniform(0.0, 1.0, size=(16, 288, 288, 3)).astype(np.float32))

    if video_path is not None:
        # Use a real video as the first batch element. Decoded frames come back
        # uint8 HxWx3; we resize to 288x288 (square) and normalize to [0, 1].
        from decord import VideoReader, cpu  # type: ignore
        import cv2

        vr = VideoReader(video_path, ctx=cpu(0))
        # Take 16 evenly-spaced frames so we cover the whole clip even if it's long.
        n_frames = len(vr)
        idx = np.linspace(0, max(n_frames - 1, 0), num=16).astype(int)
        frames = vr.get_batch(idx).asnumpy()  # (16, H, W, 3) uint8
        resized = np.stack([cv2.resize(f, (288, 288)) for f in frames])
        samples[0] = (resized.astype(np.float32) / 255.0)

    return np.stack(samples, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds — if set, builds one sample per seed (overrides batch).")
    parser.add_argument("--video-path", type=str, default=None,
                        help="If set, replaces sample 0 with frames decoded from this video.")
    args = parser.parse_args()

    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        # Replicate old behavior — `batch` independent samples from one seed
        # (numpy will produce different rows naturally when batch > 1).
        rng = np.random.default_rng(args.seed)
        video = rng.uniform(0.0, 1.0, size=(args.batch, 16, 288, 288, 3)).astype(np.float32)
        seeds = None

    if seeds is not None:
        video = _build_video_batch(seeds, video_path=args.video_path)

    flax_model, state = _load_state("videoprism_public_v1_base")
    params = state["params"]

    # ----- full forward (final output) -----
    @jax.jit
    def fwd(x):
        return flax_model.apply(state, x, train=False)

    out, _ = fwd(video)
    out = np.asarray(out)
    print(f"final output: {out.shape} dtype={out.dtype}")

    # ----- intermediate: features after spatial encoder + spatial_ln (pre temporal pos) -----
    # Manually replicate the steps of FactorizedEncoder up to spatial_ln,
    # and then up to after temporal pos addition, and after temporal_ln.

    P = 18
    b, t, h, w, c = video.shape
    reshaped = video.reshape(b * t, h, w, c)

    # _image_to_patch
    import einops
    patches = einops.rearrange(
        reshaped, "... (m p)(n q) c->...(m n)(p q c)",
        m=h // P, n=w // P, p=P, q=P, c=c,
    )

    # patch_projection (FeedForward with identity)
    pp_kernel = params["patch_projection"]["linear"]["kernel"]
    pp_bias = params["patch_projection"]["linear"]["bias"]
    feats = jnp.einsum("bnd,de->bne", patches, pp_kernel) + pp_bias

    # add spatial pos
    spatial_pos = params["spatial_pos_emb"]["emb_var"]   # (256, 768)
    feats = feats + spatial_pos

    # Spatial encoder: build a fresh VisionTransformer with the spatial sub-tree.
    # We use a parameter-only `apply` by injecting the `spatial_encoder` subtree.
    spatial_vt = encoders.VisionTransformer(
        num_tfm_layers=12, mlp_dim=3072, num_heads=12,
        atten_logit_cap=50.0, norm_policy="pre", scan=True,
    )
    se_params = {"params": params["spatial_encoder"]}
    feats_after_spatial = spatial_vt.apply(se_params, feats, paddings=None, train=False)

    # spatial_ln
    sl_params = {"params": params["spatial_ln"]}
    spatial_ln_mod = vp_layers.LayerNorm()
    spatial_features = spatial_ln_mod.apply(sl_params, feats_after_spatial)
    print(f"spatial_features: {spatial_features.shape}")

    # reshape (bt) n d -> (bn) t d
    feats_t = einshape.jax_einshape("(bt)nd->(bn)td", spatial_features, t=t)

    # add temporal pos
    temporal_pos = params["temporal_pos_emb"]["emb_var"]  # (16, 768)
    feats_t_with_pos = feats_t + temporal_pos

    # Temporal encoder
    temporal_vt = encoders.VisionTransformer(
        num_tfm_layers=4, mlp_dim=3072, num_heads=12,
        atten_logit_cap=50.0, norm_policy="pre", scan=True,
    )
    te_params = {"params": params["temporal_encoder"]}
    feats_after_temporal = temporal_vt.apply(te_params, feats_t_with_pos, paddings=None, train=False)

    tl_params = {"params": params["temporal_ln"]}
    temporal_ln_mod = vp_layers.LayerNorm()
    feats_after_temporal_ln = temporal_ln_mod.apply(tl_params, feats_after_temporal)
    print(f"after temporal_ln: {feats_after_temporal_ln.shape}")

    # Final reshape to (B, T*N, D) — sanity check this matches `fwd` output.
    n = feats_t.shape[0] // b
    final = einshape.jax_einshape("(bn)td->b(tn)d", feats_after_temporal_ln, b=b)
    diff = float(jnp.abs(final - out).max())
    print(f"manual vs jit final max abs diff: {diff:.3e}")
    # JIT op fusion in XLA produces slightly different fp32 results than a
    # step-by-step reconstruction; ~5e-4 is normal. We just want to confirm
    # the manual path isn't catastrophically wrong (e.g., transposed output).
    assert diff < 5e-3, f"manual reconstruction doesn't match jit: {diff}"

    np.savez(
        args.out,
        input=video,
        output=out,
        spatial_features=np.asarray(spatial_features),
        pre_temporal=np.asarray(feats_t_with_pos),
        temporal_out=np.asarray(feats_after_temporal_ln),
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
