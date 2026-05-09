"""Generates a JAX-side fixture for parity testing.

For each VideoPrism variant the upstream JAX repo distributes (`videoprism_public_v1_base`,
`videoprism_public_v1_large`), runs the model on a deterministic input batch and
saves an `.npz` containing the input + the full final output. The base variant
additionally saves intermediate features (post `spatial_ln` and post `temporal_ln`)
since `test_deep_parity.py` bisects against those.

Run inside a JAX venv that has the upstream `videoprism` package installed
(`pip install git+https://github.com/google-deepmind/videoprism`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp
from videoprism import models as vp
from videoprism import encoders, layers as vp_layers
import einshape


def _build_video_batch(
    seeds: list[int], num_frames: int, video_path: str | None = None
) -> np.ndarray:
    """Builds a (B, T, 288, 288, 3) float32 batch in [0, 1].

    For each seed, generates one independent random sample. If `video_path`
    is provided, replaces the first sample with frames decoded from that video.
    """
    samples = []
    for s in seeds:
        rng = np.random.default_rng(s)
        samples.append(
            rng.uniform(0.0, 1.0, size=(num_frames, 288, 288, 3)).astype(np.float32)
        )

    if video_path is not None:
        from decord import VideoReader, cpu  # type: ignore
        import cv2

        vr = VideoReader(video_path, ctx=cpu(0))
        n_frames = len(vr)
        idx = np.linspace(0, max(n_frames - 1, 0), num=num_frames).astype(int)
        frames = vr.get_batch(idx).asnumpy()  # uint8 (T, H, W, 3)
        resized = np.stack([cv2.resize(f, (288, 288)) for f in frames])
        samples[0] = resized.astype(np.float32) / 255.0

    return np.stack(samples, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model",
        default="videoprism_public_v1_base",
        choices=list(vp.MODELS),
        help="Which upstream VideoPrism configuration to run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds — if set, overrides --batch.")
    parser.add_argument("--video-path", type=str, default=None,
                        help="If set, replaces sample 0 with frames decoded from this video.")
    parser.add_argument("--save-intermediates", action="store_true",
                        help="Also save spatial_features + temporal_out (used by deep parity tests).")
    args = parser.parse_args()

    # Read the config to know T (= pos_emb_shape[0]) and architecture sizes.
    cfg = vp.CONFIGS[args.model.replace("_public", "")]
    num_frames = cfg["pos_emb_shape"][0]
    print(f"model={args.model} | T={num_frames} | model_dim={cfg['model_dim']} | "
          f"num_spatial_layers={cfg['num_spatial_layers']} | num_heads={cfg['num_heads']}")

    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",")]
        video = _build_video_batch(seeds, num_frames, video_path=args.video_path)
    else:
        rng = np.random.default_rng(args.seed)
        video = rng.uniform(0.0, 1.0,
                            size=(args.batch, num_frames, 288, 288, 3)).astype(np.float32)

    flax_model = vp.get_model(args.model)
    state = vp.load_pretrained_weights(args.model)
    params = state["params"]

    @jax.jit
    def fwd(x):
        return flax_model.apply(state, x, train=False)

    out, _ = fwd(video)
    out = np.asarray(out)
    print(f"final output: {out.shape} dtype={out.dtype}")

    save_kwargs = {"input": video, "output": out}

    if args.save_intermediates:
        # Step-by-step reconstruction to capture intermediates. Numerical drift
        # vs the JIT'd forward is normal (~5e-4) — XLA fuses ops differently.
        P = cfg["patch_size"]
        b, t, h, w, c = video.shape
        reshaped = video.reshape(b * t, h, w, c)
        import einops
        patches = einops.rearrange(
            reshaped, "... (m p)(n q) c->...(m n)(p q c)",
            m=h // P, n=w // P, p=P, q=P, c=c,
        )
        pp_kernel = params["patch_projection"]["linear"]["kernel"]
        pp_bias = params["patch_projection"]["linear"]["bias"]
        feats = jnp.einsum("bnd,de->bne", patches, pp_kernel) + pp_bias
        feats = feats + params["spatial_pos_emb"]["emb_var"]

        spatial_vt = encoders.VisionTransformer(
            num_tfm_layers=cfg["num_spatial_layers"],
            mlp_dim=cfg["mlp_dim"],
            num_heads=cfg["num_heads"],
            atten_logit_cap=cfg["atten_logit_cap"],
            norm_policy=cfg.get("norm_policy", "pre"),
            scan=cfg.get("scan", True),
        )
        feats_after_spatial = spatial_vt.apply(
            {"params": params["spatial_encoder"]}, feats, paddings=None, train=False
        )
        spatial_ln_mod = vp_layers.LayerNorm()
        spatial_features = spatial_ln_mod.apply(
            {"params": params["spatial_ln"]}, feats_after_spatial
        )

        feats_t = einshape.jax_einshape("(bt)nd->(bn)td", spatial_features, t=t)
        feats_t_with_pos = feats_t + params["temporal_pos_emb"]["emb_var"]

        temporal_vt = encoders.VisionTransformer(
            num_tfm_layers=cfg["num_temporal_layers"],
            mlp_dim=cfg["mlp_dim"],
            num_heads=cfg["num_heads"],
            atten_logit_cap=cfg["atten_logit_cap"],
            norm_policy=cfg.get("norm_policy", "pre"),
            scan=cfg.get("scan", True),
        )
        feats_after_temporal = temporal_vt.apply(
            {"params": params["temporal_encoder"]},
            feats_t_with_pos, paddings=None, train=False,
        )
        feats_after_temporal_ln = vp_layers.LayerNorm().apply(
            {"params": params["temporal_ln"]}, feats_after_temporal
        )

        save_kwargs["spatial_features"] = np.asarray(spatial_features)
        save_kwargs["pre_temporal"] = np.asarray(feats_t_with_pos)
        save_kwargs["temporal_out"] = np.asarray(feats_after_temporal_ln)
        print(f"saved intermediates: spatial_features={spatial_features.shape} "
              f"temporal_out={feats_after_temporal_ln.shape}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **save_kwargs)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
