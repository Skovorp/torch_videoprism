"""Inference-speed comparison: PyTorch port vs JAX golden reference.

Runs a 16-frame 288x288 batch through both stacks for `--iters` warmup +
`--iters` timed iterations and reports throughput. CPU-only by default —
the runpod 5090 (sm_120) needs a newer torch wheel than what ships with
runpod/pytorch:2.4.0 to use CUDA, so this benchmark reports CPU numbers.

Run inside the JAX venv on the pod (it has both jax and torch):
    source /workspace/envs/vp_jax/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python tests/videoprism/bench_inference.py --iters 5 --batch 1
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np


def bench_jax(video: np.ndarray, iters: int) -> tuple[list[float], tuple]:
    import jax
    from videoprism import models as vp

    flax_model = vp.get_model("videoprism_public_v1_base")
    state = vp.load_pretrained_weights("videoprism_public_v1_base")

    @jax.jit
    def fwd(x):
        return flax_model.apply(state, x, train=False)

    # warmup (one for JIT compile, one to ensure compile cached)
    out, _ = fwd(video)
    jax.block_until_ready(out)
    out, _ = fwd(video)
    jax.block_until_ready(out)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out, _ = fwd(video)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return times, tuple(out.shape)


def bench_torch(
    video: np.ndarray,
    iters: int,
    npz_path: str | None,
    compile_mode: str | None = None,
) -> tuple[list[float], tuple]:
    import torch
    from torch_videoprism import build_videoprism_v1_base, load_pretrained_weights

    model = build_videoprism_v1_base()
    kwargs = {"checkpoint_path": npz_path} if npz_path else {}
    load_pretrained_weights(model, **kwargs)
    model.eval()

    if compile_mode is not None:
        # Inductor compile adds a one-time cost; the warmup loop below absorbs it.
        model = torch.compile(model, mode=compile_mode)

    x = torch.from_numpy(video.copy())

    # warmup — extra iterations for compiled mode to absorb Inductor compile.
    n_warmup = 4 if compile_mode is not None else 2
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)

    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x)
            times.append(time.perf_counter() - t0)
    with torch.no_grad():
        out = model(x)
    return times, tuple(out.shape)


def fmt(times: list[float]) -> str:
    return f"mean {statistics.mean(times) * 1000:.1f}ms ± {statistics.stdev(times) * 1000:.1f}ms (n={len(times)})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--npz", type=str, default=os.environ.get("VIDEOPRISM_NPZ"))
    parser.add_argument("--skip-jax", action="store_true")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--torch-compile", choices=["default", "reduce-overhead", "max-autotune"],
                        default=None,
                        help="Run an extra Torch pass with torch.compile(mode=...).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    video = rng.uniform(0.0, 1.0, size=(args.batch, 16, 288, 288, 3)).astype(np.float32)
    print(f"input shape: {video.shape}, batch={args.batch}, iters={args.iters}")
    print()

    if not args.skip_jax:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        jax_times, jax_shape = bench_jax(video, args.iters)
        print(f"  JAX (XLA)         ({jax_shape}): {fmt(jax_times)}")

    if not args.skip_torch:
        torch_times, torch_shape = bench_torch(video, args.iters, args.npz, compile_mode=None)
        print(f"  Torch (eager)     ({torch_shape}): {fmt(torch_times)}")

    if args.torch_compile is not None:
        ct, cs = bench_torch(video, args.iters, args.npz, compile_mode=args.torch_compile)
        print(f"  Torch (compile/{args.torch_compile})  ({cs}): {fmt(ct)}")


if __name__ == "__main__":
    main()
