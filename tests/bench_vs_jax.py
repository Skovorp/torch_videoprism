"""GPU benchmark: PyTorch port vs JAX golden reference.

Forward and forward+backward for each of the 4 variants × 3 batch sizes,
fp32 throughout. Both stacks run on the same GPU (RTX 5090 in our case).

Run inside the JAX venv (which has both jax[cuda12] and torch installed).
"""
from __future__ import annotations

import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch


# ---------------------------------------------------------------------------
# PyTorch helpers
# ---------------------------------------------------------------------------


def bench_torch_fwd(compiled_model, x, *, iters=8, warmup=4):
    """Forward-only timing. Caller must pass a model that is ONLY ever invoked
    here (i.e. always inside `torch.no_grad()`); that way Dynamo compiles it
    once for grad_mode=False and never invalidates the cache."""
    with torch.no_grad():
        for _ in range(warmup):
            _ = compiled_model(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = compiled_model(x)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return statistics.mean(ts) * 1000, statistics.stdev(ts) * 1000


def bench_torch_fwd_bwd(compiled_model, params, x, *, iters=8, warmup=4):
    """Forward + full backward timing. Caller must pass a separate compiled
    model (NOT shared with the fwd-only one) so Dynamo doesn't see a grad_mode
    flip between bench calls and recompile to cap."""
    def _step():
        for p in params:
            if p.grad is not None:
                p.grad = None
        out = compiled_model(x)
        loss = out.sum() if isinstance(out, torch.Tensor) else out.last_hidden_state.sum()
        loss.backward()
        return loss

    for _ in range(warmup):
        _step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _step()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.mean(ts) * 1000, statistics.stdev(ts) * 1000


# ---------------------------------------------------------------------------
# JAX helpers
# ---------------------------------------------------------------------------


def make_jax_fns(model_name: str):
    """Return (params, fwd_fn, fwd_bwd_fn) for the given upstream model."""
    from videoprism import models as vp

    flax_model = vp.get_model(model_name)
    state = vp.load_pretrained_weights(model_name)

    if "lvt" in model_name:
        def loss_fn(params, x):
            v_emb, _, _ = flax_model.apply(
                {"params": params}, inputs=x, text_token_ids=None, text_paddings=None,
                train=False, normalize=True,
            )
            return v_emb.sum()
        def fwd_fn(params, x):
            v_emb, _, _ = flax_model.apply(
                {"params": params}, inputs=x, text_token_ids=None, text_paddings=None,
                train=False, normalize=True,
            )
            return v_emb
    else:
        def loss_fn(params, x):
            out, _ = flax_model.apply({"params": params}, x, train=False)
            return out.sum()
        def fwd_fn(params, x):
            out, _ = flax_model.apply({"params": params}, x, train=False)
            return out

    fwd_jit = jax.jit(fwd_fn)
    grad_fn = jax.value_and_grad(loss_fn)
    fwd_bwd_jit = jax.jit(grad_fn)
    return state["params"], fwd_jit, fwd_bwd_jit


def bench_jax_fn(fn, params, x, iters=8, warmup=3, returns_grads: bool = False):
    """Generic JAX benchmark with proper sync via `block_until_ready`."""
    # Warmup (also triggers JIT compile).
    for _ in range(warmup):
        out = fn(params, x)
        if returns_grads:
            loss, grads = out
            jax.block_until_ready(loss)
            jax.tree_util.tree_map(jax.block_until_ready, grads)
        else:
            jax.block_until_ready(out)

    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(params, x)
        if returns_grads:
            loss, grads = out
            jax.block_until_ready(loss)
            jax.tree_util.tree_map(jax.block_until_ready, grads)
        else:
            jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return statistics.mean(ts) * 1000, statistics.stdev(ts) * 1000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run benchmark for JAX-only OR PyTorch-only (selected via --side).

    JAX and PyTorch stacks together hit OOM on a 32 GB 5090 once `torch.compile`
    + autograd kick in (each stack hoards memory for its own kernels). The
    fair-comparison answer is to give each one the full GPU in its own process,
    then merge results. This script does either side; pass `--side jax` /
    `--side torch` (default: both, sequentially in two subprocess invocations).
    """
    import argparse
    import subprocess
    import sys

    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=["jax", "torch", "both"], default="both")
    p.add_argument("--out", default="/tmp/bench_vs_jax.json")
    args = p.parse_args()

    if args.side == "both":
        # Spawn two child processes — each gets the full GPU.
        for side in ("jax", "torch"):
            print(f"\n=== running --side {side} in subprocess ===")
            r = subprocess.run([sys.executable, __file__, "--side", side, "--out", f"/tmp/bench_vs_jax_{side}.json"], check=True)
        # Merge.
        merged = {}
        for side in ("jax", "torch"):
            with open(f"/tmp/bench_vs_jax_{side}.json") as f:
                part = json.load(f)
            for variant, by_b in part.items():
                merged.setdefault(variant, {})
                for b, fields in by_b.items():
                    merged[variant].setdefault(b, {}).update(fields)
        with open(args.out, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"\nMerged results saved to {args.out}")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}, jax devices: {jax.devices()}, torch {torch.__version__}, side={args.side}")

    torch.set_float32_matmul_precision("high")

    from torch_videoprism import (
        build_videoprism_lvt_v1_base, build_videoprism_lvt_v1_large,
        build_videoprism_v1_base, build_videoprism_v1_large,
        load_pretrained_weights,
    )

    VARIANTS = [
        ("v1_base",      "videoprism_public_v1_base",      build_videoprism_v1_base,     16),
        ("v1_large",     "videoprism_public_v1_large",     build_videoprism_v1_large,     8),
        ("lvt_v1_base",  "videoprism_lvt_public_v1_base",  build_videoprism_lvt_v1_base, 16),
        ("lvt_v1_large", "videoprism_lvt_public_v1_large", build_videoprism_lvt_v1_large, 8),
    ]
    BATCHES = [1, 4, 8]

    print()
    header = f"{'variant':<14s} {'B':>3s}  {'jax fwd':>14s}  {'torch fwd':>14s}  {'jax fwd+bwd':>14s}  {'torch fwd+bwd':>15s}"
    print(header)
    results = {}

    # Bigger variants OOM at B=8 when both JAX and PyTorch are sharing the GPU
    # (param count + activations for backward + JAX's preallocation). Drop the
    # last batch in those rows.
    print()
    print(f"{'variant':<14s} {'B':>3s}  fwd ms (mean ± std)        fwd+bwd ms (mean ± std)")

    for short_name, full_name, builder, T in VARIANTS:
        print(f"--- {short_name} (T={T}) ---")

        if args.side == "torch":
            # Bump cache_size_limit so Dynamo doesn't bail if shape/grad-mode
            # specializations stack up across the per-(variant, batch) loop.
            torch._dynamo.config.cache_size_limit = 64

            torch_model = builder().cuda()
            load_pretrained_weights(torch_model, model_name=full_name)
            # TWO compiled wrappers around the same parameters. Each is only
            # ever called in one grad_mode (fwd-only inside `no_grad`, fwd+bwd
            # outside) — that keeps Dynamo from recompiling on grad_mode flips.
            # `dynamic=False` bakes `cap=50.0` in as a Python constant; with
            # `dynamic=True`, Dynamo lifts it to a 0-d fp64 CPU tensor and
            # mismatches the CUDA graph. Per-batch-size recompile is fine —
            # `cache_size_limit=64` above covers it.
            fwd_model     = torch.compile(torch_model, dynamic=False)
            fwd_bwd_model = torch.compile(torch_model, dynamic=False)
            params = list(torch_model.parameters())
        else:
            jax_params, jax_fwd, jax_fwd_bwd = make_jax_fns(full_name)

        for B in BATCHES:
            shape = (B, T, 288, 288, 3)
            entry = {}

            try:
                if args.side == "torch":
                    x = torch.rand(*shape, dtype=torch.float32, device="cuda")
                    f_m, f_s = bench_torch_fwd(fwd_model, x)
                    b_m, b_s = bench_torch_fwd_bwd(fwd_bwd_model, params, x)
                    entry["torch_fwd"] = [f_m, f_s]
                    entry["torch_fwd_bwd"] = [b_m, b_s]
                else:
                    x = jnp.asarray(np.random.RandomState(0).rand(*shape).astype(np.float32))
                    f_m, f_s = bench_jax_fn(jax_fwd, jax_params, x, returns_grads=False)
                    b_m, b_s = bench_jax_fn(jax_fwd_bwd, jax_params, x, returns_grads=True)
                    entry["jax_fwd"] = [f_m, f_s]
                    entry["jax_fwd_bwd"] = [b_m, b_s]
            except (torch.cuda.OutOfMemoryError, MemoryError) as e:
                print(f"{short_name:<14s} {B:>3d}  OOM ({type(e).__name__}) — skipping")
                if args.side == "torch":
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                # JAX runtime OOM surfaces as JaxRuntimeError; catch broadly.
                if "RESOURCE_EXHAUSTED" in str(e) or "OutOfMemory" in str(e):
                    print(f"{short_name:<14s} {B:>3d}  OOM — skipping")
                    continue
                raise

            results.setdefault(short_name, {}).setdefault(str(B), {}).update(entry)
            print(
                f"{short_name:<14s} {B:>3d}  "
                f"{f_m:>8.1f} ± {f_s:>4.1f} ms        "
                f"{b_m:>8.1f} ± {b_s:>4.1f} ms"
            )

        if args.side == "torch":
            del torch_model, fwd_model, fwd_bwd_model, params
        else:
            del jax_params, jax_fwd, jax_fwd_bwd
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{args.side} results saved to {args.out}")


if __name__ == "__main__":
    main()
