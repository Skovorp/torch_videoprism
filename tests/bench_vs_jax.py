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


def bench_torch_fwd(model, x, iters=8, warmup=3):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return statistics.mean(ts) * 1000, statistics.stdev(ts) * 1000


def bench_torch_fwd_bwd(model, x, iters=8, warmup=3):
    """One full forward + backward step. No optimizer step (pure ops only)."""
    model.train()  # eval is fine too — we have no dropout — but match JAX semantics

    def _step():
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        out = model(x)
        # Use a tensor we can call .last_hidden_state on if HF wrapper, else raw tensor.
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
    print(f"GPU: {torch.cuda.get_device_name(0)}, jax devices: {jax.devices()}, torch {torch.__version__}")

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

    for short_name, full_name, builder, T in VARIANTS:
        print(f"--- {short_name} (T={T}) ---")
        # Build & load weights once per variant.
        torch_model = builder().cuda().eval()
        load_pretrained_weights(torch_model, model_name=full_name)
        torch_model_train = builder().cuda().train()
        load_pretrained_weights(torch_model_train, model_name=full_name)

        jax_params, jax_fwd, jax_fwd_bwd = make_jax_fns(full_name)

        for B in BATCHES:
            shape = (B, T, 288, 288, 3)
            x_torch = torch.rand(*shape, dtype=torch.float32, device="cuda")
            x_jax = jnp.asarray(x_torch.cpu().numpy())

            j_fwd, sj_fwd = bench_jax_fn(jax_fwd, jax_params, x_jax, returns_grads=False)
            t_fwd, st_fwd = bench_torch_fwd(torch_model, x_torch)
            j_bwd, sj_bwd = bench_jax_fn(jax_fwd_bwd, jax_params, x_jax, returns_grads=True)
            t_bwd, st_bwd = bench_torch_fwd_bwd(torch_model_train, x_torch)

            results.setdefault(short_name, {})[B] = {
                "jax_fwd":           [j_fwd, sj_fwd],
                "torch_fwd":         [t_fwd, st_fwd],
                "jax_fwd_bwd":       [j_bwd, sj_bwd],
                "torch_fwd_bwd":     [t_bwd, st_bwd],
            }
            print(
                f"{short_name:<14s} {B:>3d}  "
                f"{j_fwd:>9.1f}±{sj_fwd:>3.1f}ms  "
                f"{t_fwd:>9.1f}±{st_fwd:>3.1f}ms  "
                f"{j_bwd:>9.1f}±{sj_bwd:>3.1f}ms  "
                f"{t_bwd:>10.1f}±{st_bwd:>3.1f}ms"
            )

        del torch_model, torch_model_train, jax_params, jax_fwd, jax_fwd_bwd
        torch.cuda.empty_cache()

    with open("/tmp/bench_vs_jax.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /tmp/bench_vs_jax.json")


if __name__ == "__main__":
    main()
