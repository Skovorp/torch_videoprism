"""Full GPU benchmark across all 4 variants × 3 batch sizes × 3 dtype/compile modes."""
import json
import statistics
import time

import torch

from torch_videoprism import (
    build_videoprism_lvt_v1_base,
    build_videoprism_lvt_v1_large,
    build_videoprism_v1_base,
    build_videoprism_v1_large,
)


def bench(model, x, iters=10, warmup=3):
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


torch.set_float32_matmul_precision("high")

VARIANTS = [
    ("v1_base",      build_videoprism_v1_base,      16),
    ("v1_large",     build_videoprism_v1_large,      8),
    ("lvt_v1_base",  build_videoprism_lvt_v1_base,  16),
    ("lvt_v1_large", build_videoprism_lvt_v1_large,  8),
]
BATCHES = [1, 4, 8]

print(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
print()
print(f"{'variant':<14s} {'B':>3s}  {'fp32 eager':>15s}  {'fp32 compile':>16s}  {'fp16 eager':>15s}")
results = {}

for name, fn, T in VARIANTS:
    print(f"--- {name} (T={T}) ---")
    base_model = fn().cuda().eval()
    compiled_model = torch.compile(base_model, mode="default")
    fp16_model = fn().cuda().half().eval()

    for B in BATCHES:
        x32 = torch.rand(B, T, 288, 288, 3, dtype=torch.float32, device="cuda")
        x16 = x32.half()

        m32, s32 = bench(base_model, x32)
        mc, sc   = bench(compiled_model, x32, warmup=4)
        m16, s16 = bench(fp16_model, x16)

        results.setdefault(name, {})[B] = {
            "fp32_eager":   [m32, s32],
            "fp32_compile": [mc, sc],
            "fp16_eager":   [m16, s16],
        }
        print(
            f"{name:<14s} {B:>3d}  {m32:>10.1f}±{s32:>3.1f}ms  {mc:>11.1f}±{sc:>3.1f}ms  {m16:>10.1f}±{s16:>3.1f}ms"
        )

    del base_model, compiled_model, fp16_model
    torch.cuda.empty_cache()

with open("/tmp/bench_full.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to /tmp/bench_full.json")
