# torch_videoprism

PyTorch port of Google DeepMind's [VideoPrism](https://github.com/google-deepmind/videoprism) `FactorizedEncoder`.

The original release ships JAX/Flax checkpoints only ([upstream README](https://github.com/google-deepmind/videoprism) lists "Add PyTorch model support" as an open TODO). This repo provides:

- A pure-PyTorch implementation of `videoprism_public_v1_base` (114M params, ViT-B/18 @ 288 px, 16 frames).
- A loader that pulls the official Flax `.npz` checkpoint from `google/videoprism-base-f16r288` on HuggingFace and converts it to a PyTorch `state_dict`.
- A test suite that pins parity against the JAX golden reference at the **final output and at intermediate stages**, plus a `torch.hub` entry point for one-line loading.

## Install

```bash
pip install git+https://github.com/Skovorp/torch_videoprism
```

## Quick start

```python
import torch
from torch_videoprism import build_videoprism_v1_base, load_pretrained_weights

model = build_videoprism_v1_base()      # random init, 114.37M params
load_pretrained_weights(model)          # pulls Flax npz from HF + converts in-place
model.eval()

# Input: (B, T=16, H=288, W=288, C=3) in [0, 1]. Note channels-LAST, matching JAX.
video = torch.rand(1, 16, 288, 288, 3)
with torch.no_grad():
    embeddings = model(video)           # (1, 4096, 768)
```

Or via `torch.hub`, no package install required:

```python
encoder = torch.hub.load(
    "Skovorp/torch_videoprism", "videoprism_v1_base",
    pretrained=True, trust_repo=True,
)
```

## Architecture

`FactorizedEncoder` follows ViViT's factorized space-time design:

```
input         (B, T=16, H=288, W=288, C=3) in [0, 1]
  ↓ patchify (P=18)                     -> (B*T, 256, 972)
  ↓ patch_projection: Linear(972, 768)  -> (B*T, 256, 768)
  ↓ + spatial_pos_emb (256, 768)
  ↓ spatial_encoder (12× TransformerBlock, 12 heads, MLP 3072)
  ↓ spatial_ln                          -> (B*T, 256, 768)
  ↓ reshape '(b t) n d -> (b n) t d'
  ↓ + temporal_pos_emb (16, 768)        # interpolated linearly if T != 16
  ↓ temporal_encoder (4× TransformerBlock)
  ↓ temporal_ln                         -> (B*N, 16, 768)
  ↓ reshape '(b n) t d -> b (t n) d'
output        (B, 4096, 768)
```

Each transformer block is pre-LN (`y = x + Attn(LN(x)); z = y + MLP(LN(y))`) with multi-head self-attention that includes Primer-style logit capping (`50.0 * tanh(logits / 50.0)` before softmax). Activation is exact GELU (erf), not approximate.

## Test results

CPU-only run on RunPod RTX 5090 (the 5090's `sm_120` needs torch ≥ 2.6, which isn't on the runpod image — kernels still pass on CPU).

```
24 passed in 73.00s
```

| Suite | Tests | Description |
|---|---|---|
| `tests/test_components.py` | 9 | PyTorch unit tests (no checkpoint): patchify ordering, MLP residual, attention math invariants, transformer-block identity, full encoder shape / 114.37M param total / determinism. |
| `tests/test_e2e_parity.py` | 2 | PyTorch vs JAX **final output** with the loaded checkpoint, B=1. |
| `tests/test_deep_parity.py` | 5 | Multi-batch + multi-seed + real-video parity, **intermediate parity** (post-spatial_ln and post-temporal_ln), gradient flow. |

### Numerical parity vs the JAX golden reference

Three batch elements: synthetic video (decoded with decord, resized to 288²) + two random `[0,1]` samples (seeds 1 and 2).

| Stage | Shape | max abs diff | mean abs diff | cosine similarity |
|---|---|---|---|---|
| `spatial_ln` (post 12 spatial blocks) | (B·T=48, 256, 768) | 6.6e-3 | 3.6e-6 | **1.000000** |
| `temporal_ln` (post 4 temporal blocks) | (B·N=768, 16, 768) | 6.4e-3 | 4.2e-6 | **1.000000** |
| Final output, B=3 | (3, 4096, 768) | 7.7e-3 | 4.5e-6 | **1.000000** |
| Final output, single B=1 (real video row) | (1, 4096, 768) | 7.7e-3 | 1.0e-5 | 1.000000 |

Mean absolute error sits at the fp32 noise floor (~10⁻⁶). The handful of 10⁻³ outliers are XLA-vs-PyTorch matmul-fusion-order drift after 16 stacked transformer layers — they don't change the model's behavior.

## Inference benchmark

CPU, `B=1, T=16, 288²`, mean of 5 timed iterations after warmup. PyTorch 2.11.0+cpu, JAX 0.10.0 / Flax 0.12.7.

| Backend | Forward time | Relative |
|---|---|---|
| JAX/Flax (XLA, JIT-compiled) | 1.52 s ± 0.03 s | 1.00× |
| PyTorch (eager) | 2.60 s ± 0.21 s | 1.71× slower |
| **PyTorch (`torch.compile`, default mode)** | **1.42 s ± 0.10 s** | **0.93× — ~7% faster** |

GPU numbers will be added once the runpod base image catches up to a 5090-compatible torch wheel.

## Regenerating the parity fixtures

The fixtures used by `test_e2e_parity.py` and `test_deep_parity.py` aren't shipped in-repo (they're ~250 MB). To regenerate from the JAX golden reference:

```bash
pip install ".[parity]"      # installs jax, flax, the upstream videoprism package
python tests/extract_fixture_jax.py --out tests/fixtures/jax_fixture_e2e.npz
python tests/extract_fixture_jax.py --out tests/fixtures/jax_fixture_multi.npz \
    --seeds 0,1,2 --video-path path/to/some_video.mp4
pytest tests/
```

## Layout

```
torch_videoprism/
├── __init__.py            # public API
├── model.py               # FactorizedEncoder + building blocks
└── weights.py             # Flax (.npz) → PyTorch state_dict converter
tests/
├── test_components.py     # unit tests (no checkpoint, no JAX)
├── test_e2e_parity.py     # final-output parity, B=1
├── test_deep_parity.py    # intermediates, multi-batch, gradient flow
├── extract_fixture_jax.py # regenerates JAX fixtures
└── bench_inference.py     # JAX vs PyTorch (eager + compile) timings
hubconf.py                 # torch.hub entry point
pyproject.toml
LICENSE                    # Apache-2.0
```

## License

Apache 2.0 — same as the upstream [VideoPrism](https://github.com/google-deepmind/videoprism). The released checkpoint is by Google DeepMind; this repo only contains the PyTorch model code and a Flax-→-PyTorch converter.

## Citation

If you use this port, please cite the original VideoPrism paper:

```bibtex
@inproceedings{zhao2024videoprism,
  title = {VideoPrism: A Foundational Visual Encoder for Video Understanding},
  author = {Zhao, Long and Gundavarapu, Nitesh B. and Yuan, Liangzhe and Zhou, Hao and Yan, Shen and Sun, Jennifer J. and Friedman, Luke and Qian, Rui and Weyand, Tobias and Zhao, Yue and Hornung, Rachel and Schroff, Florian and Yang, Ming-Hsuan and Ross, David A. and Wang, Huisheng and Adam, Hartwig and Sirotenko, Mikhail and Liu, Ting and Gong, Boqing},
  booktitle = {ICML},
  year = {2024},
}
```
