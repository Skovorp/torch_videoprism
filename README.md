# torch_videoprism

PyTorch port of Google DeepMind's [VideoPrism](https://github.com/google-deepmind/videoprism). The original release ships JAX/Flax weights only — upstream's [README](https://github.com/google-deepmind/videoprism) lists "Add PyTorch model support" as an open TODO. This repo provides:

- A pure-PyTorch implementation of all four publicly-released VideoPrism configurations.
- A Flax-→-PyTorch checkpoint converter that pulls the `.npz` from HuggingFace and produces a numerically-equivalent PyTorch `state_dict`.
- A test suite that pins parity against the JAX reference at the **final output and at intermediate stages** (cosine similarity = 1.000000 on every variant).
- A `torch.hub` entry point per variant, and a benchmark script.

## Variants

| Name | Class | Params | Frames × Res | Output |
|---|---|---:|---:|---|
| `videoprism_v1_base`     | `FactorizedEncoder`      | 114.4M | 16 × 288² | `(B, 4096, 768)` token sequence |
| `videoprism_v1_large`    | `FactorizedEncoder`      | 354.0M |  8 × 288² | `(B, 2048, 1024)` token sequence |
| `videoprism_lvt_v1_base` | `FactorizedVideoEncoder` | 138.0M | 16 × 288² | `(B, 768)` L2-normalized embedding |
| `videoprism_lvt_v1_large`| `FactorizedVideoEncoder` | 396.0M |  8 × 288² | `(B, 1024)` L2-normalized embedding |

The two `lvt_*` variants are the vision branch of the upstream `FactorizedVideoCLIP` model — text encoder is intentionally not ported. Upstream also has a `videoprism_v1_giant` config (1408 dim, 40 layers), but no public weights have been released, so it isn't included here.

## Install

```bash
pip install git+https://github.com/Skovorp/torch_videoprism
```

## Quick start

Three loading paths — pick whichever you prefer.

**HuggingFace `AutoModel`** (recommended — one line, handles preprocessing too):

```python
import torch
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("sposiboh/videoprism-base-f16r288-pt", trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained("sposiboh/videoprism-base-f16r288-pt", trust_remote_code=True)

# Accepts: a video file path, a list of PIL Images, np.array (T, H, W, 3) of uint8 or float,
# or a torch.Tensor of one of those shapes. Frames are sampled uniformly to the model's
# native frame count and resized to 288x288. Pixels assumed in [0, 255]; pass do_rescale=False
# if your input is already in [0, 1].
inputs = processor(videos="my_clip.mp4", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
embedding = out.last_hidden_state          # (1, 4096, 768) for base
# For LvT variants, the attribute is `out.video_embeds` and the shape is (B, model_dim).
```

All 4 variants on HF (collection: [sposiboh/videoprism-pytorch-port](https://huggingface.co/collections/sposiboh/videoprism-pytorch-port-69ffbf1d5fa09a808dcfa507)):
[`sposiboh/videoprism-base-f16r288-pt`](https://huggingface.co/sposiboh/videoprism-base-f16r288-pt) ·
[`sposiboh/videoprism-large-f8r288-pt`](https://huggingface.co/sposiboh/videoprism-large-f8r288-pt) ·
[`sposiboh/videoprism-lvt-base-f16r288-pt`](https://huggingface.co/sposiboh/videoprism-lvt-base-f16r288-pt) ·
[`sposiboh/videoprism-lvt-large-f8r288-pt`](https://huggingface.co/sposiboh/videoprism-lvt-large-f8r288-pt).

**`torch.hub.load`** (no `pip install` required):

```python
encoder = torch.hub.load(
    "Skovorp/torch_videoprism", "videoprism_v1_base",
    pretrained=True, trust_repo=True,
)
# Other entry points: videoprism_v1_large, videoprism_lvt_v1_base, videoprism_lvt_v1_large.
```

**Direct package import** (most control, no HF dependency):

```python
import torch
from torch_videoprism import build_videoprism, load_pretrained_weights

# Pick any of: videoprism_public_v1_{base,large}
#              videoprism_lvt_public_v1_{base,large}
model = build_videoprism("videoprism_public_v1_base")
load_pretrained_weights(model)        # downloads + converts the Flax .npz from HF
model.eval()

# Input: (B, T, H, W, C) in [0, 1]. Channels-LAST, matching the JAX API.
video = torch.rand(1, 16, 288, 288, 3)
with torch.no_grad():
    out = model(video)                # (1, 4096, 768) for base; (1, 768) for lvt-base
```

## Architecture

**`FactorizedEncoder`** (v1 base / large) follows the ViViT factorized space-time design:

```
input         (B, T, H, W, C) in [0, 1]
  ↓ patchify (P=18)                     -> (B*T, N, P*P*C)
  ↓ patch_projection: Linear            -> (B*T, N, D)
  ↓ + spatial_pos_emb (N, D)
  ↓ spatial_encoder (12 or 24 blocks)   -> (B*T, N, D)
  ↓ spatial_ln
  ↓ reshape '(b t) n d -> (b n) t d'
  ↓ + temporal_pos_emb (T, D)           # linearly interpolated if T differs from native
  ↓ temporal_encoder (4 blocks)         -> (B*N, T, D)
  ↓ temporal_ln
  ↓ reshape '(b n) t d -> b (t n) d'
output        (B, T*N, D)
```

Each transformer block is pre-LN with multi-head self-attention; the attention applies Primer-style logit capping (`50 · tanh(logits / 50)` before softmax). Activation is exact (erf) GELU.

**`FactorizedVideoEncoder`** (lvt v1 base / large) wraps a `FactorizedEncoder`, then runs the token sequence through a 2-layer auxiliary transformer, an attentional pooler (`AttenTokenPooler`, with `PerDimScale` on the query — the only place this scaling appears), and an L2 normalize on the last axis. Output is a single embedding per video.

## Test results

48 tests pass in ~100 s on CPU. Numbers below are from the same run, comparing PyTorch outputs against the JAX golden checkpoint loaded into both stacks on identical inputs.

| Test suite | Tests | What it covers |
|---|---:|---|
| `tests/test_components.py` | 9  | PyTorch unit tests (no checkpoint, no JAX): patchify ordering, MLP residual, attention math invariants, transformer-block identity, full encoder shape / 114 M param total / determinism. |
| `tests/test_e2e_parity.py` | 12 | Final-output parity per variant (param count, output shape, cosine sim, abs error). Parameterized across all 4 variants. |
| `tests/test_deep_parity.py` | 5  | Multi-batch + multi-seed + real-video parity, **intermediate parity** (post-`spatial_ln` and post-`temporal_ln`), gradient flow. v1_base only. |
| `tests/test_pe_parity.py`  | 22 | **Direct & full-model parity** for the spatial (`_interpolate_2d_pos`) and temporal (`_interpolate_1d_pos`) pos-emb interpolation paths. Covers source-vs-target combinations for downsampling, identity (no-op), and upsampling — both at the helper level and through a full forward pass against JAX. |

Per-variant final-output parity vs JAX (B=2 random seeds, fp32 throughout):

| Variant | Output shape | max abs diff | mean abs diff | cosine sim |
|---|---|---:|---:|---:|
| `v1_base`       | `(2, 4096, 768)`  | 1.3e-3 | 2.3e-6 | **1.000000** |
| `v1_large`      | `(2, 2048, 1024)` | 2.2e-3 | 1.9e-6 | **1.000000** |
| `lvt_v1_base`   | `(2, 768)`        | 2.7e-7 | 3.0e-8 | **1.000000** |
| `lvt_v1_large`  | `(2, 1024)`       | 8.5e-7 | 7.9e-8 | **1.000000** |

The order-of-magnitude jump in absolute error for the LvT variants is because their output is L2-normalized to unit norm — the values are bounded in `[-1, 1]` rather than the unbounded post-LN distribution of the token-output variants. The `1e-3`-level outliers on the token variants are XLA-vs-PyTorch matmul-fusion-order drift, not a port bug.

`test_deep_parity.py` additionally compares post-`spatial_ln` (after 12 spatial transformer layers) and post-`temporal_ln` (after 4 temporal transformer layers) intermediates for v1_base — both come back at cosine sim = 1.000000 with mean abs error at the fp32 noise floor.

## Inference benchmark — JAX vs PyTorch on the same GPU

RTX 5090 (sm_120), torch 2.8.0+cu128, JAX 0.10.0 / Flax 0.12.7, fp32 throughout. Each side runs in its own subprocess so it gets the full GPU; both stacks are JIT-compiled (`jax.jit` and `torch.compile(dynamic=False)` over the whole model). Mean ± stdev of 8 timed iterations after warmup. Reproduce with `python tests/bench_vs_jax.py --side both`.

| variant | B | jax fwd | **torch fwd** | jax fwd+bwd | **torch fwd+bwd** |
|---|---:|---:|---:|---:|---:|
| `v1_base`       | 1 |  12.9 ms |   14.6 ms |  53.0 ms |  **45.4 ms** |
| `v1_base`       | 4 |  53.8 ms |  **47.8 ms** | 211.8 ms | **149.1 ms** |
| `v1_base`       | 8 | 112.0 ms |  **92.3 ms** | 430.4 ms | **285.9 ms** |
| `v1_large`      | 1 |  19.4 ms |  **19.1 ms** |  77.5 ms |  **61.6 ms** |
| `v1_large`      | 4 |  77.4 ms |  **74.4 ms** | 301.6 ms | **230.0 ms** |
| `v1_large`      | 8 | 157.4 ms | **142.3 ms** | 604.8 ms | **453.7 ms** |
| `lvt_v1_base`   | 1 |  19.3 ms |  **18.6 ms** |  79.1 ms |  **60.6 ms** |
| `lvt_v1_base`   | 4 |  77.8 ms |  **60.4 ms** | 309.9 ms | **203.3 ms** |
| `lvt_v1_base`   | 8 |  OOM     | **116.7 ms** |   OOM    | **398.5 ms** |
| `lvt_v1_large`  | 1 |  22.5 ms |  **21.5 ms** |  90.5 ms |  **70.2 ms** |
| `lvt_v1_large`  | 4 |  89.6 ms |  **83.2 ms** | 348.1 ms | **260.9 ms** |
| `lvt_v1_large`  | 8 | **181.5 ms** |   OOM   | **693.2 ms** |   OOM    |

Take-aways:
- **Forward**: PyTorch is flat-to-+29 % faster than JAX. Only B=1 v1_base is slightly slower (-13 %) where launch overhead dominates a 15 ms cell.
- **Forward + backward**: PyTorch is +17–52 % faster than JAX across the board. Most of the win comes from the FlexAttention fused backward kernel.
- **Why FlexAttention?** The attention math has a Primer-style logit cap (`50 · tanh(logits / 50)` before softmax) — vanilla `F.scaled_dot_product_attention` doesn't expose a hook for that. `flex_attention(score_mod=…)` does, and is fused via `torch.compile` to match JAX's XLA path.
- **OOMs at B=8** (both directions) are memory-architecture differences, not perf bugs: JAX preallocates its slice up-front (so `lvt_v1_base` runs out before forward completes), PyTorch holds the full activation graph for backward (so `lvt_v1_large` runs out at the end). Both are real-world rough edges, not regressions.

For the CPU-only path (no 5090), see `tests/bench_inference.py` — JAX-XLA and PyTorch+`torch.compile` are within ~7 % on `v1_base`.

## Preprocessing

If you don't want the HF processor, the preprocessing pipeline is short and explicit. The model expects:

- **Shape**: `(B, T, H, W, C)` — channels **last** (matching the JAX API), `H == W`.
- **Frame count**: `T = 16` for `v1_base` / `lvt_v1_base`, `T = 8` for `v1_large` / `lvt_v1_large`. Other values are accepted via 1D linear interpolation of the temporal pos-emb (parity-tested at `T ∈ {8, 12, 24}` for `v1_base`).
- **Spatial size**: native `288×288`. Other sizes (multiples of `patch_size=18`) are accepted via 2D bilinear interpolation of the spatial pos-emb (parity-tested at `144 / 216 / 432 / 576`).
- **Pixel range**: float32 in `[0, 1]`. **No ImageNet mean/std normalization** — just divide raw `uint8` frames by 255.

Minimal framework-agnostic preprocessing (~15 lines):

```python
import numpy as np
import cv2

def preprocess_video(uint8_frames, num_frames=16, image_size=288):
    """Convert a (T_input, H, W, 3) uint8 video to (1, num_frames, image_size, image_size, 3) float32 in [0, 1].

    `uint8_frames` is the decoded video — e.g. `decord.VideoReader(path).get_batch(...).asnumpy()`
    or `np.stack([np.array(pil_img) for pil_img in frames])`.
    """
    T_in = uint8_frames.shape[0]
    idx = np.linspace(0, max(T_in - 1, 0), num=num_frames).astype(int)   # uniform sampling
    sampled = uint8_frames[idx]                                          # (num_frames, H, W, 3)
    resized = np.stack([
        cv2.resize(f, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        for f in sampled
    ])                                                                    # bilinear resize
    return (resized.astype(np.float32) / 255.0)[None]                     # add batch axis, [0,1]
```

That's it — no per-channel normalization, no center crop, no temporal jitter. Feed the result to `model(...)` directly.

## Limitations

- **No upstream `videoprism_v1_giant` support** — Google has not released those weights publicly.
- **LvT text encoder not ported** — by design (this is a vision-only port of the LvT models).
- **Patch size is fixed at 18** — input height & width must be multiples of 18.

## Regenerating the parity fixtures

The parity fixtures aren't shipped in-repo (~250 MB total). Without them, only the 9 unit tests in `tests/test_components.py` run; the 17 parity tests are skipped silently. To regenerate from the JAX reference:

```bash
pip install ".[parity]"      # installs jax, flax, the upstream videoprism package, decord, opencv

# Generate one fixture per variant (single-clip pair of seeds + intermediates for v1_base).
python tests/extract_fixture_jax.py --out tests/fixtures/v1_base.npz       --model videoprism_public_v1_base       --seeds 0,1 --save-intermediates
python tests/extract_fixture_jax.py --out tests/fixtures/v1_large.npz      --model videoprism_public_v1_large      --seeds 0,1
python tests/extract_fixture_jax.py --out tests/fixtures/lvt_v1_base.npz   --model videoprism_lvt_public_v1_base   --seeds 0,1
python tests/extract_fixture_jax.py --out tests/fixtures/lvt_v1_large.npz  --model videoprism_lvt_public_v1_large  --seeds 0,1

# `tests/test_deep_parity.py` needs a separate 3-batch fixture for v1_base.
python tests/extract_fixture_jax.py --out tests/fixtures/v1_base_multi.npz --model videoprism_public_v1_base       --seeds 0,1,2 --save-intermediates

# `tests/test_pe_parity.py` needs its own fixture covering many (h_p, w_p) and T values.
python tests/extract_pe_fixture_jax.py --out tests/fixtures/pe.npz

pytest tests/
```

Each test fixture path can be overridden via env var (defaults are
`tests/fixtures/<name>.npz`):

| Env var | Used by | Default |
|---|---|---|
| `VIDEOPRISM_FIXTURE_BASE`       | `test_e2e_parity.py` (v1 base)         | `tests/fixtures/v1_base.npz`       |
| `VIDEOPRISM_FIXTURE_LARGE`      | `test_e2e_parity.py` (v1 large)        | `tests/fixtures/v1_large.npz`      |
| `VIDEOPRISM_FIXTURE_LVT_BASE`   | `test_e2e_parity.py` (lvt base)        | `tests/fixtures/lvt_v1_base.npz`   |
| `VIDEOPRISM_FIXTURE_LVT_LARGE`  | `test_e2e_parity.py` (lvt large)       | `tests/fixtures/lvt_v1_large.npz`  |
| `VIDEOPRISM_MULTI_FIXTURE`      | `test_deep_parity.py`                  | `tests/fixtures/v1_base_multi.npz` |
| `VIDEOPRISM_PE_FIXTURE`         | `test_pe_parity.py`                    | `tests/fixtures/pe.npz`            |
| `VIDEOPRISM_NPZ_<VARIANT>`      | optional override of the Flax `.npz` checkpoint path (otherwise pulled from HF) |

## Layout

```
torch_videoprism/
├── __init__.py                 # public API
├── model.py                    # FactorizedEncoder, FactorizedVideoEncoder, building blocks
├── weights.py                  # Flax (.npz) -> PyTorch state_dict converter
└── hf/                         # HuggingFace integration (`pip install '.[hf]'`)
    ├── configuration_videoprism.py
    ├── modeling_videoprism.py  # PreTrainedModel wrappers
    ├── processing_videoprism.py # ImageProcessor
    └── build_repo.py           # produces self-contained trust_remote_code repo dirs
tests/
├── test_components.py          # unit tests (no checkpoint, no JAX)
├── test_e2e_parity.py          # final-output parity, all 4 variants
├── test_deep_parity.py         # intermediates, multi-batch, gradient flow (v1_base)
├── test_pe_parity.py           # spatial + temporal PE interpolation parity vs JAX
├── extract_fixture_jax.py      # regenerates the e2e/deep JAX fixtures
├── extract_pe_fixture_jax.py   # regenerates the PE-interpolation JAX fixture
├── bench_inference.py          # CPU benchmark (JAX vs PyTorch eager / compile)
└── bench_inference_full.py     # GPU benchmark (4 variants × batch × dtype/compile)
hubconf.py                      # torch.hub entry points
pyproject.toml                  # extras: [test], [hf], [parity]
LICENSE                         # Apache-2.0
```

## License

Apache 2.0 — same as the upstream [VideoPrism](https://github.com/google-deepmind/videoprism). The released checkpoints are by Google DeepMind; this repo only contains the PyTorch model code and the Flax-→-PyTorch converter.

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
