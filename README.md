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

model = AutoModel.from_pretrained("Skovorp/videoprism-base-f16r288-pt", trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained("Skovorp/videoprism-base-f16r288-pt", trust_remote_code=True)

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

Available HF repos: `videoprism-base-f16r288-pt`, `videoprism-large-f8r288-pt`, `videoprism-lvt-base-f16r288-pt`, `videoprism-lvt-large-f8r288-pt`.

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

26 tests pass in ~80s on CPU (5090 box, torch 2.8.0+cu128). Numbers below are from the same run, comparing PyTorch outputs against the JAX golden checkpoint loaded into both stacks on identical inputs.

| Test suite | Tests | What it covers |
|---|---:|---|
| `tests/test_components.py` | 9 | PyTorch unit tests (no checkpoint, no JAX): patchify ordering, MLP residual, attention math invariants, transformer-block identity, full encoder shape / 114M param total / determinism. |
| `tests/test_e2e_parity.py` | 12 | Final-output parity per variant (param count, output shape, cosine sim, abs error). Parameterized across all 4 variants. |
| `tests/test_deep_parity.py` | 5 | Multi-batch + multi-seed + real-video parity, **intermediate parity** (post-`spatial_ln` and post-`temporal_ln`), gradient flow. v1_base only. |

Per-variant final-output parity vs JAX (B=2 random seeds, fp32 throughout):

| Variant | Output shape | max abs diff | mean abs diff | cosine sim |
|---|---|---:|---:|---:|
| `v1_base`       | `(2, 4096, 768)`  | 1.3e-3 | 2.3e-6 | **1.000000** |
| `v1_large`      | `(2, 2048, 1024)` | 2.2e-3 | 1.9e-6 | **1.000000** |
| `lvt_v1_base`   | `(2, 768)`        | 2.7e-7 | 3.0e-8 | **1.000000** |
| `lvt_v1_large`  | `(2, 1024)`       | 8.5e-7 | 7.9e-8 | **1.000000** |

The order-of-magnitude jump in absolute error for the LvT variants is because their output is L2-normalized to unit norm — the values are bounded in `[-1, 1]` rather than the unbounded post-LN distribution of the token-output variants. The `1e-3`-level outliers on the token variants are XLA-vs-PyTorch matmul-fusion-order drift, not a port bug.

`test_deep_parity.py` additionally compares post-`spatial_ln` (after 12 spatial transformer layers) and post-`temporal_ln` (after 4 temporal transformer layers) intermediates for v1_base — both come back at cosine sim = 1.000000 with mean abs error at the fp32 noise floor.

## Inference benchmark

CPU vs GPU, `B=1, T=native, 288²`, mean of 5–10 timed iterations after warmup. PyTorch 2.8.0+cu128 / JAX 0.10.0 / Flax 0.12.7. RTX 5090 (sm_120).

| Variant | JAX (CPU XLA) | PyTorch (CPU eager) | PyTorch (CPU `torch.compile`) | PyTorch (GPU eager) |
|---|---:|---:|---:|---:|
| `v1_base`       | 1.52 s | 2.60 s | **1.42 s** | **29 ms** |
| `v1_large`      | —      | —      | —          | **35 ms** |
| `lvt_v1_base`   | —      | —      | —          | **45 ms** |
| `lvt_v1_large`  | —      | —      | —          | **42 ms** |

`torch.compile` makes PyTorch ~7% faster than JAX-XLA on CPU. On GPU (5090, eager), `v1_base` runs at ~34 fps for 16-frame clips end-to-end.

## Limitations

- **Spatial position-embedding interpolation is not implemented.** All current variants use 16×16 spatial patches at 288 px, which already matches the trained pos-emb. Calling the model with a non-native spatial resolution will assert.
- **Temporal pos-emb interpolation works** (linear, matching JAX's `bilinear` for 1D), but is not exercised in the parity tests — running at non-native frame counts is on a believed-equivalent path.
- **No upstream `videoprism_v1_giant` support** — Google has not released those weights.
- **LvT text encoder not ported** — by design.

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
├── extract_fixture_jax.py      # regenerates JAX fixtures
└── bench_inference.py          # JAX vs PyTorch (eager + compile) timings
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
