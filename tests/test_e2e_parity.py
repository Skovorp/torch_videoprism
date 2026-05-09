"""End-to-end parity test: PyTorch port vs JAX golden reference.

Parameterized across the available variants (`videoprism_public_v1_base`,
`videoprism_public_v1_large`). Each case loads a fixture saved by
`tests/extract_fixture_jax.py` plus the Flax checkpoint, runs the PyTorch
port forward, and compares the final-output tensor.

Fixture layout (default tests/fixtures/<model>.npz, override via env var):
  VIDEOPRISM_FIXTURE_BASE   : tests/fixtures/v1_base.npz
  VIDEOPRISM_FIXTURE_LARGE  : tests/fixtures/v1_large.npz
The Flax checkpoint path can also be overridden:
  VIDEOPRISM_NPZ_BASE
  VIDEOPRISM_NPZ_LARGE
(if unset, the loader downloads from HuggingFace).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from torch_videoprism import build_videoprism, CONFIGS
from torch_videoprism.weights import load_pretrained_weights


@dataclass(frozen=True)
class VariantSpec:
    model_name: str          # e.g. "videoprism_public_v1_base"
    expected_params: tuple   # (low, high) inclusive bounds for total param count
    expected_output: tuple   # output shape per (B=1) sample (excluding the batch axis)
    fixture_env: str
    npz_env: str
    fixture_default: str


VARIANTS = {
    "base":  VariantSpec(
        model_name="videoprism_public_v1_base",
        expected_params=(114_300_000, 114_500_000),
        expected_output=(4096, 768),  # 16 frames * 256 spatial patches, D=768
        fixture_env="VIDEOPRISM_FIXTURE_BASE",
        npz_env="VIDEOPRISM_NPZ_BASE",
        fixture_default=os.path.join(os.path.dirname(__file__), "fixtures", "v1_base.npz"),
    ),
    "large": VariantSpec(
        model_name="videoprism_public_v1_large",
        expected_params=(353_900_000, 354_100_000),
        expected_output=(2048, 1024),  # 8 frames * 256 spatial patches, D=1024
        fixture_env="VIDEOPRISM_FIXTURE_LARGE",
        npz_env="VIDEOPRISM_NPZ_LARGE",
        fixture_default=os.path.join(os.path.dirname(__file__), "fixtures", "v1_large.npz"),
    ),
}


def _summarize(name: str, ref: np.ndarray, got: torch.Tensor) -> tuple[float, float]:
    got_np = got.detach().cpu().float().numpy()
    diff = np.abs(got_np - ref).astype(np.float64)
    flat_g = got_np.flatten().astype(np.float64)
    flat_r = ref.flatten().astype(np.float64)
    cos = float(np.dot(flat_g, flat_r) / (np.linalg.norm(flat_g) * np.linalg.norm(flat_r) + 1e-12))
    print(
        f"[{name}] shape={tuple(got.shape)} "
        f"max_abs={diff.max():.3e} mean_abs={diff.mean():.3e} cos={cos:.6f}"
    )
    return float(diff.max()), cos


@pytest.fixture(scope="module", params=sorted(VARIANTS), ids=sorted(VARIANTS))
def variant(request):
    return VARIANTS[request.param]


@pytest.fixture(scope="module")
def fixture_data(variant):
    path = os.environ.get(variant.fixture_env, variant.fixture_default)
    if not os.path.exists(path):
        pytest.skip(f"fixture for {variant.model_name} not found at {path} — "
                    f"run tests/extract_fixture_jax.py --model {variant.model_name}")
    return np.load(path)


@pytest.fixture(scope="module")
def loaded_model(variant, fixture_data):
    """Build the variant's FactorizedEncoder and load the matching weights."""
    npz_path = os.environ.get(variant.npz_env)
    model = build_videoprism(variant.model_name)
    kwargs = {"checkpoint_path": npz_path} if npz_path else {"model_name": variant.model_name}
    load_pretrained_weights(model, **kwargs)
    model.eval()
    return model


def test_param_count(variant, loaded_model):
    n = sum(p.numel() for p in loaded_model.parameters())
    lo, hi = variant.expected_params
    assert lo <= n <= hi, f"{variant.model_name}: {n} params outside [{lo}, {hi}]"


def test_output_shape(variant, fixture_data):
    """The fixture's stored output should already match the model's expected shape."""
    out = fixture_data["output"]
    b = out.shape[0]
    assert out.shape[1:] == variant.expected_output, (
        f"{variant.model_name} fixture: shape {out.shape[1:]} != expected {variant.expected_output}"
    )


def test_full_output_parity(variant, loaded_model, fixture_data):
    video = torch.from_numpy(fixture_data["input"].copy())
    with torch.no_grad():
        out = loaded_model(video)
    ref = fixture_data["output"]
    assert out.shape == ref.shape
    max_abs, cos = _summarize(variant.model_name, ref, out)
    # fp32 cross-framework drift: cosine similarity is the load-bearing check.
    assert cos > 0.9999, f"cosine sim too low: {cos}"
    # Per-element max abs cap is loose to allow XLA-vs-PyTorch fusion-order outliers.
    assert max_abs < 1e-2, f"max abs diff too high: {max_abs}"
