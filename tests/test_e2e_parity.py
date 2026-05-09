"""End-to-end parity test: PyTorch port vs JAX golden reference.

Reads a fixture saved by `scripts/jax_extract_fixture.py` (input + JAX outputs
at several intermediate stages) and verifies the PyTorch port produces the
same outputs after loading the same Flax checkpoint.

The fixture path can be overridden via the `VIDEOPRISM_JAX_FIXTURE` env var
(default: `tests/fixtures/jax_fixture_e2e.npz`). The Flax checkpoint
path can be overridden via `VIDEOPRISM_NPZ` (default: HF download).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torch_videoprism.model import build_videoprism_v1_base
from torch_videoprism.weights import load_pretrained_weights


_DEFAULT_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "jax_fixture_e2e.npz")
FIXTURE = os.environ.get("VIDEOPRISM_JAX_FIXTURE", _DEFAULT_FIXTURE)
NPZ = os.environ.get("VIDEOPRISM_NPZ")  # if None, falls back to HF download


def _video_to_torch_input(video_npy: np.ndarray) -> torch.Tensor:
    """Fixture stores (B, T, H, W, C) — same as the model expects."""
    return torch.from_numpy(np.asarray(video_npy).copy())


@pytest.fixture(scope="module")
def loaded_model():
    if not os.path.exists(FIXTURE):
        pytest.skip(f"fixture not found at {FIXTURE}; run scripts/jax_extract_fixture.py first")
    model = build_videoprism_v1_base()
    kwargs = {"checkpoint_path": NPZ} if NPZ else {}
    load_pretrained_weights(model, **kwargs)
    model.eval()
    return model


@pytest.fixture(scope="module")
def fixture_data():
    if not os.path.exists(FIXTURE):
        pytest.skip(f"fixture not found at {FIXTURE}")
    return np.load(FIXTURE)


def _summarize(name: str, ref: np.ndarray, got: torch.Tensor):
    diff = (got.detach().cpu().numpy() - ref).astype(np.float64)
    abs_err = np.abs(diff)
    cos = float(
        np.sum(got.detach().cpu().numpy().reshape(-1) * ref.reshape(-1))
        / (np.linalg.norm(got.detach().cpu().numpy()) * np.linalg.norm(ref) + 1e-12)
    )
    print(
        f"[{name}] shape={tuple(got.shape)} "
        f"max_abs={abs_err.max():.3e} mean_abs={abs_err.mean():.3e} cos={cos:.6f}"
    )
    return abs_err, cos


def test_full_output_parity(loaded_model, fixture_data):
    video = _video_to_torch_input(fixture_data["input"])
    with torch.no_grad():
        out = loaded_model(video)
    ref = fixture_data["output"]
    assert out.shape == ref.shape
    abs_err, cos = _summarize("final", ref, out)
    # Cross-framework fp32 parity tolerance: matmul order differs slightly.
    assert cos > 0.9999, f"cosine sim too low: {cos}"
    assert abs_err.max() < 5e-3, f"max abs diff too high: {abs_err.max()}"


def test_param_count(loaded_model):
    n = sum(p.numel() for p in loaded_model.parameters())
    # Trained checkpoint must round-trip to the same param total as random init.
    assert 114_300_000 < n < 114_500_000, n
