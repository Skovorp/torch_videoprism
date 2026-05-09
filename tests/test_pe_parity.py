"""Comprehensive parity tests for positional-embedding interpolation.

Three layers of coverage for the interpolation paths in `model.py`:

  1. Direct interpolation parity: feed the same flattened source pos-emb to
     PyTorch `_interpolate_2d_pos` / `_interpolate_1d_pos` and JAX
     `_interpolate_emb_2d` / `_interpolate_emb_1d`, compare element-wise.

  2. Full-model parity at non-native spatial resolutions (144/216/432/576)
     and non-native frame counts (8/12/24) — exercises the interp path inside
     `FactorizedEncoder.forward`.

  3. No-op cases: at native (16×16, T=16) the interpolation should be the
     identity, so PyTorch output should match JAX bit-for-bit (as before the
     refactor).

Fixture: `tests/extract_pe_fixture_jax.py` — env var
`VIDEOPRISM_PE_FIXTURE` overrides the path.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torch_videoprism import build_videoprism_v1_base, load_pretrained_weights
from torch_videoprism.model import _interpolate_2d_pos, _interpolate_1d_pos


_DEFAULT_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "pe.npz")
FIXTURE = os.environ.get("VIDEOPRISM_PE_FIXTURE", _DEFAULT_FIXTURE)


@pytest.fixture(scope="module")
def fixture_data():
    if not os.path.exists(FIXTURE):
        pytest.skip(f"PE fixture not found at {FIXTURE}; run tests/extract_pe_fixture_jax.py")
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def loaded_model():
    if not os.path.exists(FIXTURE):
        pytest.skip(f"PE fixture not found at {FIXTURE}")
    m = build_videoprism_v1_base()
    load_pretrained_weights(m)
    m.eval()
    return m


def _close(name: str, ref: np.ndarray, got: torch.Tensor | np.ndarray):
    got_np = got.detach().cpu().float().numpy() if isinstance(got, torch.Tensor) else got
    diff = np.abs(got_np - ref).astype(np.float64)
    flat_g = got_np.flatten().astype(np.float64)
    flat_r = ref.flatten().astype(np.float64)
    cos = float(np.dot(flat_g, flat_r) / (np.linalg.norm(flat_g) * np.linalg.norm(flat_r) + 1e-12))
    print(f"[{name}] max_abs={diff.max():.3e} mean_abs={diff.mean():.3e} cos={cos:.7f}")
    return float(diff.max()), cos


# --------------------------------------------------------------------------
# 1. Direct interpolation parity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hp,wp", [(8, 8), (12, 12), (16, 16), (20, 20), (24, 24), (32, 32)])
def test_interpolate_2d_matches_jax(fixture_data, hp, wp):
    src = torch.from_numpy(fixture_data["source_2d"].copy())  # (256, 768)
    ref = fixture_data[f"target_2d_{hp}x{wp}"]
    got = _interpolate_2d_pos(src, (16, 16), (hp, wp))
    assert got.shape == (hp * wp, src.shape[-1])
    max_abs, cos = _close(f"2D ({hp}x{wp})", ref, got)
    # Both `jax.image.resize(method='bilinear')` and
    # `F.interpolate(mode='bilinear', align_corners=False)` use half-pixel-center
    # — they should be near-identical to fp32 noise floor.
    assert cos > 0.99999, f"2D ({hp}x{wp}) cos={cos}"
    assert max_abs < 1e-5, f"2D ({hp}x{wp}) max_abs={max_abs}"


def test_interpolate_2d_native_is_identity(fixture_data):
    """At source==target the interpolation should be the literal source tensor (no copy)."""
    src = torch.from_numpy(fixture_data["source_2d"].copy())
    out = _interpolate_2d_pos(src, (16, 16), (16, 16))
    torch.testing.assert_close(out, src, rtol=0, atol=0)


@pytest.mark.parametrize("tlen", [4, 8, 12, 16, 24, 32])
def test_interpolate_1d_matches_jax(fixture_data, tlen):
    src = torch.from_numpy(fixture_data["source_1d"].copy())  # (16, 768)
    ref = fixture_data[f"target_1d_{tlen}"]
    got = _interpolate_1d_pos(src, tlen)
    assert got.shape == (tlen, src.shape[-1])
    max_abs, cos = _close(f"1D (len={tlen})", ref, got)
    assert cos > 0.99999, f"1D len={tlen} cos={cos}"
    assert max_abs < 1e-5, f"1D len={tlen} max_abs={max_abs}"


def test_interpolate_1d_native_is_identity(fixture_data):
    src = torch.from_numpy(fixture_data["source_1d"].copy())
    out = _interpolate_1d_pos(src, 16)
    torch.testing.assert_close(out, src, rtol=0, atol=0)


# --------------------------------------------------------------------------
# 2. Full-model parity at non-native sizes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("img_size", [144, 216, 288, 432, 576])
def test_full_forward_parity_non_native_spatial(loaded_model, fixture_data, img_size):
    """Model run at non-native image_size matches JAX. T=16 native here so we
    isolate the 2D-PE-interpolation path."""
    inp = torch.from_numpy(fixture_data[f"full_{img_size}_input"].copy())
    ref = fixture_data[f"full_{img_size}"]
    with torch.no_grad():
        out = loaded_model(inp)
    assert out.shape == ref.shape
    max_abs, cos = _close(f"full img_size={img_size}", ref, out)
    # Small-cap is loose because a 12-layer spatial encoder amplifies any
    # residual interpolation drift; cos is the load-bearing check.
    assert cos > 0.9999, f"img_size={img_size} cos={cos}"
    assert max_abs < 2e-2, f"img_size={img_size} max_abs={max_abs}"


@pytest.mark.parametrize("n_frames", [8, 12, 24])
def test_full_forward_parity_non_native_temporal(loaded_model, fixture_data, n_frames):
    """Model run at non-native T (with native 288 px). Isolates the 1D-temporal-PE path."""
    inp = torch.from_numpy(fixture_data[f"full_T{n_frames}_input"].copy())
    ref = fixture_data[f"full_T{n_frames}"]
    with torch.no_grad():
        out = loaded_model(inp)
    assert out.shape == ref.shape
    max_abs, cos = _close(f"full T={n_frames}", ref, out)
    assert cos > 0.9999, f"T={n_frames} cos={cos}"
    assert max_abs < 2e-2, f"T={n_frames} max_abs={max_abs}"
