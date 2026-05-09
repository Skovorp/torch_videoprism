"""Deep parity tests: PyTorch port vs JAX golden reference at multiple stages.

Uses the multi-batch fixture saved by `extract_fixture_jax.py --seeds 0,1,2
--video-path tests/fixtures/videos/synthetic_00.mp4`. Tests:

  - intermediate spatial_features parity (after the 12-block spatial encoder + spatial_ln)
  - intermediate temporal_out parity (after the 4-block temporal encoder + temporal_ln,
    pre final reshape)
  - final-output parity for B=3 (multi-batch, multi-seed, includes a real-video sample)
  - per-batch-element parity (each row of B independently matches the JAX run)
  - gradient flow: backward through the full model, gradients reach all param groups

Set `VIDEOPRISM_NPZ` to the cached Flax checkpoint path and
`VIDEOPRISM_MULTI_FIXTURE` to the .npz path produced by the extractor.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from torch_videoprism.model import build_videoprism_v1_base
from torch_videoprism.weights import load_pretrained_weights


_DEFAULT_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "jax_fixture_multi.npz"
)
FIXTURE = os.environ.get("VIDEOPRISM_MULTI_FIXTURE", _DEFAULT_FIXTURE)
NPZ = os.environ.get("VIDEOPRISM_NPZ")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_data():
    if not os.path.exists(FIXTURE):
        pytest.skip(f"multi fixture not found at {FIXTURE}")
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def loaded_model():
    if not os.path.exists(FIXTURE):
        pytest.skip(f"multi fixture not found at {FIXTURE}")
    model = build_videoprism_v1_base()
    kwargs = {"checkpoint_path": NPZ} if NPZ else {}
    load_pretrained_weights(model, **kwargs)
    model.eval()
    return model


@contextmanager
def _capture(module: torch.nn.Module):
    """Forward-hook context manager that captures the output of `module`."""
    captured: list[torch.Tensor] = []

    def hook(_mod, _inp, out):
        captured.append(out.detach())

    h = module.register_forward_hook(hook)
    try:
        yield captured
    finally:
        h.remove()


def _close(name: str, ref: np.ndarray, got: torch.Tensor, *, max_atol: float, min_cos: float):
    got_np = got.detach().cpu().float().numpy()
    diff = np.abs(got_np - ref).astype(np.float64)
    flat_g = got_np.reshape(-1).astype(np.float64)
    flat_r = ref.reshape(-1).astype(np.float64)
    cos = float(np.dot(flat_g, flat_r) / (np.linalg.norm(flat_g) * np.linalg.norm(flat_r) + 1e-12))
    print(
        f"[{name}] shape={tuple(got.shape)} max_abs={diff.max():.3e} "
        f"mean_abs={diff.mean():.3e} cos={cos:.6f}"
    )
    assert diff.max() < max_atol, f"{name}: max_abs {diff.max()} > {max_atol}"
    assert cos > min_cos, f"{name}: cosine {cos} < {min_cos}"


# ---------------------------------------------------------------------------
# Final-output parity at B=3 (multi-seed + real video)
# ---------------------------------------------------------------------------


def test_final_output_parity_multibatch(loaded_model, fixture_data):
    video = torch.from_numpy(fixture_data["input"].copy())  # (3, 16, 288, 288, 3)
    assert video.shape[0] == 3
    with torch.no_grad():
        out = loaded_model(video)
    # 16 stacked transformer layers (12 spatial + 4 temporal), softmax + tanh
    # logit cap, GELU — fp32 matmul order between PyTorch and JAX/XLA differs
    # enough that absolute-error outliers reach ~1e-2 on isolated entries even
    # though > 99.99% of entries are within a few 1e-4. Cosine similarity is
    # the load-bearing check for end-to-end correctness.
    _close("final B=3", fixture_data["output"], out, max_atol=2e-2, min_cos=0.9999)


def test_final_output_parity_per_batch_element(loaded_model, fixture_data):
    """Each row of the B=3 batch should match its single-batch counterpart.

    This makes batch-axis broadcasting / reshape bugs visible — they would
    manifest as cross-element bleed.
    """
    video_np = fixture_data["input"]  # (3, 16, 288, 288, 3)
    ref = fixture_data["output"]      # (3, 4096, 768)
    for i in range(video_np.shape[0]):
        x = torch.from_numpy(video_np[i:i + 1].copy())
        with torch.no_grad():
            y = loaded_model(x)
        _close(f"row {i}", ref[i:i + 1], y, max_atol=2e-2, min_cos=0.9999)


# ---------------------------------------------------------------------------
# Intermediate-output parity
# ---------------------------------------------------------------------------


def test_spatial_features_parity(loaded_model, fixture_data):
    """Captures `spatial_ln` output and compares against the JAX intermediate."""
    video = torch.from_numpy(fixture_data["input"].copy())
    with _capture(loaded_model.spatial_ln) as buf, torch.no_grad():
        loaded_model(video)
    assert len(buf) == 1
    pt_spatial = buf[0]  # (B*T, 256, 768)
    ref = fixture_data["spatial_features"]  # (B*T, 256, 768)
    assert pt_spatial.shape == ref.shape, (pt_spatial.shape, ref.shape)
    # After 12 spatial transformer layers, fp32 drift can hit ~1e-2 on isolated entries.
    _close("spatial_ln", ref, pt_spatial, max_atol=2e-2, min_cos=0.9999)


def test_temporal_out_parity(loaded_model, fixture_data):
    """Captures `temporal_ln` output and compares against the JAX intermediate.

    JAX saves it pre-final-reshape with shape (B*N, T, D) = (B*256, 16, 768).
    """
    video = torch.from_numpy(fixture_data["input"].copy())
    with _capture(loaded_model.temporal_ln) as buf, torch.no_grad():
        loaded_model(video)
    assert len(buf) == 1
    pt_temporal = buf[0]  # (B*N, T, D)
    ref = fixture_data["temporal_out"]  # (B*N, T, D)
    assert pt_temporal.shape == ref.shape, (pt_temporal.shape, ref.shape)
    _close("temporal_ln", ref, pt_temporal, max_atol=2e-2, min_cos=0.9999)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flow_reaches_every_param(fixture_data):
    """Forward + backward; verify gradients are finite and flow to every leaf param.

    Uses random init (no checkpoint) — gradient flow is a structural property
    of the architecture, not the weights.
    """
    model = build_videoprism_v1_base().train()
    x = torch.from_numpy(fixture_data["input"][:1].copy())  # one sample
    x.requires_grad = False
    out = model(x)
    loss = out.sum()  # any non-trivial scalar reduction
    loss.backward()

    bad = []
    zero = []
    for name, p in model.named_parameters():
        if p.grad is None:
            bad.append((name, "no grad"))
            continue
        if not torch.isfinite(p.grad).all():
            bad.append((name, "non-finite"))
            continue
        if p.grad.abs().max().item() == 0.0:
            zero.append(name)
    assert not bad, f"params with bad grads: {bad[:5]}"
    # `temporal_pos_emb` won't get gradient if input batch happens to be the
    # zero element of the temporal axis, but with sum() reduction every
    # learnable tensor should see a non-zero gradient.
    assert not zero, f"params with zero grad: {zero[:10]}"
