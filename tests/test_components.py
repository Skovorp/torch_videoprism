"""PyTorch-only unit tests for the VideoPrism port building blocks.

These tests use random init (no checkpoint) and verify shape correctness,
basic invariants, and determinism. They run quickly without JAX or HF
network access. Parity against the JAX checkpoint is in test_e2e_parity.py.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from torch_videoprism.model import (
    FactorizedEncoder,
    MLP,
    MultiHeadSelfAttention,
    TransformerBlock,
    TransformerStack,
    VIDEOPRISM_V1_BASE_CONFIG,
    build_videoprism_v1_base,
    patchify_image,
)


def _seed(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------- Patchify ----------


def test_patchify_shape_and_ordering():
    """Patchify of a 288x288x3 image with patch_size=18 produces (256, 972) with row-major (m, n) ordering."""
    g = _seed()
    x = torch.randn(2, 288, 288, 3, generator=g)
    patches = patchify_image(x, patch_size=18)
    assert patches.shape == (2, 256, 972)

    # The patch at (m, n) should equal a flat row-major slice over (p, q, c).
    m, n = 3, 5
    expected = x[:, m * 18:(m + 1) * 18, n * 18:(n + 1) * 18, :].reshape(2, -1)
    actual = patches[:, m * 16 + n, :]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# ---------- MLP ----------


def test_mlp_forward_shape_and_residual():
    g = _seed()
    mlp = MLP(dim=64, mlp_dim=128)
    x = torch.randn(2, 7, 64, generator=g)
    y = mlp(x)
    assert y.shape == x.shape

    # If we zero-out fc2 weights/bias, MLP is exactly the residual.
    with torch.no_grad():
        mlp.ffn_layer2.weight.zero_()
        mlp.ffn_layer2.bias.zero_()
    y2 = mlp(x)
    torch.testing.assert_close(y2, x, rtol=0, atol=0)


# ---------- Attention ----------


def test_attention_matches_manual():
    """The CPU einsum path equals a hand-derived `q · kᵀ → cap → softmax → · v`.

    Pins the math (query scaled by `1/sqrt(d_h)`, Primer cap before softmax,
    softmax in fp32) to a reference computation with the same weights.
    """
    g = _seed()
    attn = MultiHeadSelfAttention(dim=64, num_heads=4, atten_logit_cap=50.0).eval()
    x = torch.randn(2, 5, 64, generator=g)
    y = attn(x)

    cap = 50.0
    q = attn.query(x).view(2, 5, 4, 16) * (16 ** -0.5)
    k = attn.key(x).view(2, 5, 4, 16)
    v = attn.value(x).view(2, 5, 4, 16)
    logits = torch.einsum("btnh,bsnh->bnts", q, k)
    logits = cap * torch.tanh(logits / cap)
    probs = torch.softmax(logits.float(), dim=-1).to(v.dtype)
    encoded = torch.einsum("bnts,bsnh->btnh", probs, v).reshape(2, 5, 64)
    expected = attn.post(encoded)
    torch.testing.assert_close(y, expected, rtol=1e-6, atol=1e-6)


# ---------- TransformerBlock / Stack ----------


def test_transformer_block_zero_residual_when_outputs_zero():
    g = _seed()
    blk = TransformerBlock(dim=64, num_heads=4, mlp_dim=128)
    x = torch.randn(2, 5, 64, generator=g)
    # zero out attention output projection AND mlp fc2 -> block becomes identity.
    with torch.no_grad():
        blk.self_attention.post.weight.zero_()
        blk.self_attention.post.bias.zero_()
        blk.ff_layer.ffn_layer2.weight.zero_()
        blk.ff_layer.ffn_layer2.bias.zero_()
    y = blk(x)
    torch.testing.assert_close(y, x, rtol=0, atol=0)


def test_transformer_stack_forward_shape():
    stack = TransformerStack(num_layers=3, dim=64, num_heads=4, mlp_dim=128)
    x = torch.randn(2, 7, 64)
    y = stack(x)
    assert y.shape == x.shape


# ---------- FactorizedEncoder ----------


def test_factorized_encoder_shape_v1_base_random_init():
    """Sanity check on the full architecture's output shape with random weights."""
    model = build_videoprism_v1_base()
    model.eval()
    # Smaller batch to keep RAM reasonable on CPU.
    x = torch.rand(1, 16, 288, 288, 3)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 4096, 768)


def test_factorized_encoder_param_count_matches_jax():
    model = build_videoprism_v1_base()
    n = sum(p.numel() for p in model.parameters())
    # 114.37M expected (matches Flax checkpoint total).
    assert 114_300_000 < n < 114_500_000, n


def test_factorized_encoder_determinism():
    model = build_videoprism_v1_base().eval()
    g = _seed(42)
    x = torch.rand(1, 16, 288, 288, 3, generator=g)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    torch.testing.assert_close(y1, y2, rtol=0, atol=0)
