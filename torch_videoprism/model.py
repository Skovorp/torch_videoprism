"""PyTorch port of VideoPrism FactorizedEncoder.

Mirrors the Flax architecture in google-deepmind/videoprism's
encoders.FactorizedEncoder + supporting layers, so weight names
map 1:1 to the published checkpoint (see weights.py for the converter).

Architecture for `videoprism_public_v1_base`
--------------------------------------------
Input:  (B, T=16, H=288, W=288, C=3) in [0, 1].
Output: (B, T*N=4096, D=768).

  patchify(P=18)                 -> (B*T, 256, 972)
  patch_projection: Linear       -> (B*T, 256, 768)
  + spatial_pos_emb (256, 768)
  spatial_encoder (12 blocks)    -> (B*T, 256, 768)
  spatial_ln
  reshape '(B T) N D -> (B N) T D'
  + temporal_pos_emb (16, 768)
  temporal_encoder (4 blocks)    -> (B*N, 16, 768)
  temporal_ln
  reshape '(B N) T D -> B (T N) D' -> (B, 4096, 768)

Each transformer block (norm_policy='pre'):
    y = x + Attn(LN(x))
    z = y + MLP(LN(y))
where MLP is two Linear layers separated by exact (erf) GELU,
and Attn is multi-head scaled dot-product with `atten_logit_cap=50.0`
applied as `cap * tanh(logits / cap)` BEFORE softmax. Query is scaled
by 1/sqrt(dim_per_head); no extra scale on logits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VIDEOPRISM_V1_BASE_CONFIG = dict(
    patch_size=18,
    pos_emb_shape=(16, 16, 16),  # (T, H_p, W_p) — 16 frames, 16x16 spatial patches
    model_dim=768,
    num_spatial_layers=12,
    num_temporal_layers=4,
    num_heads=12,
    mlp_dim=3072,
    atten_logit_cap=50.0,
)

VIDEOPRISM_V1_LARGE_CONFIG = dict(
    patch_size=18,
    pos_emb_shape=(8, 16, 16),   # (T, H_p, W_p) — 8 frames @ 288px (vs 16 in base)
    model_dim=1024,
    num_spatial_layers=24,
    num_temporal_layers=4,
    num_heads=16,
    mlp_dim=4096,
    atten_logit_cap=50.0,
)

CONFIGS = {
    "videoprism_public_v1_base": VIDEOPRISM_V1_BASE_CONFIG,
    "videoprism_public_v1_large": VIDEOPRISM_V1_LARGE_CONFIG,
}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Pre-LN feed-forward block matching `layers.TransformerFeedForward`.

    Applies `LN -> Linear(D, mlp_dim) -> GELU -> Linear(mlp_dim, D)` then
    adds the residual. No dropout (we only port eval-mode behavior).
    """

    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn_layer1 = nn.Linear(dim, mlp_dim, bias=True)
        self.ffn_layer2 = nn.Linear(mlp_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.layer_norm(x)
        h = self.ffn_layer1(h)
        h = F.gelu(h, approximate="none")  # exact GELU (erf-based)
        h = self.ffn_layer2(h)
        return residual + h


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention matching `layers.DotProductAttention`.

    Configuration matches videoprism v1 base:
      - internal_enable_per_dim_scale=False, scale_query_by_dim_per_head=False
        => query is scaled by `1/sqrt(dim_per_head)` (== `1/sqrt(D/N)`).
      - atten_logit_cap=50.0 applied as `cap * tanh(logits / cap)` BEFORE softmax.
      - softmax computed in fp32 then cast back.
      - no dropout, no causal mask, no padding mask.
    """

    def __init__(self, dim: int, num_heads: int, atten_logit_cap: float = 50.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.dim_per_head = dim // num_heads
        self.atten_logit_cap = atten_logit_cap

        # Each of Q/K/V/post is `nn.Linear(dim, dim)` viewed as (D -> N*H).
        # Naming matches the Flax `self_attention/{query,key,value,post}` subtrees.
        self.query = nn.Linear(dim, dim, bias=True)
        self.key = nn.Linear(dim, dim, bias=True)
        self.value = nn.Linear(dim, dim, bias=True)
        self.post = nn.Linear(dim, dim, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, T, N, H)
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.dim_per_head)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, H) -> (B, T, D)
        b, t, _, _ = x.shape
        return x.reshape(b, t, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        # Query scale: matches `_scale_query` with internal_enable_per_dim_scale=False.
        q = q * (self.dim_per_head ** -0.5)

        # logits[b, n, t, s] = sum_h q[b, t, n, h] * k[b, s, n, h]
        logits = torch.einsum("btnh,bsnh->bnts", q, k)

        # Logit cap (Primer-style tanh cap), then softmax in fp32.
        if self.atten_logit_cap and self.atten_logit_cap > 0:
            cap = self.atten_logit_cap
            logits = cap * torch.tanh(logits / cap)
        probs = F.softmax(logits.float(), dim=-1).to(v.dtype)

        # encoded[b, t, n, h] = sum_s probs[b, n, t, s] * v[b, s, n, h]
        encoded = torch.einsum("bnts,bsnh->btnh", probs, v)
        return self.post(self._merge_heads(encoded))


class TransformerBlock(nn.Module):
    """Single transformer block matching `layers.Transformer` (norm_policy='pre').

    y = x + Attn(LN(x))
    z = y + MLP(LN(y))      <- MLP includes its own pre-LN
    """

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, atten_logit_cap: float = 50.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.self_attention = MultiHeadSelfAttention(dim, num_heads, atten_logit_cap)
        self.ff_layer = MLP(dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm(x))
        return self.ff_layer(x)


class TransformerStack(nn.Module):
    """A stack of N TransformerBlocks (no final LN).

    The Flax checkpoint stores all N blocks under a single `x_layers` key with
    a leading layer axis; the converter splits along that axis when loading.
    """

    def __init__(self, num_layers: int, dim: int, num_heads: int, mlp_dim: int, atten_logit_cap: float = 50.0):
        super().__init__()
        self.x_layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_dim, atten_logit_cap)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.x_layers:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Patchification
# ---------------------------------------------------------------------------


def patchify_image(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """`encoders._image_to_patch` ported to PyTorch.

    Input:  (..., H, W, C). Channels-last like the JAX code.
    Output: (..., (H/P)*(W/P), P*P*C). Patches are row-major over (m, n);
    pixels within a patch are row-major over (p, q, c) — same as
    einops' '... (m p)(n q) c -> ... (m n) (p q c)'.
    """
    *lead, h, w, c = x.shape
    assert h % patch_size == 0 and w % patch_size == 0, (h, w, patch_size)
    m = h // patch_size
    n = w // patch_size
    p = patch_size
    # Reshape so (m, p) factor H and (n, q) factor W.
    x = x.reshape(*lead, m, p, n, p, c)
    # Bring m, n together (row-major over patches), then p, q, c together.
    perm = list(range(len(lead))) + [
        len(lead) + 0,  # m
        len(lead) + 2,  # n
        len(lead) + 1,  # p
        len(lead) + 3,  # q
        len(lead) + 4,  # c
    ]
    x = x.permute(*perm).contiguous()
    return x.reshape(*lead, m * n, p * p * c)


# ---------------------------------------------------------------------------
# FactorizedEncoder
# ---------------------------------------------------------------------------


@dataclass
class FactorizedEncoderConfig:
    patch_size: int = 18
    pos_emb_shape: tuple = (16, 16, 16)  # (T, H_p, W_p)
    model_dim: int = 768
    num_spatial_layers: int = 12
    num_temporal_layers: int = 4
    num_heads: int = 12
    mlp_dim: int = 3072
    atten_logit_cap: float = 50.0


class FactorizedEncoder(nn.Module):
    """Port of `encoders.FactorizedEncoder` for the v1 base config.

    Forward expects a video tensor in **channels-last** order matching the
    JAX API:

        inputs: (B, T, H, W, C) in [0.0, 1.0]
        output: (B, T*N, D)

    For the `videoprism_public_v1_base` defaults this means:
        in:  (B, 16, 288, 288, 3)
        out: (B, 4096, 768)

    Use `torch_videoprism.weights.load_pretrained_weights` to populate
    the parameters from the published Flax checkpoint.
    """

    def __init__(self, **kwargs):
        super().__init__()
        cfg = FactorizedEncoderConfig(**kwargs)
        self.config = cfg

        t_p, h_p, w_p = cfg.pos_emb_shape
        spatial_seq = h_p * w_p
        temporal_seq = t_p
        patch_dim = cfg.patch_size * cfg.patch_size * 3

        # Patch projection: (P*P*C) -> D, no activation (FeedForward with identity).
        self.patch_projection = nn.Linear(patch_dim, cfg.model_dim, bias=True)

        # TrainablePositionalEmbedding stores (max_seq_length, D).
        self.spatial_pos_emb = nn.Parameter(torch.zeros(spatial_seq, cfg.model_dim))
        self.temporal_pos_emb = nn.Parameter(torch.zeros(temporal_seq, cfg.model_dim))

        self.spatial_encoder = TransformerStack(
            num_layers=cfg.num_spatial_layers,
            dim=cfg.model_dim,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            atten_logit_cap=cfg.atten_logit_cap,
        )
        self.spatial_ln = nn.LayerNorm(cfg.model_dim, eps=1e-6)

        self.temporal_encoder = TransformerStack(
            num_layers=cfg.num_temporal_layers,
            dim=cfg.model_dim,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            atten_logit_cap=cfg.atten_logit_cap,
        )
        self.temporal_ln = nn.LayerNorm(cfg.model_dim, eps=1e-6)

    @property
    def hidden_dim(self) -> int:
        return self.config.model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        b, t, h, w, c = x.shape
        assert h == w, "height and width must match"
        assert c == 3, "input must be channels-last RGB"

        # Tokenize: (B, T, H, W, 3) -> (B*T, N, P*P*3)
        x = x.reshape(b * t, h, w, c)
        x = patchify_image(x, cfg.patch_size)              # (B*T, N, patch_dim)

        # Project + add spatial position embedding.
        x = self.patch_projection(x)                       # (B*T, N, D)
        spatial_pos = self.spatial_pos_emb                 # (N, D)
        # NOTE: The reference config has pos_emb_shape (16,16,16) and patch_size 18,
        # so spatial_pos already matches (h//P)*(w//P)=256. We don't support 2D
        # interpolation in the port — assert instead.
        assert spatial_pos.shape[0] == x.shape[1], (
            f"spatial pos length {spatial_pos.shape[0]} != num patches {x.shape[1]}"
        )
        x = x + spatial_pos.unsqueeze(0)                   # broadcast over batch

        # Spatial encoder + LN.
        x = self.spatial_encoder(x)                        # (B*T, N, D)
        x = self.spatial_ln(x)

        # Reshape '(b t) n d -> (b n) t d'.
        bt, n, d = x.shape
        assert bt == b * t
        x = x.reshape(b, t, n, d).permute(0, 2, 1, 3).contiguous().reshape(b * n, t, d)

        # Temporal pos. If T differs from the embedding's stored length,
        # interpolate linearly — mirrors the JAX `_interpolate_emb_1d` path
        # (jax.image.resize(method='bilinear') is just linear in 1D).
        temporal_pos = self.temporal_pos_emb               # (T_native, D)
        if temporal_pos.shape[0] != t:
            # (T_native, D) -> (1, D, T_native) -> interpolate -> (1, D, T) -> (T, D)
            tp = temporal_pos.t().unsqueeze(0)
            tp = F.interpolate(tp, size=t, mode="linear", align_corners=False)
            temporal_pos = tp.squeeze(0).t().contiguous()
        x = x + temporal_pos.unsqueeze(0)

        # Temporal encoder + LN.
        x = self.temporal_encoder(x)                       # (B*N, T, D)
        x = self.temporal_ln(x)

        # Reshape '(b n) t d -> b (t n) d'.
        bn, t2, d2 = x.shape
        assert bn == b * n and t2 == t and d2 == d
        x = x.reshape(b, n, t, d).permute(0, 2, 1, 3).contiguous().reshape(b, t * n, d)
        return x


def build_videoprism_v1_base() -> FactorizedEncoder:
    """Builds a fresh FactorizedEncoder with the v1 base config (random init)."""
    return FactorizedEncoder(**VIDEOPRISM_V1_BASE_CONFIG)


def build_videoprism_v1_large() -> FactorizedEncoder:
    """Builds a fresh FactorizedEncoder with the v1 large config (random init)."""
    return FactorizedEncoder(**VIDEOPRISM_V1_LARGE_CONFIG)


def build_videoprism(model_name: str) -> FactorizedEncoder:
    """Generic factory: build a FactorizedEncoder by VideoPrism model name.

    Args:
      model_name: one of `CONFIGS` (e.g. 'videoprism_public_v1_base',
        'videoprism_public_v1_large').
    """
    if model_name not in CONFIGS:
        raise ValueError(f"unknown VideoPrism model {model_name!r}. Known: {sorted(CONFIGS)}")
    return FactorizedEncoder(**CONFIGS[model_name])
