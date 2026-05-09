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

VIDEOPRISM_LVT_V1_BASE_CONFIG = dict(
    # Vision branch matches videoprism_v1_base.
    patch_size=18,
    pos_emb_shape=(16, 16, 16),
    model_dim=768,
    num_spatial_layers=12,
    num_temporal_layers=4,
    num_heads=12,
    mlp_dim=3072,
    atten_logit_cap=50.0,
    # LvT-specific (video-only port; we ignore the text branch).
    num_auxiliary_layers=2,
    pooler_hidden_dim=3072,  # = model_dim * 4
)

VIDEOPRISM_LVT_V1_LARGE_CONFIG = dict(
    patch_size=18,
    pos_emb_shape=(8, 16, 16),
    model_dim=1024,
    num_spatial_layers=24,
    num_temporal_layers=4,
    num_heads=16,
    mlp_dim=4096,
    atten_logit_cap=50.0,
    num_auxiliary_layers=2,
    pooler_hidden_dim=4096,
)

CONFIGS = {
    "videoprism_public_v1_base":     VIDEOPRISM_V1_BASE_CONFIG,
    "videoprism_public_v1_large":    VIDEOPRISM_V1_LARGE_CONFIG,
    "videoprism_lvt_public_v1_base": VIDEOPRISM_LVT_V1_BASE_CONFIG,
    "videoprism_lvt_public_v1_large": VIDEOPRISM_LVT_V1_LARGE_CONFIG,
}

LVT_CONFIGS = {"videoprism_lvt_public_v1_base", "videoprism_lvt_public_v1_large"}


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


# ---------------------------------------------------------------------------
# LvT video branch (cross-modal models, video side only)
# ---------------------------------------------------------------------------


class PerDimScale(nn.Module):
    """Per-dimension scaling matching `layers.PerDimScale`.

    Computes `scale = (1.0 / softplus(0)) / sqrt(D) * softplus(per_dim_scale)`
    and multiplies the input by it elementwise on the last dim. With
    per_dim_scale init to zero this equals the standard 1/sqrt(D) scaling, but
    becomes per-dimension after training.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.per_dim_scale = nn.Parameter(torch.zeros(dim))
        # 1.0 / softplus(0) = 1 / log(2) = 1.442695041
        self.register_buffer("_const_scale",
                             torch.tensor(1.442695041 / math.sqrt(dim), dtype=torch.float32),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self._const_scale * F.softplus(self.per_dim_scale))


class AttenTokenPooler(nn.Module):
    """Cross-attention token pooler matching `layers.AttenTokenPoolingLayer`.

    Has a learnable `pooling_attention_query` of shape (num_queries, model_dim)
    that attends over the input tokens. Uses `PerDimScale` to scale the query
    (matching `internal_enable_per_dim_scale=True` in the upstream pooler — the
    only place in this port where PerDimScale is on, the main encoder uses
    plain 1/sqrt(d_per_head)).
    """
    def __init__(self, model_dim: int, hidden_dim: int, num_heads: int, num_queries: int = 1):
        super().__init__()
        assert hidden_dim % num_heads == 0, (hidden_dim, num_heads)
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_per_head = hidden_dim // num_heads
        self.num_queries = num_queries

        # Learnable pool query.
        self.pooling_attention_query = nn.Parameter(torch.zeros(num_queries, model_dim))

        # Cross-attention projections. Q maps from `model_dim` to `hidden_dim`;
        # K, V map from input token dim (also `model_dim`) to `hidden_dim`;
        # post maps back from `hidden_dim` to `model_dim`.
        self.query = nn.Linear(model_dim, hidden_dim, bias=True)
        self.key = nn.Linear(model_dim, hidden_dim, bias=True)
        self.value = nn.Linear(model_dim, hidden_dim, bias=True)
        self.post = nn.Linear(hidden_dim, model_dim, bias=True)
        self.per_dim_scale = PerDimScale(self.dim_per_head)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, hidden) -> (B, T, num_heads, dim_per_head)
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.dim_per_head)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        # Tile the learnable query across batch.
        q_in = self.pooling_attention_query.unsqueeze(0).expand(b, -1, -1)  # (B, num_queries, D)
        q = self._split(self.query(q_in))                                   # (B, Q, N, H)
        k = self._split(self.key(tokens))                                   # (B, T, N, H)
        v = self._split(self.value(tokens))                                 # (B, T, N, H)

        q = self.per_dim_scale(q)  # internal_enable_per_dim_scale=True in JAX
        logits = torch.einsum("bqnh,btnh->bnqt", q, k)
        # No logit cap on the pooler — JAX passes atten_logit_cap=0 by default
        # and FactorizedVideoCLIP doesn't override it for the pooler.
        probs = F.softmax(logits.float(), dim=-1).to(v.dtype)
        encoded = torch.einsum("bnqt,btnh->bqnh", probs, v)                 # (B, Q, N, H)
        encoded = encoded.reshape(b, self.num_queries, self.hidden_dim)
        out = self.post(encoded)                                            # (B, Q, model_dim)
        out = self.layer_norm(out)
        return out


@dataclass
class FactorizedVideoEncoderConfig:
    """LvT vision-only config — extends FactorizedEncoderConfig with pooler bits."""
    patch_size: int = 18
    pos_emb_shape: tuple = (16, 16, 16)
    model_dim: int = 768
    num_spatial_layers: int = 12
    num_temporal_layers: int = 4
    num_heads: int = 12
    mlp_dim: int = 3072
    atten_logit_cap: float = 50.0
    num_auxiliary_layers: int = 2
    pooler_hidden_dim: int = 3072
    pooler_num_queries: int = 1


class FactorizedVideoEncoder(nn.Module):
    """Vision-only port of `encoders.FactorizedVideoCLIP`.

    Pipeline:
        FactorizedEncoder           -> (B, T*N, D)
        auxiliary_encoder (2-layer) -> (B, T*N, D)
        AttenTokenPooler            -> (B, num_queries=1, D)
        squeeze axis 1              -> (B, D)
        L2 normalize on last axis   -> (B, D)
    """

    def __init__(self, **kwargs):
        super().__init__()
        cfg = FactorizedVideoEncoderConfig(**kwargs)
        self.config = cfg

        # `vision_encoder` has the same internals as the v1 base/large port —
        # we re-use FactorizedEncoder. The Flax checkpoint uses the same
        # parameter naming under `vision_encoder/...` so the loader maps cleanly.
        self.vision_encoder = FactorizedEncoder(
            patch_size=cfg.patch_size,
            pos_emb_shape=cfg.pos_emb_shape,
            model_dim=cfg.model_dim,
            num_spatial_layers=cfg.num_spatial_layers,
            num_temporal_layers=cfg.num_temporal_layers,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            atten_logit_cap=cfg.atten_logit_cap,
        )
        # Auxiliary encoder is a `VisionTransformer`, which in this port maps
        # to a TransformerStack with no final LayerNorm (matches JAX).
        self.auxiliary_encoder = TransformerStack(
            num_layers=cfg.num_auxiliary_layers,
            dim=cfg.model_dim,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            atten_logit_cap=cfg.atten_logit_cap,
        )
        self.contrastive_vision_pooler = AttenTokenPooler(
            model_dim=cfg.model_dim,
            hidden_dim=cfg.pooler_hidden_dim,
            num_heads=cfg.num_heads,
            num_queries=cfg.pooler_num_queries,
        )

    @property
    def hidden_dim(self) -> int:
        return self.config.model_dim

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Returns video embeddings of shape (B, model_dim) — L2-normalized by default."""
        feats = self.vision_encoder(x)             # (B, T*N, D)
        feats = self.auxiliary_encoder(feats)      # (B, T*N, D)
        pooled = self.contrastive_vision_pooler(feats)  # (B, Q=1, D)
        out = pooled.squeeze(-2)                    # (B, D)
        if normalize:
            # Match `_l2_normalize` in JAX: norm computed in fp32 for stability.
            orig_dtype = out.dtype
            out_f = out.float()
            out = (out_f / (out_f.norm(dim=-1, keepdim=True) + 1e-12)).to(orig_dtype)
        return out


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_videoprism_v1_base() -> FactorizedEncoder:
    """Builds a fresh FactorizedEncoder with the v1 base config (random init)."""
    return FactorizedEncoder(**VIDEOPRISM_V1_BASE_CONFIG)


def build_videoprism_v1_large() -> FactorizedEncoder:
    """Builds a fresh FactorizedEncoder with the v1 large config (random init)."""
    return FactorizedEncoder(**VIDEOPRISM_V1_LARGE_CONFIG)


def build_videoprism_lvt_v1_base() -> FactorizedVideoEncoder:
    """Vision-only port of the LvT v1 base model (~138M params)."""
    return FactorizedVideoEncoder(**VIDEOPRISM_LVT_V1_BASE_CONFIG)


def build_videoprism_lvt_v1_large() -> FactorizedVideoEncoder:
    """Vision-only port of the LvT v1 large model (~395M params)."""
    return FactorizedVideoEncoder(**VIDEOPRISM_LVT_V1_LARGE_CONFIG)


def build_videoprism(model_name: str) -> nn.Module:
    """Generic factory: builds a fresh model (random init) for any name in `CONFIGS`.

    Returns a `FactorizedVideoEncoder` for `videoprism_lvt_public_v1_*` names
    (vision-only port, output is a single (B, D) embedding); a `FactorizedEncoder`
    for `videoprism_public_v1_*` (token output of shape (B, T*N, D)).
    """
    if model_name not in CONFIGS:
        raise ValueError(f"unknown VideoPrism model {model_name!r}. Known: {sorted(CONFIGS)}")
    if model_name in LVT_CONFIGS:
        return FactorizedVideoEncoder(**CONFIGS[model_name])
    return FactorizedEncoder(**CONFIGS[model_name])
