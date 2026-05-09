"""HuggingFace `PretrainedConfig` for VideoPrism PyTorch port.

Two model types:
  - "videoprism"     : v1 base / v1 large       — output is a token sequence
  - "videoprism_lvt" : lvt v1 base / lvt v1 large — output is a single L2-normalized embedding (vision-only port)
"""
from __future__ import annotations

from transformers import PretrainedConfig


class VideoPrismConfig(PretrainedConfig):
    """Config for the v1 base / v1 large `FactorizedEncoder` variant.

    The port reads pixel inputs in [0, 1] with shape (B, T, H, W, C) — see
    `processing_videoprism.VideoPrismVideoProcessor`. Output is token features
    of shape `(B, T * H_p * W_p, model_dim)` where `H_p = W_p = image_size // patch_size`.
    """
    model_type = "videoprism"

    def __init__(
        self,
        patch_size: int = 18,
        image_size: int = 288,
        num_frames: int = 16,
        model_dim: int = 768,
        num_spatial_layers: int = 12,
        num_temporal_layers: int = 4,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        atten_logit_cap: float = 50.0,
        **kwargs,
    ):
        # `pos_emb_shape` is (T, H_p, W_p) in the underlying port, derived from
        # the standalone fields above so JSON config stays flat.
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_frames = num_frames
        self.model_dim = model_dim
        self.num_spatial_layers = num_spatial_layers
        self.num_temporal_layers = num_temporal_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.atten_logit_cap = atten_logit_cap
        super().__init__(**kwargs)

    @property
    def pos_emb_shape(self) -> tuple[int, int, int]:
        side = self.image_size // self.patch_size
        return (self.num_frames, side, side)

    @property
    def hidden_size(self) -> int:
        # Provided for HF tooling compatibility.
        return self.model_dim


class VideoPrismLvtConfig(VideoPrismConfig):
    """Config for the LvT (CLIP-style) variants — vision branch only.

    Adds the auxiliary 2-layer encoder + attentional pooler used by the
    upstream `FactorizedVideoCLIP`. We do not include the text encoder.
    """
    model_type = "videoprism_lvt"

    def __init__(
        self,
        num_auxiliary_layers: int = 2,
        pooler_hidden_dim: int | None = None,  # defaults to 4 * model_dim if None
        pooler_num_queries: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_auxiliary_layers = num_auxiliary_layers
        self.pooler_hidden_dim = pooler_hidden_dim or 4 * self.model_dim
        self.pooler_num_queries = pooler_num_queries
