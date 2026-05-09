"""HuggingFace `PreTrainedModel` wrappers around the VideoPrism PyTorch port.

Two model classes corresponding to the two configurations:
  - `VideoPrismModel`    : wraps `FactorizedEncoder`      (token-output variant)
  - `VideoPrismLvtModel` : wraps `FactorizedVideoEncoder` (single-embedding variant)

Both accept a `pixel_values` tensor of shape (B, T, H, W, C) in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from torch_videoprism.model import FactorizedEncoder, FactorizedVideoEncoder

# When loaded via `AutoConfig.from_pretrained(..., trust_remote_code=True)`,
# `transformers` imports this file alongside `configuration_videoprism.py` from
# the same HF repo directory; both packages and "remote_code" loading work.
try:
    from .configuration_videoprism import VideoPrismConfig, VideoPrismLvtConfig
except ImportError:                                       # pragma: no cover
    from configuration_videoprism import VideoPrismConfig, VideoPrismLvtConfig  # type: ignore


@dataclass
class VideoPrismOutput(ModelOutput):
    """Output of `VideoPrismModel`. Token sequence under `last_hidden_state`."""
    last_hidden_state: torch.FloatTensor | None = None


@dataclass
class VideoPrismLvtOutput(ModelOutput):
    """Output of `VideoPrismLvtModel`. Single L2-normalized embedding."""
    video_embeds: torch.FloatTensor | None = None


def _factorized_encoder_kwargs(cfg: VideoPrismConfig) -> dict:
    return dict(
        patch_size=cfg.patch_size,
        pos_emb_shape=cfg.pos_emb_shape,
        model_dim=cfg.model_dim,
        num_spatial_layers=cfg.num_spatial_layers,
        num_temporal_layers=cfg.num_temporal_layers,
        num_heads=cfg.num_heads,
        mlp_dim=cfg.mlp_dim,
        atten_logit_cap=cfg.atten_logit_cap,
    )


class VideoPrismModel(PreTrainedModel):
    """HuggingFace wrapper for the v1 base / v1 large `FactorizedEncoder`."""
    config_class = VideoPrismConfig
    main_input_name = "pixel_values"
    base_model_prefix = "videoprism"

    def __init__(self, config: VideoPrismConfig):
        super().__init__(config)
        self.videoprism = FactorizedEncoder(**_factorized_encoder_kwargs(config))
        # Required by PreTrainedModel.post_init; we ship pretrained weights so
        # the random init done here is overwritten by `from_pretrained`.
        self.post_init()

    def _init_weights(self, module: nn.Module):
        # Pretrained checkpoints are loaded by `from_pretrained`, but HF still
        # calls _init_weights on freshly built modules. Match the upstream
        # default-kernel init shape (truncated normal stddev=0.02).
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True,
        **_: dict,
    ) -> VideoPrismOutput | tuple:
        """Pixel values: (B, T, H, W, C) in [0, 1]."""
        last_hidden_state = self.videoprism(pixel_values)
        if not return_dict:
            return (last_hidden_state,)
        return VideoPrismOutput(last_hidden_state=last_hidden_state)


class VideoPrismLvtModel(PreTrainedModel):
    """HuggingFace wrapper for the LvT (vision-only) variants.

    Output is a single per-video embedding of shape (B, model_dim), L2-normalized
    on the last axis. Useful for retrieval / classification heads.
    """
    config_class = VideoPrismLvtConfig
    main_input_name = "pixel_values"
    base_model_prefix = "videoprism"

    def __init__(self, config: VideoPrismLvtConfig):
        super().__init__(config)
        self.videoprism = FactorizedVideoEncoder(
            **_factorized_encoder_kwargs(config),
            num_auxiliary_layers=config.num_auxiliary_layers,
            pooler_hidden_dim=config.pooler_hidden_dim,
            pooler_num_queries=config.pooler_num_queries,
        )
        self.post_init()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(
        self,
        pixel_values: torch.Tensor,
        normalize: bool = True,
        return_dict: bool = True,
        **_: dict,
    ) -> VideoPrismLvtOutput | tuple:
        """Pixel values: (B, T, H, W, C) in [0, 1]. Returns a single embedding per video."""
        video_embeds = self.videoprism(pixel_values, normalize=normalize)
        if not return_dict:
            return (video_embeds,)
        return VideoPrismLvtOutput(video_embeds=video_embeds)
