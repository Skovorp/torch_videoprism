"""PyTorch port of Google DeepMind's VideoPrism.

Ports the four publicly-released variants (`videoprism_public_v1_{base,large}`,
`videoprism_lvt_public_v1_{base,large}`). LvT variants are vision-only — the
text encoder is intentionally not ported.

For HuggingFace `AutoModel.from_pretrained(..., trust_remote_code=True)`
compatibility, see the `torch_videoprism.hf` subpackage and
`torch_videoprism.hf.build_repo.build_hf_repo`.

Original: https://github.com/google-deepmind/videoprism (JAX/Flax).
"""
from torch_videoprism.model import (
    CONFIGS,
    LVT_CONFIGS,
    FactorizedEncoder,
    FactorizedVideoEncoder,
    VIDEOPRISM_V1_BASE_CONFIG,
    VIDEOPRISM_V1_LARGE_CONFIG,
    VIDEOPRISM_LVT_V1_BASE_CONFIG,
    VIDEOPRISM_LVT_V1_LARGE_CONFIG,
    build_videoprism,
    build_videoprism_v1_base,
    build_videoprism_v1_large,
    build_videoprism_lvt_v1_base,
    build_videoprism_lvt_v1_large,
)
from torch_videoprism.weights import load_pretrained_weights

__all__ = [
    "CONFIGS",
    "LVT_CONFIGS",
    "FactorizedEncoder",
    "FactorizedVideoEncoder",
    "VIDEOPRISM_V1_BASE_CONFIG",
    "VIDEOPRISM_V1_LARGE_CONFIG",
    "VIDEOPRISM_LVT_V1_BASE_CONFIG",
    "VIDEOPRISM_LVT_V1_LARGE_CONFIG",
    "build_videoprism",
    "build_videoprism_v1_base",
    "build_videoprism_v1_large",
    "build_videoprism_lvt_v1_base",
    "build_videoprism_lvt_v1_large",
    "load_pretrained_weights",
]
