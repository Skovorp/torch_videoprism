"""PyTorch port of Google DeepMind's VideoPrism (FactorizedEncoder).

Original: https://github.com/google-deepmind/videoprism (JAX/Flax).
Public weights: https://huggingface.co/google/videoprism-base-f16r288.
"""
from torch_videoprism.model import (
    CONFIGS,
    FactorizedEncoder,
    VIDEOPRISM_V1_BASE_CONFIG,
    VIDEOPRISM_V1_LARGE_CONFIG,
    build_videoprism,
    build_videoprism_v1_base,
    build_videoprism_v1_large,
)
from torch_videoprism.weights import load_pretrained_weights

__all__ = [
    "CONFIGS",
    "FactorizedEncoder",
    "VIDEOPRISM_V1_BASE_CONFIG",
    "VIDEOPRISM_V1_LARGE_CONFIG",
    "build_videoprism",
    "build_videoprism_v1_base",
    "build_videoprism_v1_large",
    "load_pretrained_weights",
]
