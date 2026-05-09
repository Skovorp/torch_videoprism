"""PyTorch port of Google DeepMind's VideoPrism (FactorizedEncoder).

Original: https://github.com/google-deepmind/videoprism (JAX/Flax).
Public weights: https://huggingface.co/google/videoprism-base-f16r288.
"""
from torch_videoprism.model import (
    FactorizedEncoder,
    VIDEOPRISM_V1_BASE_CONFIG,
    build_videoprism_v1_base,
)
from torch_videoprism.weights import load_pretrained_weights

__all__ = [
    "FactorizedEncoder",
    "VIDEOPRISM_V1_BASE_CONFIG",
    "build_videoprism_v1_base",
    "load_pretrained_weights",
]
