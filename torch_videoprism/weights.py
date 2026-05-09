"""Convert VideoPrism Flax checkpoints to the PyTorch port.

The checkpoint shipped on Hugging Face (`flax_base_f16r288_repeated.npz`)
is a flat `.npz` saved by `videoprism.utils.save_checkpoint`, with keys like
``params/spatial_encoder/transformers_stack/x_layers/layer_norm/scale``.
Layers stacked under `Repeat` carry a leading layer axis (e.g. 12 for the
spatial encoder, 4 for the temporal encoder).

Flax → PyTorch layout differences handled here:
  - `nn.Dense.kernel` (in, out) → `nn.Linear.weight` (out, in)  (transpose)
  - `LayerNorm.scale` is `direct_scale=False`, so the active scale at
    runtime is `(scale + 1.0)`. We bake that +1 into PyTorch's `.weight`.
  - `AttentionProjection` Q/K/V `w` (D, N, H) → `nn.Linear.weight` (N*H, D)
    via `.transpose(1,2,0).reshape(N*H, D)`. Biases (N, H) → (N*H,).
  - `AttentionProjection` post `w` (D, N, H) → `nn.Linear.weight` (D, N*H)
    via `.reshape(D, N*H)`. Bias is already (D,).
"""
from __future__ import annotations

import os
from typing import Mapping

import numpy as np
import torch

from torch_videoprism.model import FactorizedEncoder, build_videoprism_v1_base


_NPZ_PARAM_PREFIX = "params/"


def _load_npz_flat(path: str) -> dict[str, np.ndarray]:
    """Loads the .npz checkpoint into a flat dict keyed by Flax param path.

    Drops the leading `params/` prefix for ergonomic key matching.
    """
    raw = np.load(path)
    out: dict[str, np.ndarray] = {}
    for key in raw.files:
        if not key.startswith(_NPZ_PARAM_PREFIX):
            continue
        out[key[len(_NPZ_PARAM_PREFIX):]] = np.asarray(raw[key])
    return out


# Mapping from VideoPrism model name to its (HF repo_id, npz filename).
# Mirrors `videoprism.models.CHECKPOINTS` upstream.
HF_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "videoprism_public_v1_base": (
        "google/videoprism-base-f16r288",
        "flax_base_f16r288_repeated.npz",
    ),
    "videoprism_public_v1_large": (
        "google/videoprism-large-f8r288",
        "flax_large_f8r288_repeated.npz",
    ),
}


def _hf_download(model_name: str) -> str:
    """Downloads the VideoPrism Flax .npz from Hugging Face and returns the local path."""
    if model_name not in HF_CHECKPOINTS:
        raise ValueError(
            f"Unknown VideoPrism checkpoint name: {model_name!r}. "
            f"Known: {sorted(HF_CHECKPOINTS)}"
        )
    repo_id, filename = HF_CHECKPOINTS[model_name]
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


# ---------------------------------------------------------------------------
# Per-component conversion helpers
# ---------------------------------------------------------------------------


def _to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr).copy())


def _convert_linear(kernel: np.ndarray, bias: np.ndarray | None) -> dict[str, torch.Tensor]:
    """Flax Dense (kernel: (in, out)) -> PyTorch Linear (weight: (out, in))."""
    out = {"weight": _to_torch(kernel.T)}
    if bias is not None:
        out["bias"] = _to_torch(bias)
    return out


def _convert_layernorm(scale: np.ndarray, bias: np.ndarray) -> dict[str, torch.Tensor]:
    """Flax LayerNorm (direct_scale=False) -> PyTorch LayerNorm.

    Flax uses `(scale + 1.0)` as the effective gain; PyTorch's `weight` IS the
    effective gain, so we add 1.0 here.
    """
    return {"weight": _to_torch(scale + 1.0), "bias": _to_torch(bias)}


def _convert_qkv_input_proj(w: np.ndarray, b: np.ndarray) -> dict[str, torch.Tensor]:
    """Flax AttentionProjection (input): w (D, N, H), b (N, H) ->
    PyTorch Linear: weight (N*H, D), bias (N*H,).
    """
    d, n, h = w.shape
    weight = w.transpose(1, 2, 0).reshape(n * h, d)
    bias = b.reshape(n * h)
    return {"weight": _to_torch(weight), "bias": _to_torch(bias)}


def _convert_post_proj(w: np.ndarray, b: np.ndarray) -> dict[str, torch.Tensor]:
    """Flax AttentionProjection (output): w (D, N, H), b (D,) ->
    PyTorch Linear: weight (D, N*H), bias (D,).
    """
    d, n, h = w.shape
    weight = w.reshape(d, n * h)
    return {"weight": _to_torch(weight), "bias": _to_torch(b)}


# ---------------------------------------------------------------------------
# Top-level conversion
# ---------------------------------------------------------------------------


def _convert_block(
    flax_params: Mapping[str, np.ndarray],
    encoder_prefix: str,
    layer_idx: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Returns a dict of `{submodule_path: state_dict}` for one transformer block."""
    p = lambda key: flax_params[f"{encoder_prefix}/transformers_stack/x_layers/{key}"][layer_idx]

    return {
        "layer_norm": _convert_layernorm(p("layer_norm/scale"), p("layer_norm/bias")),
        "self_attention.query": _convert_qkv_input_proj(p("self_attention/query/w"), p("self_attention/query/b")),
        "self_attention.key":   _convert_qkv_input_proj(p("self_attention/key/w"),   p("self_attention/key/b")),
        "self_attention.value": _convert_qkv_input_proj(p("self_attention/value/w"), p("self_attention/value/b")),
        "self_attention.post":  _convert_post_proj(p("self_attention/post/w"),       p("self_attention/post/b")),
        "ff_layer.layer_norm": _convert_layernorm(
            p("ff_layer/layer_norm/scale"), p("ff_layer/layer_norm/bias")
        ),
        "ff_layer.ffn_layer1": _convert_linear(
            p("ff_layer/ffn_layer1/linear/kernel"), p("ff_layer/ffn_layer1/linear/bias")
        ),
        "ff_layer.ffn_layer2": _convert_linear(
            p("ff_layer/ffn_layer2/linear/kernel"), p("ff_layer/ffn_layer2/linear/bias")
        ),
    }


def flax_params_to_state_dict(
    flax_params: Mapping[str, np.ndarray], num_spatial_layers: int, num_temporal_layers: int
) -> dict[str, torch.Tensor]:
    """Builds a PyTorch `state_dict` for `FactorizedEncoder` from flat Flax params."""
    sd: dict[str, torch.Tensor] = {}

    # Patch projection.
    pp = _convert_linear(
        flax_params["patch_projection/linear/kernel"],
        flax_params["patch_projection/linear/bias"],
    )
    sd["patch_projection.weight"] = pp["weight"]
    sd["patch_projection.bias"] = pp["bias"]

    # Position embeddings.
    sd["spatial_pos_emb"] = _to_torch(flax_params["spatial_pos_emb/emb_var"])
    sd["temporal_pos_emb"] = _to_torch(flax_params["temporal_pos_emb/emb_var"])

    # Final per-encoder LayerNorms.
    for jax_name, torch_prefix in [("spatial_ln", "spatial_ln"), ("temporal_ln", "temporal_ln")]:
        ln = _convert_layernorm(
            flax_params[f"{jax_name}/scale"], flax_params[f"{jax_name}/bias"]
        )
        sd[f"{torch_prefix}.weight"] = ln["weight"]
        sd[f"{torch_prefix}.bias"] = ln["bias"]

    # Encoder blocks.
    for prefix, num_layers, torch_root in [
        ("spatial_encoder", num_spatial_layers, "spatial_encoder.x_layers"),
        ("temporal_encoder", num_temporal_layers, "temporal_encoder.x_layers"),
    ]:
        for i in range(num_layers):
            block = _convert_block(flax_params, prefix, i)
            for sub, sub_sd in block.items():
                for k, v in sub_sd.items():
                    sd[f"{torch_root}.{i}.{sub}.{k}"] = v
    return sd


def load_pretrained_weights(
    model: FactorizedEncoder | None = None,
    *,
    model_name: str = "videoprism_public_v1_base",
    checkpoint_path: str | None = None,
    strict: bool = True,
) -> FactorizedEncoder:
    """Loads VideoPrism-B Flax weights into a PyTorch FactorizedEncoder.

    Args:
      model: optional existing FactorizedEncoder to fill in-place. If None, a
        fresh one is built with the v1 base config.
      model_name: HuggingFace VideoPrism model name (only used when
        `checkpoint_path` is None).
      checkpoint_path: optional path to a local `.npz` checkpoint.
      strict: if True, asserts every parameter in the model receives a value.

    Returns:
      The model (same instance as the `model` arg if provided).
    """
    if model is None:
        model = build_videoprism_v1_base()

    if checkpoint_path is None:
        checkpoint_path = _hf_download(model_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    flax_params = _load_npz_flat(checkpoint_path)
    state_dict = flax_params_to_state_dict(
        flax_params,
        num_spatial_layers=model.config.num_spatial_layers,
        num_temporal_layers=model.config.num_temporal_layers,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"state_dict mismatch — missing: {missing}, unexpected: {unexpected}"
        )
    return model
