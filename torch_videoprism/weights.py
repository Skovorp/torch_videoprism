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

from torch_videoprism.model import (
    FactorizedEncoder,
    FactorizedVideoEncoder,
    build_videoprism_v1_base,
)


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
    "videoprism_lvt_public_v1_base": (
        "google/videoprism-lvt-base-f16r288",
        "flax_lvt_base_f16r288_repeated.npz",
    ),
    "videoprism_lvt_public_v1_large": (
        "google/videoprism-lvt-large-f8r288",
        "flax_lvt_large_f8r288_repeated.npz",
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


def _factorized_encoder_state_dict(
    flax_params: Mapping[str, np.ndarray],
    *,
    num_spatial_layers: int,
    num_temporal_layers: int,
    flax_prefix: str = "",
    torch_prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Builds the PyTorch state_dict for one FactorizedEncoder.

    `flax_prefix` is prepended to every Flax key — used when the FactorizedEncoder
    sits under e.g. `vision_encoder/...` in an LvT checkpoint.
    `torch_prefix` is prepended to every PyTorch key — used when the FactorizedEncoder
    is the `vision_encoder` submodule of a FactorizedVideoEncoder.
    """
    fp = (lambda key: flax_params[f"{flax_prefix}{key}"])
    sd: dict[str, torch.Tensor] = {}

    pp = _convert_linear(fp("patch_projection/linear/kernel"), fp("patch_projection/linear/bias"))
    sd[f"{torch_prefix}patch_projection.weight"] = pp["weight"]
    sd[f"{torch_prefix}patch_projection.bias"] = pp["bias"]

    sd[f"{torch_prefix}spatial_pos_emb"] = _to_torch(fp("spatial_pos_emb/emb_var"))
    sd[f"{torch_prefix}temporal_pos_emb"] = _to_torch(fp("temporal_pos_emb/emb_var"))

    for jax_name, torch_name in [("spatial_ln", "spatial_ln"), ("temporal_ln", "temporal_ln")]:
        ln = _convert_layernorm(fp(f"{jax_name}/scale"), fp(f"{jax_name}/bias"))
        sd[f"{torch_prefix}{torch_name}.weight"] = ln["weight"]
        sd[f"{torch_prefix}{torch_name}.bias"] = ln["bias"]

    for jax_enc, num_layers, torch_root in [
        ("spatial_encoder", num_spatial_layers, "spatial_encoder.x_layers"),
        ("temporal_encoder", num_temporal_layers, "temporal_encoder.x_layers"),
    ]:
        for i in range(num_layers):
            # _convert_block keys flax_params with `{encoder_prefix}/...`, so we
            # set encoder_prefix to the full Flax path including any outer prefix.
            block = _convert_block(flax_params, f"{flax_prefix}{jax_enc}", i)
            for sub, sub_sd in block.items():
                for k, v in sub_sd.items():
                    sd[f"{torch_prefix}{torch_root}.{i}.{sub}.{k}"] = v
    return sd


def _aux_encoder_state_dict(
    flax_params: Mapping[str, np.ndarray],
    *,
    num_layers: int,
    torch_prefix: str = "auxiliary_encoder.",
) -> dict[str, torch.Tensor]:
    """Builds the PyTorch state_dict for the LvT 2-layer auxiliary encoder."""
    sd: dict[str, torch.Tensor] = {}
    for i in range(num_layers):
        block = _convert_block(flax_params, "auxiliary_encoder", i)
        for sub, sub_sd in block.items():
            for k, v in sub_sd.items():
                sd[f"{torch_prefix}x_layers.{i}.{sub}.{k}"] = v
    return sd


def _pooler_state_dict(
    flax_params: Mapping[str, np.ndarray],
    *,
    torch_prefix: str = "contrastive_vision_pooler.",
) -> dict[str, torch.Tensor]:
    """Builds the PyTorch state_dict for the LvT contrastive_vision_pooler."""
    sd: dict[str, torch.Tensor] = {}
    base = "contrastive_vision_pooler"

    sd[f"{torch_prefix}pooling_attention_query"] = _to_torch(
        flax_params[f"{base}/pooling_attention_query"]
    )
    # Q/K/V/post projections.
    for name in ("query", "key", "value"):
        proj = _convert_qkv_input_proj(
            flax_params[f"{base}/pooling_attention/{name}/w"],
            flax_params[f"{base}/pooling_attention/{name}/b"],
        )
        sd[f"{torch_prefix}{name}.weight"] = proj["weight"]
        sd[f"{torch_prefix}{name}.bias"] = proj["bias"]
    post = _convert_post_proj(
        flax_params[f"{base}/pooling_attention/post/w"],
        flax_params[f"{base}/pooling_attention/post/b"],
    )
    sd[f"{torch_prefix}post.weight"] = post["weight"]
    sd[f"{torch_prefix}post.bias"] = post["bias"]

    # Per-dim scale parameter (on the query, with shape (dim_per_head,)).
    sd[f"{torch_prefix}per_dim_scale.per_dim_scale"] = _to_torch(
        flax_params[f"{base}/pooling_attention/per_dim_scale/per_dim_scale"]
    )

    ln = _convert_layernorm(
        flax_params[f"{base}/pooling_attention_layer_norm/scale"],
        flax_params[f"{base}/pooling_attention_layer_norm/bias"],
    )
    sd[f"{torch_prefix}layer_norm.weight"] = ln["weight"]
    sd[f"{torch_prefix}layer_norm.bias"] = ln["bias"]
    return sd


# Public alias for the v1 (non-LvT) state-dict builder; kept for backwards-compat.
def flax_params_to_state_dict(
    flax_params: Mapping[str, np.ndarray], num_spatial_layers: int, num_temporal_layers: int
) -> dict[str, torch.Tensor]:
    """Builds a PyTorch `state_dict` for a v1 base/large `FactorizedEncoder`."""
    return _factorized_encoder_state_dict(
        flax_params,
        num_spatial_layers=num_spatial_layers,
        num_temporal_layers=num_temporal_layers,
    )


def flax_params_to_state_dict_lvt(
    flax_params: Mapping[str, np.ndarray],
    *,
    num_spatial_layers: int,
    num_temporal_layers: int,
    num_auxiliary_layers: int,
) -> dict[str, torch.Tensor]:
    """Builds a PyTorch `state_dict` for a `FactorizedVideoEncoder` (LvT video-only).

    Reads only the `vision_encoder/`, `auxiliary_encoder/`, and
    `contrastive_vision_pooler/` subtrees; the LvT checkpoint's `text_encoder/`
    params are silently ignored.
    """
    sd: dict[str, torch.Tensor] = {}
    sd.update(_factorized_encoder_state_dict(
        flax_params,
        num_spatial_layers=num_spatial_layers,
        num_temporal_layers=num_temporal_layers,
        flax_prefix="vision_encoder/",
        torch_prefix="vision_encoder.",
    ))
    sd.update(_aux_encoder_state_dict(flax_params, num_layers=num_auxiliary_layers))
    sd.update(_pooler_state_dict(flax_params))
    return sd


def load_pretrained_weights(
    model: "FactorizedEncoder | FactorizedVideoEncoder | None" = None,
    *,
    model_name: str = "videoprism_public_v1_base",
    checkpoint_path: str | None = None,
    strict: bool = True,
):
    """Loads VideoPrism Flax weights into a PyTorch model in-place.

    Dispatches based on `model` type:
      - `FactorizedEncoder`      -> reads v1 base/large checkpoint params
      - `FactorizedVideoEncoder` -> reads LvT vision_encoder + auxiliary_encoder
                                    + contrastive_vision_pooler params; ignores
                                    text_encoder.

    Args:
      model: optional existing model to fill. If None, a fresh v1 base is built.
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

    if isinstance(model, FactorizedVideoEncoder):
        state_dict = flax_params_to_state_dict_lvt(
            flax_params,
            num_spatial_layers=model.config.num_spatial_layers,
            num_temporal_layers=model.config.num_temporal_layers,
            num_auxiliary_layers=model.config.num_auxiliary_layers,
        )
    elif isinstance(model, FactorizedEncoder):
        state_dict = flax_params_to_state_dict(
            flax_params,
            num_spatial_layers=model.config.num_spatial_layers,
            num_temporal_layers=model.config.num_temporal_layers,
        )
    else:
        raise TypeError(f"Cannot load weights into {type(model).__name__}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"state_dict mismatch — missing: {missing}, unexpected: {unexpected}"
        )
    return model
