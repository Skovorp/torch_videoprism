"""torch.hub entry points for torch_videoprism.

Lets external users load the PyTorch port of VideoPrism without installing
the package, mirroring the pattern Meta uses for V-JEPA via torch.hub:

    encoder = torch.hub.load(
        "Skovorp/torch_videoprism", "videoprism_v1_base",
        pretrained=True, trust_repo=True,
    )
    out = encoder(video_in_0_to_1)  # video shape: (B, T, H, W, C)

All four publicly-released VideoPrism variants are exposed below. The two
`videoprism_lvt_v1_*` entries return a vision-only port of the LvT
(CLIP-style) models; the upstream text encoder is intentionally not ported.
"""
dependencies = ["torch", "numpy", "huggingface_hub"]


def _build(model_name: str, pretrained: bool):
    """Internal: build by upstream model name and (optionally) load weights."""
    from torch_videoprism import build_videoprism, load_pretrained_weights
    model = build_videoprism(model_name)
    if pretrained:
        load_pretrained_weights(model, model_name=model_name)
    return model


def videoprism_v1_base(pretrained: bool = True):
    """VideoPrism v1 base — `FactorizedEncoder`, 114M params, 16 frames @ 288².
    Output: (B, 4096, 768) token sequence."""
    return _build("videoprism_public_v1_base", pretrained)


def videoprism_v1_large(pretrained: bool = True):
    """VideoPrism v1 large — `FactorizedEncoder`, 354M params, 8 frames @ 288².
    Output: (B, 2048, 1024) token sequence."""
    return _build("videoprism_public_v1_large", pretrained)


def videoprism_lvt_v1_base(pretrained: bool = True):
    """VideoPrism LvT v1 base (vision-only) — `FactorizedVideoEncoder`,
    138M params, 16 frames @ 288². Output: (B, 768) L2-normalized embedding."""
    return _build("videoprism_lvt_public_v1_base", pretrained)


def videoprism_lvt_v1_large(pretrained: bool = True):
    """VideoPrism LvT v1 large (vision-only) — `FactorizedVideoEncoder`,
    396M params, 8 frames @ 288². Output: (B, 1024) L2-normalized embedding."""
    return _build("videoprism_lvt_public_v1_large", pretrained)
