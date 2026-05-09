"""torch.hub entry points for torch_videoprism.

Lets external users load the PyTorch port of VideoPrism without installing
the package, mirroring the pattern Meta uses for V-JEPA via torch.hub:

    encoder = torch.hub.load(
        "Skovorp/torch_videoprism", "videoprism_v1_base",
        pretrained=True, trust_repo=True,
    )
    out = encoder(video_in_0_to_1)  # video shape: (B, T, H, W, C)
"""
dependencies = ["torch", "numpy", "huggingface_hub"]


def videoprism_v1_base(pretrained: bool = True):
    """Builds the VideoPrism v1 base FactorizedEncoder.

    Args:
      pretrained: if True (default), downloads the Flax checkpoint from
        google/videoprism-base-f16r288 on Hugging Face and converts it.

    Returns:
      A `torch_videoprism.FactorizedEncoder` (114M params) accepting videos
      of shape (B, T=16, H=288, W=288, C=3) in [0, 1] and returning
      embeddings of shape (B, 4096, 768).
    """
    from torch_videoprism import build_videoprism_v1_base, load_pretrained_weights

    model = build_videoprism_v1_base()
    if pretrained:
        load_pretrained_weights(model, model_name="videoprism_public_v1_base")
    return model
