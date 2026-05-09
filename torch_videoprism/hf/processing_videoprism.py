"""HuggingFace `BaseImageProcessor` for VideoPrism.

Handles the video → tensor conversion expected by `VideoPrismModel` and
`VideoPrismLvtModel`:

  - Accepts: a video file path, a list of PIL Images, a numpy array of shape
    (T, H, W, C) or (B, T, H, W, C) (uint8 or float), or a torch.Tensor.
  - Produces: `pixel_values` torch.Tensor of shape (B, T, H, W, C), float32 in [0, 1].

Frame sampling: linear (uniform) sampling of `num_frames` indices over the
input clip. For lists of frames whose length already matches `num_frames`,
no sampling is done.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature


def _ensure_uint8_array(frames: Any) -> np.ndarray:
    """Coerce a single video to a uint8 numpy array of shape (T, H, W, 3)."""
    if isinstance(frames, np.ndarray):
        arr = frames
    elif isinstance(frames, torch.Tensor):
        arr = frames.detach().cpu().numpy()
    elif isinstance(frames, (list, tuple)):
        # List of PIL.Images or np arrays. Import PIL lazily — wrapping in
        # try/except also keeps `transformers.dynamic_module_utils.check_imports`
        # from rejecting the file when PIL isn't installed.
        try:
            from PIL import Image  # type: ignore
            is_pil = bool(frames) and isinstance(frames[0], Image.Image)
        except ImportError:
            is_pil = False
        if is_pil:
            arr = np.stack([np.asarray(f.convert("RGB")) for f in frames])
        else:
            arr = np.stack([np.asarray(f) for f in frames])
    else:
        raise TypeError(f"Unsupported video frames type: {type(frames)}")

    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"Expected video as (T, H, W, 3); got shape {arr.shape}"
        )
    if arr.dtype != np.uint8:
        # Float in [0, 1] also accepted, convert.
        if arr.dtype.kind == "f":
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def _decode_video(path: str, num_frames: int) -> np.ndarray:
    """Decode a video file to (num_frames, H, W, 3) uint8.

    Imports `decord` lazily — wrapped in try/except so users who only feed in
    pre-decoded frame arrays don't need decord installed (and HF's dynamic
    module loader doesn't flag it as a hard dep).
    """
    try:
        from decord import VideoReader, cpu  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Decoding video files needs `decord`. Install it (`pip install decord`) "
            "or pass already-decoded frames (np.array of shape (T, H, W, 3))."
        ) from e

    vr = VideoReader(path, ctx=cpu(0))
    n = len(vr)
    if n == 0:
        raise ValueError(f"empty video: {path}")
    idx = np.linspace(0, max(n - 1, 0), num=num_frames).astype(int)
    return vr.get_batch(idx).asnumpy()


def _sample_uniform(arr: np.ndarray, num_frames: int) -> np.ndarray:
    """Uniformly sample `num_frames` from `arr` of shape (T, H, W, 3)."""
    t = arr.shape[0]
    if t == num_frames:
        return arr
    idx = np.linspace(0, max(t - 1, 0), num=num_frames).astype(int)
    return arr[idx]


def _resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize a single (H, W, 3) uint8 frame to (size, size, 3) bilinearly.

    Tries `cv2` first, falls back to PIL. Either is fine — both produce a
    bilinear resize. Wrapped in try/except so HF's dynamic loader doesn't
    require either as a hard dep.
    """
    try:
        import cv2  # type: ignore
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        pass
    try:
        from PIL import Image  # type: ignore
        return np.asarray(
            Image.fromarray(frame).resize((size, size), Image.BILINEAR)
        )
    except ImportError as e:
        raise ImportError(
            "Resizing frames needs either `opencv-python` or `pillow`. "
            "Install one (`pip install opencv-python` or `pip install pillow`) "
            "or pass frames already at the target spatial size."
        ) from e


class VideoPrismVideoProcessor(BaseImageProcessor):
    """Preprocessor for VideoPrism inputs.

    Args:
      image_size: target spatial size (frames are bilinearly resized to size×size).
      num_frames: number of frames sampled uniformly per clip.
      do_resize: if False, frames are not resized (must already be the right shape).
      do_rescale: if False, frames are not divided by 255 (must already be in [0, 1]).
      rescale_factor: divisor for uint8 → [0, 1] conversion (default 1/255).
    """
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int = 288,
        num_frames: int = 16,
        do_resize: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1.0 / 255.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.num_frames = num_frames
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor

    def _preprocess_one(self, video: Any) -> np.ndarray:
        """Returns a (T, H, W, 3) float32 array in [0, 1]."""
        if isinstance(video, str) and os.path.exists(video):
            arr = _decode_video(video, num_frames=self.num_frames)
        else:
            arr = _ensure_uint8_array(video)
            arr = _sample_uniform(arr, self.num_frames)

        if self.do_resize and (arr.shape[1] != self.image_size or arr.shape[2] != self.image_size):
            arr = np.stack([_resize_frame(f, self.image_size) for f in arr])

        out = arr.astype(np.float32)
        if self.do_rescale:
            out = out * self.rescale_factor
        return out

    def __call__(
        self,
        videos: Any = None,
        return_tensors: str | None = "pt",
        **_: Any,
    ) -> BatchFeature:
        if videos is None:
            raise ValueError("VideoPrismVideoProcessor requires a `videos` argument.")

        # Wrap a single video into a batch.
        is_batched = (
            isinstance(videos, list)
            and videos
            and not isinstance(videos[0], (np.ndarray, torch.Tensor))
            and not (
                hasattr(videos[0], "__class__") and videos[0].__class__.__name__ == "Image"
            )
        ) or (isinstance(videos, np.ndarray) and videos.ndim == 5) or (
            isinstance(videos, torch.Tensor) and videos.ndim == 5
        )
        # Special case: list-of-frames where each frame is np.array (T, H, W, 3) batched.
        if isinstance(videos, list) and videos and isinstance(videos[0], np.ndarray) and videos[0].ndim == 4:
            is_batched = True

        if not is_batched:
            videos = [videos]

        processed = np.stack([self._preprocess_one(v) for v in videos])  # (B, T, H, W, 3)

        if return_tensors == "pt":
            pixel_values = torch.from_numpy(processed)
        elif return_tensors is None:
            pixel_values = processed
        else:
            raise ValueError(f"Only return_tensors='pt' or None is supported (got {return_tensors!r}).")

        return BatchFeature(data={"pixel_values": pixel_values}, tensor_type=return_tensors)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(
            image_size=self.image_size,
            num_frames=self.num_frames,
            do_resize=self.do_resize,
            do_rescale=self.do_rescale,
            rescale_factor=self.rescale_factor,
        )
        return d
