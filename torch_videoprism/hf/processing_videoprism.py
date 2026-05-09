"""HuggingFace processor for VideoPrism inputs.

Follows the standard `BaseImageProcessor` convention:

  - Input pixels are assumed to be in **[0, 255]** by default; `do_rescale=True`
    multiplies by `rescale_factor=1/255` to put them in [0, 1] before the model.
  - If your frames are already in [0, 1], pass `do_rescale=False`.

Accepts videos as: a path to a video file, a list of PIL images, a numpy array
of shape `(T, H, W, 3)` or `(B, T, H, W, 3)`, or a torch.Tensor with one of
those shapes. Returns a `BatchFeature` with `pixel_values` of shape
`(B, T, image_size, image_size, 3)`, `float32`, in `[0, 1]` by default.

Example:
    >>> proc = VideoPrismVideoProcessor()                  # image_size=288, num_frames=16
    >>> inputs = proc(videos="my_clip.mp4", return_tensors="pt")
    >>> inputs["pixel_values"].shape, inputs["pixel_values"].dtype
    (torch.Size([1, 16, 288, 288, 3]), torch.float32)

    # Already-decoded frames in [0, 1]:
    >>> proc(videos=frames01, do_rescale=False, return_tensors="pt")
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature


# ---------------------------------------------------------------------------
# Helpers — all "heavy" decoder/resizer imports are wrapped in try/except so
# `transformers.dynamic_module_utils.check_imports` doesn't reject this file
# when running with a minimal environment (and so users who feed in already-
# preprocessed frames don't need decord/cv2/PIL installed).
# ---------------------------------------------------------------------------


def _to_array(frames: Any) -> np.ndarray:
    """Coerce any single video to a numpy array of shape (T, H, W, 3).

    Preserves the input dtype — `do_rescale` decides whether/how to rescale.
    """
    if isinstance(frames, np.ndarray):
        arr = frames
    elif isinstance(frames, torch.Tensor):
        arr = frames.detach().cpu().numpy()
    elif isinstance(frames, (list, tuple)):
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
            f"Expected a single video shaped (T, H, W, 3); got {arr.shape}. "
            f"For batched input pass a list of videos or a (B, T, H, W, 3) tensor."
        )
    return arr


def _decode_video(path: str, num_frames: int) -> np.ndarray:
    """Decode a video file to (num_frames, H, W, 3) uint8 frames."""
    try:
        from decord import VideoReader, cpu  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Decoding video files needs `decord`. "
            "Install it (`pip install decord`) or pass already-decoded frames."
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
    """Bilinear resize one (H, W, 3) frame to (size, size, 3). Preserves dtype.

    Tries cv2 first, falls back to PIL. Both wrapped so neither is a hard dep.
    """
    if frame.shape[0] == size and frame.shape[1] == size:
        return frame
    try:
        import cv2  # type: ignore
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        pass
    try:
        from PIL import Image  # type: ignore
        # PIL needs uint8; round-trip through float for any non-uint8 input.
        if frame.dtype != np.uint8:
            f = np.clip(frame, 0, 255).round().astype(np.uint8)
        else:
            f = frame
        out = np.asarray(Image.fromarray(f).resize((size, size), Image.BILINEAR))
        return out.astype(frame.dtype)
    except ImportError as e:
        raise ImportError(
            "Resizing frames needs either `opencv-python` or `pillow`. "
            "Install one or pass frames already at the target spatial size."
        ) from e


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


def _coerce_to_batch(videos: Any) -> list:
    """Wrap `videos` into a list of single-clip items.

    A "clip" is one of: a video file path, a 4D numpy/torch tensor `(T, H, W, 3)`,
    or a list of frames (PIL or 3D arrays). A 5D numpy/torch `(B, T, H, W, 3)` is
    a batch and gets unstacked. A list whose elements are themselves clips is
    already a batch.
    """
    if isinstance(videos, str):
        return [videos]
    if isinstance(videos, (np.ndarray, torch.Tensor)) and videos.ndim == 5:
        return list(videos)
    if isinstance(videos, (list, tuple)) and videos:
        first = videos[0]
        if isinstance(first, str):
            return list(videos)
        if isinstance(first, (np.ndarray, torch.Tensor)) and first.ndim == 4:
            return list(videos)
    return [videos]


class VideoPrismVideoProcessor(BaseImageProcessor):
    """Preprocessor for VideoPrism. See module docstring for the convention.

    Args:
      image_size:    target spatial size — frames are bilinearly resized to size×size.
      num_frames:    number of frames sampled uniformly from each clip.
      do_resize:     if False, frames are not resized (must already match `image_size`).
      do_rescale:    if True (default), multiplies pixels by `rescale_factor`. Set
                     to False if your frames are already in [0, 1].
      rescale_factor: multiplier applied when `do_rescale=True` (default `1/255`).
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

    def _preprocess_one(self, video: Any, *, do_rescale: bool, rescale_factor: float) -> np.ndarray:
        """Return a (T, H, W, 3) float32 array, in [0, 1] when `do_rescale=True`."""
        if isinstance(video, str) and os.path.exists(video):
            arr = _decode_video(video, num_frames=self.num_frames)
        else:
            arr = _to_array(video)
            arr = _sample_uniform(arr, self.num_frames)

        if self.do_resize:
            arr = np.stack([_resize_frame(f, self.image_size) for f in arr])

        out = arr.astype(np.float32, copy=False)
        if do_rescale:
            out = out * float(rescale_factor)
        return out

    # `__call__` is not part of BaseImageProcessor's public API, but the standard
    # HF `Auto*Processor` flow goes through it.
    def __call__(
        self,
        videos: Any = None,
        return_tensors: str | None = "pt",
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        **_: Any,
    ) -> BatchFeature:
        if videos is None:
            raise ValueError("VideoPrismVideoProcessor requires a `videos` argument.")
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor

        batch = _coerce_to_batch(videos)

        processed = np.stack(
            [self._preprocess_one(v, do_rescale=do_rescale, rescale_factor=rescale_factor)
             for v in batch]
        )  # (B, T, H, W, 3)

        if return_tensors == "pt":
            pixel_values = torch.from_numpy(processed)
        elif return_tensors is None:
            pixel_values = processed
        else:
            raise ValueError(
                f"Only return_tensors='pt' or None is supported (got {return_tensors!r})."
            )

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
