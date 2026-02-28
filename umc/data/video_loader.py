"""Load and save video as float32 arrays for UMC compression.

Videos are loaded frame-by-frame and reshaped to (n_frames, height*width, channels)
for storage tier compression. Uses OpenCV when available, falls back to imageio.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_video(
    path: str,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    grayscale: bool = False,
) -> np.ndarray:
    """Load video file as float32 array.

    Args:
        path: Path to video file (.mp4, .avi, .mov, .mkv, .webm).
        max_frames: Maximum number of frames to load (None = all).
        resize: Optional (width, height) to resize frames.
        grayscale: Convert to grayscale.

    Returns:
        (n_frames, height*width, channels) float32 array in [0, 1].
        Metadata dict is stored as attributes if possible.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    try:
        return _load_video_cv2(str(p), max_frames, resize, grayscale)
    except ImportError:
        return _load_video_imageio(str(p), max_frames, resize, grayscale)


def _load_video_cv2(
    path: str,
    max_frames: Optional[int],
    resize: Optional[Tuple[int, int]],
    grayscale: bool,
) -> np.ndarray:
    """Load video using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame[:, :, np.newaxis]
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if resize is not None:
            frame = cv2.resize(frame, resize)

        frames.append(frame)
        count += 1
        if max_frames is not None and count >= max_frames:
            break

    cap.release()

    if not frames:
        raise ValueError(f"No frames read from: {path}")

    arr = np.stack(frames).astype(np.float32) / 255.0
    n, h, w, c = arr.shape
    return arr.reshape(n, h * w, c)


def _load_video_imageio(
    path: str,
    max_frames: Optional[int],
    resize: Optional[Tuple[int, int]],
    grayscale: bool,
) -> np.ndarray:
    """Load video using imageio (fallback)."""
    import imageio.v3 as iio

    frames_raw = iio.imread(path, plugin="pyav")

    if max_frames is not None:
        frames_raw = frames_raw[:max_frames]

    frames = []
    for frame in frames_raw:
        if grayscale and frame.ndim == 3 and frame.shape[2] == 3:
            frame = np.mean(frame, axis=2, keepdims=True).astype(np.uint8)
        if resize is not None:
            from PIL import Image
            img = Image.fromarray(frame)
            img = img.resize(resize, Image.LANCZOS)
            frame = np.asarray(img)
        if frame.ndim == 2:
            frame = frame[:, :, np.newaxis]
        frames.append(frame)

    arr = np.stack(frames).astype(np.float32) / 255.0
    n, h, w, c = arr.shape
    return arr.reshape(n, h * w, c)


def save_video(
    data: np.ndarray,
    path: str,
    height: int,
    width: int,
    fps: float = 30.0,
) -> None:
    """Save float32 array back to video file.

    Args:
        data: (n_frames, height*width, channels) float32 in [0, 1].
        path: Output video file path.
        height: Frame height in pixels.
        width: Frame width in pixels.
        fps: Frames per second.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data.reshape(1, data.shape[0], data.shape[1])

    n_frames, n_pixels, channels = data.shape

    try:
        _save_video_cv2(data, str(p), height, width, channels, fps)
    except ImportError:
        _save_video_imageio(data, str(p), height, width, channels, fps)


def _save_video_cv2(
    data: np.ndarray, path: str,
    height: int, width: int, channels: int, fps: float,
) -> None:
    """Save video using OpenCV."""
    import cv2

    ext = Path(path).suffix.lower()
    fourcc_map = {
        ".mp4": "mp4v",
        ".avi": "XVID",
        ".mov": "mp4v",
        ".mkv": "mp4v",
    }
    fourcc = cv2.VideoWriter_fourcc(*fourcc_map.get(ext, "mp4v"))
    is_color = channels >= 3

    writer = cv2.VideoWriter(path, fourcc, fps, (width, height), isColor=is_color)
    if not writer.isOpened():
        raise IOError(f"Cannot create video writer for: {path}")

    for i in range(len(data)):
        frame = data[i].reshape(height, width, channels)
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        if channels >= 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif channels == 1:
            frame = frame.squeeze(-1)
        writer.write(frame)

    writer.release()


def _save_video_imageio(
    data: np.ndarray, path: str,
    height: int, width: int, channels: int, fps: float,
) -> None:
    """Save video using imageio (fallback)."""
    import imageio.v3 as iio

    frames = []
    for i in range(len(data)):
        frame = data[i].reshape(height, width, channels)
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        if channels == 1:
            frame = frame.squeeze(-1)
        frames.append(frame)

    iio.imwrite(path, np.stack(frames), fps=fps, plugin="pyav")


def get_video_info(path: str) -> dict:
    """Get basic video metadata without loading all frames.

    Returns:
        Dict with keys: width, height, fps, n_frames, duration_sec.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open: {path}")
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        info["duration_sec"] = info["n_frames"] / max(info["fps"], 1)
        return info
    except ImportError:
        import imageio.v3 as iio
        meta = iio.immeta(path, plugin="pyav")
        return {
            "width": meta.get("size", [0, 0])[0],
            "height": meta.get("size", [0, 0])[1],
            "fps": meta.get("fps", 30),
            "n_frames": meta.get("n_images", 0),
            "duration_sec": meta.get("duration", 0),
        }
