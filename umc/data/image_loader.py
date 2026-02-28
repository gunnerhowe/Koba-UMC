"""Load and save images as float32 arrays for UMC compression."""

import numpy as np
from pathlib import Path
from typing import Optional


def load_images(path: str, max_size: Optional[int] = None) -> np.ndarray:
    """Load image(s) as float32 array in [0, 1].

    Single image -> (1, height*width, channels)
    Directory of images -> (n_images, height*width, channels)

    Args:
        path: Path to image file or directory of images.
        max_size: Optional max dimension (resizes if larger).

    Returns:
        (n_images, height*width, channels) float32 array.
    """
    from PIL import Image

    p = Path(path)

    if p.is_file():
        files = [p]
    elif p.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
        if not files:
            raise ValueError(f"No image files found in {path}")
    else:
        raise FileNotFoundError(f"Image path not found: {path}")

    arrays = []
    for f in files:
        img = Image.open(f)
        if max_size and max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]  # grayscale -> (H, W, 1)
        h, w, c = arr.shape
        arrays.append(arr.reshape(1, h * w, c))

    return np.concatenate(arrays, axis=0)


def save_images(
    data: np.ndarray,
    path: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> None:
    """Save float32 array back to image file(s).

    Args:
        data: Array from load_images or decompress. If 3D (n, pixels, channels),
              height/width must be provided or inferred as square.
        path: Output path. For multiple images, uses path as directory.
        height: Image height (pixels). If None, assumes square.
        width: Image width (pixels). If None, assumes square.
    """
    from PIL import Image

    if data.ndim == 2:
        data = data.reshape(1, data.shape[0], data.shape[1])

    n_images, n_pixels, channels = data.shape

    if height is None or width is None:
        side = int(np.sqrt(n_pixels))
        if side * side == n_pixels:
            height, width = side, side
        else:
            height = 1
            width = n_pixels

    p = Path(path)

    for i in range(n_images):
        arr = data[i].reshape(height, width, channels)
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        if channels == 1:
            arr = arr.squeeze(-1)
        img = Image.fromarray(arr)

        if n_images == 1:
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(p))
        else:
            p.mkdir(parents=True, exist_ok=True)
            img.save(str(p / f"image_{i:04d}.png"))
