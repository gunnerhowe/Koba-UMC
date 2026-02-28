"""Pre-trained model hub for Universal Manifold Codec (UMC).

Provides infrastructure for discovering, downloading, caching, and loading
pre-trained UMC models.  Model checkpoints are downloaded on first use and
stored in a local cache directory (``~/.cache/umc/models/`` by default).

Quick start::

    from umc.hub import list_models, load_model

    # See what's available
    for info in list_models():
        print(info["name"], "-", info["description"])

    # Download and instantiate a codec
    codec = load_model("financial-v1")
    codec.encode_to_mnf(windows, "compressed.mnf")

Note
----
The model files referenced in :class:`ModelRegistry` may not be published
yet.  Attempting to download an unavailable model will raise a clear error
with the HTTP status code so that users know the release has not landed.
"""

from __future__ import annotations

import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from umc._neural import TieredManifoldCodec

__all__ = [
    "ModelRegistry",
    "list_models",
    "load_model",
    "download_model",
    "clear_cache",
]

# ---------------------------------------------------------------------------
# Default cache location
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "umc" / "models"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Registry of available pre-trained UMC models.

    Each entry maps a human-readable model name to a dictionary of metadata:

    * **description** -- short prose description of the training data.
    * **url** -- download URL for the ``.pt`` checkpoint file.
    * **size_mb** -- approximate file size in megabytes.
    * **window_size** -- temporal window length the model was trained with.
    * **n_features** -- number of input features (channels) per time step.
    * **recommended_mode** -- suggested ``storage_mode`` for
      :meth:`~umc._neural.TieredManifoldCodec.from_checkpoint`.
    """

    MODELS: dict[str, dict] = {
        "financial-v1": {
            "description": "Pre-trained on 10 years of S&P 500 daily OHLCV data",
            "url": (
                "https://github.com/umccodec/umc-models/releases/"
                "download/v0.2.0/financial-v1.pt"
            ),
            "size_mb": 2.5,
            "window_size": 32,
            "n_features": 5,
            "recommended_mode": "near_lossless",
        },
        "iot-sensors-v1": {
            "description": "Pre-trained on multi-channel IoT sensor streams",
            "url": (
                "https://github.com/umccodec/umc-models/releases/"
                "download/v0.2.0/iot-sensors-v1.pt"
            ),
            "size_mb": 3.3,
            "window_size": 64,
            "n_features": 8,
            "recommended_mode": "near_lossless_turbo",
        },
        "audio-v1": {
            "description": "Pre-trained on 16kHz mono audio spectrograms",
            "url": (
                "https://github.com/umccodec/umc-models/releases/"
                "download/v0.2.0/audio-v1.pt"
            ),
            "size_mb": 4.2,
            "window_size": 1024,
            "n_features": 1,
            "recommended_mode": "near_lossless",
        },
        "scientific-v1": {
            "description": "Pre-trained on multi-physics simulation data (5 channels)",
            "url": (
                "https://github.com/umccodec/umc-models/releases/"
                "download/v0.2.0/scientific-v1.pt"
            ),
            "size_mb": 3.0,
            "window_size": 64,
            "n_features": 5,
            "recommended_mode": "near_lossless",
        },
    }

    @classmethod
    def get(cls, name: str) -> dict:
        """Return metadata for *name*, raising ``KeyError`` with a helpful
        message if the model is not registered."""
        if name not in cls.MODELS:
            available = ", ".join(sorted(cls.MODELS))
            raise KeyError(
                f"Unknown model {name!r}. Available models: {available}"
            )
        return cls.MODELS[name]

    @classmethod
    def names(cls) -> list[str]:
        """Return a sorted list of registered model names."""
        return sorted(cls.MODELS)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_cache_dir(cache_dir: Optional[str] = None) -> Path:
    """Return the cache directory as a :class:`~pathlib.Path`, creating it
    if necessary."""
    path = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format_bytes(n_bytes: int) -> str:
    """Return a human-readable string for a byte count."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024  # type: ignore[assignment]
    return f"{n_bytes:.1f} TB"


def _download_with_progress(url: str, dest: Path, expected_mb: float) -> None:
    """Download *url* to *dest*, printing a progress bar to stderr.

    Parameters
    ----------
    url:
        Remote URL to fetch.
    dest:
        Local file path to write to.  A temporary ``.part`` suffix is used
        during download to avoid leaving corrupt partial files.
    expected_mb:
        Approximate expected file size in megabytes (used for the progress
        bar when the server does not send a ``Content-Length`` header).
    """
    part_path = dest.with_suffix(dest.suffix + ".part")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "umc-hub/0.2"})
        response = urllib.request.urlopen(req)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(
                f"Model checkpoint not found at {url}. "
                f"The model may not have been released yet. "
                f"(HTTP {exc.code})"
            ) from None
        raise RuntimeError(
            f"Failed to download model from {url} (HTTP {exc.code}): {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Could not connect to {url}: {exc.reason}"
        ) from exc

    # Determine total size -- prefer Content-Length, fall back to expected_mb.
    content_length = response.headers.get("Content-Length")
    total_bytes = int(content_length) if content_length else int(expected_mb * 1024 * 1024)

    downloaded = 0
    chunk_size = 64 * 1024  # 64 KB
    start_time = time.monotonic()

    try:
        with open(part_path, "wb") as fout:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                fout.write(chunk)
                downloaded += len(chunk)

                # Progress reporting
                elapsed = time.monotonic() - start_time
                speed = downloaded / elapsed if elapsed > 0 else 0
                pct = min(downloaded / total_bytes * 100, 100.0) if total_bytes else 0
                bar_width = 30
                filled = int(bar_width * pct / 100)
                bar = "#" * filled + "-" * (bar_width - filled)
                sys.stderr.write(
                    f"\r  [{bar}] {pct:5.1f}%  "
                    f"{_format_bytes(downloaded)} / {_format_bytes(total_bytes)}  "
                    f"({_format_bytes(int(speed))}/s)"
                )
                sys.stderr.flush()

        sys.stderr.write("\n")
        sys.stderr.flush()

        # Atomic-ish rename so a crash never leaves a half-written file
        # under the final name.
        part_path.replace(dest)

    except BaseException:
        # Clean up the partial download on any failure (including Ctrl-C).
        if part_path.exists():
            part_path.unlink()
        raise


def _verify_file(path: Path, expected_mb: float) -> None:
    """Run basic integrity checks on a downloaded checkpoint.

    Currently this verifies that the file is non-empty and its size is in
    the right ballpark (within 50 % of the advertised size).  A future
    version may add SHA-256 verification once checksums are published.

    Raises
    ------
    RuntimeError
        If the file appears corrupt or truncated.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected checkpoint at {path} but file does not exist")

    actual_bytes = path.stat().st_size
    if actual_bytes == 0:
        path.unlink()
        raise RuntimeError(
            f"Downloaded file {path} is empty (0 bytes). "
            "The download may have failed silently."
        )

    expected_bytes = expected_mb * 1024 * 1024
    if actual_bytes < expected_bytes * 0.5:
        raise RuntimeError(
            f"Downloaded file {path} is suspiciously small "
            f"({_format_bytes(actual_bytes)} vs expected ~{expected_mb:.1f} MB). "
            "The file may be truncated. Delete it and try again."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_models() -> list[dict]:
    """List all available pre-trained models.

    Returns
    -------
    list[dict]
        Each dictionary contains the model ``name`` plus all metadata fields
        from :attr:`ModelRegistry.MODELS` (``description``, ``url``,
        ``size_mb``, ``window_size``, ``n_features``, ``recommended_mode``).

    Examples
    --------
    >>> from umc.hub import list_models
    >>> for m in list_models():
    ...     print(f"{m['name']:20s}  {m['size_mb']:.1f} MB  {m['description']}")
    audio-v1              4.2 MB  Pre-trained on 16kHz mono audio spectrograms
    financial-v1          2.4 MB  Pre-trained on 10 years of S&P 500 daily OHLCV data
    iot-sensors-v1        3.1 MB  Pre-trained on multi-channel IoT sensor streams
    """
    results: list[dict] = []
    for name in sorted(ModelRegistry.MODELS):
        entry = {"name": name, **ModelRegistry.MODELS[name]}
        results.append(entry)
    return results


def download_model(name: str, *, cache_dir: Optional[str] = None) -> Path:
    """Download a model checkpoint to the local cache without loading it.

    If the checkpoint already exists in the cache directory the download is
    skipped and the cached path is returned immediately.

    Parameters
    ----------
    name:
        Registered model name (e.g. ``"financial-v1"``).
    cache_dir:
        Override the default cache directory (``~/.cache/umc/models/``).

    Returns
    -------
    pathlib.Path
        Absolute path to the downloaded ``.pt`` file.

    Raises
    ------
    KeyError
        If *name* is not a registered model.
    FileNotFoundError
        If the remote file does not exist (HTTP 404).
    ConnectionError
        If the host cannot be reached.
    RuntimeError
        If the download fails for any other HTTP reason or integrity checks
        fail.
    """
    info = ModelRegistry.get(name)
    cache = _resolve_cache_dir(cache_dir)
    dest = cache / f"{name}.pt"

    if dest.exists():
        print(f"Using cached model: {dest}", file=sys.stderr)
        return dest

    print(f"Downloading {name} ({info['size_mb']:.1f} MB) ...", file=sys.stderr)
    _download_with_progress(info["url"], dest, info["size_mb"])

    _verify_file(dest, info["size_mb"])
    print(f"Saved to {dest}", file=sys.stderr)

    return dest


def load_model(
    name: str,
    *,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    local_path: Optional[str] = None,
) -> "TieredManifoldCodec":
    """Download (if needed) and load a pre-trained UMC model.

    The checkpoint is fetched from the URL in :class:`ModelRegistry` and
    stored under ``~/.cache/umc/models/`` (or *cache_dir*).  Subsequent
    calls with the same *name* reuse the cached file.

    Parameters
    ----------
    name:
        Registered model name (e.g. ``"financial-v1"``).
    device:
        Torch device string passed to
        :meth:`~umc._neural.TieredManifoldCodec.from_checkpoint`
        (default ``"cpu"``).
    cache_dir:
        Override the default cache directory.
    local_path:
        Path to a local ``.pt`` checkpoint file. If provided, skips the
        download entirely and loads directly from this path.

    Returns
    -------
    TieredManifoldCodec
        A ready-to-use codec instance.

    Examples
    --------
    >>> from umc.hub import load_model
    >>> codec = load_model("financial-v1", device="cuda")
    >>> codec.encode_to_mnf(windows, "out.mnf")

    Load from a local checkpoint:

    >>> codec = load_model("financial-v1", local_path="checkpoints/pretrained/financial-v1.pt")
    """
    if local_path is not None:
        checkpoint_path = Path(local_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Local checkpoint not found: {local_path}")
    else:
        checkpoint_path = download_model(name, cache_dir=cache_dir)

    info = ModelRegistry.get(name)
    storage_mode = info.get("recommended_mode", "lossless")

    # Lazy import -- keeps `import umc.hub` lightweight for environments
    # without torch installed.
    from umc._neural import TieredManifoldCodec  # noqa: WPS433

    return TieredManifoldCodec.from_checkpoint(
        str(checkpoint_path),
        device=device,
        storage_mode=storage_mode,
    )


def clear_cache(cache_dir: Optional[str] = None) -> int:
    """Remove all cached model checkpoints.

    Parameters
    ----------
    cache_dir:
        Override the default cache directory.

    Returns
    -------
    int
        Total number of bytes freed.

    Examples
    --------
    >>> from umc.hub import clear_cache
    >>> freed = clear_cache()
    >>> print(f"Freed {freed / 1024 / 1024:.1f} MB")
    """
    cache = _resolve_cache_dir(cache_dir)
    freed = 0

    if not cache.exists():
        return freed

    for item in cache.iterdir():
        if item.is_file():
            freed += item.stat().st_size
            item.unlink()
            print(f"Removed {item.name}", file=sys.stderr)

    # Also remove stale .part files one level up (shouldn't normally exist).
    for part in cache.glob("*.part"):
        freed += part.stat().st_size
        part.unlink()

    if freed:
        print(f"Freed {_format_bytes(freed)}", file=sys.stderr)
    else:
        print("Cache is already empty.", file=sys.stderr)

    return freed
