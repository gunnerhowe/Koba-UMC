"""Universal Manifold Codec (UMC) — Structured data compression with built-in similarity search.

Lightweight API (no torch required):
    import umc
    compressed = umc.compress(data)
    recovered = umc.decompress(compressed)

Neural codec API (requires torch, faiss):
    from umc import TieredManifoldCodec
    codec = TieredManifoldCodec.from_checkpoint("model.pt")
    codec.encode_to_mnf(windows, "output.mnf")
"""

__version__ = "0.2.0"

from pathlib import Path
from typing import Optional

import numpy as np


# ---- Lazy imports for torch-dependent classes ----
# ManifoldCodec, ManifoldCodecResult, TieredManifoldCodec, UMCConfig
# are only imported when accessed, so `import umc` and `umc.compress()`
# work without torch installed.

_LAZY_IMPORTS = {
    "ManifoldCodec": "_neural",
    "ManifoldCodecResult": "_neural",
    "TieredManifoldCodec": "_neural",
    "UMCConfig": "config",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(f".{module_name}", __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---- Model-free compression API (lightweight, numpy-only) ----

_ALL_STORAGE_MODES = [
    "lossless", "near_lossless", "lossless_zstd", "lossless_lzma",
    "normalized_lossless", "normalized_lossless_zstd",
    "near_lossless_turbo", "quantized_8", "optimal", "optimal_fast",
    "lossless_fast",
]

# ---- Multi-dtype support (float32, float16, bfloat16) ----

_DTYPE_FLOAT32 = 0
_DTYPE_FLOAT16 = 1
_DTYPE_BFLOAT16 = 2

_DTYPE_INFO = {
    _DTYPE_FLOAT32: (np.float32, 4),
    _DTYPE_FLOAT16: (np.float16, 2),
    _DTYPE_BFLOAT16: (np.uint16, 2),  # stored as uint16 bit pattern
}


def _bfloat16_to_float32(uint16_arr: np.ndarray) -> np.ndarray:
    """Convert bfloat16 bit pattern (as uint16) to float32 (lossless widening)."""
    f32_bits = uint16_arr.astype(np.uint32) << 16
    return f32_bits.view(np.float32)


def _detect_dtype(data) -> tuple[int, type, int]:
    """Detect input dtype. Returns (dtype_code, np_dtype, element_size)."""
    try:
        import torch
        if isinstance(data, torch.Tensor):
            if data.dtype == torch.bfloat16:
                return _DTYPE_BFLOAT16, np.uint16, 2
            elif data.dtype == torch.float16:
                return _DTYPE_FLOAT16, np.float16, 2
            return _DTYPE_FLOAT32, np.float32, 4
    except ImportError:
        pass

    arr = np.asarray(data)
    if arr.dtype == np.float16:
        return _DTYPE_FLOAT16, np.float16, 2
    # Check for ml_dtypes bfloat16
    if 'bfloat16' in getattr(arr.dtype, 'name', ''):
        return _DTYPE_BFLOAT16, np.uint16, 2
    return _DTYPE_FLOAT32, np.float32, 4


def _prepare_array(data, dtype_code: int) -> np.ndarray:
    """Convert input data to numpy array with appropriate dtype."""
    if dtype_code == _DTYPE_BFLOAT16:
        try:
            import torch
            if isinstance(data, torch.Tensor):
                return data.view(torch.int16).numpy().view(np.uint16)
        except ImportError:
            pass
        arr = np.asarray(data)
        if arr.dtype != np.uint16:
            arr = arr.view(np.uint16)
        return arr
    elif dtype_code == _DTYPE_FLOAT16:
        try:
            import torch
            if isinstance(data, torch.Tensor):
                return data.numpy()
        except ImportError:
            pass
        return np.asarray(data, dtype=np.float16)
    else:
        try:
            import torch
            if isinstance(data, torch.Tensor):
                return data.float().numpy()
        except ImportError:
            pass
        return np.asarray(data, dtype=np.float32)


def _to_3d(data: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Reshape any numeric array to 3D for storage compression.

    Returns (reshaped_3d, original_shape).
    Preserves the input dtype.
    """
    original_shape = data.shape

    if data.ndim == 1:
        data = data.reshape(1, -1, 1)
    elif data.ndim == 2:
        # (rows, cols) -> (1, rows, cols)
        data = data.reshape(1, data.shape[0], data.shape[1])
    elif data.ndim == 3:
        pass
    else:
        # Flatten higher dims into features
        n = data.shape[0]
        data = data.reshape(n, data.shape[1], -1)

    return data, original_shape


def compress(data, mode: str = "lossless") -> bytes:
    """Compress a numeric array to bytes (no model needed).

    Only requires numpy — no torch, no training, no config.
    Automatically detects input dtype (float32, float16, bfloat16).

    Args:
        data: Numeric array (numpy array, list, torch tensor, or anything
              np.asarray accepts). Supports float32, float16, and bfloat16.
        mode: Compression mode. Options:
            'lossless'               - Byte transpose + zlib (bit-exact, ~1.3x)
            'lossless_fast'          - Delta + zstd-3 (bit-exact, ~1.7x, fastest)
            'lossless_zstd'          - Byte transpose + zstd (bit-exact)
            'lossless_lzma'          - Byte transpose + lzma (bit-exact, slow)
            'near_lossless'          - Float16 + zlib (<0.01% error, ~2.5x) [float32 only]
            'near_lossless_turbo'    - Float16 + BT + zlib (<0.05% error, ~3x) [float32 only]
            'quantized_8'            - Uint8 + delta + zlib (~2% error, ~5-9x) [float32 only]
            'optimal'                - 128-strategy competition (bit-exact, best ratio)
            'optimal_fast'           - 20-strategy competition (bit-exact, fast)
            'normalized_lossless'    - Normalize [0,1] + BT + zlib [float32 only]
            'normalized_lossless_zstd' - Normalize [0,1] + BT + zstd [float32 only]

    Returns:
        Compressed bytes.

    Example:
        >>> import numpy as np
        >>> import umc
        >>> data = np.random.randn(100, 32, 5).astype(np.float32)
        >>> compressed = umc.compress(data)
        >>> recovered = umc.decompress(compressed)
        >>> np.array_equal(data, recovered)
        True
        >>> # Float16 support
        >>> f16 = np.random.randn(100, 128).astype(np.float16)
        >>> compressed = umc.compress(f16)
        >>> recovered = umc.decompress(compressed)
        >>> np.array_equal(f16, recovered)
        True
    """
    from .codec.tiered import _compress_storage
    import struct

    if mode not in _ALL_STORAGE_MODES:
        raise ValueError(
            f"Unknown mode: {mode!r}. Choose from: {_ALL_STORAGE_MODES}"
        )

    # Detect input dtype
    dtype_code, np_dtype, element_size = _detect_dtype(data)
    arr = _prepare_array(data, dtype_code)

    if arr.size == 0:
        raise ValueError("Cannot compress empty array")

    # Handle NaN/Inf (only for float types, not bfloat16 uint16 view)
    if dtype_code == _DTYPE_FLOAT32:
        if np.any(np.isnan(arr)):
            import warnings
            warnings.warn("Input contains NaN values; replacing with 0.0")
            arr = np.nan_to_num(arr, nan=0.0)
        if np.any(np.isinf(arr)):
            import warnings
            warnings.warn("Input contains Inf values; clipping to float32 range")
            arr = np.clip(arr, np.finfo(np.float32).min, np.finfo(np.float32).max)
    elif dtype_code == _DTYPE_FLOAT16:
        if np.any(np.isnan(arr)):
            import warnings
            warnings.warn("Input contains NaN values; replacing with 0.0")
            arr = np.nan_to_num(arr, nan=0.0)
        if np.any(np.isinf(arr)):
            import warnings
            warnings.warn("Input contains Inf values; clipping to float16 range")
            arr = np.clip(arr, np.finfo(np.float16).min, np.finfo(np.float16).max)

    arr_3d, original_shape = _to_3d(arr)
    compressed = _compress_storage(arr_3d, mode, element_size=element_size)

    if dtype_code == _DTYPE_FLOAT32:
        # Legacy UMCZ format for backward compatibility
        magic = b"UMCZ"
        shape_header = struct.pack("<B", len(original_shape))
        for dim in original_shape:
            shape_header += struct.pack("<I", dim)
        return magic + shape_header + compressed
    else:
        # UMC2 format with dtype byte
        magic = b"UMC2"
        shape_header = struct.pack("<BB", dtype_code, len(original_shape))
        for dim in original_shape:
            shape_header += struct.pack("<I", dim)
        return magic + shape_header + compressed


def decompress(data: bytes) -> np.ndarray:
    """Decompress bytes back to a numpy array.

    Restores the original array shape and dtype from the embedded metadata.
    Float32 and float16 are returned in their original dtype.
    Bfloat16 is returned as float32 (lossless widening, since numpy lacks bfloat16).

    Args:
        data: Compressed bytes from compress().

    Returns:
        Numpy array with original shape. Dtype matches input
        (float32 or float16; bfloat16 returns as float32).

    Example:
        >>> compressed = umc.compress(my_array)
        >>> recovered = umc.decompress(compressed)
    """
    from .codec.tiered import _decompress_storage
    import struct

    if len(data) < 4:
        raise ValueError("Data too short to be UMC compressed")

    magic = data[:4]

    if magic == b"UMCZ":
        # Legacy float32 format
        offset = 4
        ndim = struct.unpack("<B", data[offset:offset + 1])[0]
        offset += 1
        original_shape = []
        for _ in range(ndim):
            dim = struct.unpack("<I", data[offset:offset + 4])[0]
            original_shape.append(dim)
            offset += 4
        original_shape = tuple(original_shape)

        arr = _decompress_storage(data[offset:])
        return arr.reshape(original_shape)

    elif magic == b"UMC2":
        # Multi-dtype format
        offset = 4
        dtype_code = struct.unpack("<B", data[offset:offset + 1])[0]
        offset += 1
        ndim = struct.unpack("<B", data[offset:offset + 1])[0]
        offset += 1
        original_shape = []
        for _ in range(ndim):
            dim = struct.unpack("<I", data[offset:offset + 4])[0]
            original_shape.append(dim)
            offset += 4
        original_shape = tuple(original_shape)

        np_dtype, element_size = _DTYPE_INFO[dtype_code]
        arr = _decompress_storage(
            data[offset:], element_size=element_size, out_dtype=np_dtype
        )

        if dtype_code == _DTYPE_BFLOAT16:
            # Convert uint16 bfloat16 bits to float32 (lossless widening)
            arr = _bfloat16_to_float32(arr.view(np.uint16))

        return arr.reshape(original_shape)

    else:
        raise ValueError(
            "Not a UMC compressed stream (missing UMCZ/UMC2 magic). "
            "If this is a .mnf file, use TieredManifoldCodec.decode_from_mnf() instead."
        )


def compress_optimal(data) -> dict:
    """Compress with strategy competition and return an optimality certificate.

    Tries multiple (preprocessing, compressor) combinations and picks the
    smallest output.  Returns both the compressed bytes and a certificate
    proving how close the result is to the Shannon entropy limit.

    Args:
        data: Numeric array (numpy array, list, etc.).

    Returns:
        Dict with keys:
            compressed: bytes — the compressed data (same as compress(data, mode='optimal'))
            certificate: dict — optimality proof with entropy_gap, ratio, etc.

    Example:
        >>> result = umc.compress_optimal(data)
        >>> print(f"Ratio: {result['certificate']['ratio']:.2f}x")
        >>> print(f"Entropy gap: {result['certificate']['entropy_gap_pct']:.1f}%")
    """
    from .codec.tiered import _compress_storage
    from .codec.optimal import optimal_compress

    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Cannot compress empty array")
    if np.any(np.isnan(arr)):
        arr = np.nan_to_num(arr, nan=0.0)
    if np.any(np.isinf(arr)):
        arr = np.clip(arr, np.finfo(np.float32).min, np.finfo(np.float32).max)

    f32, original_shape = _to_3d(arr)

    # Get certificate directly from optimal_compress
    _payload, certificate = optimal_compress(f32)

    # Build the full compressed stream (with UMCZ header)
    compressed = compress(data, mode="optimal")

    return {
        "compressed": compressed,
        "certificate": {
            "entropy_h0": certificate.entropy_h0,
            "achieved_bpb": certificate.achieved_bpb,
            "entropy_gap_bpb": certificate.entropy_gap_bpb,
            "entropy_gap_pct": certificate.entropy_gap_pct,
            "randomness_p_value": certificate.randomness_p_value,
            "transform": certificate.transform_id,
            "compressor": certificate.compressor_id,
            "original_size": certificate.original_size,
            "compressed_size": certificate.compressed_size,
            "ratio": certificate.ratio,
        },
    }


def compress_raw(data: bytes, mode: str = "optimal") -> bytes:
    """Compress arbitrary binary data (any file type) to bytes.

    Works on any binary data: executables, 3D models, documents, etc.
    Uses strategy competition to find the best compression approach.

    Args:
        data: Raw bytes to compress.
        mode: Compression mode. For raw bytes, 'optimal' is recommended.

    Returns:
        Compressed bytes with UMCR magic header.

    Example:
        >>> raw = open("model.stl", "rb").read()
        >>> compressed = umc.compress_raw(raw)
        >>> recovered = umc.decompress_raw(compressed)
        >>> assert raw == recovered
    """
    from .codec.optimal import optimal_compress_bytes

    if len(data) == 0:
        raise ValueError("Cannot compress empty data")

    magic = b"UMCR"  # R for Raw
    import struct
    size_header = struct.pack("<Q", len(data))

    payload, _cert = optimal_compress_bytes(data)
    return magic + size_header + payload


def decompress_raw(data: bytes) -> bytes:
    """Decompress raw binary data compressed with compress_raw().

    Args:
        data: Compressed bytes from compress_raw().

    Returns:
        Original raw bytes.
    """
    from .codec.optimal import optimal_decompress_bytes
    import struct

    if len(data) < 4:
        raise ValueError("Data too short to be UMC compressed")

    if data[:4] == b"UMCR":
        _original_size = struct.unpack("<Q", data[4:12])[0]
        return optimal_decompress_bytes(data[12:])
    elif data[:4] == b"UMCZ":
        # Numeric array format — decompress and return raw bytes
        arr = decompress(data)
        return arr.tobytes()
    else:
        raise ValueError(
            "Not a UMC compressed stream (missing UMCR/UMCZ magic)."
        )


def compress_file(
    input_path: str,
    output_path: str,
    mode: str = "lossless",
) -> dict:
    """Compress a file to UMC format.

    Supports ANY file type. Numeric formats (.npy, .csv, .parquet, .png,
    .jpg, .wav, .mp4) use specialized loaders. All other formats (.exe,
    .stl, .pdf, .zip, etc.) are compressed as raw binary.

    Args:
        input_path: Path to input file.
        output_path: Path for compressed output.
        mode: Compression mode (see compress()).

    Returns:
        Dict with compression stats (raw_bytes, compressed_bytes, ratio, etc.).
    """
    import time
    input_p = Path(input_path)
    suffix = input_p.suffix.lower()

    start = time.perf_counter()

    if suffix == ".npy":
        data = np.load(str(input_p))
    elif suffix in (".csv", ".tsv", ".txt"):
        import pandas as pd
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(str(input_p), sep=sep)
        numeric = df.select_dtypes(include=[np.number])
        data = numeric.values.astype(np.float32)
    elif suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(str(input_p))
        numeric = df.select_dtypes(include=[np.number])
        data = numeric.values.astype(np.float32)
    elif suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        from .data.image_loader import load_images
        data = load_images(str(input_p))
    elif suffix in (".wav",):
        from .data.audio_loader import load_audio
        data = load_audio(str(input_p))
    elif suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        from .data.video_loader import load_video
        data = load_video(str(input_p))
    else:
        # Universal: compress ANY file as raw bytes
        raw_bytes_data = input_p.read_bytes()
        compressed = compress_raw(raw_bytes_data, mode=mode)
        elapsed = time.perf_counter() - start

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(compressed)

        input_file_bytes = input_p.stat().st_size
        return {
            "input_path": str(input_p),
            "output_path": str(out_p),
            "original_shape": (len(raw_bytes_data),),
            "raw_bytes": len(raw_bytes_data),
            "input_file_bytes": input_file_bytes,
            "compressed_bytes": len(compressed),
            "ratio": len(raw_bytes_data) / max(len(compressed), 1),
            "file_ratio": input_file_bytes / max(len(compressed), 1),
            "mode": mode,
            "elapsed_sec": elapsed,
            "raw_binary": True,
        }

    compressed = compress(data, mode=mode)
    elapsed = time.perf_counter() - start

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_bytes(compressed)

    raw_bytes = data.nbytes
    comp_bytes = len(compressed)
    input_file_bytes = input_p.stat().st_size

    return {
        "input_path": str(input_p),
        "output_path": str(out_p),
        "original_shape": data.shape,
        "raw_bytes": raw_bytes,
        "input_file_bytes": input_file_bytes,
        "compressed_bytes": comp_bytes,
        "ratio": raw_bytes / max(comp_bytes, 1),
        "file_ratio": input_file_bytes / max(comp_bytes, 1),
        "mode": mode,
        "elapsed_sec": elapsed,
    }


def decompress_file(
    input_path: str,
    output_path: str,
) -> dict:
    """Decompress a UMC file back to its original format.

    Output format is determined by extension (.npy, .csv, .wav, .png).
    For raw binary files (compressed with UMCR magic), restores original bytes.

    Args:
        input_path: Path to compressed UMC file.
        output_path: Path for decompressed output.

    Returns:
        Dict with decompression stats.
    """
    import time
    input_p = Path(input_path)
    output_p = Path(output_path)
    suffix = output_p.suffix.lower()

    start = time.perf_counter()
    compressed = input_p.read_bytes()

    # Check if this is raw binary (UMCR) or numeric array (UMCZ)
    if compressed[:4] == b"UMCR":
        raw_data = decompress_raw(compressed)
        elapsed = time.perf_counter() - start
        output_p.parent.mkdir(parents=True, exist_ok=True)
        output_p.write_bytes(raw_data)
        return {
            "input_path": str(input_p),
            "output_path": str(output_p),
            "shape": (len(raw_data),),
            "dtype": "bytes",
            "elapsed_sec": elapsed,
            "raw_binary": True,
        }

    data = decompress(compressed)
    elapsed = time.perf_counter() - start

    output_p.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".npy":
        np.save(str(output_p), data)
    elif suffix in (".csv", ".tsv", ".txt"):
        import pandas as pd
        if data.ndim <= 2:
            flat = data if data.ndim == 2 else data.reshape(-1, 1)
        else:
            flat = data.reshape(-1, data.shape[-1])
        pd.DataFrame(flat).to_csv(str(output_p), index=False)
    elif suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        from .data.image_loader import save_images
        save_images(data, str(output_p))
    elif suffix == ".wav":
        from .data.audio_loader import save_audio
        save_audio(data, str(output_p))
    elif suffix in (".mp4", ".avi", ".mov", ".mkv"):
        from .data.video_loader import save_video
        # Infer frame dimensions — assume square if possible
        if data.ndim == 3:
            n_pixels = data.shape[1]
            side = int(np.sqrt(n_pixels))
            if side * side == n_pixels:
                h, w = side, side
            else:
                h, w = 1, n_pixels
        else:
            h, w = data.shape[1], data.shape[2] if data.ndim >= 3 else 1
        save_video(data, str(output_p), height=h, width=w)
    else:
        np.save(str(output_p), data)

    return {
        "input_path": str(input_p),
        "output_path": str(output_p),
        "shape": data.shape,
        "dtype": str(data.dtype),
        "elapsed_sec": elapsed,
    }


__all__ = [
    "__version__",
    "ManifoldCodec",
    "ManifoldCodecResult",
    "TieredManifoldCodec",
    "UMCConfig",
    "compress",
    "compress_optimal",
    "compress_raw",
    "decompress",
    "decompress_raw",
    "compress_file",
    "decompress_file",
]
