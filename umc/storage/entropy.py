"""Entropy coding for VQ indices.

Compresses VQ codebook indices using frequency-based coding.

Compression modes:
  - 'flat':     All indices concatenated, zlib compressed (simple, fast)
  - 'grouped':  Per-level grouping for better locality, then zlib (better ratio)
  - 'optimal':  Tries multiple strategies (grouped, delta-coded, zstd) and picks smallest

Per-level grouping works because each RVQ level has different entropy:
  - Level 0: high entropy (most codes used, ~7 bits/index)
  - Level N: lower entropy (fewer codes, residuals concentrate, ~4-5 bits/index)
Grouping same-level indices together gives zlib better runs to exploit.

Delta coding works because adjacent patches within a window often use similar
codes, so delta + zigzag encoding produces small values that compress well.
"""

import struct
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import zstandard as _zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False


@dataclass
class VQIndices:
    """VQ indices for a batch of windows."""
    top_indices: np.ndarray       # (n_windows,) uint8 or uint16
    bottom_indices: np.ndarray    # (n_windows, n_patches, n_levels) uint8 or uint16
    n_patches: int
    n_levels: int
    top_n_codes: int
    bottom_n_codes: int


def measure_entropy(indices: np.ndarray, n_codes: int) -> float:
    """Measure Shannon entropy of index array in bits per index.

    Args:
        indices: Integer array of codebook indices.
        n_codes: Total number of possible codes.

    Returns:
        Entropy in bits per index.
    """
    flat = indices.ravel()
    counts = np.bincount(flat, minlength=n_codes)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# Format version tags (first byte of compressed data after zlib decompress)
_FORMAT_FLAT = 0
_FORMAT_GROUPED = 1
_FORMAT_DELTA = 2       # Delta-coded + grouped
_FORMAT_ZSTD = 3        # zstd outer compression (tag stored BEFORE compressed data)


def _zigzag_encode(arr: np.ndarray) -> np.ndarray:
    """Zigzag-encode signed int8 to unsigned uint8 (small magnitudes -> small values)."""
    s = arr.astype(np.int16)
    return ((s << 1) ^ (s >> 15)).astype(np.uint8)


def _zigzag_decode(arr: np.ndarray) -> np.ndarray:
    """Reverse zigzag encoding."""
    u = arr.astype(np.uint16)
    return (np.right_shift(u, 1) ^ -(u & 1)).astype(np.int8)


def _delta_encode_levels(bottom_indices: np.ndarray) -> np.ndarray:
    """Delta-encode bottom indices along the patch axis.

    Adjacent patches within a window often use similar codes,
    so deltas are small and compress better.
    """
    # bottom_indices: (n_windows, n_patches, n_levels) uint8
    deltas = np.zeros_like(bottom_indices, dtype=np.int8)
    deltas[:, 0, :] = bottom_indices[:, 0, :].astype(np.int8)
    deltas[:, 1:, :] = (
        bottom_indices[:, 1:, :].astype(np.int16)
        - bottom_indices[:, :-1, :].astype(np.int16)
    ).astype(np.int8)
    return _zigzag_encode(deltas)


def _delta_decode_levels(encoded: np.ndarray) -> np.ndarray:
    """Reverse delta encoding."""
    deltas = _zigzag_decode(encoded).astype(np.int16)
    result = np.zeros_like(deltas, dtype=np.int16)
    result[:, 0, :] = deltas[:, 0, :]
    for p in range(1, deltas.shape[1]):
        result[:, p, :] = result[:, p - 1, :] + deltas[:, p, :]
    return result.astype(np.uint8)


def _build_grouped_payload(vq_idx: VQIndices, delta: bool = False) -> bytes:
    """Build level-grouped payload bytes."""
    n_windows = len(vq_idx.top_indices)
    top_bits = int(np.log2(max(vq_idx.top_n_codes, 1)))
    bottom_bits = int(np.log2(max(vq_idx.bottom_n_codes, 1)))
    tag = _FORMAT_DELTA if delta else _FORMAT_GROUPED

    header = struct.pack(
        "<BIBBBB",
        tag,
        n_windows,
        vq_idx.n_patches,
        vq_idx.n_levels,
        top_bits,
        bottom_bits,
    )

    top_bytes = vq_idx.top_indices.astype(np.uint8).tobytes()

    bottom = vq_idx.bottom_indices
    if delta:
        bottom = _delta_encode_levels(bottom)

    level_blocks = []
    for lvl in range(vq_idx.n_levels):
        level_blocks.append(bottom[:, :, lvl].astype(np.uint8).tobytes())

    return header + top_bytes + b"".join(level_blocks)


def compress_indices(vq_idx: VQIndices, mode: str = "grouped") -> bytes:
    """Compress VQ indices.

    Args:
        vq_idx: VQ indices to compress.
        mode: Compression mode:
            'flat':     v1 format, backward compatible
            'grouped':  Per-level grouping + zlib (good ratio)
            'optimal':  Tries all strategies, picks smallest result

    Returns:
        Compressed bytes.
    """
    if mode == "optimal":
        return _compress_optimal(vq_idx)

    n_windows = len(vq_idx.top_indices)
    top_bytes = vq_idx.top_indices.astype(np.uint8).tobytes()

    if mode == "grouped":
        raw = _build_grouped_payload(vq_idx, delta=False)
    elif mode == "flat":
        # Old format (v1 compatible, no format tag)
        header = struct.pack(
            "<IBBBB",
            n_windows,
            vq_idx.n_patches,
            vq_idx.n_levels,
            int(np.log2(max(vq_idx.top_n_codes, 1))),
            int(np.log2(max(vq_idx.bottom_n_codes, 1))),
        )
        bottom_bytes = vq_idx.bottom_indices.astype(np.uint8).tobytes()
        raw = header + top_bytes + bottom_bytes
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'flat', 'grouped', or 'optimal'.")

    return zlib.compress(raw, level=9)


def _compress_optimal(vq_idx: VQIndices) -> bytes:
    """Try multiple compression strategies and return the smallest result.

    Strategies tried:
        1. Grouped + zlib (baseline)
        2. Delta-coded grouped + zlib (better when adjacent patches correlate)
        3. Grouped + zstd (if available, often beats zlib)
        4. Delta-coded grouped + zstd (if available)
    """
    candidates = []

    # Strategy 1: grouped + zlib
    raw_grouped = _build_grouped_payload(vq_idx, delta=False)
    candidates.append(zlib.compress(raw_grouped, level=9))

    # Strategy 2: delta + zlib
    raw_delta = _build_grouped_payload(vq_idx, delta=True)
    candidates.append(zlib.compress(raw_delta, level=9))

    # Strategy 3 & 4: zstd variants (if available)
    if _HAS_ZSTD:
        cctx = _zstd.ZstdCompressor(level=19)
        # Tag byte prefix so decompressor knows it's zstd
        zstd_tag = struct.pack("<B", _FORMAT_ZSTD)
        candidates.append(zstd_tag + cctx.compress(raw_grouped))
        candidates.append(zstd_tag + cctx.compress(raw_delta))

    return min(candidates, key=len)


def decompress_indices(data: bytes) -> VQIndices:
    """Decompress VQ indices. Auto-detects format.

    Supported formats:
        - Flat/v1 (zlib, no format tag)
        - Grouped/v2 (zlib, format tag = 1)
        - Delta/v3 (zlib, format tag = 2)
        - Any of above with zstd outer compression (prefix tag = 3)

    Args:
        data: Compressed bytes from compress_indices().

    Returns:
        VQIndices with decompressed index arrays.
    """
    # Check for zstd prefix tag (stored BEFORE compressed data)
    if len(data) > 1 and data[0] == _FORMAT_ZSTD:
        if not _HAS_ZSTD:
            raise ImportError("zstandard package required to decompress zstd-encoded indices")
        dctx = _zstd.ZstdDecompressor()
        raw = dctx.decompress(data[1:])
    else:
        raw = zlib.decompress(data)

    new_header_size = struct.calcsize("<BIBBBB")  # 9 bytes
    old_header_size = struct.calcsize("<IBBBB")    # 8 bytes

    # Try new format: first byte is format tag (1=grouped, 2=delta)
    if len(raw) >= new_header_size:
        format_tag = struct.unpack_from("<B", raw, 0)[0]
        if format_tag in (_FORMAT_GROUPED, _FORMAT_DELTA):
            _, n_windows, n_patches, n_levels, top_bits, bottom_bits = struct.unpack(
                "<BIBBBB", raw[:new_header_size]
            )
            expected_payload = n_windows + n_windows * n_patches * n_levels
            if new_header_size + expected_payload == len(raw):
                return _decompress_new_format(
                    raw, new_header_size, format_tag,
                    n_windows, n_patches, n_levels, top_bits, bottom_bits,
                )

    # Old format (flat): <I B B B B> = 8 bytes header, no format tag
    n_windows, n_patches, n_levels, top_bits, bottom_bits = struct.unpack(
        "<IBBBB", raw[:old_header_size]
    )
    return _decompress_old_format(
        raw, old_header_size, n_windows, n_patches, n_levels, top_bits, bottom_bits,
    )


def _decompress_new_format(
    raw: bytes, header_size: int, format_tag: int,
    n_windows: int, n_patches: int, n_levels: int,
    top_bits: int, bottom_bits: int,
) -> VQIndices:
    """Decompress new format (with format tag)."""
    top_n_codes = 2 ** top_bits
    bottom_n_codes = 2 ** bottom_bits
    offset = header_size

    # Top indices
    top_end = offset + n_windows
    top_indices = np.frombuffer(raw[offset:top_end], dtype=np.uint8).copy()
    offset = top_end

    # Bottom indices: per-level blocks
    bottom_indices = np.empty(
        (n_windows, n_patches, n_levels), dtype=np.uint8
    )
    level_size = n_windows * n_patches
    for lvl in range(n_levels):
        level_data = np.frombuffer(
            raw[offset:offset + level_size], dtype=np.uint8
        ).reshape(n_windows, n_patches)
        bottom_indices[:, :, lvl] = level_data
        offset += level_size

    # Reverse delta encoding if needed
    if format_tag == _FORMAT_DELTA:
        bottom_indices = _delta_decode_levels(bottom_indices)

    return VQIndices(
        top_indices=top_indices,
        bottom_indices=bottom_indices,
        n_patches=n_patches,
        n_levels=n_levels,
        top_n_codes=top_n_codes,
        bottom_n_codes=bottom_n_codes,
    )


def _decompress_old_format(
    raw: bytes, header_size: int,
    n_windows: int, n_patches: int, n_levels: int,
    top_bits: int, bottom_bits: int,
) -> VQIndices:
    """Decompress old format (no format tag, flat layout)."""
    top_n_codes = 2 ** top_bits
    bottom_n_codes = 2 ** bottom_bits
    offset = header_size

    top_end = offset + n_windows
    top_indices = np.frombuffer(raw[offset:top_end], dtype=np.uint8).copy()
    offset = top_end

    bottom_size = n_windows * n_patches * n_levels
    bottom_flat = np.frombuffer(raw[offset:offset + bottom_size], dtype=np.uint8).copy()
    bottom_indices = bottom_flat.reshape(n_windows, n_patches, n_levels)

    return VQIndices(
        top_indices=top_indices,
        bottom_indices=bottom_indices,
        n_patches=n_patches,
        n_levels=n_levels,
        top_n_codes=top_n_codes,
        bottom_n_codes=bottom_n_codes,
    )


def compute_compression_stats(
    vq_idx: VQIndices,
    raw_window_bytes: int = 5120,
) -> dict:
    """Compute compression statistics for VQ indices.

    Args:
        vq_idx: VQ indices to analyze.
        raw_window_bytes: Size of uncompressed window in bytes (default: 256*5*4).

    Returns:
        Dict with compression metrics.
    """
    n_windows = len(vq_idx.top_indices)

    # Naive storage (1 byte per index)
    naive_bytes_per_window = 1 + vq_idx.n_patches * vq_idx.n_levels
    naive_total = naive_bytes_per_window * n_windows

    # Entropy-based theoretical minimum
    top_entropy = measure_entropy(vq_idx.top_indices, vq_idx.top_n_codes)
    top_bits_total = top_entropy * n_windows

    bottom_entropies = []
    bottom_bits_total = 0
    for lvl in range(vq_idx.n_levels):
        level_indices = vq_idx.bottom_indices[:, :, lvl]
        ent = measure_entropy(level_indices, vq_idx.bottom_n_codes)
        bottom_entropies.append(ent)
        bottom_bits_total += ent * n_windows * vq_idx.n_patches

    total_bits = top_bits_total + bottom_bits_total
    entropy_bytes_per_window = total_bits / 8 / n_windows

    # Actual compression (all modes)
    compressed_grouped = compress_indices(vq_idx, mode="grouped")
    compressed_flat = compress_indices(vq_idx, mode="flat")
    compressed_optimal = compress_indices(vq_idx, mode="optimal")
    best_compressed = min(compressed_grouped, compressed_flat, compressed_optimal, key=len)
    best_bytes_per_window = len(best_compressed) / n_windows

    return {
        "n_windows": n_windows,
        "naive_bytes_per_window": naive_bytes_per_window,
        "naive_compression": raw_window_bytes / naive_bytes_per_window,
        "entropy_bytes_per_window": entropy_bytes_per_window,
        "entropy_compression": raw_window_bytes / entropy_bytes_per_window,
        "best_bytes_per_window": best_bytes_per_window,
        "best_compression": raw_window_bytes / best_bytes_per_window,
        "top_entropy_bits": top_entropy,
        "bottom_entropy_bits": bottom_entropies,
        "compressed_size": len(best_compressed),
        "grouped_size": len(compressed_grouped),
        "flat_size": len(compressed_flat),
        "optimal_size": len(compressed_optimal),
    }
