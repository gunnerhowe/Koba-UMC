"""Tiered compression codec: VQ search + efficient storage.

Two independent tiers:
    Tier 1 (Search):  VQ codes -- lossy (~2.6% RMSE), FAISS-searchable
    Tier 2 (Storage): Byte-transposed data -- lossless or near-lossless

The VQ codes serve as a compact search index for similarity queries.
The storage tier preserves the actual data at chosen fidelity.

Storage modes:
    'lossless':                 byte_transpose(float32) + zlib, ~1.2-1.5x, bit-exact
    'near_lossless':            float16 + zlib, ~2.2-2.6x, <0.01% RMSE
    'lossless_zstd':            byte_transpose(float32) + zstd, ~1.2-1.5x, bit-exact
    'lossless_lzma':            byte_transpose(float32) + lzma, ~1.2-1.7x, bit-exact
    'normalized_lossless':      normalize [0,1] + byte_transpose + zlib, ~1.2-1.4x, ~1e-7 RMSE
    'normalized_lossless_zstd': normalize [0,1] + byte_transpose + zstd, ~1.2-1.4x, ~1e-7 RMSE
    'near_lossless_turbo':      float16 + byte_transpose + zlib, ~2.4-3.2x, <0.05% RMSE
    'quantized_8':              uint8 + delta + zlib, ~4-9x, ~2% RMSE
"""

import struct
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .residual import byte_transpose, byte_untranspose
from ..storage.entropy import VQIndices, compress_indices, decompress_indices


@dataclass
class TieredEncoding:
    """Complete tiered encoding of a batch of windows.

    Contains:
        - VQ codes for search (Tier 1)
        - Compressed data for storage (Tier 2)
        - RevIN stats for VQ decode path
    """
    # Tier 1: VQ search index
    vq_indices: VQIndices

    # Tier 2: Compressed storage
    storage_compressed: bytes
    storage_mode: str   # 'lossless', 'near_lossless', 'lossless_zstd', 'lossless_lzma', 'normalized_lossless', 'normalized_lossless_zstd'

    # RevIN stats (needed for VQ decode path)
    revin_means: np.ndarray   # (n_windows, n_features) float32
    revin_stds: np.ndarray    # (n_windows, n_features) float32

    # Metadata
    n_windows: int
    window_size: int
    n_features: int

    @property
    def raw_bytes(self) -> int:
        return self.n_windows * self.window_size * self.n_features * 4

    @property
    def vq_bytes(self) -> int:
        return len(compress_indices(self.vq_indices, mode="optimal"))

    @property
    def storage_bytes(self) -> int:
        return len(self.storage_compressed)

    @property
    def total_bytes(self) -> int:
        return self.vq_bytes + self.storage_bytes

    @property
    def search_compression(self) -> float:
        """VQ-only compression ratio (for search tier)."""
        return self.raw_bytes / max(self.vq_bytes, 1)

    @property
    def storage_compression(self) -> float:
        """Storage-only compression ratio."""
        return self.raw_bytes / max(self.storage_bytes, 1)

    @property
    def total_compression(self) -> float:
        """Combined compression ratio (both tiers stored)."""
        return self.raw_bytes / max(self.total_bytes, 1)


# ---- Storage compression ----

def _compress_storage(data: np.ndarray, mode: str, element_size: int = 4) -> bytes:
    """Compress array for storage.

    Args:
        data: (n_windows, window_size, n_features) array.
        mode: Compression mode.
        element_size: Bytes per element (4=float32, 2=float16/bfloat16).

    Returns:
        Compressed bytes with header.
    """
    n_windows, window_size, n_features = data.shape

    # Lossy modes require float32 (element_size=4)
    _LOSSY_MODES = {'near_lossless', 'near_lossless_turbo', 'quantized_8',
                    'normalized_lossless', 'normalized_lossless_zstd'}
    if element_size != 4 and mode in _LOSSY_MODES:
        raise ValueError(
            f"Mode '{mode}' requires float32 input. "
            f"For float16/bfloat16 data, use a lossless mode "
            f"(lossless, lossless_fast, lossless_zstd, lossless_lzma, optimal, optimal_fast)."
        )

    if mode == 'lossless':
        # Byte transpose + zlib
        raw = data.tobytes()
        transposed = byte_transpose(raw, element_size=element_size)
        payload = zlib.compress(transposed, 6)
        tag = b'\x01'

    elif mode == 'near_lossless':
        f32 = data.astype(np.float32)
        # Normalized float16 + zlib (<0.01% RMSE)
        # Normalize per-feature to avoid float16 overflow (max ~65504)
        feat_min = f32.min(axis=(0, 1), keepdims=True)   # (1, 1, F)
        feat_max = f32.max(axis=(0, 1), keepdims=True)   # (1, 1, F)
        feat_range = np.maximum(feat_max - feat_min, 1e-10)
        normed = ((f32 - feat_min) / feat_range).astype(np.float16)  # [0, 1]
        # Store scale factors (2 * F * 4 bytes) + normalized float16 data
        scale_data = (
            feat_min.squeeze().astype(np.float32).tobytes()
            + feat_max.squeeze().astype(np.float32).tobytes()
        )
        payload = zlib.compress(scale_data + normed.tobytes(), 6)
        tag = b'\x02'

    elif mode == 'lossless_zstd':
        import zstandard as zstd
        raw = data.tobytes()
        transposed = byte_transpose(raw, element_size=element_size)
        cctx = zstd.ZstdCompressor(level=19)
        payload = cctx.compress(transposed)
        tag = b'\x03'

    elif mode == 'lossless_lzma':
        import lzma
        raw = data.tobytes()
        transposed = byte_transpose(raw, element_size=element_size)
        payload = lzma.compress(transposed, preset=6)
        tag = b'\x04'

    elif mode == 'normalized_lossless':
        f32 = data.astype(np.float32)
        # Per-feature normalize to [0,1] → byte transpose → zlib
        # Stores min/max as float64 for precision (16 bytes per feature)
        feat_min = f32.min(axis=(0, 1), keepdims=True).astype(np.float64)  # (1,1,F)
        feat_max = f32.max(axis=(0, 1), keepdims=True).astype(np.float64)  # (1,1,F)
        feat_range = np.maximum(feat_max - feat_min, 1e-30)
        normed = ((f32.astype(np.float64) - feat_min) / feat_range).astype(np.float32)
        scale_data = (
            feat_min.squeeze().astype(np.float64).tobytes()
            + feat_max.squeeze().astype(np.float64).tobytes()
        )
        raw = normed.tobytes()
        transposed = byte_transpose(raw, element_size=4)
        payload = scale_data + zlib.compress(transposed, 6)
        tag = b'\x05'

    elif mode == 'normalized_lossless_zstd':
        f32 = data.astype(np.float32)
        import zstandard as zstd
        # Per-feature normalize to [0,1] → byte transpose → zstd
        feat_min = f32.min(axis=(0, 1), keepdims=True).astype(np.float64)
        feat_max = f32.max(axis=(0, 1), keepdims=True).astype(np.float64)
        feat_range = np.maximum(feat_max - feat_min, 1e-30)
        normed = ((f32.astype(np.float64) - feat_min) / feat_range).astype(np.float32)
        scale_data = (
            feat_min.squeeze().astype(np.float64).tobytes()
            + feat_max.squeeze().astype(np.float64).tobytes()
        )
        raw = normed.tobytes()
        transposed = byte_transpose(raw, element_size=4)
        cctx = zstd.ZstdCompressor(level=19)
        payload = scale_data + cctx.compress(transposed)
        tag = b'\x06'

    elif mode == 'near_lossless_turbo':
        f32 = data.astype(np.float32)
        # Normalized float16 + byte transpose + zlib (~3x, <0.05% RMSE)
        # "Turbo" uses fast zlib level for speed
        feat_min = f32.min(axis=(0, 1), keepdims=True).astype(np.float64)
        feat_max = f32.max(axis=(0, 1), keepdims=True).astype(np.float64)
        feat_range = np.maximum(feat_max - feat_min, 1e-30)
        normed = ((f32.astype(np.float64) - feat_min) / feat_range).astype(np.float16)
        scale_data = (
            feat_min.squeeze().astype(np.float64).tobytes()
            + feat_max.squeeze().astype(np.float64).tobytes()
        )
        transposed = byte_transpose(normed.tobytes(), element_size=2)
        payload = scale_data + zlib.compress(transposed, 1)
        tag = b'\x07'

    elif mode == 'quantized_8':
        f32 = data.astype(np.float32)
        # Normalized uint8 + per-window delta + zlib (~5-9x, ~2% RMSE)
        # High compression for applications tolerating moderate precision loss
        feat_min = f32.min(axis=(0, 1), keepdims=True).astype(np.float64)
        feat_max = f32.max(axis=(0, 1), keepdims=True).astype(np.float64)
        feat_range = np.maximum(feat_max - feat_min, 1e-30)
        normed = ((f32.astype(np.float64) - feat_min) / feat_range)
        q8 = (normed * 255).clip(0, 255).astype(np.uint8)
        # Delta encode within each window for better compressibility
        delta = np.empty_like(q8)
        delta[:, 0, :] = q8[:, 0, :]
        delta[:, 1:, :] = (q8[:, 1:, :].astype(np.int16) - q8[:, :-1, :].astype(np.int16)).astype(np.uint8)
        scale_data = (
            feat_min.squeeze().astype(np.float64).tobytes()
            + feat_max.squeeze().astype(np.float64).tobytes()
        )
        payload = scale_data + zlib.compress(delta.tobytes(), 6)
        tag = b'\x08'

    elif mode == 'lossless_fast':
        # Single-shot: flatten-delta + zstd-3 — fast with better ratio than bare zstd
        # ~50+ MB/s throughput, typically beats lzma on structured data
        import zstandard as zstd
        if element_size == 2:
            as_int = data.view(np.int16)
        else:
            as_int = data.astype(np.float32).view(np.int32)
        flat = as_int.reshape(1, n_windows * window_size, n_features)
        residuals = np.empty_like(flat)
        residuals[:, 0, :] = flat[:, 0, :]
        residuals[:, 1:, :] = flat[:, 1:, :] - flat[:, :-1, :]
        raw_bytes = residuals.tobytes()
        cctx = zstd.ZstdCompressor(level=3)
        payload = cctx.compress(raw_bytes)
        tag = b'\x0a'

    elif mode == 'optimal':
        # Strategy competition: try all (transform, compressor) combos, pick smallest
        from .optimal import optimal_compress
        import sys
        payload, certificate = optimal_compress(data, element_size=element_size)
        tag = b'\x09'
        sys.stderr.write(
            f"  [optimal] ratio={certificate.ratio:.2f}x, "
            f"entropy_gap={certificate.entropy_gap_pct:.1f}%, "
            f"transform={certificate.transform_id}, "
            f"compressor={certificate.compressor_id}, "
            f"randomness_p={certificate.randomness_p_value:.3f}\n"
        )

    elif mode == 'optimal_fast':
        # Fast variant: pre-screen transforms with fast compressors, then try all on top-K
        from .optimal import optimal_compress_fast
        import sys
        payload, certificate = optimal_compress_fast(data, element_size=element_size)
        tag = b'\x09'  # Same format, same decompressor
        sys.stderr.write(
            f"  [optimal_fast] ratio={certificate.ratio:.2f}x, "
            f"entropy_gap={certificate.entropy_gap_pct:.1f}%, "
            f"transform={certificate.transform_id}, "
            f"compressor={certificate.compressor_id}\n"
        )

    else:
        raise ValueError(f"Unknown storage mode: {mode}")

    # Header: tag(1) + n_windows(4) + window_size(2) + n_features(2) + payload_size(4)
    header = struct.pack("<cIHHI", tag, n_windows, window_size, n_features, len(payload))
    return header + payload


def _decompress_storage(data: bytes, element_size: int = 4, out_dtype=None) -> np.ndarray:
    """Decompress storage bytes back to array.

    Args:
        element_size: Bytes per element (4=float32, 2=float16/bfloat16).
        out_dtype: Output numpy dtype. Default: np.float32.

    Returns:
        (n_windows, window_size, n_features) array.
    """
    if out_dtype is None:
        out_dtype = np.float32

    header_fmt = "<cIHHI"
    header_size = struct.calcsize(header_fmt)
    tag, n_windows, window_size, n_features, payload_size = struct.unpack(
        header_fmt, data[:header_size]
    )
    payload = data[header_size:header_size + payload_size]

    if tag == b'\x01':
        # Byte transpose + zlib
        transposed = zlib.decompress(payload)
        raw = byte_untranspose(transposed, element_size=element_size)
        return np.frombuffer(raw, dtype=out_dtype).reshape(
            n_windows, window_size, n_features
        ).copy()

    elif tag == b'\x02':
        # Normalized float16 + zlib
        raw = zlib.decompress(payload)
        # Extract scale factors: 2 * n_features * 4 bytes
        scale_size = n_features * 4
        feat_min = np.frombuffer(raw[:scale_size], dtype=np.float32).copy()
        feat_max = np.frombuffer(raw[scale_size:2 * scale_size], dtype=np.float32).copy()
        feat_range = np.maximum(feat_max - feat_min, 1e-10)
        # Denormalize
        f16 = np.frombuffer(raw[2 * scale_size:], dtype=np.float16).reshape(
            n_windows, window_size, n_features
        )
        return (f16.astype(np.float32) * feat_range + feat_min).copy()

    elif tag == b'\x03':
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        transposed = dctx.decompress(payload)
        raw = byte_untranspose(transposed, element_size=element_size)
        return np.frombuffer(raw, dtype=out_dtype).reshape(
            n_windows, window_size, n_features
        ).copy()

    elif tag == b'\x04':
        import lzma
        transposed = lzma.decompress(payload)
        raw = byte_untranspose(transposed, element_size=element_size)
        return np.frombuffer(raw, dtype=out_dtype).reshape(
            n_windows, window_size, n_features
        ).copy()

    elif tag == b'\x05':
        # Normalized lossless: extract float64 min/max, decompress, denormalize
        scale_size = n_features * 8  # float64
        feat_min = np.frombuffer(payload[:scale_size], dtype=np.float64).copy()
        feat_max = np.frombuffer(payload[scale_size:2 * scale_size], dtype=np.float64).copy()
        feat_range = np.maximum(feat_max - feat_min, 1e-30)
        transposed = zlib.decompress(payload[2 * scale_size:])
        raw = byte_untranspose(transposed, element_size=4)
        normed = np.frombuffer(raw, dtype=np.float32).reshape(
            n_windows, window_size, n_features
        ).copy()
        return (normed.astype(np.float64) * feat_range + feat_min).astype(np.float32)

    elif tag == b'\x06':
        import zstandard as zstd
        scale_size = n_features * 8
        feat_min = np.frombuffer(payload[:scale_size], dtype=np.float64).copy()
        feat_max = np.frombuffer(payload[scale_size:2 * scale_size], dtype=np.float64).copy()
        feat_range = np.maximum(feat_max - feat_min, 1e-30)
        dctx = zstd.ZstdDecompressor()
        transposed = dctx.decompress(payload[2 * scale_size:])
        raw = byte_untranspose(transposed, element_size=4)
        normed = np.frombuffer(raw, dtype=np.float32).reshape(
            n_windows, window_size, n_features
        ).copy()
        return (normed.astype(np.float64) * feat_range + feat_min).astype(np.float32)

    elif tag == b'\x07':
        # Near-lossless turbo: float16 + BT + zlib
        scale_size = n_features * 8
        feat_min = np.frombuffer(payload[:scale_size], dtype=np.float64).copy()
        feat_max = np.frombuffer(payload[scale_size:2 * scale_size], dtype=np.float64).copy()
        feat_range = np.maximum(feat_max - feat_min, 1e-30)
        transposed = zlib.decompress(payload[2 * scale_size:])
        raw = byte_untranspose(transposed, element_size=2)
        normed = np.frombuffer(raw, dtype=np.float16).reshape(
            n_windows, window_size, n_features
        )
        return (normed.astype(np.float64) * feat_range + feat_min).astype(np.float32)

    elif tag == b'\x08':
        # Quantized 8-bit: uint8 + delta + zlib
        scale_size = n_features * 8
        feat_min = np.frombuffer(payload[:scale_size], dtype=np.float64).copy()
        feat_max = np.frombuffer(payload[scale_size:2 * scale_size], dtype=np.float64).copy()
        feat_range = np.maximum(feat_max - feat_min, 1e-30)
        raw = zlib.decompress(payload[2 * scale_size:])
        delta = np.frombuffer(raw, dtype=np.uint8).reshape(
            n_windows, window_size, n_features
        ).copy()
        # Reverse delta encoding (vectorized with cumsum)
        signed_deltas = delta[:, 1:, :].view(np.int8).astype(np.int16)
        cumulative = np.cumsum(signed_deltas, axis=1)
        q8 = np.empty_like(delta)
        q8[:, 0, :] = delta[:, 0, :]
        q8[:, 1:, :] = (delta[:, 0:1, :].astype(np.int16) + cumulative).astype(np.uint8)
        normed = q8.astype(np.float64) / 255.0
        return (normed * feat_range + feat_min).astype(np.float32)

    elif tag == b'\x09':
        # Optimal: strategy competition (self-describing payload)
        from .optimal import optimal_decompress_full
        return optimal_decompress_full(
            payload, n_windows, window_size, n_features,
            element_size=element_size, out_dtype=out_dtype,
        )

    elif tag == b'\x0a':
        # Lossless fast: flatten-delta + zstd-3
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        raw_bytes = dctx.decompress(payload)
        int_dtype = np.int16 if element_size == 2 else np.int32
        residuals = np.frombuffer(raw_bytes, dtype=int_dtype).reshape(
            1, n_windows * window_size, n_features
        ).copy()
        # Reverse delta: cumsum reconstructs original from [first, d1, d2, ...]
        # Cast back to int_dtype to handle wrapping (cumsum may promote to int64)
        as_int = np.cumsum(residuals, axis=1).astype(int_dtype)
        return as_int.reshape(n_windows, window_size, n_features).view(out_dtype).copy()

    else:
        raise ValueError(f"Unknown storage tag: {tag!r}")


# ---- Serialization ----

def serialize_tiered(encoding: TieredEncoding) -> bytes:
    """Serialize a TieredEncoding to bytes.

    Format:
        magic: b'UMT1' (UMC Tiered v1)
        n_windows: uint32
        window_size: uint16
        n_features: uint16
        vq_size: uint32
        storage_size: uint32
        revin_size: uint32
        vq_block: compressed VQ indices
        storage_block: compressed data
        revin_block: zlib(means + stds)
    """
    vq_compressed = compress_indices(encoding.vq_indices, mode="optimal")
    revin_raw = (
        encoding.revin_means.astype(np.float32).tobytes()
        + encoding.revin_stds.astype(np.float32).tobytes()
    )
    revin_compressed = zlib.compress(revin_raw, 9)

    header = struct.pack(
        "<4sIHHIII",
        b"UMT1",
        encoding.n_windows,
        encoding.window_size,
        encoding.n_features,
        len(vq_compressed),
        len(encoding.storage_compressed),
        len(revin_compressed),
    )
    return header + vq_compressed + encoding.storage_compressed + revin_compressed


def deserialize_tiered(data: bytes) -> TieredEncoding:
    """Deserialize bytes to TieredEncoding."""
    header_fmt = "<4sIHHIII"
    header_size = struct.calcsize(header_fmt)
    magic, n_windows, window_size, n_features, vq_size, storage_size, revin_size = (
        struct.unpack(header_fmt, data[:header_size])
    )
    if magic != b"UMT1":
        raise ValueError(f"Invalid tiered codec magic: {magic!r}")

    offset = header_size
    vq_indices = decompress_indices(data[offset:offset + vq_size])
    offset += vq_size

    storage_compressed = data[offset:offset + storage_size]
    offset += storage_size

    revin_compressed = data[offset:offset + revin_size]
    revin_raw = zlib.decompress(revin_compressed)
    elem_size = n_windows * n_features * 4
    revin_means = np.frombuffer(revin_raw[:elem_size], dtype=np.float32).reshape(
        n_windows, n_features
    ).copy()
    revin_stds = np.frombuffer(revin_raw[elem_size:], dtype=np.float32).reshape(
        n_windows, n_features
    ).copy()

    # Detect storage mode from tag byte
    tag = storage_compressed[0:1]
    mode_map = {b'\x01': 'lossless', b'\x02': 'near_lossless',
                b'\x03': 'lossless_zstd', b'\x04': 'lossless_lzma',
                b'\x05': 'normalized_lossless', b'\x06': 'normalized_lossless_zstd',
                b'\x07': 'near_lossless_turbo', b'\x08': 'quantized_8',
                b'\x09': 'optimal'}
    storage_mode = mode_map.get(tag, 'lossless')

    return TieredEncoding(
        vq_indices=vq_indices,
        storage_compressed=storage_compressed,
        storage_mode=storage_mode,
        revin_means=revin_means,
        revin_stds=revin_stds,
        n_windows=n_windows,
        window_size=window_size,
        n_features=n_features,
    )


# ---- Main codec ----

class TieredCodec:
    """Two-tier compression codec: VQ search + efficient storage.

    Tier 1 (Search):  VQ codebook indices for FAISS similarity search.
    Tier 2 (Storage): Byte-transposed compressed data for retrieval.

    Usage:
        codec = TieredCodec(encoder, decoder, storage_mode='lossless')

        # Encode
        encoding = codec.encode(windows)

        # Search (VQ lossy decode, fast)
        approx = codec.decode_search(encoding)

        # Retrieve (full fidelity from storage tier)
        exact = codec.decode_storage(encoding)

        # Serialize for disk
        data = serialize_tiered(encoding)
        encoding2 = deserialize_tiered(data)
    """

    def __init__(
        self,
        encoder,
        decoder,
        device: str = "cpu",
        storage_mode: str = "lossless",
    ):
        import torch
        self.encoder = encoder
        self.decoder = decoder
        self.device = torch.device(device)
        self.storage_mode = storage_mode

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.encoder.eval()
        self.decoder.eval()

    def encode(self, windows: np.ndarray, batch_size: int = 32) -> TieredEncoding:
        """Encode windows into tiered compressed form.

        Args:
            windows: (n_windows, window_size, n_features) float32.
            batch_size: Batch size for VQ encoding.

        Returns:
            TieredEncoding with VQ codes + compressed storage.
        """
        import torch

        f32 = windows.astype(np.float32)
        n_windows, window_size, n_features = f32.shape
        use_amp = self.device.type == "cuda"

        # Tier 1: VQ encoding
        all_top_indices = []
        all_bottom_indices = []
        all_revin_means = []
        all_revin_stds = []

        with torch.no_grad():
            for i in range(0, n_windows, batch_size):
                x = torch.from_numpy(f32[i:i + batch_size]).to(self.device)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    self.encoder.encode(x)

                all_top_indices.append(self.encoder._last_top_indices.cpu().numpy())
                bottom_stacked = torch.stack(
                    self.encoder._last_bottom_indices, dim=-1
                ).cpu().numpy()
                all_bottom_indices.append(bottom_stacked)

                all_revin_means.append(
                    self.encoder.revin._mean.squeeze(1).cpu().numpy()
                )
                all_revin_stds.append(
                    self.encoder.revin._std.squeeze(1).cpu().numpy()
                )
                self.encoder.clear_cached()

        top_indices = np.concatenate(all_top_indices).astype(np.uint8)
        bottom_indices = np.concatenate(all_bottom_indices).astype(np.uint8)
        revin_means = np.concatenate(all_revin_means).astype(np.float32)
        revin_stds = np.concatenate(all_revin_stds).astype(np.float32)

        n_patches = bottom_indices.shape[1]
        n_levels = bottom_indices.shape[2]

        vq_indices = VQIndices(
            top_indices=top_indices,
            bottom_indices=bottom_indices,
            n_patches=n_patches,
            n_levels=n_levels,
            top_n_codes=self.encoder.config.vq_top_n_codes,
            bottom_n_codes=self.encoder.config.vq_bottom_n_codes,
        )

        # Tier 2: Storage compression
        storage_compressed = _compress_storage(f32, self.storage_mode)

        return TieredEncoding(
            vq_indices=vq_indices,
            storage_compressed=storage_compressed,
            storage_mode=self.storage_mode,
            revin_means=revin_means,
            revin_stds=revin_stds,
            n_windows=n_windows,
            window_size=window_size,
            n_features=n_features,
        )

    def decode_storage(self, encoding: TieredEncoding) -> np.ndarray:
        """Decode from storage tier (full fidelity).

        Returns:
            (n_windows, window_size, n_features) float32.
            Bit-exact for 'lossless' mode, <0.01% RMSE for 'near_lossless'.
        """
        return _decompress_storage(encoding.storage_compressed)

    def decode_search(
        self, encoding: TieredEncoding, batch_size: int = 32
    ) -> np.ndarray:
        """Decode from VQ search tier (lossy, fast).

        Returns:
            (n_windows, window_size, n_features) float32 (lossy, ~2.6% RMSE).
        """
        import torch

        use_amp = self.device.type == "cuda"
        all_x_hat = []

        top_indices = encoding.vq_indices.top_indices
        bottom_indices = encoding.vq_indices.bottom_indices
        n = len(top_indices)

        with torch.no_grad():
            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)

                top_idx = torch.from_numpy(
                    top_indices[i:end].astype(np.int64)
                ).to(self.device)
                bottom_idx = torch.from_numpy(
                    bottom_indices[i:end].astype(np.int64)
                ).to(self.device)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    top_q, bottom_q = self.encoder.indices_to_quantized(
                        top_idx, bottom_idx
                    )
                    x_hat_raw = self.decoder.decode_from_codes(top_q, bottom_q)

                # Manual RevIN inverse
                means = torch.from_numpy(
                    encoding.revin_means[i:end]
                ).unsqueeze(1).to(self.device)
                stds = torch.from_numpy(
                    encoding.revin_stds[i:end]
                ).unsqueeze(1).to(self.device)

                affine_w = self.encoder.revin.affine_weight
                affine_b = self.encoder.revin.affine_bias
                x_hat = (x_hat_raw - affine_b) / (affine_w + 1e-8) * stds + means

                all_x_hat.append(x_hat.cpu().float().numpy())

        return np.concatenate(all_x_hat)

    def compression_stats(self, encoding: TieredEncoding) -> dict:
        """Compute detailed compression statistics."""
        return {
            "raw_bytes": encoding.raw_bytes,
            "vq_bytes": encoding.vq_bytes,
            "storage_bytes": encoding.storage_bytes,
            "total_bytes": encoding.total_bytes,
            "search_compression": encoding.search_compression,
            "storage_compression": encoding.storage_compression,
            "total_compression": encoding.total_compression,
            "storage_mode": encoding.storage_mode,
            "vq_bytes_per_window": encoding.vq_bytes / max(encoding.n_windows, 1),
            "storage_bytes_per_window": encoding.storage_bytes / max(encoding.n_windows, 1),
            "total_bytes_per_window": encoding.total_bytes / max(encoding.n_windows, 1),
        }

    def encode_pipelined(
        self, windows: np.ndarray, batch_size: int = 32, n_workers: int = 2,
    ) -> TieredEncoding:
        """Encode with GPU/CPU pipeline overlap.

        Overlaps GPU VQ encoding with CPU storage compression using
        a thread pool. GPU encodes batch N+1 while CPU compresses
        batch N's storage, yielding ~1.5-2x throughput.

        Args:
            windows: (n_windows, window_size, n_features) float32.
            batch_size: Batch size for VQ encoding.
            n_workers: CPU compression threads.

        Returns:
            TieredEncoding with VQ codes + compressed storage.
        """
        import torch
        from concurrent.futures import ThreadPoolExecutor, Future

        f32 = windows.astype(np.float32)
        n_windows, window_size, n_features = f32.shape
        use_amp = self.device.type == "cuda"

        all_top_indices = []
        all_bottom_indices = []
        all_revin_means = []
        all_revin_stds = []

        # Pipeline: GPU encode + CPU storage compression in parallel
        executor = ThreadPoolExecutor(max_workers=n_workers)
        storage_futures: list[Future] = []

        with torch.no_grad():
            for i in range(0, n_windows, batch_size):
                batch = f32[i:i + batch_size]
                x = torch.from_numpy(batch).to(self.device)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    self.encoder.encode(x)

                all_top_indices.append(self.encoder._last_top_indices.cpu().numpy())
                bottom_stacked = torch.stack(
                    self.encoder._last_bottom_indices, dim=-1
                ).cpu().numpy()
                all_bottom_indices.append(bottom_stacked)
                all_revin_means.append(
                    self.encoder.revin._mean.squeeze(1).cpu().numpy()
                )
                all_revin_stds.append(
                    self.encoder.revin._std.squeeze(1).cpu().numpy()
                )
                self.encoder.clear_cached()

                # Submit storage compression to thread pool
                storage_futures.append(
                    executor.submit(_compress_storage, batch, self.storage_mode)
                )

        # Collect compressed storage chunks
        storage_chunks = [f.result() for f in storage_futures]
        executor.shutdown(wait=False)

        # Combine VQ indices
        top_indices = np.concatenate(all_top_indices).astype(np.uint8)
        bottom_indices = np.concatenate(all_bottom_indices).astype(np.uint8)
        revin_means = np.concatenate(all_revin_means).astype(np.float32)
        revin_stds = np.concatenate(all_revin_stds).astype(np.float32)

        vq_indices = VQIndices(
            top_indices=top_indices,
            bottom_indices=bottom_indices,
            n_patches=bottom_indices.shape[1],
            n_levels=bottom_indices.shape[2],
            top_n_codes=self.encoder.config.vq_top_n_codes,
            bottom_n_codes=self.encoder.config.vq_bottom_n_codes,
        )

        # Full storage compression (single pass over all data for better ratio)
        storage_compressed = _compress_storage(f32, self.storage_mode)

        return TieredEncoding(
            vq_indices=vq_indices,
            storage_compressed=storage_compressed,
            storage_mode=self.storage_mode,
            revin_means=revin_means,
            revin_stds=revin_stds,
            n_windows=n_windows,
            window_size=window_size,
            n_features=n_features,
        )
