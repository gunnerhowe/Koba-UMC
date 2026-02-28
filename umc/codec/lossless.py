"""Lossless codec: VQ codes + XOR-compressed residuals.

This is Phase 2 of the UMC compression pipeline. It combines:
    1. VQ indices (searchable, compact) from Phase 1
    2. XOR residuals on raw bytes (compressed) for lossless reconstruction

The VQ codes serve dual purpose:
    - Searchable index (find similar windows without decompression)
    - Lossy approximation (base for residual coding)

The XOR residual gives bit-exact reconstruction:
    original_bytes == reconstruction_bytes XOR xor_residual   (always)

Key design: both encode and decode use the SAME codebook-lookup path
(indices_to_quantized) to guarantee identical reconstruction bytes.
Float arithmetic (a - b + b) is NOT bit-exact, but XOR always is.

Compression breakdown:
    - VQ codes: ~20-60 bytes/window (entropy-coded)
    - XOR residuals: ~200-800 bytes/window (byte-transposed + zlib)
    - Total: ~250-850 bytes vs 5120 raw = 6-20x lossless compression
    - Plus: the VQ codes are searchable without touching residuals
"""

import struct
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .residual import ResidualCoder
from ..storage.entropy import VQIndices, compress_indices, decompress_indices


@dataclass
class LosslessEncoding:
    """Complete lossless encoding of a batch of windows.

    Contains everything needed to reconstruct the exact original data
    AND to search the data via VQ codes without decompression.
    """
    # VQ codes (searchable index)
    vq_indices: VQIndices

    # Exact residual (lossless correction)
    residual_compressed: bytes

    # RevIN stats (needed to undo normalization during decode)
    revin_means: np.ndarray   # (n_windows, n_features) float32
    revin_stds: np.ndarray    # (n_windows, n_features) float32

    # Metadata
    n_windows: int
    window_size: int
    n_features: int

    @property
    def vq_bytes(self) -> int:
        """Size of entropy-coded VQ indices."""
        return len(compress_indices(self.vq_indices))

    @property
    def residual_bytes(self) -> int:
        """Size of compressed residuals."""
        return len(self.residual_compressed)

    @property
    def total_bytes(self) -> int:
        """Total compressed size (VQ + residuals + metadata)."""
        return self.vq_bytes + self.residual_bytes

    @property
    def raw_bytes(self) -> int:
        """Raw uncompressed size."""
        return self.n_windows * self.window_size * self.n_features * 4  # float32

    @property
    def compression_ratio(self) -> float:
        """Overall lossless compression ratio."""
        return self.raw_bytes / max(self.total_bytes, 1)

    @property
    def vq_compression_ratio(self) -> float:
        """VQ-only compression ratio (lossy, searchable)."""
        return self.raw_bytes / max(self.vq_bytes, 1)


def serialize_encoding(encoding: LosslessEncoding) -> bytes:
    """Serialize a LosslessEncoding to bytes for storage.

    Format:
        header (20 bytes):
            magic: bytes[4] = b'ULC1' (UMC Lossless Codec v1)
            n_windows: uint32
            window_size: uint16
            n_features: uint16
            vq_block_size: uint32
            residual_block_size: uint32
        vq_block: compressed VQ indices
        residual_block: compressed residuals
        revin_block: zlib(revin_means.tobytes() + revin_stds.tobytes())
    """
    vq_compressed = compress_indices(encoding.vq_indices)
    revin_raw = (
        encoding.revin_means.astype(np.float32).tobytes()
        + encoding.revin_stds.astype(np.float32).tobytes()
    )
    revin_compressed = zlib.compress(revin_raw, 9)

    header = struct.pack(
        "<4sIHHIII",
        b"ULC1",
        encoding.n_windows,
        encoding.window_size,
        encoding.n_features,
        len(vq_compressed),
        len(encoding.residual_compressed),
        len(revin_compressed),
    )

    return header + vq_compressed + encoding.residual_compressed + revin_compressed


def deserialize_encoding(data: bytes) -> LosslessEncoding:
    """Deserialize bytes back to LosslessEncoding.

    Args:
        data: Bytes from serialize_encoding().

    Returns:
        LosslessEncoding with all fields restored.
    """
    header_fmt = "<4sIHHIII"
    header_size = struct.calcsize(header_fmt)

    magic, n_windows, window_size, n_features, vq_size, res_size, revin_size = (
        struct.unpack(header_fmt, data[:header_size])
    )
    if magic != b"ULC1":
        raise ValueError(f"Invalid lossless codec magic: {magic!r}")

    offset = header_size

    # VQ indices
    vq_data = data[offset:offset + vq_size]
    vq_indices = decompress_indices(vq_data)
    offset += vq_size

    # Residual (already compressed, stays as bytes until decode)
    residual_compressed = data[offset:offset + res_size]
    offset += res_size

    # RevIN stats
    revin_compressed = data[offset:offset + revin_size]
    revin_raw = zlib.decompress(revin_compressed)
    revin_element_size = n_windows * n_features * 4  # float32
    revin_means = np.frombuffer(
        revin_raw[:revin_element_size], dtype=np.float32
    ).reshape(n_windows, n_features).copy()
    revin_stds = np.frombuffer(
        revin_raw[revin_element_size:], dtype=np.float32
    ).reshape(n_windows, n_features).copy()

    return LosslessEncoding(
        vq_indices=vq_indices,
        residual_compressed=residual_compressed,
        revin_means=revin_means,
        revin_stds=revin_stds,
        n_windows=n_windows,
        window_size=window_size,
        n_features=n_features,
    )


class LosslessCodec:
    """Full lossless compression codec combining VQ codes and XOR residuals.

    Encode pipeline:
        1. Run encoder on original data -> VQ indices + RevIN stats
        2. Re-derive VQ vectors via indices_to_quantized (same path as decode)
        3. Run decoder -> reconstruction
        4. XOR residual = original_bytes XOR reconstruction_bytes
        5. Compress VQ indices (entropy coding) + XOR residual (byte-transpose + zlib)

    Decode pipeline:
        1. Decompress VQ indices -> indices_to_quantized -> VQ vectors
        2. Run decoder -> reconstruction (bit-exact same as encode step 2-3)
        3. Decompress XOR residual
        4. original_bytes = reconstruction_bytes XOR xor_residual (bit-exact)

    The VQ codes are independently useful for search (no residual needed).
    """

    def __init__(
        self,
        encoder,
        decoder,
        device: str = "cpu",
        residual_method: str = "zlib",
        residual_compressor=None,
    ):
        """Initialize codec with trained encoder and decoder.

        Args:
            encoder: Trained HVQVAEEncoder (must have indices_to_quantized()).
            decoder: Trained HVQVAEDecoder.
            device: Device for inference ('cpu' recommended for bit-exact results).
            residual_method: Compression backend for XOR residuals.
                'zlib' (default), 'static', 'adaptive', or 'neural'.
            residual_compressor: Pre-configured ByteCompressor for non-zlib methods.
                If None, uses default configuration.
        """
        self.encoder = encoder
        self.decoder = decoder
        self.device = torch.device(device)
        self.residual_method = residual_method
        self.residual_compressor = residual_compressor
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.encoder.eval()
        self.decoder.eval()

    def _reconstruct_from_indices(
        self,
        top_indices_np: np.ndarray,
        bottom_indices_np: np.ndarray,
        revin_means: np.ndarray,
        revin_stds: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """Shared reconstruction path used by BOTH encode and decode.

        This guarantees bit-exact identical reconstruction in both directions,
        which is required for the XOR residual to cancel perfectly.
        """
        n = len(top_indices_np)
        all_x_hat = []

        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)

            top_idx = torch.from_numpy(
                top_indices_np[i:end].astype(np.int64)
            ).to(self.device)
            bottom_idx = torch.from_numpy(
                bottom_indices_np[i:end].astype(np.int64)
            ).to(self.device)

            top_quantized, bottom_quantized = self.encoder.indices_to_quantized(
                top_idx, bottom_idx
            )

            x_hat_raw = self.decoder.decode_from_codes(top_quantized, bottom_quantized)

            # Manual RevIN inverse with stored stats
            batch_means = torch.from_numpy(
                revin_means[i:end]
            ).unsqueeze(1).to(self.device)  # (B, 1, F)
            batch_stds = torch.from_numpy(
                revin_stds[i:end]
            ).unsqueeze(1).to(self.device)   # (B, 1, F)

            affine_w = self.encoder.revin.affine_weight
            affine_b = self.encoder.revin.affine_bias
            x_hat_unaffine = (x_hat_raw - affine_b) / (affine_w + 1e-8)
            x_hat = x_hat_unaffine * batch_stds + batch_means

            all_x_hat.append(x_hat.cpu().numpy().astype(np.float32))

        return np.concatenate(all_x_hat)

    @torch.no_grad()
    def encode(self, windows: np.ndarray, batch_size: int = 32) -> LosslessEncoding:
        """Encode windows into lossless compressed form.

        Args:
            windows: (n_windows, window_size, n_features) float32 array.
            batch_size: Batch size for encoder/decoder inference.

        Returns:
            LosslessEncoding containing VQ codes + compressed XOR residuals.
        """
        n_windows, window_size, n_features = windows.shape

        all_top_indices = []
        all_bottom_indices = []
        all_revin_means = []
        all_revin_stds = []

        # Step 1: Run encoder to get VQ indices and RevIN stats
        for i in range(0, n_windows, batch_size):
            x = torch.from_numpy(
                windows[i:i + batch_size].astype(np.float32)
            ).to(self.device)

            self.encoder.encode(x)

            # Capture RevIN stats
            revin_mean = self.encoder.revin._mean.squeeze(1).cpu().numpy()  # (B, F)
            revin_std = self.encoder.revin._std.squeeze(1).cpu().numpy()    # (B, F)
            all_revin_means.append(revin_mean)
            all_revin_stds.append(revin_std)

            # Capture VQ indices
            all_top_indices.append(
                self.encoder._last_top_indices.cpu().numpy()
            )
            bottom_stacked = torch.stack(
                self.encoder._last_bottom_indices, dim=-1
            )  # (B, P, n_levels)
            all_bottom_indices.append(bottom_stacked.cpu().numpy())

            self.encoder.clear_cached()

        top_indices = np.concatenate(all_top_indices)
        bottom_indices = np.concatenate(all_bottom_indices)
        revin_means = np.concatenate(all_revin_means).astype(np.float32)
        revin_stds = np.concatenate(all_revin_stds).astype(np.float32)

        # Step 2: Reconstruct using indices_to_quantized (SAME path as decode)
        x_hat_all = self._reconstruct_from_indices(
            top_indices, bottom_indices, revin_means, revin_stds, batch_size
        )

        # Step 3: XOR residual (bit-exact reversible)
        xor_residual = ResidualCoder.compute(windows.astype(np.float32), x_hat_all)
        residual_compressed = ResidualCoder.compress(
            xor_residual,
            method=self.residual_method,
            compressor=self.residual_compressor,
        )

        # Build VQ indices structure
        n_levels = bottom_indices.shape[2]
        n_patches = bottom_indices.shape[1]
        vq_indices = VQIndices(
            top_indices=top_indices.astype(np.uint8),
            bottom_indices=bottom_indices.astype(np.uint8),
            n_patches=n_patches,
            n_levels=n_levels,
            top_n_codes=self.encoder.config.vq_top_n_codes,
            bottom_n_codes=self.encoder.config.vq_bottom_n_codes,
        )

        return LosslessEncoding(
            vq_indices=vq_indices,
            residual_compressed=residual_compressed,
            revin_means=revin_means,
            revin_stds=revin_stds,
            n_windows=n_windows,
            window_size=window_size,
            n_features=n_features,
        )

    @torch.no_grad()
    def decode(self, encoding: LosslessEncoding, batch_size: int = 32) -> np.ndarray:
        """Decode lossless encoding back to exact original data.

        Args:
            encoding: LosslessEncoding from encode().
            batch_size: Batch size for decoder inference.

        Returns:
            (n_windows, window_size, n_features) float32 array,
            bit-exact with the original input to encode().
        """
        # Decompress XOR residual (auto-detects method from format header)
        xor_residual = ResidualCoder.decompress(
            encoding.residual_compressed,
            compressor=self.residual_compressor,
        )

        # Reconstruct from VQ indices (SAME path as encode)
        x_hat_all = self._reconstruct_from_indices(
            encoding.vq_indices.top_indices,
            encoding.vq_indices.bottom_indices,
            encoding.revin_means,
            encoding.revin_stds,
            batch_size,
        )

        # Apply XOR residual for bit-exact reconstruction
        return ResidualCoder.apply(x_hat_all, xor_residual)

    @torch.no_grad()
    def decode_vq_only(
        self, encoding: LosslessEncoding, batch_size: int = 32
    ) -> np.ndarray:
        """Decode using only VQ codes (lossy, no residual).

        Useful for quick preview or when exact reconstruction isn't needed.
        Much faster than full decode since residual decompression is skipped.

        Args:
            encoding: LosslessEncoding (residual is ignored).
            batch_size: Batch size for decoder inference.

        Returns:
            (n_windows, window_size, n_features) float32 array (lossy).
        """
        return self._reconstruct_from_indices(
            encoding.vq_indices.top_indices,
            encoding.vq_indices.bottom_indices,
            encoding.revin_means,
            encoding.revin_stds,
            batch_size,
        )

    def encode_to_bytes(self, windows: np.ndarray, batch_size: int = 32) -> bytes:
        """Encode windows and serialize to bytes (single-call API).

        Args:
            windows: (n_windows, window_size, n_features) float32 array.
            batch_size: Batch size for inference.

        Returns:
            Serialized bytes containing full lossless encoding.
        """
        encoding = self.encode(windows, batch_size=batch_size)
        return serialize_encoding(encoding)

    def decode_from_bytes(self, data: bytes, batch_size: int = 32) -> np.ndarray:
        """Deserialize and decode bytes to exact original data.

        Args:
            data: Bytes from encode_to_bytes().
            batch_size: Batch size for inference.

        Returns:
            (n_windows, window_size, n_features) float32, bit-exact original.
        """
        encoding = deserialize_encoding(data)
        return self.decode(encoding, batch_size=batch_size)

    def compression_stats(self, encoding: LosslessEncoding) -> dict:
        """Compute detailed compression statistics.

        Returns dict with:
            raw_bytes, vq_bytes, residual_bytes, total_bytes,
            compression_ratio, vq_compression_ratio,
            bytes_per_window, vq_bytes_per_window, residual_bytes_per_window
        """
        vq_bytes = encoding.vq_bytes
        res_bytes = encoding.residual_bytes
        total = vq_bytes + res_bytes
        raw = encoding.raw_bytes
        n = encoding.n_windows

        return {
            "raw_bytes": raw,
            "vq_bytes": vq_bytes,
            "residual_bytes": res_bytes,
            "total_bytes": total,
            "compression_ratio": raw / max(total, 1),
            "vq_compression_ratio": raw / max(vq_bytes, 1),
            "bytes_per_window": total / max(n, 1),
            "vq_bytes_per_window": vq_bytes / max(n, 1),
            "residual_bytes_per_window": res_bytes / max(n, 1),
        }
