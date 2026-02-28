"""Residual coder for lossless compression.

Uses XOR on raw float32 bytes for perfect bit-exact lossless coding.
IEEE 754 float arithmetic (a - b + b) is NOT bit-exact, but XOR is:
    a XOR b XOR b == a   (always, for any byte pattern)

The XOR residual between similar floats produces mostly-zero bytes,
which compress extremely well with byte transposition + zlib.

The residual is the bridge between lossy VQ and lossless reconstruction:
    original_bytes == reconstruction_bytes XOR xor_residual   (bit-exact)

Compression backends (Phase 3):
    - 'zlib':     zlib level 9 on byte-transposed data (default, no dependencies)
    - 'static':   Per-channel static arithmetic coding (requires constriction)
    - 'adaptive': Adaptive arithmetic coding (requires constriction)
    - 'neural':   Neural arithmetic coding (requires constriction + trained model)
"""

import struct
import zlib
from typing import Optional

import numpy as np


def byte_transpose(data: bytes, element_size: int) -> bytes:
    """Transpose byte layout for better compression.

    Reorders bytes so that byte 0 of every element is contiguous,
    then byte 1, etc. For float32 (element_size=4):

    Normal:  [B0 B1 B2 B3] [B0 B1 B2 B3] [B0 B1 B2 B3] ...
    Transposed: [B0 B0 B0 ...] [B1 B1 B1 ...] [B2 B2 B2 ...] [B3 B3 B3 ...]

    Uses C extension when available (~4x faster).
    """
    try:
        from ..cext import HAS_C_EXT, fast_byte_transpose
        if HAS_C_EXT:
            return fast_byte_transpose(data, element_size)
    except ImportError:
        pass

    n_elements = len(data) // element_size
    if n_elements == 0:
        return data
    arr = np.frombuffer(data, dtype=np.uint8).reshape(n_elements, element_size)
    return arr.T.tobytes()


def byte_untranspose(data: bytes, element_size: int) -> bytes:
    """Reverse byte transposition. Uses C extension when available."""
    try:
        from ..cext import HAS_C_EXT, fast_byte_untranspose
        if HAS_C_EXT:
            return fast_byte_untranspose(data, element_size)
    except ImportError:
        pass

    n_elements = len(data) // element_size
    if n_elements == 0:
        return data
    arr = np.frombuffer(data, dtype=np.uint8).reshape(element_size, n_elements)
    return arr.T.tobytes()


class ResidualCoder:
    """Encode and decode exact float32 residuals for lossless compression.

    Uses XOR on the raw uint32 representation of float32 values.
    XOR is its own inverse and is perfectly bit-exact, unlike float
    subtraction/addition which can lose bits to rounding.

    Compression pipeline:
        XOR residual (uint32) -> byte transpose -> compress (zlib or arithmetic)

    Supported backends:
        'zlib':     zlib level 9 (default, no extra dependencies)
        'static':   Per-channel static arithmetic coding
        'adaptive': Adaptive arithmetic coding

    Typical compression ratios on financial XOR residuals:
        - zlib:     3-8x
        - static:   5-12x (better per-channel probability modeling)
        - adaptive: 6-15x (adapts to local statistics)
    """

    @staticmethod
    def compute(original: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
        """Compute XOR residual between original and reconstruction.

        Args:
            original: Original data, any shape, float32.
            reconstruction: VQ reconstruction, same shape, float32.

        Returns:
            XOR residual as uint32 array (same shape).
        """
        orig_u32 = original.astype(np.float32).view(np.uint32)
        recon_u32 = reconstruction.astype(np.float32).view(np.uint32)
        return np.bitwise_xor(orig_u32, recon_u32)

    @staticmethod
    def apply(reconstruction: np.ndarray, xor_residual: np.ndarray) -> np.ndarray:
        """Apply XOR residual to recover exact original.

        Args:
            reconstruction: VQ reconstruction, float32.
            xor_residual: Stored XOR residual, uint32.

        Returns:
            original: bit-exact float32 recovery (XOR is perfectly reversible).
        """
        recon_u32 = reconstruction.astype(np.float32).view(np.uint32)
        orig_u32 = np.bitwise_xor(recon_u32, xor_residual)
        return orig_u32.view(np.float32)

    # Marker byte for new format (0xFF can never be ndim in old format)
    _NEW_FORMAT_MARKER = 0xFF

    @staticmethod
    def compress(
        residual: np.ndarray,
        method: str = "zlib",
        compressor=None,
    ) -> bytes:
        """Compress XOR residual array to bytes.

        New format (Phase 3):
            marker: uint8 = 0xFF  (distinguishes from old format)
            method_tag: uint8 (0=zlib, 1=static, 2=adaptive, 3=neural)
            ndim: uint8
            shape: ndim * uint32
            payload: compressed byte-transposed data

        Old format (Phase 2, backward compat):
            ndim: uint8
            shape: ndim * uint32
            payload: zlib(byte_transposed)

        Args:
            residual: uint32 XOR residual array, any shape.
            method: Compression backend ('zlib', 'static', 'adaptive', 'neural').
            compressor: Pre-configured ByteCompressor instance (for 'static',
                'adaptive', 'neural'). If None, uses default configuration.

        Returns:
            Compressed bytes.
        """
        method_tags = {"zlib": 0, "static": 1, "adaptive": 2, "neural": 3}
        if method not in method_tags:
            raise ValueError(f"Unknown compression method: {method!r}. "
                           f"Choose from {list(method_tags.keys())}")

        flat = residual.view(np.uint32) if residual.dtype == np.float32 else residual.astype(np.uint32)
        shape = flat.shape

        # Shape header: ndim + shape dimensions
        shape_header = struct.pack("<B", len(shape))
        for dim in shape:
            shape_header += struct.pack("<I", dim)

        # New format prefix: marker + method tag
        prefix = struct.pack("<BB", ResidualCoder._NEW_FORMAT_MARKER, method_tags[method])

        # Byte-transpose
        raw_bytes = flat.tobytes()
        transposed = byte_transpose(raw_bytes, 4)  # uint32 = 4 bytes

        # Compress using selected backend
        if method == "zlib":
            compressed = zlib.compress(transposed, 9)
        else:
            if compressor is None:
                compressor = _get_default_compressor(method)
            compressed = compressor.compress(transposed, element_size=4)

        return prefix + shape_header + compressed

    @staticmethod
    def decompress(data: bytes, compressor=None) -> np.ndarray:
        """Decompress bytes back to uint32 XOR residual array.

        Auto-detects old vs new format and compression method.

        Args:
            data: Compressed bytes from compress().
            compressor: ByteCompressor instance for non-zlib methods.
                If None, uses default configuration.

        Returns:
            uint32 XOR residual array with original shape.
        """
        offset = 0

        # Detect format: new format starts with 0xFF marker
        first_byte = struct.unpack_from("<B", data, offset)[0]

        if first_byte == ResidualCoder._NEW_FORMAT_MARKER:
            # New format: marker + method_tag + ndim + shape + payload
            offset += 1
            method_tag = struct.unpack_from("<B", data, offset)[0]
            offset += 1
            method_names = {0: "zlib", 1: "static", 2: "adaptive", 3: "neural"}
            method = method_names.get(method_tag, "zlib")
        else:
            # Old format: first byte is ndim, no method tag, always zlib
            method = "zlib"

        # Read shape header
        ndim = struct.unpack_from("<B", data, offset)[0]
        offset += 1
        shape = []
        for _ in range(ndim):
            dim = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            shape.append(dim)
        shape = tuple(shape)

        payload = data[offset:]

        # Decompress
        if method == "zlib":
            transposed = zlib.decompress(payload)
        else:
            if compressor is None:
                compressor = _get_default_compressor(method)
            transposed = compressor.decompress(payload, element_size=4)

        raw_bytes = byte_untranspose(transposed, 4)
        return np.frombuffer(raw_bytes, dtype=np.uint32).reshape(shape).copy()


def _get_default_compressor(method: str):
    """Get a default compressor instance for the given method."""
    from .arithmetic import StaticByteCompressor, AdaptiveByteCompressor

    if method == "static":
        return StaticByteCompressor()
    elif method == "adaptive":
        return AdaptiveByteCompressor()
    elif method == "neural":
        raise ValueError(
            "Neural compressor requires a pre-configured NeuralByteCompressor "
            "with a trained model. Pass compressor= explicitly."
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")
