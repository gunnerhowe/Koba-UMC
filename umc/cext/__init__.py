"""UMC C Extension — fast compression kernels with pure-Python fallback.

Usage:
    from umc.cext import fast_byte_transpose, fast_byte_untranspose
    from umc.cext import fast_delta_decode_order1, HAS_C_EXT

If the compiled C library is available, these functions use the C kernels.
Otherwise they fall back to the original NumPy implementations transparently.
"""

import ctypes
import os
import platform
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Try to load the compiled C library
# ---------------------------------------------------------------------------

HAS_C_EXT = False
_lib = None

def _find_library():
    """Find the compiled shared library."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    system = platform.system()

    if system == "Windows":
        names = ["_umc_kernels.dll", "umc_kernels.dll"]
    elif system == "Darwin":
        names = ["_umc_kernels.dylib", "_umc_kernels.so"]
    else:
        names = ["_umc_kernels.so"]

    # Search in cext dir, then parent (umc), then project root
    search_dirs = [
        base_dir,
        os.path.dirname(base_dir),
        os.path.dirname(os.path.dirname(base_dir)),
    ]

    for d in search_dirs:
        for name in names:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path
    return None


def _load_library():
    global _lib, HAS_C_EXT
    path = _find_library()
    if path is None:
        return

    try:
        _lib = ctypes.CDLL(path)

        # Verify version
        _lib.umc_kernels_version.restype = ctypes.c_int
        version = _lib.umc_kernels_version()
        if version < 1:
            _lib = None
            return

        # Set up function signatures
        c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
        c_int32_p = ctypes.POINTER(ctypes.c_int32)
        c_uint32_p = ctypes.POINTER(ctypes.c_uint32)
        c_int64 = ctypes.c_int64
        c_int = ctypes.c_int

        _lib.byte_transpose.argtypes = [c_uint8_p, c_uint8_p, c_int64, c_int]
        _lib.byte_transpose.restype = None

        _lib.byte_untranspose.argtypes = [c_uint8_p, c_uint8_p, c_int64, c_int]
        _lib.byte_untranspose.restype = None

        _lib.delta_encode_order1.argtypes = [c_int32_p, c_int32_p, c_int64, c_int64, c_int64]
        _lib.delta_encode_order1.restype = None

        _lib.delta_decode_order1.argtypes = [c_int32_p, c_int64, c_int64, c_int64]
        _lib.delta_decode_order1.restype = None

        _lib.delta_decode_order2.argtypes = [c_int32_p, c_int64, c_int64, c_int64]
        _lib.delta_decode_order2.restype = None

        _lib.delta_decode_order3.argtypes = [c_int32_p, c_int64, c_int64, c_int64]
        _lib.delta_decode_order3.restype = None

        _lib.delta_decode_order4.argtypes = [c_int32_p, c_int64, c_int64, c_int64]
        _lib.delta_decode_order4.restype = None

        _lib.xor_encode.argtypes = [c_uint32_p, c_uint32_p, c_int64, c_int64, c_int64]
        _lib.xor_encode.restype = None

        _lib.xor_decode.argtypes = [c_uint32_p, c_int64, c_int64, c_int64]
        _lib.xor_decode.restype = None

        _lib.zigzag_encode.argtypes = [c_int32_p, c_uint32_p, c_int64]
        _lib.zigzag_encode.restype = None

        _lib.zigzag_decode.argtypes = [c_uint32_p, c_int32_p, c_int64]
        _lib.zigzag_decode.restype = None

        HAS_C_EXT = True

    except (OSError, AttributeError):
        _lib = None


_load_library()


# ---------------------------------------------------------------------------
# Public API: C-accelerated with pure-Python fallback
# ---------------------------------------------------------------------------

def fast_byte_transpose(data: bytes, element_size: int) -> bytes:
    """Byte transpose — C-accelerated if available."""
    n_elements = len(data) // element_size
    if n_elements == 0:
        return data

    if HAS_C_EXT:
        src = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        dst = (ctypes.c_uint8 * len(data))()
        _lib.byte_transpose(src, dst, n_elements, element_size)
        return bytes(dst)

    # Fallback: NumPy
    arr = np.frombuffer(data, dtype=np.uint8).reshape(n_elements, element_size)
    return arr.T.tobytes()


def fast_byte_untranspose(data: bytes, element_size: int) -> bytes:
    """Byte untranspose — C-accelerated if available."""
    n_elements = len(data) // element_size
    if n_elements == 0:
        return data

    if HAS_C_EXT:
        src = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        dst = (ctypes.c_uint8 * len(data))()
        _lib.byte_untranspose(src, dst, n_elements, element_size)
        return bytes(dst)

    # Fallback: NumPy
    arr = np.frombuffer(data, dtype=np.uint8).reshape(element_size, n_elements)
    return arr.T.tobytes()


def fast_delta_encode_order1(data: np.ndarray) -> np.ndarray:
    """Order-1 delta encoding on int32 (N, W, F) array."""
    n, w, f = data.shape

    if HAS_C_EXT:
        src = np.ascontiguousarray(data, dtype=np.int32)
        dst = np.empty_like(src)
        _lib.delta_encode_order1(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            dst.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n, w, f,
        )
        return dst

    # Fallback: NumPy
    r = np.empty_like(data)
    r[:, 0, :] = data[:, 0, :]
    r[:, 1:, :] = data[:, 1:, :] - data[:, :-1, :]
    return r


def fast_delta_decode_order1(data: np.ndarray) -> np.ndarray:
    """Order-1 delta decoding (cumsum) on int32 (N, W, F) array. In-place."""
    n, w, f = data.shape

    if HAS_C_EXT:
        out = np.ascontiguousarray(data.copy(), dtype=np.int32)
        _lib.delta_decode_order1(
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n, w, f,
        )
        return out

    # Fallback: Python loop (sequential, cannot vectorize)
    out = data.copy()
    for t in range(1, w):
        out[:, t, :] += out[:, t - 1, :]
    return out


def fast_delta_decode_order2(data: np.ndarray) -> np.ndarray:
    """Order-2 delta decoding on int32 (N, W, F) array."""
    n, w, f = data.shape

    if HAS_C_EXT:
        out = np.ascontiguousarray(data.copy(), dtype=np.int32)
        _lib.delta_decode_order2(
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n, w, f,
        )
        return out

    # Fallback
    out = data.copy()
    if w > 1:
        out[:, 1, :] = (data[:, 1, :].astype(np.int64) + out[:, 0, :].astype(np.int64)).astype(np.int32)
    for t in range(2, w):
        predicted = (2 * out[:, t-1, :].astype(np.int64) - out[:, t-2, :].astype(np.int64))
        out[:, t, :] = (data[:, t, :].astype(np.int64) + predicted).astype(np.int32)
    return out


def fast_delta_decode_order3(data: np.ndarray) -> np.ndarray:
    """Order-3 delta decoding on int32 (N, W, F) array."""
    n, w, f = data.shape

    if HAS_C_EXT:
        out = np.ascontiguousarray(data.copy(), dtype=np.int32)
        _lib.delta_decode_order3(
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n, w, f,
        )
        return out

    # Fallback
    out = data.copy()
    if w > 1:
        out[:, 1, :] += out[:, 0, :]
    if w > 2:
        out[:, 2, :] = (data[:, 2, :].astype(np.int64)
                        + 2 * out[:, 1, :].astype(np.int64)
                        - out[:, 0, :].astype(np.int64)).astype(np.int32)
    for t in range(3, w):
        predicted = (3 * out[:, t-1, :].astype(np.int64)
                     - 3 * out[:, t-2, :].astype(np.int64)
                     + out[:, t-3, :].astype(np.int64))
        out[:, t, :] = (data[:, t, :].astype(np.int64) + predicted).astype(np.int32)
    return out


def fast_delta_decode_order4(data: np.ndarray) -> np.ndarray:
    """Order-4 delta decoding on int32 (N, W, F) array."""
    n, w, f = data.shape

    if HAS_C_EXT:
        out = np.ascontiguousarray(data.copy(), dtype=np.int32)
        _lib.delta_decode_order4(
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n, w, f,
        )
        return out

    # Fallback
    out = data.copy()
    if w > 1:
        out[:, 1, :] += out[:, 0, :]
    if w > 2:
        out[:, 2, :] = (data[:, 2, :].astype(np.int64)
                        + 2 * out[:, 1, :].astype(np.int64)
                        - out[:, 0, :].astype(np.int64)).astype(np.int32)
    if w > 3:
        out[:, 3, :] = (data[:, 3, :].astype(np.int64)
                        + 3 * out[:, 2, :].astype(np.int64)
                        - 3 * out[:, 1, :].astype(np.int64)
                        + out[:, 0, :].astype(np.int64)).astype(np.int32)
    for t in range(4, w):
        predicted = (4 * out[:, t-1, :].astype(np.int64)
                     - 6 * out[:, t-2, :].astype(np.int64)
                     + 4 * out[:, t-3, :].astype(np.int64)
                     - out[:, t-4, :].astype(np.int64))
        out[:, t, :] = (data[:, t, :].astype(np.int64) + predicted).astype(np.int32)
    return out


def fast_xor_decode(data: np.ndarray) -> np.ndarray:
    """XOR delta decoding on uint32 (N, W, F) array. In-place."""
    n, w, f = data.shape

    if HAS_C_EXT:
        out = np.ascontiguousarray(data.copy(), dtype=np.uint32)
        _lib.xor_decode(
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            n, w, f,
        )
        return out

    # Fallback
    out = data.copy()
    for t in range(1, w):
        out[:, t, :] ^= out[:, t - 1, :]
    return out


def fast_zigzag_encode(data: np.ndarray) -> np.ndarray:
    """Zigzag encoding: signed int32 -> unsigned uint32."""
    count = data.size

    if HAS_C_EXT:
        src = np.ascontiguousarray(data.ravel(), dtype=np.int32)
        dst = np.empty(count, dtype=np.uint32)
        _lib.zigzag_encode(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            count,
        )
        return dst.reshape(data.shape)

    # Fallback
    return ((data >> 31) ^ (data << 1)).view(np.uint32)


def fast_zigzag_decode(data: np.ndarray) -> np.ndarray:
    """Zigzag decoding: unsigned uint32 -> signed int32."""
    count = data.size

    if HAS_C_EXT:
        src = np.ascontiguousarray(data.ravel(), dtype=np.uint32)
        dst = np.empty(count, dtype=np.int32)
        _lib.zigzag_decode(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            dst.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            count,
        )
        return dst.reshape(data.shape)

    # Fallback
    return ((data >> 1) ^ -(data & 1).astype(np.int32)).view(np.int32)
