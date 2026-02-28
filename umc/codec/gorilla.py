"""Gorilla and Chimp time-series float compression.

Implements Facebook's Gorilla XOR-based float compression (Pelkonen et al., 2015)
and the improved Chimp variant (Liakos et al., 2022).

These operate on individual float32/float64 value sequences:
  - First value stored verbatim
  - Subsequent values XORed with previous
  - XOR result encoded with leading/trailing zero optimization

This is a baseline comparison for UMC's storage tier.
"""

import struct
from typing import Optional

import numpy as np


class _BitWriter:
    """Write individual bits to a byte buffer."""

    def __init__(self):
        self._bytes = bytearray()
        self._current = 0
        self._bit_pos = 0  # bits written in current byte (0-7)

    def write_bit(self, bit: int):
        self._current = (self._current << 1) | (bit & 1)
        self._bit_pos += 1
        if self._bit_pos == 8:
            self._bytes.append(self._current)
            self._current = 0
            self._bit_pos = 0

    def write_bits(self, value: int, n_bits: int):
        for i in range(n_bits - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def flush(self) -> bytes:
        if self._bit_pos > 0:
            self._current <<= (8 - self._bit_pos)
            self._bytes.append(self._current)
        return bytes(self._bytes)

    @property
    def total_bits(self) -> int:
        return len(self._bytes) * 8 + self._bit_pos


class _BitReader:
    """Read individual bits from a byte buffer."""

    def __init__(self, data: bytes):
        self._data = data
        self._byte_pos = 0
        self._bit_pos = 0  # next bit to read in current byte (0-7)

    def read_bit(self) -> int:
        if self._byte_pos >= len(self._data):
            return 0
        byte = self._data[self._byte_pos]
        bit = (byte >> (7 - self._bit_pos)) & 1
        self._bit_pos += 1
        if self._bit_pos == 8:
            self._bit_pos = 0
            self._byte_pos += 1
        return bit

    def read_bits(self, n_bits: int) -> int:
        value = 0
        for _ in range(n_bits):
            value = (value << 1) | self.read_bit()
        return value


def _float32_to_uint32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def _uint32_to_float32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def _leading_zeros_32(val: int) -> int:
    if val == 0:
        return 32
    n = 0
    if val & 0xFFFF0000 == 0:
        n += 16; val <<= 16
    if val & 0xFF000000 == 0:
        n += 8; val <<= 8
    if val & 0xF0000000 == 0:
        n += 4; val <<= 4
    if val & 0xC0000000 == 0:
        n += 2; val <<= 2
    if val & 0x80000000 == 0:
        n += 1
    return n


def _trailing_zeros_32(val: int) -> int:
    if val == 0:
        return 32
    n = 0
    if val & 0x0000FFFF == 0:
        n += 16; val >>= 16
    if val & 0x000000FF == 0:
        n += 8; val >>= 8
    if val & 0x0000000F == 0:
        n += 4; val >>= 4
    if val & 0x00000003 == 0:
        n += 2; val >>= 2
    if val & 0x00000001 == 0:
        n += 1
    return n


def gorilla_compress(values: np.ndarray) -> bytes:
    """Compress a 1D float32 array using Gorilla XOR encoding.

    Args:
        values: 1D float32 array.

    Returns:
        Compressed bytes.
    """
    values = values.ravel().astype(np.float32)
    n = len(values)
    if n == 0:
        return struct.pack("<I", 0)

    writer = _BitWriter()

    # Header: count
    header = struct.pack("<I", n)

    # First value: verbatim 32 bits
    prev = _float32_to_uint32(float(values[0]))
    writer.write_bits(prev, 32)

    prev_leading = 0
    prev_trailing = 0

    for i in range(1, n):
        curr = _float32_to_uint32(float(values[i]))
        xor = prev ^ curr

        if xor == 0:
            # Same value: single 0 bit
            writer.write_bit(0)
        else:
            leading = _leading_zeros_32(xor)
            trailing = _trailing_zeros_32(xor)
            meaningful = 32 - leading - trailing

            writer.write_bit(1)

            if (leading >= prev_leading and trailing >= prev_trailing
                    and prev_leading + prev_trailing > 0):
                # Fits within previous window: '10' prefix
                writer.write_bit(0)
                prev_meaningful = 32 - prev_leading - prev_trailing
                significant = (xor >> prev_trailing) & ((1 << prev_meaningful) - 1)
                writer.write_bits(significant, prev_meaningful)
            else:
                # New window: '11' prefix + 5-bit leading + 6-bit length + meaningful bits
                writer.write_bit(1)
                writer.write_bits(leading, 5)
                writer.write_bits(meaningful - 1, 6)  # -1 since meaningful >= 1
                significant = (xor >> trailing) & ((1 << meaningful) - 1)
                writer.write_bits(significant, meaningful)
                prev_leading = leading
                prev_trailing = trailing

        prev = curr

    return header + writer.flush()


def gorilla_decompress(data: bytes) -> np.ndarray:
    """Decompress Gorilla-encoded bytes to float32 array.

    Args:
        data: Compressed bytes from gorilla_compress().

    Returns:
        1D float32 array.
    """
    n = struct.unpack("<I", data[:4])[0]
    if n == 0:
        return np.array([], dtype=np.float32)

    reader = _BitReader(data[4:])
    values = np.empty(n, dtype=np.float32)

    # First value
    prev = reader.read_bits(32)
    values[0] = _uint32_to_float32(prev)

    prev_leading = 0
    prev_trailing = 0
    prev_meaningful = 32

    for i in range(1, n):
        if reader.read_bit() == 0:
            # Same value
            values[i] = _uint32_to_float32(prev)
        else:
            if reader.read_bit() == 0:
                # Reuse previous window
                significant = reader.read_bits(prev_meaningful)
                xor = significant << prev_trailing
            else:
                # New window
                leading = reader.read_bits(5)
                meaningful = reader.read_bits(6) + 1
                trailing = 32 - leading - meaningful
                significant = reader.read_bits(meaningful)
                xor = significant << trailing
                prev_leading = leading
                prev_trailing = trailing
                prev_meaningful = meaningful

            curr = prev ^ xor
            values[i] = _uint32_to_float32(curr)
            prev = curr

    return values


def chimp_compress(values: np.ndarray) -> bytes:
    """Compress a 1D float32 array using Chimp128 variant.

    Chimp improves on Gorilla by using the previous 128 values as references,
    picking the one with the smallest XOR. For simplicity, we use a window of
    the last 8 values (Chimp8) which gives most of the benefit.

    Args:
        values: 1D float32 array.

    Returns:
        Compressed bytes.
    """
    values = values.ravel().astype(np.float32)
    n = len(values)
    if n == 0:
        return struct.pack("<BII", 1, 0, 0)  # tag=1 (chimp), n=0

    # Chimp: for each value, XOR with the best of last WINDOW_SIZE values
    WINDOW = 8
    header = struct.pack("<BII", 1, n, WINDOW)

    writer = _BitWriter()

    # First value verbatim
    prev_vals = [_float32_to_uint32(float(values[0]))]
    writer.write_bits(prev_vals[0], 32)

    for i in range(1, n):
        curr = _float32_to_uint32(float(values[i]))

        # Find best reference (smallest XOR)
        best_xor = curr ^ prev_vals[-1]
        best_ref = len(prev_vals) - 1
        for j in range(max(0, len(prev_vals) - WINDOW), len(prev_vals)):
            xor = curr ^ prev_vals[j]
            lz = _leading_zeros_32(xor) if xor != 0 else 33
            best_lz = _leading_zeros_32(best_xor) if best_xor != 0 else 33
            if lz > best_lz:
                best_xor = xor
                best_ref = j

        # Encode reference index (3 bits for window of 8)
        ref_offset = len(prev_vals) - 1 - best_ref
        writer.write_bits(min(ref_offset, WINDOW - 1), 3)

        if best_xor == 0:
            writer.write_bit(0)
        else:
            writer.write_bit(1)
            leading = _leading_zeros_32(best_xor)
            trailing = _trailing_zeros_32(best_xor)
            meaningful = 32 - leading - trailing
            writer.write_bits(leading, 5)
            writer.write_bits(meaningful - 1, 6)
            significant = (best_xor >> trailing) & ((1 << meaningful) - 1)
            writer.write_bits(significant, meaningful)

        prev_vals.append(curr)

    return header + writer.flush()


def chimp_decompress(data: bytes) -> np.ndarray:
    """Decompress Chimp-encoded bytes to float32 array.

    Args:
        data: Compressed bytes from chimp_compress().

    Returns:
        1D float32 array.
    """
    tag, n, window = struct.unpack("<BII", data[:9])
    if n == 0:
        return np.array([], dtype=np.float32)

    reader = _BitReader(data[9:])
    values = np.empty(n, dtype=np.float32)

    # First value
    first = reader.read_bits(32)
    values[0] = _uint32_to_float32(first)
    prev_vals = [first]

    for i in range(1, n):
        ref_offset = reader.read_bits(3)
        ref_idx = len(prev_vals) - 1 - ref_offset
        ref_idx = max(0, ref_idx)
        ref_val = prev_vals[ref_idx]

        if reader.read_bit() == 0:
            curr = ref_val
        else:
            leading = reader.read_bits(5)
            meaningful = reader.read_bits(6) + 1
            trailing = 32 - leading - meaningful
            significant = reader.read_bits(meaningful)
            xor = significant << trailing
            curr = ref_val ^ xor

        values[i] = _uint32_to_float32(curr)
        prev_vals.append(curr)

    return values


def compress_array(data: np.ndarray, method: str = "gorilla") -> bytes:
    """Compress a float32 array (any shape) using specified method.

    Flattens the array, compresses, and stores shape metadata.

    Args:
        data: Float32 array of any shape.
        method: 'gorilla' or 'chimp'.

    Returns:
        Compressed bytes with shape header.
    """
    shape = data.shape
    flat = data.ravel().astype(np.float32)

    # Shape header: ndim + dimensions
    shape_header = struct.pack("<B", len(shape))
    for d in shape:
        shape_header += struct.pack("<I", d)

    if method == "gorilla":
        payload = gorilla_compress(flat)
    elif method == "chimp":
        payload = chimp_compress(flat)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'gorilla' or 'chimp'.")

    return shape_header + payload


def decompress_array(data: bytes) -> np.ndarray:
    """Decompress bytes from compress_array() back to original shape.

    Auto-detects Gorilla vs Chimp format.

    Args:
        data: Compressed bytes from compress_array().

    Returns:
        Float32 array restored to original shape.
    """
    ndim = struct.unpack("<B", data[:1])[0]
    offset = 1
    shape = []
    for _ in range(ndim):
        d = struct.unpack("<I", data[offset:offset + 4])[0]
        shape.append(d)
        offset += 4

    payload = data[offset:]

    # Detect format: chimp has tag byte = 1 at start
    if len(payload) >= 9 and payload[0] == 1:
        flat = chimp_decompress(payload)
    else:
        flat = gorilla_decompress(payload)

    return flat.reshape(shape)
