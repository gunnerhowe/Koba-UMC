"""Optimal compression with predictive coding, strategy competition, and optimality certificate.

Uses multi-order predictive coding (like FLAC/FPC for audio/floats) combined
with brute-force strategy competition (MDL principle) to achieve the best
possible lossless compression for structured numeric data AND arbitrary bytes.

Pipeline:
    1. Predictive transforms: predict each value from context, store residual
       - Order 0-4: raw, delta, linear, quadratic, cubic
       - Zigzag encoding maps signed residuals to small unsigned integers
       - Per-feature adaptive: pick best order per feature
       - Cross-feature: exploit inter-feature correlation
       - Flatten-delta: treat windows as continuous time series
       - Window-delta: exploit cross-window correlation
    2. Byte transposition: group same-position bytes together
    3. Compressor competition: zlib-9, zstd-22, lzma-9, brotli-11,
       per-channel lzma, arithmetic coding
    4. Optimality certificate: entropy gap, randomness test

Usage:
    from umc.codec.optimal import optimal_compress, optimal_decompress

    payload, cert = optimal_compress(data)   # data: (N, W, F) float32
    recovered = optimal_decompress(payload)  # bit-exact round-trip
    print(cert)  # entropy gap, randomness test, compression ratio
"""

from __future__ import annotations

import struct
import zlib
from collections import Counter
from dataclasses import dataclass
from math import log2
from typing import Optional

import numpy as np

from .residual import byte_transpose, byte_untranspose

# ---------------------------------------------------------------------------
# Optimality certificate
# ---------------------------------------------------------------------------


@dataclass
class OptimalityCertificate:
    """Proof of compression quality relative to theoretical limits."""

    entropy_h0: float          # Shannon H0 in bits/byte of best preprocessed stream
    achieved_bpb: float        # Actual compressed bits per byte
    entropy_gap_bpb: float     # achieved - H0  (>= 0, lower is better)
    entropy_gap_pct: float     # gap as % of H0  (0% = perfect)
    randomness_p_value: float  # Chi-squared p-value on compressed bytes
    transform_id: int          # Which preprocessing won
    compressor_id: int         # Which backend won
    original_size: int         # Input size in bytes
    compressed_size: int       # Output size in bytes
    ratio: float               # original / compressed

    # Fixed serialized size: 8*5 + 1*2 + 4*2 = 50 bytes
    _STRUCT_FMT = "<5d2B2I"
    _STRUCT_SIZE = struct.calcsize("<5d2B2I")

    def pack(self) -> bytes:
        return struct.pack(
            self._STRUCT_FMT,
            self.entropy_h0, self.achieved_bpb, self.entropy_gap_bpb,
            self.entropy_gap_pct, self.randomness_p_value,
            self.transform_id, self.compressor_id,
            self.original_size, self.compressed_size,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "OptimalityCertificate":
        vals = struct.unpack(cls._STRUCT_FMT, data[:cls._STRUCT_SIZE])
        h0, bpb, gap, gap_pct, p_val, t_id, c_id, orig, comp = vals
        return cls(
            entropy_h0=h0, achieved_bpb=bpb, entropy_gap_bpb=gap,
            entropy_gap_pct=gap_pct, randomness_p_value=p_val,
            transform_id=t_id, compressor_id=c_id,
            original_size=orig, compressed_size=comp,
            ratio=orig / max(comp, 1),
        )


# ---------------------------------------------------------------------------
# Entropy & randomness measurement
# ---------------------------------------------------------------------------


def _byte_entropy(data: bytes) -> float:
    """Shannon H0 entropy in bits per byte."""
    if len(data) == 0:
        return 0.0
    counts = Counter(data)
    n = len(data)
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * log2(p)
    return h


def _bigram_entropy(data: bytes) -> float:
    """Shannon H1 (bigram/conditional) entropy in bits per byte.

    H1 = H(X_{t} | X_{t-1}), more accurate than H0 for correlated data.
    Measures how many bits per byte are needed given the previous byte.
    """
    if len(data) < 2:
        return _byte_entropy(data)

    # Count bigrams
    bigram_counts: dict[int, dict[int, int]] = {}
    for i in range(len(data) - 1):
        a, b = data[i], data[i + 1]
        if a not in bigram_counts:
            bigram_counts[a] = {}
        bigram_counts[a][b] = bigram_counts[a].get(b, 0) + 1

    # Compute conditional entropy H(X|Y)
    n = len(data) - 1
    h1 = 0.0
    for a, successors in bigram_counts.items():
        count_a = sum(successors.values())
        p_a = count_a / n
        h_given_a = 0.0
        for count_b in successors.values():
            p_b_given_a = count_b / count_a
            if p_b_given_a > 0:
                h_given_a -= p_b_given_a * log2(p_b_given_a)
        h1 += p_a * h_given_a

    return h1


def _chi_squared_uniformity(data: bytes) -> float:
    """Chi-squared test p-value against uniform byte distribution.

    Returns p-value: if > 0.05, the bytes are statistically
    indistinguishable from random (incompressible).
    """
    if len(data) < 256:
        return 0.0

    counts = np.zeros(256, dtype=np.float64)
    for b in data:
        counts[b] += 1

    expected = len(data) / 256.0
    chi2_stat = float(np.sum((counts - expected) ** 2 / expected))
    df = 255

    z = ((chi2_stat / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / (
        (2.0 / (9.0 * df)) ** 0.5
    )
    p_value = 0.5 * (1.0 + _erf(-z / 2**0.5))
    return max(0.0, min(1.0, p_value))


def _erf(x: float) -> float:
    """Approximate error function (Abramowitz & Stegun 7.1.26)."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    )
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def _compute_certificate(
    preprocessed_bytes: bytes,
    compressed_bytes: bytes,
    original_size: int,
    transform_id: int,
    compressor_id: int,
) -> OptimalityCertificate:
    """Build an optimality certificate from compression results."""
    # Use H1 (bigram entropy) for more accurate gap measurement
    h0 = _byte_entropy(preprocessed_bytes)
    h1 = _bigram_entropy(preprocessed_bytes)
    # Use the lower (tighter) bound
    h_best = min(h0, h1)

    n_bytes_in = len(preprocessed_bytes)
    n_bytes_out = len(compressed_bytes)
    achieved_bpb = (n_bytes_out * 8.0) / max(n_bytes_in, 1)

    gap = max(achieved_bpb - h_best, 0.0)
    gap_pct = (gap / h_best * 100.0) if h_best > 0 else 0.0

    p_value = _chi_squared_uniformity(compressed_bytes)

    return OptimalityCertificate(
        entropy_h0=h_best,
        achieved_bpb=achieved_bpb,
        entropy_gap_bpb=gap,
        entropy_gap_pct=gap_pct,
        randomness_p_value=p_value,
        transform_id=transform_id,
        compressor_id=compressor_id,
        original_size=original_size,
        compressed_size=n_bytes_out,
        ratio=original_size / max(n_bytes_out, 1),
    )


# ---------------------------------------------------------------------------
# Zigzag encoding: maps signed int32 to unsigned so small values are small
# ---------------------------------------------------------------------------


def _zigzag_encode(x: np.ndarray) -> np.ndarray:
    """Zigzag encode int32 -> uint32: 0->0, -1->1, 1->2, -2->3, 2->4, ..."""
    return ((x >> 31) ^ (x << 1)).view(np.uint32)


def _zigzag_decode(x: np.ndarray) -> np.ndarray:
    """Zigzag decode uint32 -> int32."""
    x = x.view(np.uint32)
    # Both operands must be int32 to avoid int64 upcast
    shifted = (x >> 1).view(np.int32)
    mask = -((x & 1).astype(np.int32))
    return shifted ^ mask


# ---------------------------------------------------------------------------
# Int32 prediction helpers (exact integer arithmetic, no float rounding)
# ---------------------------------------------------------------------------


def _int32_residuals_order1(as_int: np.ndarray) -> np.ndarray:
    """Order-1 (delta) prediction in int32 space. C-accelerated."""
    try:
        from ..cext import HAS_C_EXT, fast_delta_encode_order1
        if HAS_C_EXT:
            return fast_delta_encode_order1(as_int)
    except ImportError:
        pass
    r = np.empty_like(as_int)
    r[:, 0, :] = as_int[:, 0, :]
    r[:, 1:, :] = as_int[:, 1:, :] - as_int[:, :-1, :]
    return r


def _int32_reconstruct_order1(r: np.ndarray) -> np.ndarray:
    """Inverse of order-1 residuals. C-accelerated (sequential loop)."""
    try:
        from ..cext import HAS_C_EXT, fast_delta_decode_order1
        if HAS_C_EXT:
            return fast_delta_decode_order1(r)
    except ImportError:
        pass
    out = r.copy()
    for t in range(1, r.shape[1]):
        out[:, t, :] += out[:, t - 1, :]
    return out


def _int32_residuals_order2(as_int: np.ndarray) -> np.ndarray:
    """Order-2 (linear) prediction: predicted[t] = 2*x[t-1] - x[t-2]."""
    r = np.empty_like(as_int)
    r[:, 0, :] = as_int[:, 0, :]
    r[:, 1, :] = as_int[:, 1, :] - as_int[:, 0, :]
    if as_int.shape[1] > 2:
        predicted = (2 * as_int[:, 1:-1, :].astype(np.int64)
                     - as_int[:, :-2, :].astype(np.int64))
        r[:, 2:, :] = (as_int[:, 2:, :].astype(np.int64) - predicted).astype(np.int32)
    return r


def _int32_reconstruct_order2(r: np.ndarray) -> np.ndarray:
    """Inverse of order-2 residuals. C-accelerated."""
    try:
        from ..cext import HAS_C_EXT, fast_delta_decode_order2
        if HAS_C_EXT:
            return fast_delta_decode_order2(r)
    except ImportError:
        pass
    out = r.copy()
    out[:, 1, :] = r[:, 1, :] + out[:, 0, :]
    for t in range(2, r.shape[1]):
        predicted = (2 * out[:, t - 1, :].astype(np.int64)
                     - out[:, t - 2, :].astype(np.int64))
        out[:, t, :] = (r[:, t, :].astype(np.int64) + predicted).astype(np.int32)
    return out


def _int32_residuals_order3(as_int: np.ndarray) -> np.ndarray:
    """Order-3 (quadratic) prediction: 3*x[t-1] - 3*x[t-2] + x[t-3]."""
    r = np.empty_like(as_int)
    r[:, 0, :] = as_int[:, 0, :]
    r[:, 1, :] = as_int[:, 1, :] - as_int[:, 0, :]
    if as_int.shape[1] > 2:
        r[:, 2, :] = (as_int[:, 2, :].astype(np.int64)
                       - 2 * as_int[:, 1, :].astype(np.int64)
                       + as_int[:, 0, :].astype(np.int64)).astype(np.int32)
    if as_int.shape[1] > 3:
        a = as_int.astype(np.int64)
        predicted = 3 * a[:, 2:-1, :] - 3 * a[:, 1:-2, :] + a[:, :-3, :]
        r[:, 3:, :] = (a[:, 3:, :] - predicted).astype(np.int32)
    return r


def _int32_reconstruct_order3(r: np.ndarray) -> np.ndarray:
    """Inverse of order-3 residuals. C-accelerated."""
    try:
        from ..cext import HAS_C_EXT, fast_delta_decode_order3
        if HAS_C_EXT:
            return fast_delta_decode_order3(r)
    except ImportError:
        pass
    out = r.copy()
    out[:, 1, :] = r[:, 1, :] + out[:, 0, :]
    if r.shape[1] > 2:
        out[:, 2, :] = (r[:, 2, :].astype(np.int64)
                        + 2 * out[:, 1, :].astype(np.int64)
                        - out[:, 0, :].astype(np.int64)).astype(np.int32)
    for t in range(3, r.shape[1]):
        predicted = (3 * out[:, t - 1, :].astype(np.int64)
                     - 3 * out[:, t - 2, :].astype(np.int64)
                     + out[:, t - 3, :].astype(np.int64))
        out[:, t, :] = (r[:, t, :].astype(np.int64) + predicted).astype(np.int32)
    return out


def _int32_residuals_order4(as_int: np.ndarray) -> np.ndarray:
    """Order-4 (cubic) prediction: 4a - 6b + 4c - d."""
    r = np.empty_like(as_int)
    a = as_int.astype(np.int64)
    r[:, 0, :] = as_int[:, 0, :]
    if as_int.shape[1] > 1:
        r[:, 1, :] = (a[:, 1, :] - a[:, 0, :]).astype(np.int32)
    if as_int.shape[1] > 2:
        r[:, 2, :] = (a[:, 2, :] - 2 * a[:, 1, :] + a[:, 0, :]).astype(np.int32)
    if as_int.shape[1] > 3:
        r[:, 3, :] = (a[:, 3, :] - 3 * a[:, 2, :] + 3 * a[:, 1, :] - a[:, 0, :]).astype(np.int32)
    if as_int.shape[1] > 4:
        predicted = (4 * a[:, 3:-1, :] - 6 * a[:, 2:-2, :]
                     + 4 * a[:, 1:-3, :] - a[:, :-4, :])
        r[:, 4:, :] = (a[:, 4:, :] - predicted).astype(np.int32)
    return r


def _int32_reconstruct_order4(r: np.ndarray) -> np.ndarray:
    """Inverse of order-4 residuals. C-accelerated."""
    try:
        from ..cext import HAS_C_EXT, fast_delta_decode_order4
        if HAS_C_EXT:
            return fast_delta_decode_order4(r)
    except ImportError:
        pass
    out = r.copy()
    if r.shape[1] > 1:
        out[:, 1, :] = (r[:, 1, :].astype(np.int64) + out[:, 0, :].astype(np.int64)).astype(np.int32)
    if r.shape[1] > 2:
        out[:, 2, :] = (r[:, 2, :].astype(np.int64) + 2 * out[:, 1, :].astype(np.int64)
                        - out[:, 0, :].astype(np.int64)).astype(np.int32)
    if r.shape[1] > 3:
        out[:, 3, :] = (r[:, 3, :].astype(np.int64) + 3 * out[:, 2, :].astype(np.int64)
                        - 3 * out[:, 1, :].astype(np.int64)
                        + out[:, 0, :].astype(np.int64)).astype(np.int32)
    for t in range(4, r.shape[1]):
        predicted = (4 * out[:, t - 1, :].astype(np.int64)
                     - 6 * out[:, t - 2, :].astype(np.int64)
                     + 4 * out[:, t - 3, :].astype(np.int64)
                     - out[:, t - 4, :].astype(np.int64))
        out[:, t, :] = (r[:, t, :].astype(np.int64) + predicted).astype(np.int32)
    return out


_RESIDUAL_FNS = [
    None,  # order 0 = raw (no residuals)
    (_int32_residuals_order1, _int32_reconstruct_order1),
    (_int32_residuals_order2, _int32_reconstruct_order2),
    (_int32_residuals_order3, _int32_reconstruct_order3),
    (_int32_residuals_order4, _int32_reconstruct_order4),
]


# ---------------------------------------------------------------------------
# Preprocessing transforms (all lossless, invertible)
# ---------------------------------------------------------------------------

_TRANSFORM_NAMES = [
    "raw_transpose",                # 0
    "delta_int32",                  # 1
    "xor_delta",                    # 2
    "linear_int32",                 # 3
    "quadratic_int32",              # 4
    "cubic_int32",                  # 5
    "adaptive_per_feature",         # 6
    "cross_feature",                # 7
    "delta_zigzag",                 # 8
    "flatten_delta_zigzag",         # 9
    "window_time_delta",            # 10
    "linear_zigzag",                # 11
    "identity",                     # 12 — no transposition, raw bytes
    "delta_identity",               # 13 — delta, no byte transpose
    "split_exp_mantissa",           # 14 — separate IEEE 754 components
    "flatten_delta_identity",       # 15 — flatten + delta, no byte transpose
]
_COMPRESSOR_NAMES = [
    "zlib-9",                       # 0
    "zstd-22",                      # 1
    "lzma-9",                       # 2
    "ac_static",                    # 3
    "ac_adaptive",                  # 4
    "per_channel_zlib",             # 5
    "per_channel_lzma",             # 6
    "brotli-11",                    # 7
]


# --- Transform 0: raw byte transpose ---

def _transform_raw(f32: np.ndarray) -> bytes:
    return byte_transpose(f32.tobytes(), element_size=4)

def _inverse_raw(data: bytes, shape: tuple) -> np.ndarray:
    raw = byte_untranspose(data, element_size=4)
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


# --- Transform 1: order-1 int32 delta ---

def _transform_delta_int32(f32: np.ndarray) -> bytes:
    r = _int32_residuals_order1(f32.view(np.int32))
    return byte_transpose(r.tobytes(), element_size=4)

def _inverse_delta_int32(data: bytes, shape: tuple) -> np.ndarray:
    raw = byte_untranspose(data, element_size=4)
    r = np.frombuffer(raw, dtype=np.int32).reshape(shape).copy()
    return _int32_reconstruct_order1(r).view(np.float32).copy()


# --- Transform 2: order-1 XOR delta ---

def _transform_xor(f32: np.ndarray) -> bytes:
    as_uint = f32.view(np.uint32)
    xored = np.empty_like(as_uint)
    xored[:, 0, :] = as_uint[:, 0, :]
    xored[:, 1:, :] = as_uint[:, 1:, :] ^ as_uint[:, :-1, :]
    return byte_transpose(xored.tobytes(), element_size=4)

def _inverse_xor(data: bytes, shape: tuple) -> np.ndarray:
    raw = byte_untranspose(data, element_size=4)
    xored = np.frombuffer(raw, dtype=np.uint32).reshape(shape).copy()
    for t in range(1, shape[1]):
        xored[:, t, :] ^= xored[:, t - 1, :]
    return xored.view(np.float32).copy()


# --- Transform 3: order-2 (linear) int32 prediction ---

def _transform_linear_int32(f32: np.ndarray) -> bytes:
    r = _int32_residuals_order2(f32.view(np.int32))
    return byte_transpose(r.tobytes(), element_size=4)

def _inverse_linear_int32(data: bytes, shape: tuple) -> np.ndarray:
    raw = byte_untranspose(data, element_size=4)
    r = np.frombuffer(raw, dtype=np.int32).reshape(shape).copy()
    return _int32_reconstruct_order2(r).view(np.float32).copy()


# --- Transform 4: order-3 (quadratic) int32 prediction ---

def _transform_quadratic_int32(f32: np.ndarray) -> bytes:
    r = _int32_residuals_order3(f32.view(np.int32))
    return byte_transpose(r.tobytes(), element_size=4)

def _inverse_quadratic_int32(data: bytes, shape: tuple) -> np.ndarray:
    raw = byte_untranspose(data, element_size=4)
    r = np.frombuffer(raw, dtype=np.int32).reshape(shape).copy()
    return _int32_reconstruct_order3(r).view(np.float32).copy()


# --- Transform 5: order-4 (cubic) int32 prediction ---

def _transform_cubic_int32(f32: np.ndarray) -> bytes:
    r = _int32_residuals_order4(f32.view(np.int32))
    return byte_transpose(r.tobytes(), element_size=4)

def _inverse_cubic_int32(data: bytes, shape: tuple) -> np.ndarray:
    raw = byte_untranspose(data, element_size=4)
    r = np.frombuffer(raw, dtype=np.int32).reshape(shape).copy()
    return _int32_reconstruct_order4(r).view(np.float32).copy()


# --- Transform 6: adaptive per-feature (best of orders 0-4) ---

def _transform_adaptive(f32: np.ndarray) -> bytes:
    """Try prediction orders 0-4 for each feature, pick the best."""
    n_win, win_sz, n_feat = f32.shape
    as_int = f32.view(np.int32)

    all_residuals = [
        as_int.copy(),
        _int32_residuals_order1(as_int),
        _int32_residuals_order2(as_int),
        _int32_residuals_order3(as_int),
        _int32_residuals_order4(as_int),
    ]

    orders = np.zeros(n_feat, dtype=np.uint8)
    best_residuals = np.empty_like(as_int)

    for f in range(n_feat):
        best_score = float("inf")
        best_order = 0
        for order_idx, res in enumerate(all_residuals):
            score = float(np.abs(res[:, :, f].astype(np.int64)).sum())
            if score < best_score:
                best_score = score
                best_order = order_idx
        orders[f] = best_order
        best_residuals[:, :, f] = all_residuals[best_order][:, :, f]

    header = orders.tobytes()
    transposed = byte_transpose(best_residuals.tobytes(), element_size=4)
    return header + transposed


def _inverse_adaptive(data: bytes, shape: tuple) -> np.ndarray:
    """Inverse of adaptive per-feature transform."""
    n_win, win_sz, n_feat = shape
    orders = np.frombuffer(data[:n_feat], dtype=np.uint8).copy()
    raw = byte_untranspose(data[n_feat:], element_size=4)
    residuals = np.frombuffer(raw, dtype=np.int32).reshape(shape).copy()

    result = np.empty(shape, dtype=np.int32)
    for f in range(n_feat):
        order = int(orders[f])
        feat_r = residuals[:, :, f:f + 1]
        if order == 0:
            result[:, :, f] = feat_r[:, :, 0]
        else:
            _, reconstruct_fn = _RESIDUAL_FNS[order]
            result[:, :, f] = reconstruct_fn(feat_r)[:, :, 0]

    return result.view(np.float32).copy()


# --- Transform 7: cross-feature decorrelation ---

def _transform_cross_feature(f32: np.ndarray) -> bytes:
    """Feature 0: temporal delta. Features 1+: cross-feature differences."""
    n_win, win_sz, n_feat = f32.shape
    as_int = f32.view(np.int32)
    residuals = np.empty_like(as_int)

    residuals[:, 0, 0] = as_int[:, 0, 0]
    if win_sz > 1:
        residuals[:, 1:, 0] = as_int[:, 1:, 0] - as_int[:, :-1, 0]

    for f in range(1, n_feat):
        residuals[:, :, f] = as_int[:, :, f] - as_int[:, :, f - 1]

    return byte_transpose(residuals.tobytes(), element_size=4)


def _inverse_cross_feature(data: bytes, shape: tuple) -> np.ndarray:
    n_win, win_sz, n_feat = shape
    raw = byte_untranspose(data, element_size=4)
    residuals = np.frombuffer(raw, dtype=np.int32).reshape(shape).copy()

    result = np.empty(shape, dtype=np.int32)

    result[:, 0, 0] = residuals[:, 0, 0]
    for t in range(1, win_sz):
        result[:, t, 0] = residuals[:, t, 0] + result[:, t - 1, 0]

    for f in range(1, n_feat):
        result[:, :, f] = residuals[:, :, f] + result[:, :, f - 1]

    return result.view(np.float32).copy()


# --- Transform 8: delta + zigzag (maps small residuals to small uints) ---

def _transform_delta_zigzag(f32: np.ndarray) -> bytes:
    r = _int32_residuals_order1(f32.view(np.int32))
    zz = _zigzag_encode(r)
    return byte_transpose(zz.tobytes(), element_size=4)

def _inverse_delta_zigzag(data: bytes, shape: tuple) -> np.ndarray:
    raw = byte_untranspose(data, element_size=4)
    zz = np.frombuffer(raw, dtype=np.uint32).reshape(shape).copy()
    r = _zigzag_decode(zz)
    return _int32_reconstruct_order1(r).view(np.float32).copy()


# --- Transform 9: flatten across windows then delta + zigzag ---

def _transform_flatten_delta_zigzag(f32: np.ndarray) -> bytes:
    """Reshape (N, W, F) -> (1, N*W, F), then delta + zigzag.

    This captures cross-window temporal continuity that per-window
    delta misses. Critical for time series where consecutive windows
    represent consecutive time periods.
    """
    n_win, win_sz, n_feat = f32.shape
    flat = f32.reshape(1, n_win * win_sz, n_feat)
    as_int = flat.view(np.int32)
    r = _int32_residuals_order1(as_int)
    zz = _zigzag_encode(r)
    return byte_transpose(zz.tobytes(), element_size=4)

def _inverse_flatten_delta_zigzag(data: bytes, shape: tuple) -> np.ndarray:
    n_win, win_sz, n_feat = shape
    flat_shape = (1, n_win * win_sz, n_feat)
    raw = byte_untranspose(data, element_size=4)
    zz = np.frombuffer(raw, dtype=np.uint32).reshape(flat_shape).copy()
    r = _zigzag_decode(zz)
    result = _int32_reconstruct_order1(r)
    return result.view(np.float32).reshape(shape).copy()


# --- Transform 10: window delta (axis 0) + time delta (axis 1) ---

def _transform_window_time_delta(f32: np.ndarray) -> bytes:
    """Delta across windows first, then across time steps.

    Exploits both cross-window and within-window correlation.
    """
    as_int = f32.view(np.int32).copy()

    # Delta across windows (axis 0)
    r = np.empty_like(as_int)
    r[0] = as_int[0]
    r[1:] = as_int[1:] - as_int[:-1]

    # Then delta across time (axis 1)
    r2 = np.empty_like(r)
    r2[:, 0, :] = r[:, 0, :]
    r2[:, 1:, :] = r[:, 1:, :] - r[:, :-1, :]

    zz = _zigzag_encode(r2)
    return byte_transpose(zz.tobytes(), element_size=4)

def _inverse_window_time_delta(data: bytes, shape: tuple) -> np.ndarray:
    raw = byte_untranspose(data, element_size=4)
    zz = np.frombuffer(raw, dtype=np.uint32).reshape(shape).copy()
    r2 = _zigzag_decode(zz)

    # Reverse time delta
    r = r2.copy()
    for t in range(1, shape[1]):
        r[:, t, :] += r[:, t - 1, :]

    # Reverse window delta
    result = r.copy()
    for w in range(1, shape[0]):
        result[w] += result[w - 1]

    return result.view(np.float32).copy()


# --- Transform 11: order-2 linear + zigzag ---

def _transform_linear_zigzag(f32: np.ndarray) -> bytes:
    r = _int32_residuals_order2(f32.view(np.int32))
    zz = _zigzag_encode(r)
    return byte_transpose(zz.tobytes(), element_size=4)

def _inverse_linear_zigzag(data: bytes, shape: tuple) -> np.ndarray:
    raw = byte_untranspose(data, element_size=4)
    zz = np.frombuffer(raw, dtype=np.uint32).reshape(shape).copy()
    r = _zigzag_decode(zz)
    return _int32_reconstruct_order2(r).view(np.float32).copy()


# --- Transform 12: identity (no byte transposition) ---

def _transform_identity(f32: np.ndarray) -> bytes:
    """Raw bytes, no transposition. Lets compressors work on native float32 layout."""
    return f32.tobytes()

def _inverse_identity(data: bytes, shape: tuple) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()


# --- Transform 13: delta + identity (delta encoding, no byte transpose) ---

def _transform_delta_identity(f32: np.ndarray) -> bytes:
    """Int32 delta encoding without byte transposition.

    Good for quantized data where successive values differ by small amounts
    but byte transposition disrupts the pattern (e.g., financial tick data).
    """
    r = _int32_residuals_order1(f32.view(np.int32))
    return r.tobytes()

def _inverse_delta_identity(data: bytes, shape: tuple) -> np.ndarray:
    r = np.frombuffer(data, dtype=np.int32).reshape(shape).copy()
    return _int32_reconstruct_order1(r).view(np.float32).copy()


# --- Transform 14: split exponent/mantissa ---

def _transform_split_em(f32: np.ndarray) -> bytes:
    """Split IEEE 754 float32 into exponent bytes and mantissa bytes.

    Exponent bytes (sign + 8-bit exponent) compress extremely well because
    nearby values share exponents. Mantissa bytes (23-bit fraction) are
    stored separately. Each group is byte-transposed independently.
    """
    as_uint = f32.view(np.uint32).ravel()
    n = len(as_uint)

    # Extract sign+exponent (top 9 bits → stored as uint16 for alignment)
    # and mantissa (bottom 23 bits → stored as uint32 with top bits zeroed)
    exp_part = (as_uint >> 23).astype(np.uint16)  # 9 bits: sign(1) + exponent(8)
    man_part = (as_uint & 0x007FFFFF).astype(np.uint32)  # 23 bits

    exp_bytes = byte_transpose(exp_part.tobytes(), element_size=2)
    man_bytes = byte_transpose(man_part.tobytes(), element_size=4)

    header = struct.pack("<I", n)
    return header + exp_bytes + man_bytes

def _inverse_split_em(data: bytes, shape: tuple) -> np.ndarray:
    n = struct.unpack("<I", data[:4])[0]
    exp_size = n * 2
    exp_bytes = byte_untranspose(data[4:4 + exp_size], element_size=2)
    man_bytes = byte_untranspose(data[4 + exp_size:], element_size=4)

    exp_part = np.frombuffer(exp_bytes, dtype=np.uint16).astype(np.uint32)
    man_part = np.frombuffer(man_bytes, dtype=np.uint32)

    as_uint = (exp_part << 23) | man_part
    return as_uint.view(np.float32).reshape(shape).copy()


# --- Transform 15: flatten + delta + identity (no byte transpose) ---

def _transform_flatten_delta_identity(f32: np.ndarray) -> bytes:
    """Flatten across windows + delta, no byte transposition.

    Captures cross-window continuity for quantized/structured data
    where byte transposition hurts.
    """
    n_win, win_sz, n_feat = f32.shape
    flat = f32.reshape(1, n_win * win_sz, n_feat)
    as_int = flat.view(np.int32)
    r = _int32_residuals_order1(as_int)
    return r.tobytes()

def _inverse_flatten_delta_identity(data: bytes, shape: tuple) -> np.ndarray:
    n_win, win_sz, n_feat = shape
    flat_shape = (1, n_win * win_sz, n_feat)
    r = np.frombuffer(data, dtype=np.int32).reshape(flat_shape).copy()
    result = _int32_reconstruct_order1(r)
    return result.view(np.float32).reshape(shape).copy()


_TRANSFORMS = [
    _transform_raw,                      # 0
    _transform_delta_int32,              # 1
    _transform_xor,                      # 2
    _transform_linear_int32,             # 3
    _transform_quadratic_int32,          # 4
    _transform_cubic_int32,              # 5
    _transform_adaptive,                 # 6
    _transform_cross_feature,            # 7
    _transform_delta_zigzag,             # 8
    _transform_flatten_delta_zigzag,     # 9
    _transform_window_time_delta,        # 10
    _transform_linear_zigzag,            # 11
    _transform_identity,                 # 12
    _transform_delta_identity,           # 13
    _transform_split_em,                 # 14
    _transform_flatten_delta_identity,   # 15
]
_INVERSES = [
    _inverse_raw,                        # 0
    _inverse_delta_int32,                # 1
    _inverse_xor,                        # 2
    _inverse_linear_int32,               # 3
    _inverse_quadratic_int32,            # 4
    _inverse_cubic_int32,                # 5
    _inverse_adaptive,                   # 6
    _inverse_cross_feature,              # 7
    _inverse_delta_zigzag,               # 8
    _inverse_flatten_delta_zigzag,       # 9
    _inverse_window_time_delta,          # 10
    _inverse_linear_zigzag,              # 11
    _inverse_identity,                   # 12
    _inverse_delta_identity,             # 13
    _inverse_split_em,                   # 14
    _inverse_flatten_delta_identity,     # 15
]


# ---------------------------------------------------------------------------
# Compressor backends
# ---------------------------------------------------------------------------


def _compress_zlib9(data: bytes) -> bytes:
    return zlib.compress(data, 9)

def _decompress_zlib(data: bytes) -> bytes:
    return zlib.decompress(data)


def _compress_zstd22(data: bytes) -> Optional[bytes]:
    try:
        import zstandard as zstd
        return zstd.ZstdCompressor(level=22).compress(data)
    except ImportError:
        return None

def _decompress_zstd(data: bytes) -> bytes:
    import zstandard as zstd
    return zstd.ZstdDecompressor().decompress(data)


def _compress_lzma9(data: bytes) -> bytes:
    import lzma
    return lzma.compress(data, preset=9)

def _decompress_lzma(data: bytes) -> bytes:
    import lzma
    return lzma.decompress(data)


def _compress_ac_static(data: bytes) -> Optional[bytes]:
    try:
        from .arithmetic import StaticByteCompressor
        return StaticByteCompressor().compress(data, element_size=4)
    except (RuntimeError, ImportError):
        return None

def _decompress_ac_static(data: bytes) -> bytes:
    from .arithmetic import StaticByteCompressor
    return StaticByteCompressor().decompress(data, element_size=4)


def _compress_ac_adaptive(data: bytes) -> Optional[bytes]:
    try:
        from .arithmetic import AdaptiveByteCompressor
        return AdaptiveByteCompressor().compress(data, element_size=4)
    except (RuntimeError, ImportError):
        return None

def _decompress_ac_adaptive(data: bytes) -> bytes:
    from .arithmetic import AdaptiveByteCompressor
    return AdaptiveByteCompressor().decompress(data, element_size=4)


def _compress_per_channel_zlib(data: bytes) -> bytes:
    """Compress each byte-transposed channel separately with zlib-9."""
    n_bytes = len(data)
    channel_size = n_bytes // 4
    if channel_size == 0:
        return b"PCZ1" + struct.pack("<I", 0) + zlib.compress(data, 9)

    parts = []
    for ch in range(4):
        start = ch * channel_size
        end = start + channel_size
        compressed_ch = zlib.compress(data[start:end], 9)
        parts.append(struct.pack("<I", len(compressed_ch)))
        parts.append(compressed_ch)

    leftover = data[4 * channel_size:]
    return b"PCZ1" + struct.pack("<I", channel_size) + b"".join(parts) + leftover


def _decompress_per_channel_zlib(data: bytes) -> bytes:
    if data[:4] != b"PCZ1":
        raise ValueError("Not per-channel zlib compressed")
    channel_size = struct.unpack("<I", data[4:8])[0]
    if channel_size == 0:
        return zlib.decompress(data[8:])

    offset = 8
    channels = []
    for _ch in range(4):
        comp_size = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        channels.append(zlib.decompress(data[offset:offset + comp_size]))
        offset += comp_size

    leftover = data[offset:]
    return b"".join(channels) + leftover


def _compress_per_channel_lzma(data: bytes) -> bytes:
    """Compress each byte-transposed channel separately with lzma-9.

    Different byte positions in float32 have vastly different entropy:
    - MSB (sign+exponent): very low entropy, compresses 100:1
    - Middle bytes: moderate entropy
    - LSB (mantissa): high entropy, near-random
    Per-channel compression lets each channel use its own model.
    """
    import lzma
    n_bytes = len(data)
    channel_size = n_bytes // 4
    if channel_size == 0:
        return b"PCL1" + struct.pack("<I", 0) + lzma.compress(data, preset=9)

    parts = []
    for ch in range(4):
        start = ch * channel_size
        end = start + channel_size
        compressed_ch = lzma.compress(data[start:end], preset=9)
        parts.append(struct.pack("<I", len(compressed_ch)))
        parts.append(compressed_ch)

    leftover = data[4 * channel_size:]
    return b"PCL1" + struct.pack("<I", channel_size) + b"".join(parts) + leftover


def _decompress_per_channel_lzma(data: bytes) -> bytes:
    import lzma
    if data[:4] != b"PCL1":
        raise ValueError("Not per-channel lzma compressed")
    channel_size = struct.unpack("<I", data[4:8])[0]
    if channel_size == 0:
        return lzma.decompress(data[8:])

    offset = 8
    channels = []
    for _ch in range(4):
        comp_size = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        channels.append(lzma.decompress(data[offset:offset + comp_size]))
        offset += comp_size

    leftover = data[offset:]
    return b"".join(channels) + leftover


def _compress_brotli11(data: bytes) -> Optional[bytes]:
    try:
        import brotli
        return brotli.compress(data, quality=11)
    except ImportError:
        return None

def _decompress_brotli(data: bytes) -> bytes:
    import brotli
    return brotli.decompress(data)


_COMPRESSORS = [
    _compress_zlib9,                 # 0
    _compress_zstd22,                # 1
    _compress_lzma9,                 # 2
    _compress_ac_static,             # 3
    _compress_ac_adaptive,           # 4
    _compress_per_channel_zlib,      # 5
    _compress_per_channel_lzma,      # 6
    _compress_brotli11,              # 7
]

_DECOMPRESSORS = [
    _decompress_zlib,                # 0
    _decompress_zstd,                # 1
    _decompress_lzma,                # 2
    _decompress_ac_static,           # 3
    _decompress_ac_adaptive,         # 4
    _decompress_per_channel_zlib,    # 5
    _decompress_per_channel_lzma,    # 6
    _decompress_brotli,              # 7
]


# ---------------------------------------------------------------------------
# Raw bytes compression (for arbitrary binary files)
# ---------------------------------------------------------------------------


def optimal_compress_bytes(raw: bytes) -> tuple[bytes, OptimalityCertificate]:
    """Compress arbitrary byte data using best-of-N strategy competition.

    For raw binary data (exe, 3D files, etc.) — no float32 assumption.
    Tries multiple compressor backends and picks the smallest output.

    Args:
        raw: Raw bytes to compress.

    Returns:
        (payload_bytes, certificate)
    """
    original_size = len(raw)

    # For raw bytes, only use non-float-specific compressors
    raw_compressors = [
        (0, _compress_zlib9),
        (1, _compress_zstd22),
        (2, _compress_lzma9),
        (7, _compress_brotli11),
    ]

    best_size = float("inf")
    best_compressor_id = 0
    best_compressed = b""

    for c_id, compress_fn in raw_compressors:
        compressed = compress_fn(raw)
        if compressed is None:
            continue
        if len(compressed) < best_size:
            best_size = len(compressed)
            best_compressor_id = c_id
            best_compressed = compressed

    # Transform ID 255 = raw bytes (no float transform)
    transform_id = 255
    cert = _compute_certificate(
        raw, best_compressed, original_size, transform_id, best_compressor_id,
    )

    cert_bytes = cert.pack()
    header = struct.pack(_PAYLOAD_HEADER_FMT, transform_id, best_compressor_id, len(cert_bytes))
    payload = header + cert_bytes + best_compressed

    return payload, cert


def optimal_decompress_bytes(payload: bytes) -> bytes:
    """Decompress raw bytes from optimal-mode payload."""
    t_id, c_id, cert_size = struct.unpack(
        _PAYLOAD_HEADER_FMT, payload[:_PAYLOAD_HEADER_SIZE]
    )
    offset = _PAYLOAD_HEADER_SIZE + cert_size
    compressed = payload[offset:]

    decompressor = _DECOMPRESSORS[c_id]
    return decompressor(compressed)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

_PAYLOAD_HEADER_FMT = "<BBH"
_PAYLOAD_HEADER_SIZE = struct.calcsize("<BBH")

def _float16_transforms(data: np.ndarray, element_size: int) -> list:
    """Generate simplified transforms for non-float32 data (float16/bfloat16)."""
    results = []
    # Transform 0: raw byte transpose
    try:
        preprocessed = byte_transpose(data.tobytes(), element_size=element_size)
        results.append((0, preprocessed))
    except Exception:
        pass
    # Transform 1: int16 delta + byte transpose
    try:
        as_int = data.view(np.int16)
        r = np.empty_like(as_int)
        r[:, 0, :] = as_int[:, 0, :]
        r[:, 1:, :] = as_int[:, 1:, :] - as_int[:, :-1, :]
        preprocessed = byte_transpose(r.tobytes(), element_size=element_size)
        results.append((1, preprocessed))
    except Exception:
        pass
    # Transform 9: flatten + int16 delta + byte transpose
    try:
        n_win, win_sz, n_feat = data.shape
        flat = data.reshape(1, n_win * win_sz, n_feat)
        as_int = flat.view(np.int16)
        r = np.empty_like(as_int)
        r[:, 0, :] = as_int[:, 0, :]
        r[:, 1:, :] = as_int[:, 1:, :] - as_int[:, :-1, :]
        preprocessed = byte_transpose(r.tobytes(), element_size=element_size)
        results.append((9, preprocessed))
    except Exception:
        pass
    # Transform 12: identity (raw bytes, no transposition)
    try:
        results.append((12, data.tobytes()))
    except Exception:
        pass
    return results


def _float16_inverse(t_id: int, preprocessed: bytes, shape: tuple,
                     element_size: int, out_dtype) -> np.ndarray:
    """Inverse transforms for non-float32 data."""
    if t_id == 0:
        raw = byte_untranspose(preprocessed, element_size=element_size)
        return np.frombuffer(raw, dtype=out_dtype).reshape(shape).copy()
    elif t_id == 1:
        raw = byte_untranspose(preprocessed, element_size=element_size)
        r = np.frombuffer(raw, dtype=np.int16).reshape(shape).copy()
        for t in range(1, shape[1]):
            r[:, t, :] += r[:, t - 1, :]
        return r.view(out_dtype).copy()
    elif t_id == 9:
        n_win, win_sz, n_feat = shape
        flat_shape = (1, n_win * win_sz, n_feat)
        raw = byte_untranspose(preprocessed, element_size=element_size)
        r = np.frombuffer(raw, dtype=np.int16).reshape(flat_shape).copy()
        for t in range(1, flat_shape[1]):
            r[:, t, :] += r[:, t - 1, :]
        return r.view(out_dtype).reshape(shape).copy()
    elif t_id == 12:
        return np.frombuffer(preprocessed, dtype=out_dtype).reshape(shape).copy()
    else:
        raise ValueError(f"Unknown float16 transform ID: {t_id}")


def optimal_compress(data: np.ndarray, element_size: int = 4) -> tuple[bytes, OptimalityCertificate]:
    """Compress data using predictive coding + strategy competition.

    Pipeline:
        1. Run all transforms (16 preprocessors for float32, 4 for float16)
        2. Try all compressor backends (8 compressors) on each transform
        3. Pick the (transform, compressor) pair with smallest output
        4. Compute optimality certificate

    Brute-force: tries all combinations to guarantee the best result.
    Uses thread parallelism for speed.

    Args:
        data: (n_windows, window_size, n_features) array.
        element_size: Bytes per element (4=float32, 2=float16/bfloat16).

    Returns:
        (payload_bytes, certificate)
    """
    if element_size == 4:
        f32 = data.astype(np.float32)
    else:
        f32 = data
    original_size = f32.nbytes

    # Stage 1: Run all transforms (fast, numpy-bound — sequential is fine)
    if element_size != 4:
        transform_results = _float16_transforms(f32, element_size)
    else:
        transform_results = []
        for t_id, transform_fn in enumerate(_TRANSFORMS):
            try:
                preprocessed = transform_fn(f32)
                transform_results.append((t_id, preprocessed))
            except Exception:
                continue

    # Stage 2: Try ALL compressors on ALL transforms (brute-force, parallelized)
    # Use threads since compressors release the GIL (zlib, lzma, zstd, brotli)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _try_compress(t_id, preprocessed, c_id, compress_fn):
        try:
            compressed = compress_fn(preprocessed)
        except Exception:
            compressed = None
        if compressed is None:
            return None
        return (len(compressed), t_id, c_id, compressed, preprocessed)

    best_size = float("inf")
    best_transform_id = 0
    best_compressor_id = 0
    best_compressed = b""
    best_preprocessed = b""

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for t_id, preprocessed in transform_results:
            for c_id, compress_fn in enumerate(_COMPRESSORS):
                futures.append(
                    executor.submit(_try_compress, t_id, preprocessed, c_id, compress_fn)
                )

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            size, t_id, c_id, compressed, preprocessed = result
            if size < best_size:
                best_size = size
                best_transform_id = t_id
                best_compressor_id = c_id
                best_compressed = compressed
                best_preprocessed = preprocessed

    # Stage 4: Build certificate
    cert = _compute_certificate(
        best_preprocessed, best_compressed, original_size,
        best_transform_id, best_compressor_id,
    )

    # Pack payload
    cert_bytes = cert.pack()
    header = struct.pack(
        _PAYLOAD_HEADER_FMT,
        best_transform_id, best_compressor_id, len(cert_bytes),
    )
    payload = header + cert_bytes + best_compressed

    return payload, cert


_TOP_K_FAST = 4  # Number of top transforms to fully evaluate in fast mode
_IDENTITY_T_ID = 12  # Always include identity as a candidate (safety net)
# Fast compressors used in screening (skip slow AC and per-channel variants)
_FAST_SCREEN_COMPRESSORS = [0, 1, 2, 7]  # zlib-9, zstd-22, lzma-9, brotli-11


def optimal_compress_fast(data: np.ndarray, element_size: int = 4) -> tuple[bytes, OptimalityCertificate]:
    """Fast variant of optimal_compress with two-phase screening.

    Phase 1: Screen all transforms with cheap zlib-6 to rank them.
    Phase 2: Try fast compressors (zlib, zstd, lzma, brotli) on top-K transforms.
             Always includes identity transform as a safety net.

    Args:
        data: (n_windows, window_size, n_features) array.
        element_size: Bytes per element (4=float32, 2=float16/bfloat16).

    Returns:
        (payload_bytes, certificate)
    """
    if element_size == 4:
        f32 = data.astype(np.float32)
    else:
        f32 = data
    original_size = f32.nbytes

    # Phase 1: Screen all transforms with fast zlib-6
    if element_size != 4:
        # For float16/bfloat16, use simplified transforms
        all_transforms = _float16_transforms(f32, element_size)
        screen_results = []
        identity_entry = None
        for t_id, preprocessed in all_transforms:
            screened = zlib.compress(preprocessed, 6)
            entry = (len(screened), t_id, preprocessed)
            screen_results.append(entry)
            if t_id == _IDENTITY_T_ID:
                identity_entry = entry
    else:
        screen_results = []
        identity_entry = None
        for t_id, transform_fn in enumerate(_TRANSFORMS):
            try:
                preprocessed = transform_fn(f32)
                screened = zlib.compress(preprocessed, 6)
                entry = (len(screened), t_id, preprocessed)
                screen_results.append(entry)
                if t_id == _IDENTITY_T_ID:
                    identity_entry = entry
            except Exception:
                continue

    # Phase 2: Top-K transforms + always include identity
    screen_results.sort(key=lambda x: x[0])
    top_transforms = screen_results[:_TOP_K_FAST]

    # Ensure identity is always included (critical for matching standard compressors)
    top_t_ids = {entry[1] for entry in top_transforms}
    if identity_entry is not None and _IDENTITY_T_ID not in top_t_ids:
        top_transforms.append(identity_entry)

    best_size = float("inf")
    best_transform_id = 0
    best_compressor_id = 0
    best_compressed = b""
    best_preprocessed = b""

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _try_compress(t_id, preprocessed, c_id, compress_fn):
        try:
            compressed = compress_fn(preprocessed)
        except Exception:
            compressed = None
        if compressed is None:
            return None
        return (len(compressed), t_id, c_id, compressed, preprocessed)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for _score, t_id, preprocessed in top_transforms:
            for c_id in _FAST_SCREEN_COMPRESSORS:
                futures.append(
                    executor.submit(
                        _try_compress, t_id, preprocessed,
                        c_id, _COMPRESSORS[c_id],
                    )
                )

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            size, t_id, c_id, compressed, preprocessed = result
            if size < best_size:
                best_size = size
                best_transform_id = t_id
                best_compressor_id = c_id
                best_compressed = compressed
                best_preprocessed = preprocessed

    cert = _compute_certificate(
        best_preprocessed, best_compressed, original_size,
        best_transform_id, best_compressor_id,
    )

    cert_bytes = cert.pack()
    header = struct.pack(
        _PAYLOAD_HEADER_FMT,
        best_transform_id, best_compressor_id, len(cert_bytes),
    )
    payload = header + cert_bytes + best_compressed

    return payload, cert


def optimal_decompress(payload: bytes):
    """Decompress optimal-mode payload (inner step)."""
    t_id, c_id, cert_size = struct.unpack(
        _PAYLOAD_HEADER_FMT, payload[:_PAYLOAD_HEADER_SIZE]
    )
    offset = _PAYLOAD_HEADER_SIZE + cert_size
    compressed = payload[offset:]

    decompressor = _DECOMPRESSORS[c_id]
    preprocessed = decompressor(compressed)

    return t_id, preprocessed


def optimal_decompress_full(
    payload: bytes, n_windows: int, window_size: int, n_features: int,
    element_size: int = 4, out_dtype=None,
) -> np.ndarray:
    """Full decompress: payload -> array with correct shape and dtype."""
    if out_dtype is None:
        out_dtype = np.float32

    t_id, preprocessed = optimal_decompress(payload)

    # Raw bytes mode (t_id=255): no float transform
    if t_id == 255:
        return np.frombuffer(preprocessed, dtype=out_dtype).reshape(
            n_windows, window_size, n_features
        ).copy()

    shape = (n_windows, window_size, n_features)

    if element_size != 4:
        return _float16_inverse(t_id, preprocessed, shape, element_size, out_dtype)

    inverse_fn = _INVERSES[t_id]
    return inverse_fn(preprocessed, shape)


def read_certificate(payload: bytes) -> OptimalityCertificate:
    """Read the optimality certificate from a payload without decompressing."""
    _t_id, _c_id, cert_size = struct.unpack(
        _PAYLOAD_HEADER_FMT, payload[:_PAYLOAD_HEADER_SIZE]
    )
    cert_bytes = payload[_PAYLOAD_HEADER_SIZE:_PAYLOAD_HEADER_SIZE + cert_size]
    return OptimalityCertificate.unpack(cert_bytes)
