"""Tests for optimal compression mode with strategy competition."""

import numpy as np
import pytest
import zlib

import umc
from umc.codec.optimal import (
    OptimalityCertificate,
    _byte_entropy,
    _chi_squared_uniformity,
    optimal_compress,
    optimal_decompress_full,
    read_certificate,
    _TRANSFORM_NAMES,
    _COMPRESSOR_NAMES,
)
from umc.codec.tiered import _compress_storage, _decompress_storage
from umc.codec.residual import byte_transpose


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def financial_data():
    """Realistic financial-like data: (20, 32, 5)."""
    rng = np.random.RandomState(42)
    base = 100.0 + rng.randn(20, 1, 5).cumsum(axis=0) * 0.5
    noise = rng.randn(20, 32, 5) * 0.01
    return (base + noise).astype(np.float32)


@pytest.fixture
def iot_data():
    """IoT sensor-like data: (50, 64, 8)."""
    rng = np.random.RandomState(123)
    t = np.linspace(0, 10, 64).reshape(1, 64, 1)
    signals = np.sin(t * np.arange(1, 9).reshape(1, 1, 8)) + rng.randn(50, 64, 8) * 0.05
    return signals.astype(np.float32)


@pytest.fixture
def audio_data():
    """Audio-like data: (10, 1024, 1)."""
    rng = np.random.RandomState(7)
    t = np.linspace(0, 2 * np.pi, 1024).reshape(1, 1024, 1)
    waves = np.sin(440 * t) + 0.3 * np.sin(880 * t) + rng.randn(10, 1024, 1) * 0.01
    return waves.astype(np.float32)


@pytest.fixture
def small_data():
    """Minimal data: (1, 32, 5)."""
    rng = np.random.RandomState(0)
    return rng.randn(1, 32, 5).astype(np.float32)


# ---------------------------------------------------------------------------
# Round-trip tests (bit-exact lossless)
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Verify optimal mode is bit-exact lossless."""

    def test_roundtrip_financial(self, financial_data):
        compressed = _compress_storage(financial_data, "optimal")
        recovered = _decompress_storage(compressed)
        assert np.array_equal(financial_data, recovered)

    def test_roundtrip_iot(self, iot_data):
        compressed = _compress_storage(iot_data, "optimal")
        recovered = _decompress_storage(compressed)
        assert np.array_equal(iot_data, recovered)

    def test_roundtrip_audio(self, audio_data):
        compressed = _compress_storage(audio_data, "optimal")
        recovered = _decompress_storage(compressed)
        assert np.array_equal(audio_data, recovered)

    def test_roundtrip_small(self, small_data):
        compressed = _compress_storage(small_data, "optimal")
        recovered = _decompress_storage(compressed)
        assert np.array_equal(small_data, recovered)

    def test_roundtrip_via_umc_api(self, financial_data):
        """Test through the top-level umc.compress/decompress API."""
        compressed = umc.compress(financial_data, mode="optimal")
        recovered = umc.decompress(compressed)
        assert np.array_equal(financial_data, recovered)

    def test_roundtrip_constant_data(self):
        """Constant data should compress well and round-trip perfectly."""
        data = np.full((5, 16, 3), 42.0, dtype=np.float32)
        compressed = umc.compress(data, mode="optimal")
        recovered = umc.decompress(compressed)
        assert np.array_equal(data, recovered)

    def test_roundtrip_negative_values(self):
        """Negative values should round-trip correctly."""
        rng = np.random.RandomState(99)
        data = (rng.randn(10, 32, 5) * 100).astype(np.float32)
        compressed = umc.compress(data, mode="optimal")
        recovered = umc.decompress(compressed)
        assert np.array_equal(data, recovered)


# ---------------------------------------------------------------------------
# Certificate tests
# ---------------------------------------------------------------------------

class TestCertificate:
    """Verify optimality certificate contents."""

    def test_certificate_fields(self, financial_data):
        payload, cert = optimal_compress(financial_data)
        assert cert.entropy_h0 >= 0
        assert cert.achieved_bpb > 0
        assert cert.entropy_gap_bpb >= 0
        assert cert.entropy_gap_pct >= 0
        assert 0 <= cert.randomness_p_value <= 1
        assert cert.transform_id in range(len(_TRANSFORM_NAMES)) or cert.transform_id == 255
        assert cert.compressor_id in range(len(_COMPRESSOR_NAMES))
        assert cert.original_size == financial_data.nbytes
        assert cert.compressed_size > 0
        assert cert.ratio > 0

    def test_certificate_pack_unpack(self, financial_data):
        _, cert = optimal_compress(financial_data)
        packed = cert.pack()
        unpacked = OptimalityCertificate.unpack(packed)
        assert abs(unpacked.entropy_h0 - cert.entropy_h0) < 1e-10
        assert abs(unpacked.achieved_bpb - cert.achieved_bpb) < 1e-10
        assert unpacked.transform_id == cert.transform_id
        assert unpacked.compressor_id == cert.compressor_id
        assert unpacked.original_size == cert.original_size
        assert unpacked.compressed_size == cert.compressed_size

    def test_read_certificate_from_payload(self, financial_data):
        payload, cert = optimal_compress(financial_data)
        read_cert = read_certificate(payload)
        assert read_cert.transform_id == cert.transform_id
        assert read_cert.compressor_id == cert.compressor_id
        assert abs(read_cert.entropy_h0 - cert.entropy_h0) < 1e-10

    def test_compress_optimal_api(self, financial_data):
        result = umc.compress_optimal(financial_data)
        assert "compressed" in result
        assert "certificate" in result
        c = result["certificate"]
        assert "entropy_h0" in c
        assert "entropy_gap_pct" in c
        assert "ratio" in c
        assert c["ratio"] > 0
        # Verify the compressed data can be decompressed
        recovered = umc.decompress(result["compressed"])
        assert np.array_equal(financial_data, recovered)


# ---------------------------------------------------------------------------
# Entropy gap tests
# ---------------------------------------------------------------------------

class TestEntropyGap:
    """Verify entropy gap is non-negative (can't beat entropy)."""

    def test_gap_nonnegative(self, financial_data):
        _, cert = optimal_compress(financial_data)
        assert cert.entropy_gap_bpb >= 0, "Achieved bpb should be >= entropy H0"

    def test_gap_nonnegative_random(self):
        """Random data has max entropy; gap should be small or zero."""
        rng = np.random.RandomState(42)
        data = rng.randn(20, 32, 5).astype(np.float32)
        _, cert = optimal_compress(data)
        assert cert.entropy_gap_bpb >= 0


# ---------------------------------------------------------------------------
# Better-than-zlib test
# ---------------------------------------------------------------------------

class TestBetterThanBaseline:
    """Optimal should be at least as good as any single strategy."""

    def test_at_least_as_good_as_zlib(self, financial_data):
        """Optimal output should be <= zlib level 9 (since zlib is a candidate)."""
        optimal_compressed = _compress_storage(financial_data, "optimal")
        zlib_compressed = _compress_storage(financial_data, "lossless")
        # Optimal should be no larger (it tries zlib among others)
        assert len(optimal_compressed) <= len(zlib_compressed) + 100  # small header overhead


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Test entropy and randomness measurement functions."""

    def test_byte_entropy_constant(self):
        data = bytes([42] * 1000)
        assert _byte_entropy(data) == 0.0

    def test_byte_entropy_uniform(self):
        data = bytes(range(256)) * 100
        h = _byte_entropy(data)
        assert 7.9 < h <= 8.0, f"Uniform should be ~8 bits, got {h}"

    def test_byte_entropy_empty(self):
        assert _byte_entropy(b"") == 0.0

    def test_chi_squared_random(self):
        rng = np.random.RandomState(42)
        data = bytes(rng.randint(0, 256, 10000).tolist())
        p = _chi_squared_uniformity(data)
        # Random data should have high p-value (not reject uniform hypothesis)
        assert p > 0.01, f"Random bytes should be near-uniform, p={p}"

    def test_chi_squared_skewed(self):
        data = bytes([0] * 5000 + [1] * 5000)
        p = _chi_squared_uniformity(data)
        # Highly skewed data should have very low p-value
        assert p < 0.01, f"Skewed bytes should reject uniform, p={p}"


# ---------------------------------------------------------------------------
# New transform round-trip tests
# ---------------------------------------------------------------------------

class TestNewTransforms:
    """Verify new transforms (zigzag, flatten, window_time) are lossless."""

    def test_delta_zigzag_roundtrip(self, financial_data):
        from umc.codec.optimal import _transform_delta_zigzag, _inverse_delta_zigzag
        transformed = _transform_delta_zigzag(financial_data)
        recovered = _inverse_delta_zigzag(transformed, financial_data.shape)
        assert np.array_equal(financial_data, recovered)

    def test_flatten_delta_zigzag_roundtrip(self, financial_data):
        from umc.codec.optimal import (
            _transform_flatten_delta_zigzag, _inverse_flatten_delta_zigzag
        )
        transformed = _transform_flatten_delta_zigzag(financial_data)
        recovered = _inverse_flatten_delta_zigzag(transformed, financial_data.shape)
        assert np.array_equal(financial_data, recovered)

    def test_window_time_delta_roundtrip(self, iot_data):
        from umc.codec.optimal import (
            _transform_window_time_delta, _inverse_window_time_delta
        )
        transformed = _transform_window_time_delta(iot_data)
        recovered = _inverse_window_time_delta(transformed, iot_data.shape)
        assert np.array_equal(iot_data, recovered)

    def test_linear_zigzag_roundtrip(self, audio_data):
        from umc.codec.optimal import (
            _transform_linear_zigzag, _inverse_linear_zigzag
        )
        transformed = _transform_linear_zigzag(audio_data)
        recovered = _inverse_linear_zigzag(transformed, audio_data.shape)
        assert np.array_equal(audio_data, recovered)

    def test_zigzag_encode_decode(self):
        from umc.codec.optimal import _zigzag_encode, _zigzag_decode
        values = np.array([0, -1, 1, -2, 2, -128, 127, -2**31+1], dtype=np.int32)
        encoded = _zigzag_encode(values)
        decoded = _zigzag_decode(encoded)
        assert np.array_equal(values, decoded)

    def test_delta_identity_roundtrip(self, financial_data):
        from umc.codec.optimal import _transform_delta_identity, _inverse_delta_identity
        transformed = _transform_delta_identity(financial_data)
        recovered = _inverse_delta_identity(transformed, financial_data.shape)
        assert np.array_equal(financial_data, recovered)

    def test_split_em_roundtrip(self, financial_data):
        from umc.codec.optimal import _transform_split_em, _inverse_split_em
        transformed = _transform_split_em(financial_data)
        recovered = _inverse_split_em(transformed, financial_data.shape)
        assert np.array_equal(financial_data, recovered)

    def test_split_em_roundtrip_negative(self):
        """Split exponent/mantissa should handle negative values."""
        from umc.codec.optimal import _transform_split_em, _inverse_split_em
        data = np.array([[[-1.5, 0.0, 3.14], [1e-7, 1e7, -0.0]]], dtype=np.float32)
        transformed = _transform_split_em(data)
        recovered = _inverse_split_em(transformed, data.shape)
        assert np.array_equal(data, recovered)

    def test_flatten_delta_identity_roundtrip(self, financial_data):
        from umc.codec.optimal import (
            _transform_flatten_delta_identity, _inverse_flatten_delta_identity
        )
        transformed = _transform_flatten_delta_identity(financial_data)
        recovered = _inverse_flatten_delta_identity(transformed, financial_data.shape)
        assert np.array_equal(financial_data, recovered)


# ---------------------------------------------------------------------------
# Raw binary compression tests
# ---------------------------------------------------------------------------

class TestRawBinaryCompression:
    """Test compress_raw / decompress_raw for arbitrary binary data."""

    def test_raw_roundtrip_text(self):
        raw = b"Hello, World! " * 100
        compressed = umc.compress_raw(raw)
        recovered = umc.decompress_raw(compressed)
        assert raw == recovered

    def test_raw_roundtrip_binary(self):
        rng = np.random.RandomState(42)
        raw = bytes(rng.randint(0, 256, 10000).tolist())
        compressed = umc.compress_raw(raw)
        recovered = umc.decompress_raw(compressed)
        assert raw == recovered

    def test_raw_compression_ratio(self):
        # Highly compressible: repeated pattern
        raw = b"\x00\x01\x02\x03" * 10000
        compressed = umc.compress_raw(raw)
        assert len(compressed) < len(raw) / 2

    def test_raw_magic_header(self):
        raw = b"test data " * 50
        compressed = umc.compress_raw(raw)
        assert compressed[:4] == b"UMCR"


# ---------------------------------------------------------------------------
# Optimal fast mode tests
# ---------------------------------------------------------------------------

class TestOptimalFast:
    """Verify optimal_fast mode produces correct, competitive results."""

    def test_roundtrip_financial(self, financial_data):
        compressed = umc.compress(financial_data, mode="optimal_fast")
        recovered = umc.decompress(compressed)
        assert np.array_equal(financial_data, recovered)

    def test_roundtrip_small(self, small_data):
        compressed = umc.compress(small_data, mode="optimal_fast")
        recovered = umc.decompress(compressed)
        assert np.array_equal(small_data, recovered)

    def test_roundtrip_constant(self):
        data = np.full((5, 16, 3), 42.0, dtype=np.float32)
        compressed = umc.compress(data, mode="optimal_fast")
        recovered = umc.decompress(compressed)
        assert np.array_equal(data, recovered)

    def test_competitive_with_optimal(self, financial_data):
        """optimal_fast should be within 5% of optimal on compression ratio."""
        opt = umc.compress(financial_data, mode="optimal")
        fast = umc.compress(financial_data, mode="optimal_fast")
        # Allow 5% ratio loss + some absolute tolerance for small data
        assert len(fast) <= len(opt) * 1.05 + 200
