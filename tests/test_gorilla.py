"""Tests for Gorilla and Chimp time-series float compression."""

import numpy as np
import pytest

from umc.codec.gorilla import (
    gorilla_compress,
    gorilla_decompress,
    chimp_compress,
    chimp_decompress,
    compress_array,
    decompress_array,
)


class TestGorillaCompress:
    """Test Gorilla XOR-based compression."""

    def test_roundtrip_random(self):
        """Random float32 data roundtrips losslessly."""
        rng = np.random.RandomState(42)
        data = rng.randn(1000).astype(np.float32)
        compressed = gorilla_compress(data)
        decoded = gorilla_decompress(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_roundtrip_financial(self):
        """Financial-like price series roundtrips losslessly."""
        rng = np.random.RandomState(42)
        prices = 100.0 + np.cumsum(rng.randn(500) * 0.5)
        data = prices.astype(np.float32)
        compressed = gorilla_compress(data)
        decoded = gorilla_decompress(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_roundtrip_constant(self):
        """Constant values compress extremely well."""
        data = np.full(1000, 42.0, dtype=np.float32)
        compressed = gorilla_compress(data)
        decoded = gorilla_decompress(compressed)
        np.testing.assert_array_equal(data, decoded)
        # Constant data should compress very well (close to 1 bit per value)
        assert len(compressed) < data.nbytes / 4

    def test_roundtrip_empty(self):
        """Empty array roundtrips."""
        data = np.array([], dtype=np.float32)
        compressed = gorilla_compress(data)
        decoded = gorilla_decompress(compressed)
        assert len(decoded) == 0

    def test_roundtrip_single_value(self):
        """Single value roundtrips."""
        data = np.array([3.14], dtype=np.float32)
        compressed = gorilla_compress(data)
        decoded = gorilla_decompress(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_compression_ratio(self):
        """Financial data compresses better than random data."""
        rng = np.random.RandomState(42)

        # Smooth financial-like data
        prices = 100.0 + np.cumsum(rng.randn(1000) * 0.1)
        financial = prices.astype(np.float32)
        fin_compressed = gorilla_compress(financial)
        fin_ratio = financial.nbytes / len(fin_compressed)

        # Pure random data
        random_data = rng.randn(1000).astype(np.float32)
        rand_compressed = gorilla_compress(random_data)
        rand_ratio = random_data.nbytes / len(rand_compressed)

        # Financial should compress better
        assert fin_ratio > rand_ratio

    def test_special_values(self):
        """Handles special float values (zero, very small, very large)."""
        data = np.array([0.0, 1e-38, 1e38, -1e38, 1e-7, -0.0], dtype=np.float32)
        compressed = gorilla_compress(data)
        decoded = gorilla_decompress(compressed)
        np.testing.assert_array_equal(data, decoded)


class TestChimpCompress:
    """Test Chimp (improved XOR) compression."""

    def test_roundtrip_random(self):
        """Random float32 data roundtrips losslessly."""
        rng = np.random.RandomState(42)
        data = rng.randn(500).astype(np.float32)
        compressed = chimp_compress(data)
        decoded = chimp_decompress(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_roundtrip_financial(self):
        """Financial data roundtrips losslessly."""
        rng = np.random.RandomState(42)
        prices = 100.0 + np.cumsum(rng.randn(500) * 0.5)
        data = prices.astype(np.float32)
        compressed = chimp_compress(data)
        decoded = chimp_decompress(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_roundtrip_empty(self):
        """Empty array roundtrips."""
        data = np.array([], dtype=np.float32)
        compressed = chimp_compress(data)
        decoded = chimp_decompress(compressed)
        assert len(decoded) == 0

    def test_roundtrip_constant(self):
        """Constant values roundtrip."""
        data = np.full(200, 99.5, dtype=np.float32)
        compressed = chimp_compress(data)
        decoded = chimp_decompress(compressed)
        np.testing.assert_array_equal(data, decoded)


class TestCompressArray:
    """Test the high-level compress_array / decompress_array API."""

    def test_gorilla_3d(self):
        """3D array roundtrips via gorilla."""
        rng = np.random.RandomState(42)
        data = rng.randn(10, 32, 5).astype(np.float32)
        compressed = compress_array(data, method="gorilla")
        decoded = decompress_array(compressed)
        np.testing.assert_array_equal(data, decoded)
        assert decoded.shape == data.shape

    def test_chimp_2d(self):
        """2D array roundtrips via chimp."""
        rng = np.random.RandomState(42)
        data = rng.randn(50, 10).astype(np.float32)
        compressed = compress_array(data, method="chimp")
        decoded = decompress_array(compressed)
        np.testing.assert_array_equal(data, decoded)
        assert decoded.shape == data.shape

    def test_auto_detect_format(self):
        """decompress_array auto-detects gorilla vs chimp format."""
        rng = np.random.RandomState(42)
        data = rng.randn(20, 8).astype(np.float32)

        gorilla_bytes = compress_array(data, method="gorilla")
        chimp_bytes = compress_array(data, method="chimp")

        # Both should decompress correctly
        np.testing.assert_array_equal(decompress_array(gorilla_bytes), data)
        np.testing.assert_array_equal(decompress_array(chimp_bytes), data)
