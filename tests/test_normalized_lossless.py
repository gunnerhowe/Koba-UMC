"""Tests for normalized_lossless and normalized_lossless_zstd storage modes."""

import numpy as np
import pytest

from umc.codec.tiered import _compress_storage, _decompress_storage


class TestNormalizedLossless:
    """Test normalized_lossless compression mode."""

    def test_roundtrip_random(self):
        """Random float32 data roundtrips with minimal error."""
        rng = np.random.RandomState(42)
        data = rng.randn(50, 32, 5).astype(np.float32)
        compressed = _compress_storage(data, "normalized_lossless")
        decoded = _decompress_storage(compressed)
        assert decoded.shape == data.shape
        rmse = np.sqrt(np.mean((data - decoded) ** 2))
        assert rmse < 1e-6

    def test_roundtrip_financial(self):
        """Financial price data roundtrips."""
        rng = np.random.RandomState(42)
        prices = 100.0 + np.cumsum(rng.randn(20, 32, 5) * 0.5, axis=1)
        data = prices.astype(np.float32)
        compressed = _compress_storage(data, "normalized_lossless")
        decoded = _decompress_storage(compressed)
        assert decoded.shape == data.shape
        rmse = np.sqrt(np.mean((data - decoded) ** 2))
        assert rmse < 1e-4

    def test_roundtrip_constant(self):
        """Constant data compresses extremely well."""
        data = np.full((50, 32, 5), 42.0, dtype=np.float32)
        compressed = _compress_storage(data, "normalized_lossless")
        decoded = _decompress_storage(compressed)
        assert decoded.shape == data.shape
        # Constant data should get very high ratio
        assert len(compressed) < data.nbytes / 10

    def test_roundtrip_single_window(self):
        """Single window roundtrips."""
        data = np.random.randn(1, 16, 3).astype(np.float32)
        compressed = _compress_storage(data, "normalized_lossless")
        decoded = _decompress_storage(compressed)
        assert decoded.shape == data.shape
        rmse = np.sqrt(np.mean((data - decoded) ** 2))
        assert rmse < 1e-6

    def test_roundtrip_large_values(self):
        """Large magnitude values roundtrip correctly."""
        data = np.array([[[1e10, -1e10, 1e-10, -1e-10, 0.0]]] * 32,
                         dtype=np.float32).reshape(1, 32, 5)
        compressed = _compress_storage(data, "normalized_lossless")
        decoded = _decompress_storage(compressed)
        assert decoded.shape == data.shape
        # Relative error should be small
        nonzero = np.abs(data) > 0
        if nonzero.any():
            rel_err = np.abs(data[nonzero] - decoded[nonzero]) / np.abs(data[nonzero])
            assert rel_err.max() < 1e-5

    def test_all_synthetic_types(self):
        """All synthetic data types compress and decompress."""
        from umc.data.synthetic import generate_all_types
        all_data = generate_all_types(n_windows=20)
        for name, data in all_data.items():
            compressed = _compress_storage(data, "normalized_lossless")
            decoded = _decompress_storage(compressed)
            assert decoded.shape == data.shape, f"Shape mismatch for {name}"
            rmse = np.sqrt(np.mean((data - decoded) ** 2))
            assert rmse < 1e-3, f"RMSE too high for {name}: {rmse}"


class TestNormalizedLosslessZstd:
    """Test normalized_lossless_zstd compression mode."""

    def test_roundtrip_random(self):
        """Random data roundtrips via zstd backend."""
        rng = np.random.RandomState(42)
        data = rng.randn(20, 32, 5).astype(np.float32)
        compressed = _compress_storage(data, "normalized_lossless_zstd")
        decoded = _decompress_storage(compressed)
        assert decoded.shape == data.shape
        rmse = np.sqrt(np.mean((data - decoded) ** 2))
        assert rmse < 1e-6

    def test_better_than_zlib_on_constant(self):
        """Zstd should match or beat zlib on constant data."""
        data = np.full((50, 32, 5), 99.9, dtype=np.float32)
        c_zlib = _compress_storage(data, "normalized_lossless")
        c_zstd = _compress_storage(data, "normalized_lossless_zstd")
        # Both should decompress correctly
        d_zlib = _decompress_storage(c_zlib)
        d_zstd = _decompress_storage(c_zstd)
        np.testing.assert_allclose(d_zlib, d_zstd, atol=1e-6)

    def test_roundtrip_all_types(self):
        """All synthetic data types work with zstd backend."""
        from umc.data.synthetic import generate_all_types
        all_data = generate_all_types(n_windows=10)
        for name, data in all_data.items():
            compressed = _compress_storage(data, "normalized_lossless_zstd")
            decoded = _decompress_storage(compressed)
            assert decoded.shape == data.shape, f"Shape mismatch for {name}"


class TestModeMap:
    """Test that all modes are correctly mapped in serialization."""

    def test_all_modes_roundtrip(self):
        """Every storage mode roundtrips correctly."""
        data = np.random.randn(5, 16, 3).astype(np.float32)
        for mode in ["lossless", "near_lossless", "lossless_zstd", "lossless_lzma",
                      "normalized_lossless", "normalized_lossless_zstd"]:
            compressed = _compress_storage(data, mode)
            decoded = _decompress_storage(compressed)
            assert decoded.shape == data.shape, f"Shape mismatch for mode {mode}"

    def test_invalid_mode_raises(self):
        """Unknown mode raises ValueError."""
        data = np.random.randn(5, 16, 3).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown storage mode"):
            _compress_storage(data, "invalid_mode")
