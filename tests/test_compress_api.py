"""Tests for the top-level umc.compress() / umc.decompress() API."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import umc


class TestCompress:
    """Test umc.compress() function."""

    def test_3d_lossless(self):
        """3D array lossless roundtrip."""
        data = np.random.randn(10, 32, 5).astype(np.float32)
        compressed = umc.compress(data, mode="lossless")
        decoded = umc.decompress(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_2d_lossless(self):
        """2D array lossless roundtrip."""
        data = np.random.randn(100, 8).astype(np.float32)
        compressed = umc.compress(data, mode="lossless")
        decoded = umc.decompress(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_1d_lossless(self):
        """1D array lossless roundtrip."""
        data = np.random.randn(500).astype(np.float32)
        compressed = umc.compress(data, mode="lossless")
        decoded = umc.decompress(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_near_lossless(self):
        """Near-lossless mode has small error."""
        data = np.random.randn(10, 32, 5).astype(np.float32)
        compressed = umc.compress(data, mode="near_lossless")
        decoded = umc.decompress(compressed)
        assert decoded.shape == data.shape
        rmse = np.sqrt(np.mean((data - decoded) ** 2))
        assert rmse < 0.01

    def test_normalized_lossless(self):
        """Normalized lossless mode roundtrips with minimal error."""
        data = np.random.randn(10, 32, 5).astype(np.float32)
        compressed = umc.compress(data, mode="normalized_lossless")
        decoded = umc.decompress(compressed)
        assert decoded.shape == data.shape
        rmse = np.sqrt(np.mean((data - decoded) ** 2))
        assert rmse < 1e-6

    def test_auto_cast_float64(self):
        """Float64 input is auto-cast to float32."""
        data = np.random.randn(5, 16, 3)  # float64
        compressed = umc.compress(data, mode="lossless")
        decoded = umc.decompress(compressed)
        assert decoded.dtype == np.float32

    def test_auto_cast_int(self):
        """Integer input is auto-cast to float32."""
        data = np.arange(100).reshape(1, 100, 1)
        compressed = umc.compress(data, mode="lossless")
        decoded = umc.decompress(compressed)
        np.testing.assert_array_equal(data.astype(np.float32), decoded)

    def test_list_input(self):
        """List input works."""
        data = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
        compressed = umc.compress(data, mode="lossless")
        decoded = umc.decompress(compressed)
        np.testing.assert_array_equal(np.array(data, dtype=np.float32), decoded)

    def test_nan_handling(self):
        """NaN values are replaced with 0."""
        data = np.array([[[1.0, float("nan"), 3.0]]], dtype=np.float32)
        with pytest.warns(match="NaN"):
            compressed = umc.compress(data, mode="lossless")
        decoded = umc.decompress(compressed)
        assert decoded[0, 0, 1] == 0.0

    def test_inf_handling(self):
        """Inf values are clipped."""
        data = np.array([[[1.0, float("inf"), -float("inf")]]], dtype=np.float32)
        with pytest.warns(match="Inf"):
            compressed = umc.compress(data, mode="lossless")
        decoded = umc.decompress(compressed)
        assert np.isfinite(decoded).all()

    def test_empty_raises(self):
        """Empty array raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            umc.compress(np.array([]))

    def test_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        data = np.random.randn(1, 10, 3).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown mode"):
            umc.compress(data, mode="invalid")

    def test_invalid_decompress_magic(self):
        """Invalid magic bytes raise ValueError."""
        with pytest.raises(ValueError, match="UMCZ"):
            umc.decompress(b"not_valid_data")

    def test_all_modes(self):
        """All storage modes work via compress/decompress."""
        data = np.random.randn(5, 16, 3).astype(np.float32)
        for mode in umc._ALL_STORAGE_MODES:
            compressed = umc.compress(data, mode=mode)
            decoded = umc.decompress(compressed)
            assert decoded.shape == data.shape, f"Failed for mode={mode}"


class TestLosslessFast:
    """Test lossless_fast mode (flatten-delta + zstd-3)."""

    def test_roundtrip_3d(self):
        data = np.random.randn(10, 32, 5).astype(np.float32)
        compressed = umc.compress(data, mode="lossless_fast")
        decoded = umc.decompress(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_roundtrip_1d(self):
        data = np.random.randn(500).astype(np.float32)
        compressed = umc.compress(data, mode="lossless_fast")
        decoded = umc.decompress(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_roundtrip_constant(self):
        data = np.full((5, 16, 3), 42.0, dtype=np.float32)
        compressed = umc.compress(data, mode="lossless_fast")
        decoded = umc.decompress(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_roundtrip_negative(self):
        rng = np.random.RandomState(99)
        data = (rng.randn(10, 32, 5) * 100).astype(np.float32)
        compressed = umc.compress(data, mode="lossless_fast")
        decoded = umc.decompress(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_compression_ratio_structured(self):
        """Structured data should compress beyond 1.0x."""
        rng = np.random.RandomState(42)
        base = 100.0 + rng.randn(50, 1, 5).cumsum(axis=0) * 0.5
        noise = rng.randn(50, 32, 5) * 0.01
        data = (base + noise).astype(np.float32)
        compressed = umc.compress(data, mode="lossless_fast")
        ratio = data.nbytes / len(compressed)
        assert ratio > 1.2, f"Expected ratio > 1.2 for structured data, got {ratio:.2f}"


class TestCompressFile:
    """Test umc.compress_file() / umc.decompress_file()."""

    def test_npy_roundtrip(self):
        """NPY file roundtrips."""
        data = np.random.randn(10, 32, 5).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            npy_path = Path(tmp) / "test.npy"
            umc_path = Path(tmp) / "test.umc"
            out_path = Path(tmp) / "recovered.npy"

            np.save(str(npy_path), data)
            stats = umc.compress_file(str(npy_path), str(umc_path))
            assert stats["ratio"] > 0.5

            umc.decompress_file(str(umc_path), str(out_path))
            recovered = np.load(str(out_path))
            np.testing.assert_array_equal(data, recovered)

    def test_csv_roundtrip(self):
        """CSV file compresses and decompresses."""
        import pandas as pd
        data = np.random.randn(100, 5).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "test.csv"
            umc_path = Path(tmp) / "test.umc"
            out_path = Path(tmp) / "recovered.csv"

            pd.DataFrame(data, columns=[f"f{i}" for i in range(5)]).to_csv(
                str(csv_path), index=False
            )
            stats = umc.compress_file(str(csv_path), str(umc_path))
            assert stats["compressed_bytes"] > 0

            umc.decompress_file(str(umc_path), str(out_path))
            recovered = pd.read_csv(str(out_path)).values.astype(np.float32)
            np.testing.assert_allclose(data, recovered, atol=1e-5)
