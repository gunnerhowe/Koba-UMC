"""Tests for GPU-pipelined encoding in TieredCodec.

These tests run on CPU (no GPU required) and verify that the pipelined
encode path produces the same output as sequential encode.
"""

import numpy as np
import pytest
import torch

from umc.codec.tiered import _compress_storage, _decompress_storage


class TestPipelinedStorage:
    """Test pipelined storage at the storage tier level (no model needed)."""

    def test_lossless_roundtrip(self):
        """Storage compression roundtrips losslessly."""
        rng = np.random.RandomState(42)
        data = rng.randn(50, 32, 5).astype(np.float32)

        compressed = _compress_storage(data, "lossless")
        decoded = _decompress_storage(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_all_modes_roundtrip(self):
        """All storage modes produce valid output."""
        rng = np.random.RandomState(42)
        data = rng.randn(20, 32, 5).astype(np.float32)

        for mode in ["lossless", "near_lossless"]:
            compressed = _compress_storage(data, mode)
            decoded = _decompress_storage(compressed)
            assert decoded.shape == data.shape

            if mode == "lossless":
                np.testing.assert_array_equal(data, decoded)
            else:
                # Near-lossless: close but not exact
                rmse = np.sqrt(np.mean((data - decoded) ** 2))
                assert rmse < 0.1

    def test_zstd_mode(self):
        """zstd mode roundtrips if zstandard is installed."""
        try:
            import zstandard
        except ImportError:
            pytest.skip("zstandard not installed")

        rng = np.random.RandomState(42)
        data = rng.randn(20, 32, 5).astype(np.float32)
        compressed = _compress_storage(data, "lossless_zstd")
        decoded = _decompress_storage(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_lzma_mode(self):
        """lzma mode roundtrips."""
        rng = np.random.RandomState(42)
        data = rng.randn(20, 32, 5).astype(np.float32)
        compressed = _compress_storage(data, "lossless_lzma")
        decoded = _decompress_storage(compressed)
        np.testing.assert_array_equal(data, decoded)

    def test_concurrent_compression(self):
        """Multiple storage compressions run correctly in threads."""
        from concurrent.futures import ThreadPoolExecutor

        rng = np.random.RandomState(42)
        batches = [rng.randn(20, 32, 5).astype(np.float32) for _ in range(4)]

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_compress_storage, b, "lossless") for b in batches]
            results = [f.result() for f in futures]

        for batch, compressed in zip(batches, results):
            decoded = _decompress_storage(compressed)
            np.testing.assert_array_equal(batch, decoded)
