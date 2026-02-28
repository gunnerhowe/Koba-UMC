"""Performance benchmark tests."""

import numpy as np
import pytest

from umc.evaluation.benchmarks import benchmark_gzip, benchmark_numpy_raw, run_all_baselines
from umc.evaluation.metrics import (
    reconstruction_rmse,
    compression_ratio,
    effective_dimensionality,
)


@pytest.fixture
def sample_windows():
    """Sample windowed data for benchmarks."""
    rng = np.random.RandomState(42)
    return rng.randn(1000, 64, 5).astype(np.float32)


class TestCompressionBaselines:
    def test_gzip_produces_smaller(self, sample_windows):
        """gzip should compress random data somewhat."""
        result = benchmark_gzip(sample_windows)
        assert result["ratio"] > 1.0  # Should compress at least a little
        assert result["compress_time_sec"] > 0

    def test_numpy_raw_baseline(self, sample_windows):
        """Numpy binary baseline works."""
        result = benchmark_numpy_raw(sample_windows)
        assert result["binary_bytes"] > 0
        assert result["csv_bytes"] > result["binary_bytes"]

    def test_run_all_baselines(self, sample_windows):
        """All baselines run without error."""
        results = run_all_baselines(sample_windows)
        assert len(results) >= 2  # At least gzip + numpy
        for r in results:
            assert "method" in r


class TestMetrics:
    def test_rmse_zero_for_identical(self):
        """RMSE is zero when original equals reconstructed."""
        data = np.random.randn(100, 64, 5).astype(np.float32)
        assert reconstruction_rmse(data, data) == pytest.approx(0.0)

    def test_rmse_positive_for_different(self):
        """RMSE is positive for different data."""
        rng = np.random.RandomState(42)
        data = rng.randn(100, 64, 5).astype(np.float32)
        noise = rng.randn(100, 64, 5).astype(np.float32)
        assert reconstruction_rmse(data, data + noise) > 0

    def test_compression_ratio_calculation(self):
        """Compression ratio math is correct."""
        result = compression_ratio(1000000, 10000, decoder_bytes=50000)
        assert result["coordinate_ratio"] == pytest.approx(100.0)
        assert result["system_ratio"] == pytest.approx(1000000 / 60000)

    def test_effective_dimensionality(self):
        """Effective dim counts active dimensions correctly."""
        rng = np.random.RandomState(42)
        # 5 active dims (high variance) + 5 collapsed dims (near zero)
        latents = np.zeros((1000, 10), dtype=np.float32)
        latents[:, :5] = rng.randn(1000, 5)
        latents[:, 5:] = rng.randn(1000, 5) * 0.001  # Below threshold

        result = effective_dimensionality(latents, threshold=0.01)
        assert result["active_dims"] == 5
        assert result["total_dims"] == 10
        assert result["participation_ratio"] > 0
