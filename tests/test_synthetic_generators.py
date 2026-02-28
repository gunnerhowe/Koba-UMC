"""Tests for multi-domain synthetic data generators."""

import numpy as np
import pytest

from umc.data.synthetic import (
    generate_financial,
    generate_sine_waves,
    generate_iot_sensors,
    generate_weather,
    generate_ecg,
    generate_audio_spectrogram,
    generate_all_types,
)


class TestSyntheticGenerators:
    """Test that each generator produces valid data."""

    def test_financial_shape(self):
        data = generate_financial(n_windows=50, window_size=32, n_features=5)
        assert data.shape == (50, 32, 5)
        assert data.dtype == np.float32

    def test_financial_positive_prices(self):
        data = generate_financial(n_windows=100)
        # OHLC (features 0-3) should be positive
        assert (data[:, :, :4] > 0).all()

    def test_sine_waves_shape(self):
        data = generate_sine_waves(n_windows=50, window_size=32, n_features=4)
        assert data.shape == (50, 32, 4)
        assert data.dtype == np.float32

    def test_iot_sensors_shape(self):
        data = generate_iot_sensors(n_windows=50, window_size=32, n_features=3)
        assert data.shape == (50, 32, 3)
        assert data.dtype == np.float32

    def test_weather_shape(self):
        data = generate_weather(n_windows=50, window_size=32, n_features=4)
        assert data.shape == (50, 32, 4)
        assert data.dtype == np.float32

    def test_ecg_shape(self):
        data = generate_ecg(n_windows=50, window_size=64, n_features=1)
        assert data.shape == (50, 64, 1)
        assert data.dtype == np.float32

    def test_audio_spectrogram_shape(self):
        data = generate_audio_spectrogram(n_windows=50, window_size=32, n_features=16)
        assert data.shape == (50, 32, 16)
        assert data.dtype == np.float32

    def test_audio_spectrogram_non_negative(self):
        """Audio spectrogram values should be non-negative (energy)."""
        data = generate_audio_spectrogram(n_windows=100)
        assert (data >= 0).all()

    def test_reproducible(self):
        """Same seed produces same data."""
        d1 = generate_financial(n_windows=10, seed=99)
        d2 = generate_financial(n_windows=10, seed=99)
        np.testing.assert_array_equal(d1, d2)

    def test_different_seeds(self):
        """Different seeds produce different data."""
        d1 = generate_financial(n_windows=10, seed=1)
        d2 = generate_financial(n_windows=10, seed=2)
        assert not np.array_equal(d1, d2)

    def test_generate_all_types(self):
        """generate_all_types returns all 6 data types."""
        all_data = generate_all_types(n_windows=20)
        assert len(all_data) == 6
        expected = {"financial", "sine_waves", "iot_sensors", "weather", "ecg", "audio_spectrogram"}
        assert set(all_data.keys()) == expected
        for name, data in all_data.items():
            assert data.ndim == 3, f"{name} should be 3D"
            assert data.shape[0] == 20, f"{name} should have 20 windows"
            assert data.dtype == np.float32, f"{name} should be float32"

    def test_no_nans(self):
        """No generator should produce NaN values."""
        all_data = generate_all_types(n_windows=100)
        for name, data in all_data.items():
            assert not np.isnan(data).any(), f"{name} contains NaN"

    def test_no_infs(self):
        """No generator should produce infinite values."""
        all_data = generate_all_types(n_windows=100)
        for name, data in all_data.items():
            assert not np.isinf(data).any(), f"{name} contains Inf"


class TestCompressibility:
    """Test that synthetic data actually compresses with UMC storage tier."""

    def test_all_types_compress(self):
        """All data types achieve > 1x compression with lossless storage."""
        from umc.codec.tiered import _compress_storage

        all_data = generate_all_types(n_windows=200)
        for name, data in all_data.items():
            compressed = _compress_storage(data, "lossless")
            ratio = data.nbytes / len(compressed)
            assert ratio > 1.0, f"{name} did not compress (ratio={ratio:.2f})"
