"""Tests for UniversalPreprocessor."""

import numpy as np
import pytest

from umc.data.preprocessors import UniversalPreprocessor


class TestUniversalPreprocessor:
    """Test the domain-agnostic preprocessor."""

    def test_fit_transform_roundtrip(self):
        """fit -> transform -> inverse_transform preserves data."""
        rng = np.random.RandomState(42)
        data = rng.randn(200, 5).astype(np.float32) * 100 + 50
        prep = UniversalPreprocessor()
        normed = prep.fit_transform(data)
        recovered = prep.inverse_transform(normed)
        np.testing.assert_allclose(data, recovered, atol=1e-3)

    def test_normalized_stats(self):
        """Transformed data has approximately zero mean and unit std."""
        rng = np.random.RandomState(42)
        data = rng.randn(1000, 3).astype(np.float32) * 10 + 5
        prep = UniversalPreprocessor(clip_range=0)  # no clipping
        normed = prep.fit_transform(data)
        np.testing.assert_allclose(normed.mean(axis=0), 0, atol=0.05)
        np.testing.assert_allclose(normed.std(axis=0), 1, atol=0.05)

    def test_3d_input(self):
        """3D array input works."""
        rng = np.random.RandomState(42)
        data = rng.randn(50, 32, 5).astype(np.float32)
        prep = UniversalPreprocessor()
        normed = prep.fit_transform(data)
        recovered = prep.inverse_transform(normed)
        assert normed.shape == data.shape
        np.testing.assert_allclose(data, recovered, atol=1e-3)

    def test_1d_input(self):
        """1D array input works."""
        data = np.arange(100).astype(np.float32)
        prep = UniversalPreprocessor()
        normed = prep.fit_transform(data)
        recovered = prep.inverse_transform(normed)
        np.testing.assert_allclose(data, recovered, atol=1e-2)

    def test_clipping(self):
        """Values are clipped to clip_range."""
        data = np.array([[0, 0, 0, 0, 100]], dtype=np.float32)
        prep = UniversalPreprocessor(clip_range=5.0)
        # Fit on broader data so the outlier stands out
        fit_data = np.random.randn(1000, 1).astype(np.float32)
        prep.fit(fit_data)
        normed = prep.transform(data.reshape(-1, 1))
        assert normed.max() <= 5.0
        assert normed.min() >= -5.0

    def test_constant_feature(self):
        """Constant features don't cause division by zero."""
        data = np.ones((100, 3), dtype=np.float32)
        data[:, 0] = 5.0  # constant
        data[:, 1] = np.arange(100)  # varying
        data[:, 2] = 0.0  # zero
        prep = UniversalPreprocessor()
        normed = prep.fit_transform(data)
        assert np.isfinite(normed).all()

    def test_state_dict_roundtrip(self):
        """state_dict / load_state_dict preserves parameters."""
        rng = np.random.RandomState(42)
        data = rng.randn(100, 4).astype(np.float32) * 50
        prep = UniversalPreprocessor()
        prep.fit(data)
        normed1 = prep.transform(data)

        state = prep.state_dict()
        prep2 = UniversalPreprocessor()
        prep2.load_state_dict(state)
        normed2 = prep2.transform(data)

        np.testing.assert_array_equal(normed1, normed2)

    def test_transform_without_fit_raises(self):
        """Calling transform before fit raises RuntimeError."""
        prep = UniversalPreprocessor()
        with pytest.raises(RuntimeError, match="fit"):
            prep.transform(np.ones((10, 3)))

    def test_inverse_without_fit_raises(self):
        """Calling inverse_transform before fit raises RuntimeError."""
        prep = UniversalPreprocessor()
        with pytest.raises(RuntimeError, match="fit"):
            prep.inverse_transform(np.ones((10, 3)))
