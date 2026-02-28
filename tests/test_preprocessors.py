"""Tests for per-window normalization and updated WindowDataset."""

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from umc.data.preprocessors import WindowNormalizer, WindowDataset


class TestWindowNormalizer:
    def test_normalize_denormalize_roundtrip(self):
        """Normalize then denormalize recovers original data."""
        windows = np.random.randn(50, 64, 5).astype(np.float32)
        normalizer = WindowNormalizer()
        normalized, means, stds = normalizer.normalize(windows)
        recovered = normalizer.denormalize(normalized, means, stds)
        assert np.allclose(windows, recovered, atol=1e-4)

    def test_normalized_stats(self):
        """Normalized windows have ~zero mean, ~unit std per feature."""
        windows = np.random.randn(50, 64, 5).astype(np.float32) * 10 + 5
        normalizer = WindowNormalizer()
        normalized, _, _ = normalizer.normalize(windows)
        per_window_mean = normalized.mean(axis=1)
        per_window_std = normalized.std(axis=1)
        assert np.allclose(per_window_mean, 0, atol=1e-5)
        assert np.allclose(per_window_std, 1, atol=0.15)

    def test_flat_window_handling(self):
        """Windows with zero variance do not produce NaN/Inf."""
        windows = np.ones((5, 64, 5), dtype=np.float32)
        normalizer = WindowNormalizer()
        normalized, means, stds = normalizer.normalize(windows)
        assert np.all(np.isfinite(normalized))

    def test_output_shapes(self):
        """Output shapes match input dimensions."""
        n, ws, nf = 30, 128, 5
        windows = np.random.randn(n, ws, nf).astype(np.float32)
        normalizer = WindowNormalizer()
        normalized, means, stds = normalizer.normalize(windows)
        assert normalized.shape == (n, ws, nf)
        assert means.shape == (n, nf)
        assert stds.shape == (n, nf)

    def test_output_dtype(self):
        """Outputs are float32."""
        windows = np.random.randn(10, 32, 5).astype(np.float64)
        normalizer = WindowNormalizer()
        normalized, means, stds = normalizer.normalize(windows)
        assert normalized.dtype == np.float32
        assert means.dtype == np.float32
        assert stds.dtype == np.float32

    def test_clipping(self):
        """Extreme values are clipped to [-10, 10]."""
        windows = np.zeros((2, 32, 5), dtype=np.float32)
        # Make one window have a huge outlier
        windows[0, 0, 0] = 1000.0
        normalizer = WindowNormalizer()
        normalized, _, _ = normalizer.normalize(windows)
        assert normalized.max() <= 10.0
        assert normalized.min() >= -10.0


class TestWindowDatasetWithScaleFactors:
    def test_returns_tuple_when_scales_present(self):
        """WindowDataset returns (x, mean, std) tuple."""
        windows = np.random.randn(20, 32, 5).astype(np.float32)
        means = np.random.randn(20, 5).astype(np.float32)
        stds = np.abs(np.random.randn(20, 5)).astype(np.float32) + 0.1
        ds = WindowDataset(windows, means=means, stds=stds)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 3
        assert item[0].shape == (32, 5)
        assert item[1].shape == (5,)
        assert item[2].shape == (5,)

    def test_backward_compatible_without_scales(self):
        """WindowDataset returns tensor when no scales provided."""
        windows = np.random.randn(20, 32, 5).astype(np.float32)
        ds = WindowDataset(windows)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (32, 5)

    def test_length(self):
        """Dataset length matches number of windows."""
        windows = np.random.randn(42, 32, 5).astype(np.float32)
        ds = WindowDataset(windows)
        assert len(ds) == 42

    def test_dataloader_compatibility(self):
        """WindowDataset with scales works with DataLoader."""
        from torch.utils.data import DataLoader

        windows = np.random.randn(16, 32, 5).astype(np.float32)
        means = np.random.randn(16, 5).astype(np.float32)
        stds = np.abs(np.random.randn(16, 5)).astype(np.float32) + 0.1
        ds = WindowDataset(windows, means=means, stds=stds)
        loader = DataLoader(ds, batch_size=8)
        batch = next(iter(loader))
        assert isinstance(batch, (list, tuple))
        x, m, s = batch
        assert x.shape == (8, 32, 5)
        assert m.shape == (8, 5)
        assert s.shape == (8, 5)
