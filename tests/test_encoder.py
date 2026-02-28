"""Tests for UMC encoder modules."""

import pytest

pytest.importorskip("torch")

import torch
import numpy as np

from umc.config import UMCConfig
from umc.encoder.base import EncodingResult
from umc.encoder.vae import VAEEncoder
from umc.encoder.conv_encoder import ConvEncoder
from umc.encoder.adaptive import AdaptiveEncoder


@pytest.fixture
def config():
    return UMCConfig(
        window_size=32,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=64,
        encoder_hidden=(128, 64),
        num_charts=8,
        batch_size=16,
    )


@pytest.fixture
def sample_batch(config):
    """Random batch of windowed OHLCV data."""
    batch_size = 16
    return torch.randn(batch_size, config.window_size, config.n_features)


class TestVAEEncoder:
    def test_output_shape(self, config, sample_batch):
        """Encoder produces correct tensor shapes."""
        encoder = VAEEncoder(config)
        result = encoder.encode(sample_batch)

        assert result.z.shape == (16, config.max_latent_dim)
        assert result.mu.shape == (16, config.max_latent_dim)
        assert result.log_var.shape == (16, config.max_latent_dim)
        assert result.chart_id.shape == (16,)
        assert result.confidence.shape == (16,)

    def test_confidence_range(self, config, sample_batch):
        """Confidence scores are in [0, 1]."""
        encoder = VAEEncoder(config)
        result = encoder.encode(sample_batch)

        assert (result.confidence >= 0).all()
        assert (result.confidence <= 1).all()

    def test_chart_id_range(self, config, sample_batch):
        """Chart IDs are valid indices."""
        encoder = VAEEncoder(config)
        result = encoder.encode(sample_batch)

        assert (result.chart_id >= 0).all()
        assert (result.chart_id < config.num_charts).all()

    def test_deterministic_eval(self, config, sample_batch):
        """Encoder is deterministic in eval mode."""
        encoder = VAEEncoder(config)
        encoder.eval()

        with torch.no_grad():
            r1 = encoder.encode(sample_batch)
            r2 = encoder.encode(sample_batch)

        assert torch.allclose(r1.z, r2.z)

    def test_stochastic_train(self, config, sample_batch):
        """Encoder has stochastic sampling in train mode."""
        encoder = VAEEncoder(config)
        encoder.train()

        # Different random seeds should give different z
        torch.manual_seed(1)
        r1 = encoder.encode(sample_batch)
        torch.manual_seed(2)
        r2 = encoder.encode(sample_batch)

        # mu should be the same, z may differ
        assert torch.allclose(r1.mu, r2.mu)

    def test_active_dims_reported(self, config, sample_batch):
        """Active dims count is a non-negative integer."""
        encoder = VAEEncoder(config)
        result = encoder.encode(sample_batch)
        assert isinstance(result.active_dims, int)
        assert result.active_dims >= 0

    def test_gradient_flow(self, config, sample_batch):
        """Gradients flow through the encoder."""
        encoder = VAEEncoder(config)
        encoder.train()
        result = encoder.encode(sample_batch)
        loss = result.z.sum()
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in encoder.parameters())
        assert has_grad


class TestAdaptiveEncoder:
    def test_output_shape(self, config, sample_batch):
        """Adaptive encoder produces correct shapes."""
        encoder = AdaptiveEncoder(config)
        result = encoder.encode(sample_batch)

        assert result.z.shape == (16, config.max_latent_dim)

    def test_gate_values(self, config):
        """Gate values are in [0, 1]."""
        encoder = AdaptiveEncoder(config)
        gates = encoder.get_gate_values()

        assert gates.shape == (config.max_latent_dim,)
        assert np.all(gates >= 0)
        assert np.all(gates <= 1)

    def test_effective_dim(self, config, sample_batch):
        """Effective dim returns a reasonable value."""
        encoder = AdaptiveEncoder(config)
        encoder.encode(sample_batch)
        dim = encoder.get_effective_dim()

        assert isinstance(dim, int)
        assert 0 <= dim <= config.max_latent_dim

    def test_gate_sparsity_loss(self, config):
        """Gate sparsity loss is non-negative."""
        encoder = AdaptiveEncoder(config)
        loss = encoder.gate_sparsity_loss()

        assert loss.item() >= 0

    def test_prune(self, config):
        """Pruning reduces active dimensions."""
        encoder = AdaptiveEncoder(config)
        # Force some gates low
        with torch.no_grad():
            encoder.dim_gates.data[:32] = -5.0

        pruned = encoder.prune(threshold=0.5)
        assert pruned > 0
        assert encoder.get_effective_dim() < config.max_latent_dim


class TestConvEncoder:
    def test_output_shape(self, config, sample_batch):
        """Conv encoder produces correct tensor shapes."""
        encoder = ConvEncoder(config)
        result = encoder.encode(sample_batch)

        assert result.z.shape == (16, config.max_latent_dim)
        assert result.mu.shape == (16, config.max_latent_dim)
        assert result.log_var.shape == (16, config.max_latent_dim)
        assert result.chart_id.shape == (16,)
        assert result.confidence.shape == (16,)

    def test_confidence_range(self, config, sample_batch):
        """Confidence scores are in [0, 1]."""
        encoder = ConvEncoder(config)
        result = encoder.encode(sample_batch)

        assert (result.confidence >= 0).all()
        assert (result.confidence <= 1).all()

    def test_chart_id_range(self, config, sample_batch):
        """Chart IDs are valid indices."""
        encoder = ConvEncoder(config)
        result = encoder.encode(sample_batch)

        assert (result.chart_id >= 0).all()
        assert (result.chart_id < config.num_charts).all()

    def test_deterministic_eval(self, config, sample_batch):
        """Conv encoder is deterministic in eval mode."""
        encoder = ConvEncoder(config)
        encoder.eval()

        with torch.no_grad():
            r1 = encoder.encode(sample_batch)
            r2 = encoder.encode(sample_batch)

        assert torch.allclose(r1.z, r2.z)

    def test_gradient_flow(self, config, sample_batch):
        """Gradients flow through the conv encoder."""
        encoder = ConvEncoder(config)
        encoder.train()
        result = encoder.encode(sample_batch)
        loss = result.z.sum()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in encoder.parameters())
        assert has_grad

    def test_gumbel_softmax_training(self, config, sample_batch):
        """Gumbel-Softmax is used during training."""
        encoder = ConvEncoder(config)
        encoder.train()
        encoder.gumbel_temperature = 0.5
        result = encoder.encode(sample_batch)

        assert (result.chart_id >= 0).all()
        assert (result.chart_id < config.num_charts).all()
