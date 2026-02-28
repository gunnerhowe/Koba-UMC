"""Tests for UMC decoder modules."""

import pytest
import torch

from umc.config import UMCConfig
from umc.decoder.mlp_decoder import MLPDecoder
from umc.decoder.conv_decoder import ConvDecoder


@pytest.fixture
def config():
    return UMCConfig(
        window_size=32,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=64,
        decoder_hidden=(64, 128),
        num_charts=8,
        chart_embedding_dim=8,
    )


class TestMLPDecoder:
    def test_output_shape(self, config):
        """Decoder reconstructs correct shape."""
        decoder = MLPDecoder(config)
        z = torch.randn(16, config.max_latent_dim)
        chart_id = torch.randint(0, config.num_charts, (16,))

        x_hat = decoder.decode(z, chart_id)
        assert x_hat.shape == (16, config.window_size, config.n_features)

    def test_deterministic(self, config):
        """Decoder is always deterministic."""
        decoder = MLPDecoder(config)
        z = torch.randn(16, config.max_latent_dim)
        chart_id = torch.randint(0, config.num_charts, (16,))

        x1 = decoder.decode(z, chart_id)
        x2 = decoder.decode(z, chart_id)
        assert torch.allclose(x1, x2)

    def test_different_charts_different_output(self, config):
        """Different chart IDs produce different outputs."""
        decoder = MLPDecoder(config)
        z = torch.randn(1, config.max_latent_dim)

        x0 = decoder.decode(z, torch.tensor([0]))
        x1 = decoder.decode(z, torch.tensor([1]))

        # They should differ (unless extremely unlikely initialization)
        assert not torch.allclose(x0, x1, atol=1e-6)

    def test_gradient_flow(self, config):
        """Gradients flow through the decoder."""
        decoder = MLPDecoder(config)
        z = torch.randn(16, config.max_latent_dim, requires_grad=True)
        chart_id = torch.randint(0, config.num_charts, (16,))

        x_hat = decoder.decode(z, chart_id)
        loss = x_hat.sum()
        loss.backward()

        assert z.grad is not None
        assert z.grad.abs().sum() > 0

    def test_single_sample(self, config):
        """Works with batch size 1."""
        decoder = MLPDecoder(config)
        z = torch.randn(1, config.max_latent_dim)
        chart_id = torch.tensor([0])

        x_hat = decoder.decode(z, chart_id)
        assert x_hat.shape == (1, config.window_size, config.n_features)


class TestConvDecoder:
    def test_output_shape(self, config):
        """Conv decoder reconstructs correct shape."""
        decoder = ConvDecoder(config)
        z = torch.randn(16, config.max_latent_dim)
        chart_id = torch.randint(0, config.num_charts, (16,))

        x_hat = decoder.decode(z, chart_id)
        assert x_hat.shape == (16, config.window_size, config.n_features)

    def test_deterministic(self, config):
        """Conv decoder is always deterministic."""
        decoder = ConvDecoder(config)
        decoder.eval()
        z = torch.randn(16, config.max_latent_dim)
        chart_id = torch.randint(0, config.num_charts, (16,))

        x1 = decoder.decode(z, chart_id)
        x2 = decoder.decode(z, chart_id)
        assert torch.allclose(x1, x2)

    def test_different_charts_different_output(self, config):
        """Different chart IDs produce different outputs."""
        decoder = ConvDecoder(config)
        z = torch.randn(1, config.max_latent_dim)

        x0 = decoder.decode(z, torch.tensor([0]))
        x1 = decoder.decode(z, torch.tensor([1]))

        assert not torch.allclose(x0, x1, atol=1e-6)

    def test_gradient_flow(self, config):
        """Gradients flow through the conv decoder."""
        decoder = ConvDecoder(config)
        z = torch.randn(16, config.max_latent_dim, requires_grad=True)
        chart_id = torch.randint(0, config.num_charts, (16,))

        x_hat = decoder.decode(z, chart_id)
        loss = x_hat.sum()
        loss.backward()

        assert z.grad is not None
        assert z.grad.abs().sum() > 0

    def test_single_sample(self, config):
        """Works with batch size 1."""
        decoder = ConvDecoder(config)
        z = torch.randn(1, config.max_latent_dim)
        chart_id = torch.tensor([0])

        x_hat = decoder.decode(z, chart_id)
        assert x_hat.shape == (1, config.window_size, config.n_features)

    def test_roundtrip_with_conv_encoder(self, config):
        """Conv encoder + decoder roundtrip produces correct shapes."""
        from umc.encoder.conv_encoder import ConvEncoder
        encoder = ConvEncoder(config)
        decoder = ConvDecoder(config)

        x = torch.randn(8, config.window_size, config.n_features)
        enc_result = encoder.encode(x)
        x_hat = decoder.decode(enc_result.z, enc_result.chart_id)

        assert x_hat.shape == x.shape
