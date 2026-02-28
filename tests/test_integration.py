"""Integration tests: end-to-end encode -> store -> search -> decode pipeline."""

import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("vector_quantize_pytorch")

import torch
from torch.utils.data import DataLoader

from umc.config import UMCConfig
from umc.encoder.vae import VAEEncoder
from umc.encoder.conv_encoder import ConvEncoder
from umc.decoder.mlp_decoder import MLPDecoder
from umc.decoder.conv_decoder import ConvDecoder
from umc.training.trainer import UMCTrainer
from umc.training.losses import (
    total_loss, kl_divergence, reconstruction_loss,
    multiscale_reconstruction_loss, spectral_loss,
)
from umc.storage.mnf_format import MNFWriter, MNFReader
from umc.storage.manifest import DecoderManifest
from umc.processor.cluster import ManifoldCluster
from umc.processor.anomaly import ManifoldAnomalyDetector
from umc.data.synthetic import SyntheticManifoldGenerator
from umc.data.preprocessors import WindowDataset
from umc.evaluation.metrics import reconstruction_rmse, effective_dimensionality


@pytest.fixture
def small_config():
    return UMCConfig(
        window_size=16,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=16,
        encoder_hidden=(64, 32),
        decoder_hidden=(32, 64),
        num_charts=4,
        batch_size=32,
        epochs=10,
        beta_warmup_epochs=5,
        learning_rate=1e-3,
    )


@pytest.fixture
def synthetic_data(small_config):
    """Generate synthetic financial-like data."""
    gen = SyntheticManifoldGenerator(seed=42)
    data, true_dim = gen.generate_financial_synthetic(
        n_samples=500,
        intrinsic_dim=5,
        window_size=small_config.window_size,
        n_features=small_config.n_features,
    )
    return data, true_dim


class TestEndToEnd:
    def test_training_loss_decreases(self, small_config, synthetic_data):
        """Loss decreases over training epochs."""
        data, _ = synthetic_data

        train_data = data[:400]
        val_data = data[400:]

        train_loader = DataLoader(
            WindowDataset(train_data), batch_size=small_config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            WindowDataset(val_data), batch_size=small_config.batch_size
        )

        encoder = VAEEncoder(small_config)
        decoder = MLPDecoder(small_config)
        trainer = UMCTrainer(encoder, decoder, small_config)

        history = trainer.train(train_loader, val_loader, verbose=False)

        # Loss should decrease
        first_loss = history[0]["train"]["total"]
        last_loss = history[-1]["train"]["total"]
        assert last_loss < first_loss

    def test_encode_store_load_decode(self, small_config, synthetic_data):
        """Full pipeline: encode -> save .mnf -> load .mnf -> decode."""
        data, _ = synthetic_data

        # Quick train
        train_loader = DataLoader(
            WindowDataset(data), batch_size=small_config.batch_size, shuffle=True
        )
        encoder = VAEEncoder(small_config)
        decoder = MLPDecoder(small_config)
        trainer = UMCTrainer(encoder, decoder, small_config)
        trainer.train(train_loader, train_loader, verbose=False)

        # Encode
        device = trainer.device
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            x = torch.from_numpy(data).float().to(device)
            enc_result = encoder.encode(x)
            z = enc_result.z.cpu().numpy()
            chart_ids = enc_result.chart_id.cpu().numpy().astype(np.uint8)
            confidences = enc_result.confidence.cpu().numpy()

        # Store
        decoder_hash = DecoderManifest.compute_hash_bytes(decoder.state_dict())

        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            mnf_path = f.name

        try:
            writer = MNFWriter()
            writer.write(
                mnf_path, z, chart_ids, decoder_hash,
                confidences=confidences,
            )

            # Load
            reader = MNFReader()
            mnf = reader.read(mnf_path)

            # Verify coordinates survived roundtrip (float16 precision)
            loaded_z = mnf.coordinates.astype(np.float32)

            # Decode from loaded coordinates
            with torch.no_grad():
                z_tensor = torch.from_numpy(loaded_z).float().to(device)
                chart_tensor = torch.from_numpy(mnf.chart_ids.astype(np.int64)).to(device)
                x_hat = decoder.decode(z_tensor, chart_tensor).cpu().numpy()

            # Shape check
            assert x_hat.shape == data.shape

            # RMSE should be reasonable (not NaN or infinite)
            rmse = reconstruction_rmse(data, x_hat)
            assert np.isfinite(rmse)

        finally:
            os.unlink(mnf_path)

    def test_manifold_search_after_encoding(self, small_config, synthetic_data):
        """Search works on encoded coordinates."""
        data, _ = synthetic_data

        encoder = VAEEncoder(small_config)
        encoder.eval()

        with torch.no_grad():
            x = torch.from_numpy(data).float()
            enc_result = encoder.encode(x)
            z = enc_result.z.numpy()

        from umc.processor.search import ManifoldSearch
        search = ManifoldSearch(z)
        result = search.query(z[0], k=5)

        assert result.indices.shape == (1, 5)
        assert result.indices[0, 0] == 0  # Self is nearest

    def test_anomaly_detection_on_encoded(self, small_config):
        """Anomaly detection works on encoded data."""
        gen = SyntheticManifoldGenerator(seed=42)
        data, labels, _ = gen.generate_with_anomalies(
            n_normal=400, n_anomalies=100,
            intrinsic_dim=5, ambient_dim=small_config.input_dim,
        )

        # Reshape to (n, window, features) for encoding
        data_3d = data.reshape(-1, small_config.window_size, small_config.n_features)

        encoder = VAEEncoder(small_config)
        encoder.eval()

        with torch.no_grad():
            x = torch.from_numpy(data_3d).float()
            enc_result = encoder.encode(x)
            z = enc_result.z.numpy()

        from umc.processor.anomaly import ManifoldAnomalyDetector
        detector = ManifoldAnomalyDetector(z, method="isolation_forest")
        scores = detector.score(z)

        assert scores.shape == (len(data_3d),)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)


class TestLossFunctions:
    def test_total_loss_components(self, small_config):
        """Total loss returns all expected components."""
        x = torch.randn(16, small_config.window_size, small_config.n_features)
        x_hat = torch.randn(16, small_config.window_size, small_config.n_features, requires_grad=True)
        mu = torch.randn(16, small_config.max_latent_dim, requires_grad=True)
        log_var = torch.randn(16, small_config.max_latent_dim, requires_grad=True)
        z = torch.randn(16, small_config.max_latent_dim, requires_grad=True)

        loss, components = total_loss(x, x_hat, mu, log_var, z, small_config, epoch=10)

        assert loss.requires_grad
        assert "reconstruction" in components
        assert "kl" in components
        assert "sparsity" in components
        assert "beta" in components
        assert "total" in components

    def test_beta_annealing(self, small_config):
        """Beta increases during warmup."""
        x = torch.randn(16, small_config.window_size, small_config.n_features)
        x_hat = torch.randn_like(x)
        mu = torch.randn(16, small_config.max_latent_dim)
        log_var = torch.randn(16, small_config.max_latent_dim)
        z = torch.randn(16, small_config.max_latent_dim)

        _, comp_early = total_loss(x, x_hat, mu, log_var, z, small_config, epoch=0)
        _, comp_late = total_loss(x, x_hat, mu, log_var, z, small_config, epoch=small_config.beta_warmup_epochs)

        assert comp_early["beta"] <= comp_late["beta"]
        assert comp_late["beta"] == pytest.approx(small_config.beta_end)


    def test_multiscale_loss_gradient(self, small_config):
        """Multi-scale loss produces valid gradient."""
        x = torch.randn(8, small_config.window_size, small_config.n_features)
        x_hat = torch.randn(8, small_config.window_size, small_config.n_features, requires_grad=True)
        loss = multiscale_reconstruction_loss(x, x_hat, scales=(1, 4))
        loss.backward()
        assert x_hat.grad is not None
        assert x_hat.grad.abs().sum() > 0

    def test_spectral_loss_gradient(self, small_config):
        """Spectral loss produces valid gradient."""
        x = torch.randn(8, small_config.window_size, small_config.n_features)
        x_hat = torch.randn(8, small_config.window_size, small_config.n_features, requires_grad=True)
        loss = spectral_loss(x, x_hat)
        loss.backward()
        assert x_hat.grad is not None
        assert x_hat.grad.abs().sum() > 0

    def test_total_loss_with_multiscale_and_spectral(self, small_config):
        """Total loss includes multiscale and spectral when weights > 0."""
        small_config.multiscale_weight = 0.5
        small_config.spectral_weight = 0.1
        x = torch.randn(8, small_config.window_size, small_config.n_features)
        x_hat = torch.randn(8, small_config.window_size, small_config.n_features, requires_grad=True)
        mu = torch.randn(8, small_config.max_latent_dim)
        log_var = torch.randn(8, small_config.max_latent_dim)
        z = torch.randn(8, small_config.max_latent_dim)
        loss, comp = total_loss(x, x_hat, mu, log_var, z, small_config, epoch=10)
        assert "multiscale" in comp
        assert "spectral" in comp
        assert comp["multiscale"] > 0
        assert comp["spectral"] > 0
        # Reset for other tests
        small_config.multiscale_weight = 0.0
        small_config.spectral_weight = 0.0


class TestConvEndToEnd:
    def test_conv_training_loss_decreases(self, small_config, synthetic_data):
        """Conv encoder/decoder: loss decreases over training epochs."""
        data, _ = synthetic_data

        train_data = data[:400]
        val_data = data[400:]

        train_loader = DataLoader(
            WindowDataset(train_data), batch_size=small_config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            WindowDataset(val_data), batch_size=small_config.batch_size
        )

        encoder = ConvEncoder(small_config)
        decoder = ConvDecoder(small_config)
        trainer = UMCTrainer(encoder, decoder, small_config)

        history = trainer.train(train_loader, val_loader, verbose=False)

        first_loss = history[0]["train"]["total"]
        last_loss = history[-1]["train"]["total"]
        assert last_loss < first_loss

    def test_conv_encode_decode_roundtrip(self, small_config, synthetic_data):
        """Conv encoder/decoder: encode -> decode produces finite output."""
        data, _ = synthetic_data

        encoder = ConvEncoder(small_config)
        decoder = ConvDecoder(small_config)

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            x = torch.from_numpy(data[:64]).float()
            enc_result = encoder.encode(x)
            x_hat = decoder.decode(enc_result.z, enc_result.chart_id)

        assert x_hat.shape == x.shape
        assert torch.isfinite(x_hat).all()


class TestReconstructionConfidenceAnomaly:
    def test_reconstruction_confidence_method(self, small_config, synthetic_data):
        """Reconstruction confidence anomaly detection works."""
        data, _ = synthetic_data

        encoder = ConvEncoder(small_config)
        decoder = ConvDecoder(small_config)
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            x = torch.from_numpy(data[:100]).float()
            enc = encoder.encode(x)
            z = enc.z.numpy()
            x_hat = decoder.decode(enc.z, enc.chart_id).numpy()

        detector = ManifoldAnomalyDetector(z, method="reconstruction_confidence")
        detector.fit_reconstruction_confidence(data[:100], x_hat)

        per_sample_errors = np.sqrt(np.mean(
            (data[:100].reshape(100, -1) - x_hat.reshape(100, -1)) ** 2,
            axis=1,
        ))
        scores = detector.score(z, reconstruction_errors=per_sample_errors)

        assert scores.shape == (100,)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)


class TestSyntheticData:
    def test_known_dimensionality(self, small_config):
        """Synthetic data has expected structure."""
        gen = SyntheticManifoldGenerator(seed=42)
        data, true_dim = gen.generate_financial_synthetic(
            n_samples=100,
            intrinsic_dim=5,
            window_size=small_config.window_size,
            n_features=small_config.n_features,
        )

        assert data.shape == (100, small_config.window_size, small_config.n_features)
        assert true_dim == 5
        assert not np.any(np.isnan(data))
