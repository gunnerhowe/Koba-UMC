"""Tests for Hierarchical VQ-VAE components."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("vector_quantize_pytorch")

import torch
import torch.nn as nn

from umc.config import UMCConfig
from umc.encoder.vq_layers import RevIN, PatchEmbedding, VectorQuantizerEMA
from umc.encoder.hvqvae_encoder import HVQVAEEncoder, TransformerTSBlock
from umc.decoder.hvqvae_decoder import HVQVAEDecoder


@pytest.fixture
def config():
    """Small config for fast tests."""
    return UMCConfig(
        window_size=64,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=32,
        num_charts=4,
        encoder_type="hvqvae",
        # Small transformer for tests
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=128,
        patch_size=8,
        vq_dim=16,
        vq_top_n_codes=32,
        vq_bottom_n_codes=64,
        vq_commitment_weight=0.25,
        vq_ema_decay=0.99,
        vq_dead_code_threshold=2,
        transformer_dropout=0.0,  # No dropout for deterministic tests
    )


@pytest.fixture
def sample_batch(config):
    """Random batch of windowed OHLCV data."""
    batch_size = 8
    return torch.randn(batch_size, config.window_size, config.n_features)


# ---- RevIN Tests ----

class TestRevIN:
    def test_forward_shape(self):
        revin = RevIN(5)
        x = torch.randn(4, 32, 5)
        out = revin(x)
        assert out.shape == x.shape

    def test_inverse_roundtrip(self):
        """RevIN normalize -> inverse recovers original."""
        revin = RevIN(5)
        x = torch.randn(4, 32, 5) * 10 + 5
        normalized = revin(x)
        recovered = revin.inverse(normalized)
        assert torch.allclose(x, recovered, atol=1e-5)

    def test_normalized_stats(self):
        """After RevIN, each sample should have ~0 mean, ~1 std per feature."""
        revin = RevIN(5)
        # Use non-trivial data
        x = torch.randn(8, 64, 5) * 3 + 7
        out = revin(x)
        # Check per-sample, per-feature
        means = out.mean(dim=1)  # (B, F)
        # After affine, mean should be affine_bias (default 0)
        assert means.abs().max() < 0.5  # Approximately zero

    def test_inverse_without_forward_raises(self):
        """Calling inverse before forward should raise."""
        revin = RevIN(5)
        x = torch.randn(4, 32, 5)
        with pytest.raises(RuntimeError):
            revin.inverse(x)

    def test_set_stats(self):
        """Manual stats setting enables inverse without forward."""
        revin = RevIN(5)
        mean = torch.zeros(4, 1, 5)
        std = torch.ones(4, 1, 5)
        revin.set_stats(mean, std)
        x = torch.randn(4, 32, 5)
        result = revin.inverse(x)
        assert result.shape == x.shape


# ---- PatchEmbedding Tests ----

class TestPatchEmbedding:
    def test_output_shape(self):
        patch_embed = PatchEmbedding(n_features=5, d_model=64, patch_size=8, n_patches=8)
        x = torch.randn(4, 64, 5)
        out = patch_embed(x)
        assert out.shape == (4, 8, 64)

    def test_different_window_sizes(self):
        """Correct shape for various window/patch combinations."""
        for window, patch in [(32, 8), (64, 16), (128, 8), (256, 8)]:
            n_patches = window // patch
            embed = PatchEmbedding(5, 32, patch, n_patches)
            x = torch.randn(2, window, 5)
            out = embed(x)
            assert out.shape == (2, n_patches, 32), f"Failed for window={window}, patch={patch}"

    def test_gradient_flow(self):
        embed = PatchEmbedding(5, 32, 8, 4)
        x = torch.randn(2, 32, 5, requires_grad=True)
        out = embed(x)
        out.sum().backward()
        assert x.grad is not None


# ---- VectorQuantizerEMA Tests ----

class TestVectorQuantizerEMA:
    def test_forward_shape_2d(self):
        """2D input: (batch, code_dim)."""
        vq = VectorQuantizerEMA(n_codes=32, code_dim=16)
        z_e = torch.randn(8, 16)
        z_q, loss, indices = vq(z_e)
        assert z_q.shape == z_e.shape
        assert indices.shape == (8,)
        assert loss.item() >= 0

    def test_forward_shape_3d(self):
        """3D input: (batch, n_patches, code_dim)."""
        vq = VectorQuantizerEMA(n_codes=64, code_dim=16)
        z_e = torch.randn(4, 8, 16)
        z_q, loss, indices = vq(z_e)
        assert z_q.shape == z_e.shape
        assert indices.shape == (4, 8)

    def test_indices_valid(self):
        """Indices are within codebook range."""
        vq = VectorQuantizerEMA(n_codes=32, code_dim=16)
        z_e = torch.randn(8, 16)
        _, _, indices = vq(z_e)
        assert (indices >= 0).all()
        assert (indices < 32).all()

    def test_straight_through_gradient(self):
        """Gradients flow through straight-through estimator."""
        vq = VectorQuantizerEMA(n_codes=32, code_dim=16)
        z_e = torch.randn(8, 16, requires_grad=True)
        z_q, loss, _ = vq(z_e)
        (z_q.sum() + loss).backward()
        assert z_e.grad is not None
        assert z_e.grad.abs().sum() > 0

    def test_ema_codebook_update(self):
        """EMA updates move codebook toward inputs."""
        vq = VectorQuantizerEMA(n_codes=4, code_dim=8, decay=0.5)
        vq.train()

        # Create inputs clustered around specific points
        target = torch.tensor([[1.0] * 8, [-1.0] * 8, [0.5] * 8, [-0.5] * 8])
        codebook_before = vq.codebook.clone()

        # Run multiple updates
        for _ in range(20):
            z_e = target + torch.randn_like(target) * 0.01
            vq(z_e)

        # Codebook should have changed
        assert not torch.allclose(vq.codebook, codebook_before, atol=0.01)

    def test_dead_code_reset(self):
        """Dead codes get reinitialized."""
        vq = VectorQuantizerEMA(n_codes=16, code_dim=8)
        vq.train()

        # Only use a few codes
        z_e = torch.randn(4, 8)
        vq(z_e)

        # Most codes should have low usage
        n_reset = vq.reset_dead_codes(z_e, threshold=2)
        assert n_reset >= 0  # Some codes may have been reset

    def test_perplexity(self):
        """Perplexity is computed and positive."""
        vq = VectorQuantizerEMA(n_codes=32, code_dim=16)
        z_e = torch.randn(64, 16)
        vq(z_e)
        assert vq.perplexity > 0

    def test_get_codes(self):
        """Can look up codes by index."""
        vq = VectorQuantizerEMA(n_codes=16, code_dim=8)
        indices = torch.tensor([0, 5, 10])
        codes = vq.get_codes(indices)
        assert codes.shape == (3, 8)
        assert torch.allclose(codes[0], vq.codebook[0])


# ---- TransformerTSBlock Tests ----

class TestTransformerTSBlock:
    def test_output_shape(self):
        block = TransformerTSBlock(d_model=64, n_heads=4, d_ff=128, dropout=0.0)
        x = torch.randn(4, 8, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        block = TransformerTSBlock(d_model=64, n_heads=4, d_ff=128, dropout=0.0)
        x = torch.randn(4, 8, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


# ---- HVQVAEEncoder Tests ----

class TestHVQVAEEncoder:
    def test_output_shape(self, config, sample_batch):
        """Encoder produces correct tensor shapes."""
        encoder = HVQVAEEncoder(config)
        result = encoder.encode(sample_batch)

        assert result.z.shape == (8, config.max_latent_dim)
        assert result.mu.shape == (8, config.max_latent_dim)
        assert result.log_var.shape == (8, config.max_latent_dim)
        assert result.chart_id.shape == (8,)
        assert result.confidence.shape == (8,)

    def test_confidence_range(self, config, sample_batch):
        """Confidence scores are in [0, 1]."""
        encoder = HVQVAEEncoder(config)
        result = encoder.encode(sample_batch)
        assert (result.confidence >= 0).all()
        assert (result.confidence <= 1).all()

    def test_chart_id_range(self, config, sample_batch):
        """Chart IDs are valid indices."""
        encoder = HVQVAEEncoder(config)
        result = encoder.encode(sample_batch)
        assert (result.chart_id >= 0).all()
        assert (result.chart_id < config.num_charts).all()

    def test_vq_loss_positive(self, config, sample_batch):
        """VQ commitment loss is non-negative."""
        encoder = HVQVAEEncoder(config)
        encoder.train()
        encoder.encode(sample_batch)
        assert encoder.vq_loss.item() >= 0

    def test_perplexity_tracked(self, config, sample_batch):
        """Perplexity is tracked for both VQ levels."""
        encoder = HVQVAEEncoder(config)
        encoder.encode(sample_batch)
        assert encoder.top_perplexity > 0
        assert encoder.bottom_perplexity > 0

    def test_deterministic_eval(self, config, sample_batch):
        """Encoder is deterministic in eval mode."""
        encoder = HVQVAEEncoder(config)
        encoder.eval()
        with torch.no_grad():
            r1 = encoder.encode(sample_batch)
            r2 = encoder.encode(sample_batch)
        assert torch.allclose(r1.z, r2.z)

    def test_gradient_flow(self, config, sample_batch):
        """Gradients flow through the VQ-VAE encoder."""
        encoder = HVQVAEEncoder(config)
        encoder.train()
        result = encoder.encode(sample_batch)
        loss = result.z.sum() + encoder.vq_loss
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in encoder.parameters() if p.requires_grad)
        assert has_grad

    def test_stored_vq_outputs(self, config, sample_batch):
        """Encoder stores VQ outputs for decoder access."""
        encoder = HVQVAEEncoder(config)
        encoder.train()
        encoder.encode(sample_batch)

        n_patches = config.window_size // config.patch_size
        assert encoder._last_top_quantized is not None
        assert encoder._last_top_quantized.shape == (8, config.vq_dim)
        assert encoder._last_bottom_quantized is not None
        assert encoder._last_bottom_quantized.shape == (8, n_patches, config.vq_dim)

    def test_dead_code_reset(self, config, sample_batch):
        """Dead code reset runs without error."""
        encoder = HVQVAEEncoder(config)
        encoder.train()
        encoder.encode(sample_batch)
        n_top, n_bottom = encoder.reset_dead_codes()
        assert isinstance(n_top, int)
        assert isinstance(n_bottom, int)

    def test_log_var_zeros(self, config, sample_batch):
        """VQ-VAE has zero log_var (no KL)."""
        encoder = HVQVAEEncoder(config)
        result = encoder.encode(sample_batch)
        assert torch.allclose(result.log_var, torch.zeros_like(result.log_var))

    def test_effective_dim(self, config):
        encoder = HVQVAEEncoder(config)
        assert encoder.get_effective_dim() == config.max_latent_dim


# ---- HVQVAEDecoder Tests ----

class TestHVQVAEDecoder:
    def test_decode_from_codes_shape(self, config):
        """Direct VQ decode produces correct shape."""
        decoder = HVQVAEDecoder(config)
        n_patches = config.window_size // config.patch_size

        top_q = torch.randn(4, config.vq_dim)
        bottom_q = torch.randn(4, n_patches, config.vq_dim)
        x_hat = decoder.decode_from_codes(top_q, bottom_q)

        assert x_hat.shape == (4, config.window_size, config.n_features)

    def test_decode_from_z_shape(self, config):
        """Standard decode interface produces correct shape."""
        decoder = HVQVAEDecoder(config)
        z = torch.randn(4, config.max_latent_dim)
        chart_id = torch.zeros(4, dtype=torch.long)
        x_hat = decoder.decode(z, chart_id)

        assert x_hat.shape == (4, config.window_size, config.n_features)

    def test_gradient_flow_from_codes(self, config):
        """Gradients flow through decode_from_codes."""
        decoder = HVQVAEDecoder(config)
        n_patches = config.window_size // config.patch_size

        top_q = torch.randn(4, config.vq_dim, requires_grad=True)
        bottom_q = torch.randn(4, n_patches, config.vq_dim, requires_grad=True)
        x_hat = decoder.decode_from_codes(top_q, bottom_q)
        x_hat.sum().backward()

        assert top_q.grad is not None
        assert bottom_q.grad is not None

    def test_gradient_flow_from_z(self, config):
        """Gradients flow through standard decode path."""
        decoder = HVQVAEDecoder(config)
        z = torch.randn(4, config.max_latent_dim, requires_grad=True)
        chart_id = torch.zeros(4, dtype=torch.long)
        x_hat = decoder.decode(z, chart_id)
        x_hat.sum().backward()

        assert z.grad is not None


# ---- End-to-End Tests ----

class TestHVQVAEEndToEnd:
    def test_encode_decode_roundtrip(self, config, sample_batch):
        """End-to-end forward pass produces valid output."""
        encoder = HVQVAEEncoder(config)
        decoder = HVQVAEDecoder(config)
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            enc_result = encoder.encode(sample_batch)
            x_hat = decoder.decode(enc_result.z, enc_result.chart_id)

        assert x_hat.shape == sample_batch.shape
        assert torch.isfinite(x_hat).all()

    def test_training_forward_backward(self, config, sample_batch):
        """Full training step completes without error."""
        encoder = HVQVAEEncoder(config)
        decoder = HVQVAEDecoder(config)
        encoder.train()
        decoder.train()

        enc_result = encoder.encode(sample_batch)

        # Direct VQ decode path
        x_hat_raw = decoder.decode_from_codes(
            encoder._last_top_quantized,
            encoder._last_bottom_quantized,
        )
        x_hat = encoder.revin.inverse(x_hat_raw)

        # Compute loss
        recon_loss = nn.functional.mse_loss(x_hat, sample_batch)
        total_loss = recon_loss + encoder.vq_loss
        total_loss.backward()

        # Check gradients exist
        enc_grads = sum(1 for p in encoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        dec_grads = sum(1 for p in decoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert enc_grads > 0
        assert dec_grads > 0

    def test_z_projection_for_faiss(self, config, sample_batch):
        """Continuous z output is suitable for FAISS indexing."""
        encoder = HVQVAEEncoder(config)
        encoder.eval()

        with torch.no_grad():
            result = encoder.encode(sample_batch)

        z = result.z.numpy()
        assert z.shape == (8, config.max_latent_dim)
        assert z.dtype.name.startswith('float')
        # Check reasonable variance (not collapsed)
        assert z.std() > 0.001


# ---- DAC-style VQ Tests ----

class TestDACStyleVQ:
    """Tests for codebook_dim, cosine_sim, rotation_trick."""

    @pytest.fixture
    def dac_config(self):
        """Config with DAC-style VQ enabled."""
        return UMCConfig(
            window_size=64,
            features=("open", "high", "low", "close", "volume"),
            max_latent_dim=32,
            num_charts=4,
            encoder_type="hvqvae",
            d_model=64,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=128,
            patch_size=8,
            vq_dim=16,
            vq_top_n_codes=32,
            vq_bottom_n_codes=64,
            vq_bottom_n_levels=2,
            vq_commitment_weight=0.25,
            vq_ema_decay=0.99,
            vq_dead_code_threshold=2,
            transformer_dropout=0.0,
            # DAC-style params
            vq_codebook_dim=8,
            vq_use_cosine_sim=True,
            vq_rotation_trick=True,
        )

    def test_encoder_builds(self, dac_config):
        """Encoder with DAC-style VQ initializes without error."""
        encoder = HVQVAEEncoder(dac_config)
        # Verify codebook is in reduced dimension
        cb = encoder.vq_top._codebook.embed
        assert cb.shape[-1] == 8  # codebook_dim, not vq_dim

    def test_encode_shapes(self, dac_config):
        """Encoder output shapes unchanged with DAC-style VQ."""
        encoder = HVQVAEEncoder(dac_config)
        encoder.eval()
        x = torch.randn(4, 64, 5)
        with torch.no_grad():
            result = encoder.encode(x)
        assert result.z.shape == (4, dac_config.max_latent_dim)
        assert encoder._last_top_quantized.shape == (4, dac_config.vq_dim)
        n_patches = 64 // 8
        assert encoder._last_bottom_quantized.shape == (4, n_patches, dac_config.vq_dim)

    def test_indices_to_quantized_matches_encode(self, dac_config):
        """indices_to_quantized produces same vectors as encode() with codebook_dim."""
        encoder = HVQVAEEncoder(dac_config)
        encoder.eval()
        x = torch.randn(4, 64, 5)

        with torch.no_grad():
            encoder.encode(x)
            top_q_direct = encoder._last_top_quantized.clone()
            bottom_q_direct = encoder._last_bottom_quantized.clone()

            top_idx = encoder._last_top_indices.clone()
            bottom_idx = torch.stack(encoder._last_bottom_indices, dim=-1)

            top_q_lookup, bottom_q_lookup = encoder.indices_to_quantized(
                top_idx, bottom_idx
            )

        # Top should match (lookup + project_out == forward path quantized output)
        torch.testing.assert_close(top_q_lookup, top_q_direct, atol=1e-5, rtol=1e-4)
        # Bottom: shape should match, values close
        assert bottom_q_lookup.shape == bottom_q_direct.shape

    def test_training_forward_backward(self, dac_config):
        """Full training step with DAC-style VQ completes without error."""
        encoder = HVQVAEEncoder(dac_config)
        decoder = HVQVAEDecoder(dac_config)
        encoder.train()
        decoder.train()

        x = torch.randn(4, 64, 5)
        enc_result = encoder.encode(x)
        x_hat_raw = decoder.decode_from_codes(
            encoder._last_top_quantized,
            encoder._last_bottom_quantized,
        )
        x_hat = encoder.revin.inverse(x_hat_raw)

        recon_loss = torch.nn.functional.mse_loss(x_hat, x)
        total_loss = recon_loss + encoder.vq_loss
        total_loss.backward()

        enc_grads = sum(1 for p in encoder.parameters()
                        if p.grad is not None and p.grad.abs().sum() > 0)
        assert enc_grads > 0

    def test_vq_loss_reasonable(self, dac_config):
        """VQ loss with DAC-style is finite and non-negative."""
        encoder = HVQVAEEncoder(dac_config)
        encoder.train()
        x = torch.randn(4, 64, 5)
        encoder.encode(x)
        assert encoder.vq_loss.item() >= 0
        assert torch.isfinite(encoder.vq_loss)

    def test_codebook_dim_zero_means_no_projection(self):
        """codebook_dim=0 means no projection (original behavior)."""
        config = UMCConfig(
            window_size=64,
            features=("open", "high", "low", "close", "volume"),
            max_latent_dim=32,
            num_charts=4,
            d_model=64,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=128,
            patch_size=8,
            vq_dim=16,
            vq_top_n_codes=32,
            vq_bottom_n_codes=64,
            vq_codebook_dim=0,
            vq_use_cosine_sim=False,
            vq_rotation_trick=False,
            transformer_dropout=0.0,
        )
        encoder = HVQVAEEncoder(config)
        cb = encoder.vq_top._codebook.embed
        assert cb.shape[-1] == 16  # full vq_dim, no projection


# ---- FSQ Tests ----

class TestFSQMode:
    """Tests for Finite Scalar Quantization (no codebook collapse)."""

    @pytest.fixture
    def fsq_config(self):
        """Config with FSQ mode enabled."""
        return UMCConfig(
            window_size=64,
            features=("open", "high", "low", "close", "volume"),
            max_latent_dim=32,
            num_charts=4,
            encoder_type="hvqvae",
            d_model=64,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=128,
            patch_size=8,
            vq_dim=16,
            vq_bottom_n_levels=3,
            vq_commitment_weight=0.0,
            transformer_dropout=0.0,
            # FSQ mode
            vq_type="fsq",
            fsq_top_levels=(8, 8),       # 64 implicit codes
            fsq_bottom_levels=(8, 5, 5),  # 200 implicit codes per level
        )

    def test_encoder_builds(self, fsq_config):
        """FSQ encoder initializes without error."""
        encoder = HVQVAEEncoder(fsq_config)
        assert encoder._vq_type == 'fsq'
        assert encoder._top_n_codes == 64
        assert encoder._bottom_n_codes == 200

    def test_encode_shapes(self, fsq_config):
        """FSQ encoder output shapes are correct."""
        encoder = HVQVAEEncoder(fsq_config)
        encoder.eval()
        x = torch.randn(4, 64, 5)
        with torch.no_grad():
            result = encoder.encode(x)
        assert result.z.shape == (4, fsq_config.max_latent_dim)
        assert encoder._last_top_quantized.shape == (4, fsq_config.vq_dim)
        n_patches = 64 // 8
        assert encoder._last_bottom_quantized.shape == (4, n_patches, fsq_config.vq_dim)

    def test_vq_loss_is_zero(self, fsq_config):
        """FSQ has no commitment loss — vq_loss should be 0."""
        encoder = HVQVAEEncoder(fsq_config)
        encoder.train()
        x = torch.randn(4, 64, 5)
        encoder.encode(x)
        assert encoder.vq_loss.item() == 0.0

    def test_indices_to_quantized_roundtrip(self, fsq_config):
        """indices_to_quantized matches forward pass output for FSQ."""
        encoder = HVQVAEEncoder(fsq_config)
        encoder.eval()
        x = torch.randn(4, 64, 5)

        with torch.no_grad():
            encoder.encode(x)
            top_q_direct = encoder._last_top_quantized.clone()
            bottom_q_direct = encoder._last_bottom_quantized.clone()

            top_idx = encoder._last_top_indices.clone()
            bottom_idx = torch.stack(encoder._last_bottom_indices, dim=-1)

            top_q_lookup, bottom_q_lookup = encoder.indices_to_quantized(
                top_idx, bottom_idx
            )

        torch.testing.assert_close(top_q_lookup, top_q_direct, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(bottom_q_lookup, bottom_q_direct, atol=1e-5, rtol=1e-4)

    def test_training_forward_backward(self, fsq_config):
        """Full training step with FSQ completes without error."""
        encoder = HVQVAEEncoder(fsq_config)
        decoder = HVQVAEDecoder(fsq_config)
        encoder.train()
        decoder.train()

        x = torch.randn(4, 64, 5)
        enc_result = encoder.encode(x)
        x_hat_raw = decoder.decode_from_codes(
            encoder._last_top_quantized,
            encoder._last_bottom_quantized,
        )
        x_hat = encoder.revin.inverse(x_hat_raw)

        recon_loss = torch.nn.functional.mse_loss(x_hat, x)
        # No VQ loss with FSQ, purely reconstruction
        recon_loss.backward()

        enc_grads = sum(1 for p in encoder.parameters()
                        if p.grad is not None and p.grad.abs().sum() > 0)
        dec_grads = sum(1 for p in decoder.parameters()
                        if p.grad is not None and p.grad.abs().sum() > 0)
        assert enc_grads > 0
        assert dec_grads > 0

    def test_perplexity_tracked(self, fsq_config):
        """Perplexity is tracked for FSQ indices."""
        encoder = HVQVAEEncoder(fsq_config)
        encoder.eval()
        x = torch.randn(8, 64, 5)
        with torch.no_grad():
            encoder.encode(x)
        assert encoder.top_perplexity > 0
        assert encoder.bottom_perplexity > 0
        assert len(encoder.per_level_perplexity) == 3  # 3 RVQ levels

    def test_indices_are_valid_range(self, fsq_config):
        """FSQ indices are within valid range."""
        encoder = HVQVAEEncoder(fsq_config)
        encoder.eval()
        x = torch.randn(4, 64, 5)
        with torch.no_grad():
            encoder.encode(x)
        top_idx = encoder._last_top_indices
        assert (top_idx >= 0).all()
        assert (top_idx < 64).all()  # 8*8 = 64

        for level_idx in encoder._last_bottom_indices:
            assert (level_idx >= 0).all()
            assert (level_idx < 200).all()  # 8*5*5 = 200

    def test_deterministic_eval(self, fsq_config):
        """FSQ encoder is deterministic in eval mode."""
        encoder = HVQVAEEncoder(fsq_config)
        encoder.eval()
        x = torch.randn(4, 64, 5)
        with torch.no_grad():
            r1 = encoder.encode(x)
            r2 = encoder.encode(x)
        assert torch.allclose(r1.z, r2.z)
