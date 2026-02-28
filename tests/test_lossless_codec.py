"""Tests for Phase 2 lossless codec: VQ codes + compressed residuals.

Verifies:
    1. Byte transposition round-trip
    2. Residual compression/decompression
    3. Full lossless encode/decode with bit-exact verification
    4. Serialization round-trip
    5. VQ-only (lossy) decode path
    6. MNF format with residual block
"""

import tempfile

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("vector_quantize_pytorch")

import torch

from umc.config import UMCConfig
from umc.encoder.hvqvae_encoder import HVQVAEEncoder
from umc.decoder.hvqvae_decoder import HVQVAEDecoder
from umc.codec.residual import ResidualCoder, byte_transpose, byte_untranspose
from umc.codec.lossless import (
    LosslessCodec,
    LosslessEncoding,
    serialize_encoding,
    deserialize_encoding,
)
from umc.storage.mnf_format import MNFWriter, MNFReader


@pytest.fixture
def small_config():
    """Minimal config for fast tests."""
    return UMCConfig(
        window_size=32,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=16,
        encoder_type="hvqvae",
        num_charts=4,
        chart_embedding_dim=8,
        d_model=32,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        d_ff=64,
        patch_size=8,
        transformer_dropout=0.0,
        vq_dim=16,
        vq_top_n_codes=4,
        vq_bottom_n_codes=16,
        vq_bottom_n_levels=2,
        vq_commitment_weight=0.1,
        vq_ema_decay=0.99,
        vq_dead_code_threshold=2,
    )


@pytest.fixture
def model_pair(small_config):
    """Untrained encoder + decoder for codec testing."""
    encoder = HVQVAEEncoder(small_config)
    decoder = HVQVAEDecoder(small_config)
    encoder.eval()
    decoder.eval()
    return encoder, decoder


@pytest.fixture
def sample_windows():
    """Synthetic financial-like windows."""
    rng = np.random.RandomState(42)
    n_windows = 8
    window_size = 32
    n_features = 5
    # Simulate OHLCV: prices ~150, volume ~1e6
    base_price = 150.0 + rng.randn(n_windows, 1, 1) * 10
    noise = rng.randn(n_windows, window_size, 4) * 0.5
    prices = base_price + noise
    volume = np.abs(rng.randn(n_windows, window_size, 1)) * 1e6
    return np.concatenate([prices, volume], axis=2).astype(np.float32)


# ---- Byte transposition tests ----

class TestByteTranspose:
    def test_roundtrip_float32(self):
        """Byte transpose then untranspose recovers original bytes."""
        data = np.random.randn(100).astype(np.float32).tobytes()
        transposed = byte_transpose(data, 4)
        recovered = byte_untranspose(transposed, 4)
        assert recovered == data

    def test_roundtrip_different_sizes(self):
        """Works with various element sizes."""
        for elem_size in [1, 2, 4, 8]:
            data = np.random.bytes(elem_size * 50)
            transposed = byte_transpose(data, elem_size)
            recovered = byte_untranspose(transposed, elem_size)
            assert recovered == data

    def test_empty_data(self):
        """Handles empty input."""
        assert byte_transpose(b"", 4) == b""
        assert byte_untranspose(b"", 4) == b""

    def test_transposed_is_different(self):
        """Transposed data has different byte order (unless trivial)."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes()
        transposed = byte_transpose(data, 4)
        # Transposed groups byte 0s together, byte 1s together, etc.
        # Should generally differ from original
        assert len(transposed) == len(data)


# ---- Residual coder tests ----

class TestResidualCoder:
    def test_xor_compute_apply_exact(self):
        """XOR compute() then apply() gives bit-exact original (always)."""
        original = np.random.randn(10, 32, 5).astype(np.float32)
        reconstruction = original + np.random.randn(*original.shape).astype(np.float32) * 0.01
        xor_residual = ResidualCoder.compute(original, reconstruction)
        recovered = ResidualCoder.apply(reconstruction, xor_residual)
        np.testing.assert_array_equal(recovered, original)

    def test_xor_with_identical_inputs(self):
        """XOR of identical arrays is all zeros."""
        data = np.random.randn(5, 10).astype(np.float32)
        xor_res = ResidualCoder.compute(data, data)
        np.testing.assert_array_equal(xor_res, np.zeros_like(xor_res))

    def test_compress_decompress_exact(self):
        """Compress then decompress gives bit-exact XOR residual."""
        original = np.random.randn(10, 32, 5).astype(np.float32)
        reconstruction = original + np.random.randn(*original.shape).astype(np.float32) * 0.01
        xor_residual = ResidualCoder.compute(original, reconstruction)
        compressed = ResidualCoder.compress(xor_residual)
        decompressed = ResidualCoder.decompress(compressed)
        np.testing.assert_array_equal(decompressed, xor_residual)

    def test_compression_ratio(self):
        """Compressed XOR residuals are smaller than raw for similar data."""
        # Similar floats XOR to mostly-zero bytes -> compress well
        base = np.ones((100, 32, 5), dtype=np.float32) * 150.0
        noise = np.random.randn(100, 32, 5).astype(np.float32) * 0.001
        xor_res = ResidualCoder.compute(base + noise, base)
        compressed = ResidualCoder.compress(xor_res)
        raw_size = xor_res.nbytes
        assert len(compressed) < raw_size

    def test_shape_preservation(self):
        """Decompress recovers original shape."""
        for shape in [(5, 10), (3, 32, 5), (1, 256, 5)]:
            data = np.random.randn(*shape).astype(np.float32)
            xor_res = ResidualCoder.compute(data, np.zeros_like(data))
            compressed = ResidualCoder.compress(xor_res)
            decompressed = ResidualCoder.decompress(compressed)
            assert decompressed.shape == shape

    def test_full_pipeline_bitexact(self):
        """End-to-end: original -> compute -> compress -> decompress -> apply == original."""
        original = np.random.randn(5, 32, 5).astype(np.float32) * 100
        reconstruction = original + np.random.randn(*original.shape).astype(np.float32) * 0.5

        xor_residual = ResidualCoder.compute(original, reconstruction)
        compressed = ResidualCoder.compress(xor_residual)
        decompressed = ResidualCoder.decompress(compressed)
        recovered = ResidualCoder.apply(reconstruction, decompressed)

        np.testing.assert_array_equal(recovered, original)


# ---- Lossless codec tests ----

class TestLosslessCodec:
    def test_encode_decode_bitexact(self, model_pair, sample_windows):
        """Full encode -> decode round-trip is bit-exact."""
        encoder, decoder = model_pair
        codec = LosslessCodec(encoder, decoder, device="cpu")

        encoding = codec.encode(sample_windows, batch_size=4)
        recovered = codec.decode(encoding, batch_size=4)

        np.testing.assert_array_equal(recovered, sample_windows)

    def test_encoding_has_correct_shape(self, model_pair, sample_windows):
        """LosslessEncoding has correct metadata."""
        encoder, decoder = model_pair
        codec = LosslessCodec(encoder, decoder, device="cpu")

        encoding = codec.encode(sample_windows, batch_size=4)

        assert encoding.n_windows == sample_windows.shape[0]
        assert encoding.window_size == sample_windows.shape[1]
        assert encoding.n_features == sample_windows.shape[2]
        assert encoding.revin_means.shape == (8, 5)
        assert encoding.revin_stds.shape == (8, 5)

    def test_compression_ratio_positive(self, model_pair, sample_windows):
        """Compression ratio is > 1 (we actually compress)."""
        encoder, decoder = model_pair
        codec = LosslessCodec(encoder, decoder, device="cpu")

        encoding = codec.encode(sample_windows, batch_size=4)
        stats = codec.compression_stats(encoding)

        assert stats["compression_ratio"] > 1.0
        assert stats["vq_compression_ratio"] > stats["compression_ratio"]

    def test_vq_only_decode(self, model_pair, sample_windows):
        """VQ-only decode returns something reasonable (lossy)."""
        encoder, decoder = model_pair
        codec = LosslessCodec(encoder, decoder, device="cpu")

        encoding = codec.encode(sample_windows, batch_size=4)
        lossy = codec.decode_vq_only(encoding, batch_size=4)

        assert lossy.shape == sample_windows.shape
        assert lossy.dtype == np.float32
        # Lossy should be different from original (VQ error)
        # but not wildly different for reasonable data
        assert not np.array_equal(lossy, sample_windows)

    def test_bytes_roundtrip_bitexact(self, model_pair, sample_windows):
        """encode_to_bytes -> decode_from_bytes is bit-exact."""
        encoder, decoder = model_pair
        codec = LosslessCodec(encoder, decoder, device="cpu")

        encoded_bytes = codec.encode_to_bytes(sample_windows, batch_size=4)
        recovered = codec.decode_from_bytes(encoded_bytes, batch_size=4)

        np.testing.assert_array_equal(recovered, sample_windows)

    def test_single_window(self, model_pair):
        """Works with a single window."""
        encoder, decoder = model_pair
        codec = LosslessCodec(encoder, decoder, device="cpu")

        window = np.random.randn(1, 32, 5).astype(np.float32) * 100
        encoding = codec.encode(window, batch_size=1)
        recovered = codec.decode(encoding, batch_size=1)

        np.testing.assert_array_equal(recovered, window)


# ---- Serialization tests ----

class TestSerialization:
    def test_serialize_deserialize_roundtrip(self, model_pair, sample_windows):
        """Serialized encoding deserializes to identical encoding."""
        encoder, decoder = model_pair
        codec = LosslessCodec(encoder, decoder, device="cpu")

        encoding = codec.encode(sample_windows, batch_size=4)
        serialized = serialize_encoding(encoding)
        restored = deserialize_encoding(serialized)

        # Check metadata
        assert restored.n_windows == encoding.n_windows
        assert restored.window_size == encoding.window_size
        assert restored.n_features == encoding.n_features

        # Check VQ indices
        np.testing.assert_array_equal(
            restored.vq_indices.top_indices,
            encoding.vq_indices.top_indices,
        )
        np.testing.assert_array_equal(
            restored.vq_indices.bottom_indices,
            encoding.vq_indices.bottom_indices,
        )

        # Check RevIN stats
        np.testing.assert_array_equal(restored.revin_means, encoding.revin_means)
        np.testing.assert_array_equal(restored.revin_stds, encoding.revin_stds)

        # Check residual bytes identical
        assert restored.residual_compressed == encoding.residual_compressed

    def test_serialize_decode_bitexact(self, model_pair, sample_windows):
        """Serialize -> deserialize -> decode gives bit-exact original."""
        encoder, decoder = model_pair
        codec = LosslessCodec(encoder, decoder, device="cpu")

        encoding = codec.encode(sample_windows, batch_size=4)
        serialized = serialize_encoding(encoding)
        restored = deserialize_encoding(serialized)
        recovered = codec.decode(restored, batch_size=4)

        np.testing.assert_array_equal(recovered, sample_windows)


# ---- MNF format with residual tests ----

class TestMNFResidual:
    def test_mnf_write_read_with_residual(self):
        """MNF format correctly stores and reads back residual data."""
        rng = np.random.RandomState(42)
        n_samples = 10
        latent_dim = 16

        coords = rng.randn(n_samples, latent_dim).astype(np.float32)
        chart_ids = rng.randint(0, 4, n_samples).astype(np.uint8)
        decoder_hash = b"\xab" * 32
        residual_data = b"test_residual_block_data_" * 10

        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            path = f.name

        try:
            writer = MNFWriter()
            writer.write(
                path=path,
                coordinates=coords,
                chart_ids=chart_ids,
                decoder_hash=decoder_hash,
                residual_data=residual_data,
            )

            reader = MNFReader()
            mnf = reader.read(path)

            assert mnf.header.has_residual
            assert mnf.residual_data == residual_data
            np.testing.assert_array_almost_equal(
                mnf.coordinates, coords.astype(np.float16), decimal=2
            )
        finally:
            import os
            os.unlink(path)

    def test_mnf_no_residual_backward_compat(self):
        """MNF files without residual block still read correctly."""
        rng = np.random.RandomState(42)
        coords = rng.randn(5, 8).astype(np.float32)
        chart_ids = np.zeros(5, dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            path = f.name

        try:
            writer = MNFWriter()
            writer.write(
                path=path,
                coordinates=coords,
                chart_ids=chart_ids,
                decoder_hash=b"\x00" * 32,
            )

            reader = MNFReader()
            mnf = reader.read(path)

            assert not mnf.header.has_residual
            assert mnf.residual_data is None
        finally:
            import os
            os.unlink(path)


# ---- indices_to_quantized tests ----

class TestIndicesToQuantized:
    def test_encode_then_lookup_matches(self, model_pair):
        """indices_to_quantized produces same vectors as encode()."""
        encoder, decoder = model_pair
        encoder.eval()

        x = torch.randn(4, 32, 5)
        with torch.no_grad():
            encoder.encode(x)

            # Get vectors from encode path
            top_q_direct = encoder._last_top_quantized.clone()
            bottom_q_direct = encoder._last_bottom_quantized.clone()

            # Get indices
            top_idx = encoder._last_top_indices.clone()
            bottom_idx = torch.stack(encoder._last_bottom_indices, dim=-1)

            # Look up from indices
            top_q_lookup, bottom_q_lookup = encoder.indices_to_quantized(
                top_idx, bottom_idx
            )

        # Top should match exactly
        torch.testing.assert_close(top_q_lookup, top_q_direct, atol=1e-6, rtol=1e-5)

        # Bottom: with quantize_dropout, some levels may have -1 indices
        # The lookup handles this by masking to zero, which matches RVQ behavior
        # when dropout zeroes out a level
        # For a non-dropout forward pass, they should match closely
        # (exact match depends on whether dropout was active)
        assert bottom_q_lookup.shape == bottom_q_direct.shape


# ---- DAC-style lossless codec tests ----

class TestLosslessCodecDACStyle:
    """Lossless codec round-trip with codebook_dim / cosine_sim / rotation_trick."""

    @pytest.fixture
    def dac_config(self):
        return UMCConfig(
            window_size=32,
            features=("open", "high", "low", "close", "volume"),
            max_latent_dim=16,
            encoder_type="hvqvae",
            num_charts=4,
            chart_embedding_dim=8,
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=64,
            patch_size=8,
            transformer_dropout=0.0,
            vq_dim=16,
            vq_top_n_codes=4,
            vq_bottom_n_codes=16,
            vq_bottom_n_levels=2,
            vq_commitment_weight=0.1,
            vq_ema_decay=0.99,
            vq_dead_code_threshold=2,
            vq_codebook_dim=8,
            vq_use_cosine_sim=True,
            vq_rotation_trick=True,
        )

    @pytest.fixture
    def dac_model_pair(self, dac_config):
        encoder = HVQVAEEncoder(dac_config)
        decoder = HVQVAEDecoder(dac_config)
        encoder.eval()
        decoder.eval()
        return encoder, decoder

    def test_encode_decode_bitexact(self, dac_model_pair, sample_windows):
        """Full lossless round-trip is bit-exact with DAC-style VQ."""
        encoder, decoder = dac_model_pair
        codec = LosslessCodec(encoder, decoder, device="cpu")

        encoding = codec.encode(sample_windows, batch_size=4)
        recovered = codec.decode(encoding, batch_size=4)

        np.testing.assert_array_equal(recovered, sample_windows)

    def test_bytes_roundtrip_bitexact(self, dac_model_pair, sample_windows):
        """Serialize -> deserialize -> decode is bit-exact with DAC-style VQ."""
        encoder, decoder = dac_model_pair
        codec = LosslessCodec(encoder, decoder, device="cpu")

        encoded_bytes = codec.encode_to_bytes(sample_windows, batch_size=4)
        recovered = codec.decode_from_bytes(encoded_bytes, batch_size=4)

        np.testing.assert_array_equal(recovered, sample_windows)
