"""Tests for tiered compression codec: VQ search + efficient storage.

Verifies:
    1. Storage compression/decompression round-trip (lossless)
    2. Storage compression near-lossless mode (float16)
    3. Full tiered encode/decode
    4. Serialization round-trip
    5. VQ search decode path
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("vector_quantize_pytorch")

import torch

from umc.config import UMCConfig
from umc.encoder.hvqvae_encoder import HVQVAEEncoder
from umc.decoder.hvqvae_decoder import HVQVAEDecoder
from umc.codec.tiered import (
    TieredCodec,
    TieredEncoding,
    serialize_tiered,
    deserialize_tiered,
    _compress_storage,
    _decompress_storage,
)


@pytest.fixture
def small_config():
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
    encoder = HVQVAEEncoder(small_config)
    decoder = HVQVAEDecoder(small_config)
    encoder.eval()
    decoder.eval()
    return encoder, decoder


@pytest.fixture
def sample_windows():
    rng = np.random.RandomState(42)
    n_windows = 8
    window_size = 32
    base_price = 150.0 + rng.randn(n_windows, 1, 1) * 10
    noise = rng.randn(n_windows, window_size, 4) * 0.5
    prices = base_price + noise
    volume = np.abs(rng.randn(n_windows, window_size, 1)) * 1e6
    return np.concatenate([prices, volume], axis=2).astype(np.float32)


class TestStorageCompression:
    """Test storage tier compression/decompression."""

    def test_lossless_roundtrip(self, sample_windows):
        compressed = _compress_storage(sample_windows, 'lossless')
        recovered = _decompress_storage(compressed)
        assert np.array_equal(
            sample_windows.view(np.uint32),
            recovered.view(np.uint32),
        ), "Lossless storage is not bit-exact"

    def test_near_lossless_roundtrip(self, sample_windows):
        compressed = _compress_storage(sample_windows, 'near_lossless')
        recovered = _decompress_storage(compressed)
        # Float16 precision: should be close but not bit-exact
        assert recovered.shape == sample_windows.shape
        assert recovered.dtype == np.float32
        rmse = np.sqrt(np.mean((sample_windows - recovered) ** 2))
        data_range = sample_windows.max() - sample_windows.min()
        rmse_pct = rmse / data_range * 100
        assert rmse_pct < 0.1, f"Near-lossless RMSE {rmse_pct:.4f}% exceeds 0.1%"

    def test_lossless_compresses(self, sample_windows):
        compressed = _compress_storage(sample_windows, 'lossless')
        raw_size = sample_windows.nbytes
        assert len(compressed) < raw_size, "Lossless storage did not compress"

    def test_near_lossless_smaller(self, sample_windows):
        lossless = _compress_storage(sample_windows, 'lossless')
        near = _compress_storage(sample_windows, 'near_lossless')
        assert len(near) < len(lossless), "Near-lossless should be smaller than lossless"


class TestTieredCodec:
    """Test full tiered codec pipeline."""

    def test_encode_decode_storage_lossless(self, model_pair, sample_windows):
        encoder, decoder = model_pair
        codec = TieredCodec(encoder, decoder, device="cpu", storage_mode="lossless")
        encoding = codec.encode(sample_windows, batch_size=4)

        recovered = codec.decode_storage(encoding)
        assert np.array_equal(
            sample_windows.view(np.uint32),
            recovered.view(np.uint32),
        ), "Storage decode is not bit-exact in lossless mode"

    def test_encode_decode_storage_near_lossless(self, model_pair, sample_windows):
        encoder, decoder = model_pair
        codec = TieredCodec(encoder, decoder, device="cpu", storage_mode="near_lossless")
        encoding = codec.encode(sample_windows, batch_size=4)

        recovered = codec.decode_storage(encoding)
        rmse = np.sqrt(np.mean((sample_windows - recovered) ** 2))
        data_range = sample_windows.max() - sample_windows.min()
        assert rmse / data_range * 100 < 0.1

    def test_search_decode(self, model_pair, sample_windows):
        encoder, decoder = model_pair
        codec = TieredCodec(encoder, decoder, device="cpu", storage_mode="lossless")
        encoding = codec.encode(sample_windows, batch_size=4)

        approx = codec.decode_search(encoding, batch_size=4)
        assert approx.shape == sample_windows.shape
        assert approx.dtype == np.float32
        # VQ decode should be lossy but reasonable
        assert not np.array_equal(sample_windows, approx)

    def test_compression_stats(self, model_pair, sample_windows):
        encoder, decoder = model_pair
        codec = TieredCodec(encoder, decoder, device="cpu", storage_mode="lossless")
        encoding = codec.encode(sample_windows, batch_size=4)

        stats = codec.compression_stats(encoding)
        assert stats["raw_bytes"] == sample_windows.nbytes
        assert stats["vq_bytes"] > 0
        assert stats["storage_bytes"] > 0
        assert stats["search_compression"] > 1.0
        assert stats["storage_mode"] == "lossless"

    def test_encoding_properties(self, model_pair, sample_windows):
        encoder, decoder = model_pair
        codec = TieredCodec(encoder, decoder, device="cpu", storage_mode="lossless")
        encoding = codec.encode(sample_windows, batch_size=4)

        assert encoding.n_windows == 8
        assert encoding.window_size == 32
        assert encoding.n_features == 5
        assert encoding.raw_bytes == 8 * 32 * 5 * 4
        assert encoding.total_bytes == encoding.vq_bytes + encoding.storage_bytes


class TestSerialization:
    """Test tiered encoding serialization."""

    def test_serialize_roundtrip(self, model_pair, sample_windows):
        encoder, decoder = model_pair
        codec = TieredCodec(encoder, decoder, device="cpu", storage_mode="lossless")
        encoding = codec.encode(sample_windows, batch_size=4)

        serialized = serialize_tiered(encoding)
        encoding2 = deserialize_tiered(serialized)

        # Verify storage tier
        recovered = codec.decode_storage(encoding2)
        assert np.array_equal(
            sample_windows.view(np.uint32),
            recovered.view(np.uint32),
        )

        # Verify VQ indices preserved
        assert np.array_equal(
            encoding.vq_indices.top_indices,
            encoding2.vq_indices.top_indices,
        )
        assert np.array_equal(
            encoding.vq_indices.bottom_indices,
            encoding2.vq_indices.bottom_indices,
        )

    def test_serialize_near_lossless(self, model_pair, sample_windows):
        encoder, decoder = model_pair
        codec = TieredCodec(encoder, decoder, device="cpu", storage_mode="near_lossless")
        encoding = codec.encode(sample_windows, batch_size=4)

        serialized = serialize_tiered(encoding)
        encoding2 = deserialize_tiered(serialized)

        recovered = codec.decode_storage(encoding2)
        rmse = np.sqrt(np.mean((sample_windows - recovered) ** 2))
        data_range = sample_windows.max() - sample_windows.min()
        assert rmse / data_range * 100 < 0.1
