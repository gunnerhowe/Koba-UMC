"""Integration tests for tiered codec + .mnf format pipeline.

Verifies:
    1. MNF write/read round-trip with tiered data
    2. TieredManifoldCodec encode_to_mnf / decode_from_mnf
    3. Search via VQ codes in .mnf files
    4. Compression stats from .mnf
    5. Optimal entropy coding round-trip
    6. Delta-coded VQ index round-trip
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from umc.config import UMCConfig
from umc.encoder.hvqvae_encoder import HVQVAEEncoder
from umc.decoder.hvqvae_decoder import HVQVAEDecoder
from umc.codec.tiered import (
    TieredCodec,
    serialize_tiered,
    deserialize_tiered,
    _compress_storage,
    _decompress_storage,
)
from umc.storage.mnf_format import MNFWriter, MNFReader, MNFHeader
from umc.storage.entropy import (
    VQIndices,
    compress_indices,
    decompress_indices,
    _delta_encode_levels,
    _delta_decode_levels,
)
from umc import TieredManifoldCodec


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


class TestOptimalEntropy:
    """Test optimal entropy coding for VQ indices."""

    def test_optimal_roundtrip(self):
        """Optimal compression produces valid decompressible output."""
        rng = np.random.RandomState(123)
        n, p, l = 16, 4, 3
        vq = VQIndices(
            top_indices=rng.randint(0, 4, size=n).astype(np.uint8),
            bottom_indices=rng.randint(0, 16, size=(n, p, l)).astype(np.uint8),
            n_patches=p,
            n_levels=l,
            top_n_codes=4,
            bottom_n_codes=16,
        )
        compressed = compress_indices(vq, mode="optimal")
        recovered = decompress_indices(compressed)

        assert np.array_equal(vq.top_indices, recovered.top_indices)
        assert np.array_equal(vq.bottom_indices, recovered.bottom_indices)
        assert recovered.n_patches == p
        assert recovered.n_levels == l

    def test_optimal_not_larger_than_grouped(self):
        """Optimal mode should be <= grouped mode size."""
        rng = np.random.RandomState(456)
        n, p, l = 64, 8, 4
        vq = VQIndices(
            top_indices=rng.randint(0, 8, size=n).astype(np.uint8),
            bottom_indices=rng.randint(0, 256, size=(n, p, l)).astype(np.uint8),
            n_patches=p,
            n_levels=l,
            top_n_codes=8,
            bottom_n_codes=256,
        )
        grouped = compress_indices(vq, mode="grouped")
        optimal = compress_indices(vq, mode="optimal")
        assert len(optimal) <= len(grouped)

    def test_delta_encode_decode_roundtrip(self):
        """Delta encoding is perfectly reversible."""
        rng = np.random.RandomState(789)
        bottom = rng.randint(0, 256, size=(10, 8, 4)).astype(np.uint8)
        encoded = _delta_encode_levels(bottom)
        decoded = _delta_decode_levels(encoded)
        assert np.array_equal(bottom, decoded)

    def test_delta_encode_correlated_data(self):
        """Delta coding produces smaller values for correlated patches."""
        # Create correlated data (adjacent patches similar)
        n, p, l = 32, 8, 4
        base = np.random.RandomState(42).randint(0, 256, size=(n, 1, l)).astype(np.uint8)
        noise = np.random.RandomState(42).randint(-3, 4, size=(n, p, l)).astype(np.int16)
        bottom = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        encoded = _delta_encode_levels(bottom)
        # Delta-encoded values should have more zeros and small values
        n_small = np.sum(encoded < 10)
        n_small_orig = np.sum(bottom < 10)
        assert n_small > n_small_orig


class TestMNFTiered:
    """Test .mnf format with tiered data block."""

    def test_mnf_write_read_tiered(self, model_pair, sample_windows):
        """MNF round-trip preserves tiered encoding data."""
        encoder, decoder = model_pair
        codec = TieredCodec(encoder, decoder, device="cpu", storage_mode="lossless")
        encoding = codec.encode(sample_windows, batch_size=4)
        tiered_bytes = serialize_tiered(encoding)

        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            tmp_path = f.name

        try:
            # Write
            n = encoding.n_windows
            writer = MNFWriter()
            writer.write(
                path=tmp_path,
                coordinates=np.zeros((n, 1), dtype=np.float16),
                chart_ids=np.zeros(n, dtype=np.uint8),
                decoder_hash=b"\x00" * 32,
                tiered_data=tiered_bytes,
            )

            # Read
            reader = MNFReader()
            mnf = reader.read(tmp_path)

            assert mnf.header.has_tiered
            assert mnf.tiered_data is not None
            assert mnf.tiered_data == tiered_bytes

            # Verify encoding round-trip through MNF
            encoding2 = deserialize_tiered(mnf.tiered_data)
            recovered = codec.decode_storage(encoding2)
            assert np.array_equal(
                sample_windows.view(np.uint32),
                recovered.view(np.uint32),
            )
        finally:
            os.unlink(tmp_path)

    def test_mnf_header_flags(self):
        """Tiered flag bit is correctly set in header."""
        header = MNFHeader(
            magic=b"MNF1",
            version=1,
            domain_id=0,
            n_samples=10,
            latent_dim=1,
            n_charts=1,
            coord_dtype=0,
            decoder_hash=b"\x00" * 32,
            flags=0x20,  # has_tiered only
        )
        assert header.has_tiered
        assert not header.has_index
        assert not header.has_confidence
        assert not header.has_vq_codes
        assert not header.has_residual

    def test_mnf_backward_compat(self):
        """MNF files without tiered block still read correctly."""
        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            tmp_path = f.name

        try:
            n = 5
            writer = MNFWriter()
            writer.write(
                path=tmp_path,
                coordinates=np.random.randn(n, 4).astype(np.float16),
                chart_ids=np.zeros(n, dtype=np.uint8),
                decoder_hash=b"\x00" * 32,
            )

            reader = MNFReader()
            mnf = reader.read(tmp_path)
            assert not mnf.header.has_tiered
            assert mnf.tiered_data is None
        finally:
            os.unlink(tmp_path)


class TestTieredManifoldCodec:
    """Test high-level TieredManifoldCodec API."""

    def test_encode_decode_mnf_lossless(self, model_pair, sample_windows, small_config):
        """Full encode -> .mnf -> decode round-trip (lossless)."""
        encoder, decoder = model_pair

        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            tmp_path = f.name

        try:
            codec = TieredManifoldCodec(
                encoder, decoder, small_config,
                device="cpu", storage_mode="lossless",
            )

            # Encode to .mnf
            bytes_written = codec.encode_to_mnf(sample_windows, tmp_path, batch_size=4)
            assert bytes_written > 0
            assert os.path.getsize(tmp_path) == bytes_written

            # Decode from .mnf (storage mode)
            recovered = codec.decode_from_mnf(tmp_path, mode="storage")
            assert np.array_equal(
                sample_windows.view(np.uint32),
                recovered.view(np.uint32),
            ), "Lossless storage decode through .mnf is not bit-exact"

            # Decode from .mnf (search mode)
            approx = codec.decode_from_mnf(tmp_path, mode="search", batch_size=4)
            assert approx.shape == sample_windows.shape
            assert approx.dtype == np.float32
        finally:
            os.unlink(tmp_path)

    def test_encode_decode_mnf_near_lossless(self, model_pair, sample_windows, small_config):
        """Near-lossless mode through .mnf."""
        encoder, decoder = model_pair

        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            tmp_path = f.name

        try:
            codec = TieredManifoldCodec(
                encoder, decoder, small_config,
                device="cpu", storage_mode="near_lossless",
            )

            codec.encode_to_mnf(sample_windows, tmp_path, batch_size=4)
            recovered = codec.decode_from_mnf(tmp_path, mode="storage")

            rmse = np.sqrt(np.mean((sample_windows - recovered) ** 2))
            data_range = sample_windows.max() - sample_windows.min()
            rmse_pct = rmse / data_range * 100
            assert rmse_pct < 0.1, f"Near-lossless RMSE {rmse_pct:.4f}% exceeds 0.1%"
        finally:
            os.unlink(tmp_path)

    def test_stats_from_mnf(self, model_pair, sample_windows, small_config):
        """Compression stats are computed correctly from .mnf file."""
        encoder, decoder = model_pair

        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            tmp_path = f.name

        try:
            codec = TieredManifoldCodec(
                encoder, decoder, small_config,
                device="cpu", storage_mode="lossless",
            )

            codec.encode_to_mnf(sample_windows, tmp_path, batch_size=4)
            stats = codec.stats_from_mnf(tmp_path)

            assert stats["raw_bytes"] == sample_windows.nbytes
            assert stats["vq_bytes"] > 0
            assert stats["storage_bytes"] > 0
            assert stats["storage_compression"] > 1.0
            assert stats["mnf_file_bytes"] > 0
            assert stats["mnf_compression"] > 0
            assert stats["storage_mode"] == "lossless"
        finally:
            os.unlink(tmp_path)

    def test_search_from_mnf(self, model_pair, sample_windows, small_config):
        """VQ search through .mnf finds reasonable matches."""
        encoder, decoder = model_pair

        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            tmp_path = f.name

        try:
            codec = TieredManifoldCodec(
                encoder, decoder, small_config,
                device="cpu", storage_mode="lossless",
            )

            codec.encode_to_mnf(sample_windows, tmp_path, batch_size=4)

            # Search with first 2 windows as queries
            query = sample_windows[:2]
            result = codec.search_from_mnf(tmp_path, query, k=3, batch_size=4)

            assert result.indices.shape == (2, 3)
            assert result.distances.shape == (2, 3)
            # First match for each query should be itself (distance ~0)
            assert result.indices[0, 0] == 0
            assert result.indices[1, 0] == 1
            assert result.distances[0, 0] < 1e-6
            assert result.distances[1, 0] < 1e-6
        finally:
            os.unlink(tmp_path)

    def test_from_checkpoint_format(self, model_pair, small_config):
        """TieredManifoldCodec loads from experiment checkpoint format."""
        encoder, decoder = model_pair

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        try:
            # Save in experiment checkpoint format
            torch.save({
                "config": small_config,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "epoch": 10,
                "val_mse": 0.5,
            }, tmp_path)

            # Load
            codec = TieredManifoldCodec.from_checkpoint(tmp_path)
            assert codec.config == small_config
        finally:
            os.unlink(tmp_path)

    def test_from_codec_format(self, model_pair, small_config):
        """TieredManifoldCodec loads from ManifoldCodec checkpoint format."""
        encoder, decoder = model_pair

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        try:
            # Save in ManifoldCodec format
            torch.save({
                "config": small_config,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
            }, tmp_path)

            # Load
            codec = TieredManifoldCodec.from_checkpoint(tmp_path)
            assert codec.config == small_config
        finally:
            os.unlink(tmp_path)
