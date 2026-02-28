"""Tests for .mnf file format and storage layer."""

import os
import tempfile

import numpy as np
import pytest

from umc.storage.mnf_format import MNFWriter, MNFReader, MNFHeader, MAGIC
from umc.storage.entropy import (
    VQIndices, compress_indices, decompress_indices,
    measure_entropy, compute_compression_stats,
)


@pytest.fixture
def sample_data():
    """Generate sample manifold data for storage tests."""
    rng = np.random.RandomState(42)
    n_samples = 100
    latent_dim = 32

    return {
        "coordinates": rng.randn(n_samples, latent_dim).astype(np.float32),
        "chart_ids": rng.randint(0, 8, n_samples).astype(np.uint8),
        "confidences": rng.rand(n_samples).astype(np.float32),
        "decoder_hash": b"\xab" * 32,
    }


class TestMNFFormat:
    def test_write_read_roundtrip(self, sample_data):
        """Write then read produces byte-exact coordinates."""
        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            path = f.name

        try:
            writer = MNFWriter()
            writer.write(
                path=path,
                coordinates=sample_data["coordinates"],
                chart_ids=sample_data["chart_ids"],
                decoder_hash=sample_data["decoder_hash"],
                confidences=sample_data["confidences"],
                coord_dtype="float16",
            )

            reader = MNFReader()
            mnf = reader.read(path)

            # Coordinates roundtrip (float16 precision)
            expected = sample_data["coordinates"].astype(np.float16)
            assert np.array_equal(mnf.coordinates, expected)

            # Chart IDs exact
            assert np.array_equal(mnf.chart_ids, sample_data["chart_ids"])

            # Confidences (float16 precision)
            expected_conf = sample_data["confidences"].astype(np.float16)
            assert np.array_equal(mnf.confidences, expected_conf)

        finally:
            os.unlink(path)

    def test_header_fields(self, sample_data):
        """Header contains correct metadata."""
        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            path = f.name

        try:
            writer = MNFWriter()
            writer.write(
                path=path,
                coordinates=sample_data["coordinates"],
                chart_ids=sample_data["chart_ids"],
                decoder_hash=sample_data["decoder_hash"],
                domain_id=0,
                version=1,
            )

            reader = MNFReader()
            header = reader.read_header(path)

            assert header.magic == MAGIC
            assert header.version == 1
            assert header.domain_id == 0
            assert header.n_samples == 100
            assert header.latent_dim == 32
            assert header.decoder_hash == sample_data["decoder_hash"]

        finally:
            os.unlink(path)

    def test_float32_dtype(self, sample_data):
        """Float32 coordinate storage works."""
        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            path = f.name

        try:
            writer = MNFWriter()
            writer.write(
                path=path,
                coordinates=sample_data["coordinates"],
                chart_ids=sample_data["chart_ids"],
                decoder_hash=sample_data["decoder_hash"],
                coord_dtype="float32",
            )

            reader = MNFReader()
            mnf = reader.read(path)

            assert mnf.coordinates.dtype == np.float32
            assert np.allclose(mnf.coordinates, sample_data["coordinates"])

        finally:
            os.unlink(path)

    def test_read_coordinates_with_indices(self, sample_data):
        """Selective coordinate reading works."""
        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            path = f.name

        try:
            writer = MNFWriter()
            writer.write(
                path=path,
                coordinates=sample_data["coordinates"],
                chart_ids=sample_data["chart_ids"],
                decoder_hash=sample_data["decoder_hash"],
            )

            reader = MNFReader()
            indices = np.array([0, 5, 10, 50])
            coords = reader.read_coordinates(path, indices=indices)

            assert coords.shape == (4, 32)

        finally:
            os.unlink(path)

    def test_no_confidence_flag(self, sample_data):
        """File without confidence block works."""
        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            path = f.name

        try:
            writer = MNFWriter()
            writer.write(
                path=path,
                coordinates=sample_data["coordinates"],
                chart_ids=sample_data["chart_ids"],
                decoder_hash=sample_data["decoder_hash"],
                confidences=None,
            )

            reader = MNFReader()
            mnf = reader.read(path)

            assert mnf.confidences is None
            assert not mnf.header.has_confidence

        finally:
            os.unlink(path)

    def test_write_read_with_scale_factors(self, sample_data):
        """Scale factors survive .mnf roundtrip."""
        rng = np.random.RandomState(123)
        n_features = 5
        n = sample_data["coordinates"].shape[0]
        means = rng.randn(n, n_features).astype(np.float32)
        stds = np.abs(rng.randn(n, n_features)).astype(np.float32) + 0.1

        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            path = f.name

        try:
            writer = MNFWriter()
            writer.write(
                path=path,
                coordinates=sample_data["coordinates"],
                chart_ids=sample_data["chart_ids"],
                decoder_hash=sample_data["decoder_hash"],
                confidences=sample_data["confidences"],
                scale_means=means,
                scale_stds=stds,
            )

            reader = MNFReader()
            mnf = reader.read(path)

            assert mnf.header.has_scale_factors
            expected_means = means.astype(np.float16)
            expected_stds = stds.astype(np.float16)
            assert np.array_equal(mnf.scale_means, expected_means)
            assert np.array_equal(mnf.scale_stds, expected_stds)
            assert mnf.scale_means.shape == (n, n_features)
            assert mnf.scale_stds.shape == (n, n_features)

            # Coordinates and other data still correct
            assert np.array_equal(mnf.chart_ids, sample_data["chart_ids"])
            assert mnf.header.n_samples == n

        finally:
            os.unlink(path)

    def test_backward_compatible_without_scale_factors(self, sample_data):
        """Files without scale factors still read correctly."""
        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            path = f.name

        try:
            writer = MNFWriter()
            writer.write(
                path=path,
                coordinates=sample_data["coordinates"],
                chart_ids=sample_data["chart_ids"],
                decoder_hash=sample_data["decoder_hash"],
            )

            reader = MNFReader()
            mnf = reader.read(path)

            assert not mnf.header.has_scale_factors
            assert mnf.scale_means is None
            assert mnf.scale_stds is None

        finally:
            os.unlink(path)

    def test_write_read_with_vq_codes(self, sample_data):
        """VQ code data survives .mnf roundtrip."""
        # Simulate compressed VQ indices
        vq_code_data = b"\x01\x02\x03\x04" * 50  # 200 bytes of fake compressed data

        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            path = f.name

        try:
            writer = MNFWriter()
            writer.write(
                path=path,
                coordinates=sample_data["coordinates"],
                chart_ids=sample_data["chart_ids"],
                decoder_hash=sample_data["decoder_hash"],
                confidences=sample_data["confidences"],
                vq_code_data=vq_code_data,
            )

            reader = MNFReader()
            mnf = reader.read(path)

            assert mnf.header.has_vq_codes
            assert mnf.vq_code_data == vq_code_data

            # Other fields still correct
            expected = sample_data["coordinates"].astype(np.float16)
            assert np.array_equal(mnf.coordinates, expected)
            assert np.array_equal(mnf.chart_ids, sample_data["chart_ids"])

        finally:
            os.unlink(path)

    def test_write_read_all_optional_blocks(self, sample_data):
        """All optional blocks (confidence + scale + VQ + index) roundtrip."""
        rng = np.random.RandomState(99)
        n = sample_data["coordinates"].shape[0]
        means = rng.randn(n, 5).astype(np.float32)
        stds = np.abs(rng.randn(n, 5)).astype(np.float32) + 0.1
        vq_code_data = b"\xAB\xCD" * 100
        index_data = b"\xFF" * 256

        with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
            path = f.name

        try:
            writer = MNFWriter()
            writer.write(
                path=path,
                coordinates=sample_data["coordinates"],
                chart_ids=sample_data["chart_ids"],
                decoder_hash=sample_data["decoder_hash"],
                confidences=sample_data["confidences"],
                scale_means=means,
                scale_stds=stds,
                vq_code_data=vq_code_data,
                index_data=index_data,
            )

            reader = MNFReader()
            mnf = reader.read(path)

            assert mnf.header.has_confidence
            assert mnf.header.has_scale_factors
            assert mnf.header.has_vq_codes
            assert mnf.header.has_index
            assert mnf.vq_code_data == vq_code_data
            assert mnf.index_data == index_data
            assert mnf.scale_means.shape == (n, 5)

        finally:
            os.unlink(path)

    def test_file_size_proportional(self):
        """File size scales with n_samples and latent_dim."""
        rng = np.random.RandomState(42)

        sizes = {}
        for n in [100, 1000]:
            coords = rng.randn(n, 32).astype(np.float32)
            chart_ids = rng.randint(0, 4, n).astype(np.uint8)

            with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
                path = f.name

            writer = MNFWriter()
            writer.write(path, coords, chart_ids, b"\x00" * 32)
            sizes[n] = os.path.getsize(path)
            os.unlink(path)

        # 10x more samples should give roughly 10x larger file
        ratio = sizes[1000] / sizes[100]
        assert 8 < ratio < 12  # Allow some overhead for header


class TestEntropyCoding:
    def test_compress_decompress_roundtrip(self):
        """Compressed VQ indices decompress to original values."""
        rng = np.random.RandomState(42)
        n_windows = 200
        n_patches = 16
        n_levels = 3

        vq_idx = VQIndices(
            top_indices=rng.randint(0, 16, n_windows).astype(np.uint8),
            bottom_indices=rng.randint(0, 256, (n_windows, n_patches, n_levels)).astype(np.uint8),
            n_patches=n_patches,
            n_levels=n_levels,
            top_n_codes=16,
            bottom_n_codes=256,
        )

        compressed = compress_indices(vq_idx)
        recovered = decompress_indices(compressed)

        assert np.array_equal(recovered.top_indices, vq_idx.top_indices)
        assert np.array_equal(recovered.bottom_indices, vq_idx.bottom_indices)
        assert recovered.n_patches == n_patches
        assert recovered.n_levels == n_levels
        assert recovered.top_n_codes == 16
        assert recovered.bottom_n_codes == 256

    def test_compression_reduces_size(self):
        """Compressed size is smaller than raw size for realistic VQ indices."""
        rng = np.random.RandomState(42)
        n_windows = 500
        n_patches = 16
        n_levels = 8

        # Realistic: non-uniform distributions (VQ codebooks have concentrated usage)
        top_indices = rng.choice(16, n_windows, p=np.array([0.3, 0.2, 0.1] + [0.4/13]*13)).astype(np.uint8)
        # Bottom: use 128 codes with ~50% utilization (realistic for trained RVQ)
        active_codes = rng.choice(128, 64, replace=False)
        bottom_indices = rng.choice(
            active_codes, (n_windows, n_patches, n_levels)
        ).astype(np.uint8)

        vq_idx = VQIndices(
            top_indices=top_indices,
            bottom_indices=bottom_indices,
            n_patches=n_patches,
            n_levels=n_levels,
            top_n_codes=16,
            bottom_n_codes=128,
        )

        compressed = compress_indices(vq_idx)
        raw_size = n_windows + n_windows * n_patches * n_levels
        assert len(compressed) < raw_size

    def test_grouped_vs_flat_roundtrip(self):
        """Both compression modes round-trip correctly."""
        rng = np.random.RandomState(42)
        vq_idx = VQIndices(
            top_indices=rng.randint(0, 16, 100).astype(np.uint8),
            bottom_indices=rng.randint(0, 128, (100, 8, 3)).astype(np.uint8),
            n_patches=8,
            n_levels=3,
            top_n_codes=16,
            bottom_n_codes=128,
        )

        for mode in ["flat", "grouped"]:
            compressed = compress_indices(vq_idx, mode=mode)
            recovered = decompress_indices(compressed)
            assert np.array_equal(recovered.top_indices, vq_idx.top_indices), f"{mode} top mismatch"
            assert np.array_equal(recovered.bottom_indices, vq_idx.bottom_indices), f"{mode} bottom mismatch"

    def test_measure_entropy_uniform(self):
        """Uniform distribution has maximum entropy."""
        n_codes = 256
        indices = np.arange(n_codes).astype(np.uint8)
        ent = measure_entropy(indices, n_codes)
        # Uniform over 256 codes = 8 bits
        assert abs(ent - 8.0) < 0.01

    def test_measure_entropy_concentrated(self):
        """Concentrated distribution has low entropy."""
        indices = np.zeros(1000, dtype=np.uint8)  # All same code
        ent = measure_entropy(indices, 256)
        assert ent < 0.01

    def test_compression_stats(self):
        """Compression stats dict has expected keys."""
        rng = np.random.RandomState(42)
        vq_idx = VQIndices(
            top_indices=rng.randint(0, 16, 100).astype(np.uint8),
            bottom_indices=rng.randint(0, 256, (100, 8, 2)).astype(np.uint8),
            n_patches=8,
            n_levels=2,
            top_n_codes=16,
            bottom_n_codes=256,
        )

        stats = compute_compression_stats(vq_idx)
        assert "n_windows" in stats
        assert "best_compression" in stats
        assert "entropy_bytes_per_window" in stats
        assert "optimal_size" in stats
        assert stats["n_windows"] == 100
        assert stats["best_compression"] > 1.0
