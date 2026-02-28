"""Tests for float16 and bfloat16 compression support."""

import numpy as np
import pytest

import umc


class TestFloat16Lossless:
    """Float16 round-trip tests for lossless modes."""

    def _make_f16(self, shape):
        rng = np.random.default_rng(42)
        return rng.standard_normal(shape).astype(np.float16)

    def test_lossless_roundtrip(self):
        data = self._make_f16((10, 32, 5))
        compressed = umc.compress(data, mode="lossless")
        recovered = umc.decompress(compressed)
        assert recovered.dtype == np.float16
        assert np.array_equal(data, recovered)

    def test_lossless_1d(self):
        data = self._make_f16((128,))
        compressed = umc.compress(data, mode="lossless")
        recovered = umc.decompress(compressed)
        assert recovered.dtype == np.float16
        assert recovered.shape == (128,)
        assert np.array_equal(data, recovered)

    def test_lossless_2d(self):
        data = self._make_f16((50, 128))
        compressed = umc.compress(data, mode="lossless")
        recovered = umc.decompress(compressed)
        assert recovered.dtype == np.float16
        assert recovered.shape == (50, 128)
        assert np.array_equal(data, recovered)

    def test_lossless_fast_roundtrip(self):
        data = self._make_f16((10, 32, 5))
        compressed = umc.compress(data, mode="lossless_fast")
        recovered = umc.decompress(compressed)
        assert recovered.dtype == np.float16
        assert np.array_equal(data, recovered)

    def test_lossless_zstd_roundtrip(self):
        data = self._make_f16((10, 32, 5))
        compressed = umc.compress(data, mode="lossless_zstd")
        recovered = umc.decompress(compressed)
        assert recovered.dtype == np.float16
        assert np.array_equal(data, recovered)

    def test_lossless_lzma_roundtrip(self):
        data = self._make_f16((10, 32, 5))
        compressed = umc.compress(data, mode="lossless_lzma")
        recovered = umc.decompress(compressed)
        assert recovered.dtype == np.float16
        assert np.array_equal(data, recovered)

    def test_optimal_roundtrip(self):
        data = self._make_f16((5, 16, 3))
        compressed = umc.compress(data, mode="optimal")
        recovered = umc.decompress(compressed)
        assert recovered.dtype == np.float16
        assert np.array_equal(data, recovered)

    def test_optimal_fast_roundtrip(self):
        data = self._make_f16((5, 16, 3))
        compressed = umc.compress(data, mode="optimal_fast")
        recovered = umc.decompress(compressed)
        assert recovered.dtype == np.float16
        assert np.array_equal(data, recovered)

    def test_smaller_than_float32(self):
        """Float16 compressed should be smaller than same data as float32."""
        data_f16 = self._make_f16((100, 64, 8))
        data_f32 = data_f16.astype(np.float32)

        comp_f16 = umc.compress(data_f16, mode="lossless")
        comp_f32 = umc.compress(data_f32, mode="lossless")

        assert len(comp_f16) < len(comp_f32)

    def test_lossy_mode_raises(self):
        """Lossy modes should raise ValueError for float16 input."""
        data = self._make_f16((10, 32, 5))
        with pytest.raises(ValueError, match="requires float32"):
            umc.compress(data, mode="near_lossless")
        with pytest.raises(ValueError, match="requires float32"):
            umc.compress(data, mode="quantized_8")

    def test_umc2_magic(self):
        """Float16 compressed data should use UMC2 magic."""
        data = self._make_f16((10, 32, 5))
        compressed = umc.compress(data, mode="lossless")
        assert compressed[:4] == b"UMC2"

    def test_float32_backward_compat(self):
        """Float32 should still use UMCZ magic for backward compatibility."""
        data = np.random.randn(10, 32, 5).astype(np.float32)
        compressed = umc.compress(data, mode="lossless")
        assert compressed[:4] == b"UMCZ"


class TestBfloat16:
    """Bfloat16 compression tests (requires torch)."""

    @pytest.fixture(autouse=True)
    def _check_torch(self):
        pytest.importorskip("torch")

    def test_bfloat16_roundtrip(self):
        import torch
        data = torch.randn(10, 32, 5, dtype=torch.bfloat16)
        compressed = umc.compress(data, mode="lossless")
        recovered = umc.decompress(compressed)
        # bfloat16 -> float32 on decompress (lossless widening)
        assert recovered.dtype == np.float32
        # Convert back to bfloat16 to verify exact bits
        recovered_bf16 = torch.from_numpy(recovered).to(torch.bfloat16)
        assert torch.equal(data, recovered_bf16)

    def test_bfloat16_lossless_fast(self):
        import torch
        data = torch.randn(10, 32, 5, dtype=torch.bfloat16)
        compressed = umc.compress(data, mode="lossless_fast")
        recovered = umc.decompress(compressed)
        recovered_bf16 = torch.from_numpy(recovered).to(torch.bfloat16)
        assert torch.equal(data, recovered_bf16)

    def test_bfloat16_umc2_magic(self):
        import torch
        data = torch.randn(10, 32, 5, dtype=torch.bfloat16)
        compressed = umc.compress(data, mode="lossless")
        assert compressed[:4] == b"UMC2"

    def test_bfloat16_2d_embeddings(self):
        """Typical embedding use case: (n_vectors, dim) bfloat16."""
        import torch
        data = torch.randn(1000, 768, dtype=torch.bfloat16)
        compressed = umc.compress(data, mode="lossless")
        recovered = umc.decompress(compressed)
        recovered_bf16 = torch.from_numpy(recovered).to(torch.bfloat16)
        assert torch.equal(data, recovered_bf16)
        # Should be meaningfully compressed
        raw_size = data.numel() * 2  # 2 bytes per bfloat16
        assert len(compressed) < raw_size


class TestFloat16Embeddings:
    """Real-world embedding compression scenarios."""

    def test_embedding_batch(self):
        """Compress a batch of float16 embedding vectors."""
        rng = np.random.default_rng(123)
        # Typical: 1000 vectors of dimension 384 (sentence-transformers)
        embeddings = rng.standard_normal((1000, 384)).astype(np.float16)
        compressed = umc.compress(embeddings, mode="lossless")
        recovered = umc.decompress(compressed)
        assert recovered.dtype == np.float16
        assert np.array_equal(embeddings, recovered)

    def test_normalized_embeddings(self):
        """Normalized embeddings (unit vectors) should compress well."""
        rng = np.random.default_rng(456)
        raw = rng.standard_normal((500, 1536)).astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        normalized = (raw / norms).astype(np.float16)
        compressed = umc.compress(normalized, mode="lossless")
        recovered = umc.decompress(compressed)
        assert np.array_equal(normalized, recovered)
