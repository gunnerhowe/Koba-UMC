"""Tests for streaming chunk-based codec."""

import numpy as np
import pytest

from umc.codec.streaming import (
    StreamingStorageCodec,
    serialize_chunks,
    deserialize_chunks,
)


class TestStreamingCodec:
    """Test streaming storage codec."""

    def _make_windows(self, n=100, seed=42):
        rng = np.random.RandomState(seed)
        return rng.randn(n, 32, 5).astype(np.float32)

    def test_push_and_flush_lossless(self):
        """Push windows one at a time, verify lossless roundtrip."""
        windows = self._make_windows(100)
        codec = StreamingStorageCodec(chunk_size=32, mode="lossless")

        chunks = []
        for i in range(len(windows)):
            chunk = codec.push(windows[i])
            if chunk is not None:
                chunks.append(chunk)
        final = codec.flush()
        if final is not None:
            chunks.append(final)

        # Should have ceil(100/32) = 4 chunks (32+32+32+4)
        assert len(chunks) == 4

        decoded = StreamingStorageCodec.decode_all(chunks)
        np.testing.assert_array_equal(windows, decoded)

    def test_push_batch(self):
        """Push entire batch at once."""
        windows = self._make_windows(64)
        codec = StreamingStorageCodec(chunk_size=32, mode="lossless")

        chunks = codec.push_batch(windows)
        final = codec.flush()
        if final is not None:
            chunks.append(final)

        assert len(chunks) == 2
        decoded = StreamingStorageCodec.decode_all(chunks)
        np.testing.assert_array_equal(windows, decoded)

    def test_exact_chunk_boundary(self):
        """No flush needed when windows exactly fill chunks."""
        windows = self._make_windows(64)
        codec = StreamingStorageCodec(chunk_size=32, mode="lossless")

        chunks = codec.push_batch(windows)
        final = codec.flush()

        assert len(chunks) == 2
        assert final is None  # No leftover

        decoded = StreamingStorageCodec.decode_all(chunks)
        np.testing.assert_array_equal(windows, decoded)

    def test_single_window_chunks(self):
        """chunk_size=1 creates one chunk per window."""
        windows = self._make_windows(5)
        codec = StreamingStorageCodec(chunk_size=1, mode="lossless")

        chunks = codec.push_batch(windows)
        assert len(chunks) == 5

        decoded = StreamingStorageCodec.decode_all(chunks)
        np.testing.assert_array_equal(windows, decoded)

    def test_near_lossless_mode(self):
        """Near-lossless mode works in streaming."""
        windows = self._make_windows(50)
        codec = StreamingStorageCodec(chunk_size=25, mode="near_lossless")

        chunks = codec.push_batch(windows)
        assert len(chunks) == 2

        decoded = StreamingStorageCodec.decode_all(chunks)
        assert decoded.shape == windows.shape
        # Near-lossless: very close but not exact
        rmse = np.sqrt(np.mean((windows - decoded) ** 2))
        assert rmse < 0.1  # Should be very small

    def test_iter_decode(self):
        """iter_decode yields chunks one at a time."""
        windows = self._make_windows(96)
        codec = StreamingStorageCodec(chunk_size=32, mode="lossless")
        chunks = codec.push_batch(windows)

        decoded_parts = list(codec.iter_decode(chunks))
        assert len(decoded_parts) == 3
        for part in decoded_parts:
            assert part.shape == (32, 32, 5)

        full = np.concatenate(decoded_parts, axis=0)
        np.testing.assert_array_equal(windows, full)

    def test_reset(self):
        """Reset clears the buffer."""
        windows = self._make_windows(10)
        codec = StreamingStorageCodec(chunk_size=32, mode="lossless")

        for w in windows:
            codec.push(w)
        codec.reset()

        assert codec.flush() is None  # Buffer empty


class TestChunkSerialization:
    """Test chunk stream serialization."""

    def test_serialize_roundtrip(self):
        """Chunks survive serialization to/from a byte stream."""
        windows = np.random.RandomState(42).randn(100, 32, 5).astype(np.float32)
        codec = StreamingStorageCodec(chunk_size=25, mode="lossless")

        chunks = codec.push_batch(windows)
        final = codec.flush()
        if final:
            chunks.append(final)

        # Serialize -> bytes -> deserialize
        stream = serialize_chunks(chunks)
        restored = deserialize_chunks(stream)

        assert len(restored) == len(chunks)
        for orig, rest in zip(chunks, restored):
            assert orig == rest

        # Full roundtrip
        decoded = StreamingStorageCodec.decode_all(restored)
        np.testing.assert_array_equal(windows, decoded)

    def test_invalid_magic(self):
        """Invalid stream magic raises ValueError."""
        with pytest.raises(ValueError, match="Invalid stream magic"):
            deserialize_chunks(b"XXXX" + b"\x00" * 10)
