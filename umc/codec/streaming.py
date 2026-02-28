"""Streaming chunk-based codec for real-time compression.

Processes windows in fixed-size chunks, allowing incremental encoding
without buffering the entire dataset. Slightly worse compression ratio
than full-batch (~5-10%) due to smaller compressor context, but enables
real-time pipelines.

Usage:
    codec = StreamingStorageCodec(chunk_size=64, mode='lossless')

    # Encode streaming
    for window in window_stream:
        chunk = codec.push(window)
        if chunk is not None:
            send(chunk)
    final = codec.flush()
    if final:
        send(final)

    # Decode
    for windows in codec.iter_decode(all_chunks):
        process(windows)
"""

import struct
from typing import Iterator, List, Optional

import numpy as np

from .tiered import _compress_storage, _decompress_storage


class StreamingStorageCodec:
    """Chunk-based streaming codec wrapping the storage tier.

    Buffers incoming windows and compresses when buffer reaches chunk_size.
    Each chunk is independently decompressible.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        mode: str = "lossless",
    ):
        """Initialize streaming codec.

        Args:
            chunk_size: Number of windows per chunk.
            mode: Storage compression mode ('lossless', 'near_lossless',
                  'lossless_zstd', 'lossless_lzma', 'normalized_lossless',
                  'normalized_lossless_zstd').
        """
        self.chunk_size = chunk_size
        self.mode = mode
        self._buffer: List[np.ndarray] = []

    def push(self, window: np.ndarray) -> Optional[bytes]:
        """Add a window to the buffer.

        Args:
            window: Single window array (window_size, n_features) float32.

        Returns:
            Compressed chunk bytes when buffer reaches chunk_size, else None.
        """
        self._buffer.append(window.astype(np.float32))
        if len(self._buffer) >= self.chunk_size:
            return self._flush_buffer()
        return None

    def push_batch(self, windows: np.ndarray) -> List[bytes]:
        """Add multiple windows, returning any completed chunks.

        Args:
            windows: (n_windows, window_size, n_features) float32.

        Returns:
            List of compressed chunk bytes (may be empty).
        """
        chunks = []
        for i in range(len(windows)):
            result = self.push(windows[i])
            if result is not None:
                chunks.append(result)
        return chunks

    def flush(self) -> Optional[bytes]:
        """Compress any remaining buffered windows.

        Returns:
            Compressed chunk bytes if buffer is non-empty, else None.
        """
        if len(self._buffer) > 0:
            return self._flush_buffer()
        return None

    def _flush_buffer(self) -> bytes:
        """Compress current buffer and clear it."""
        stacked = np.stack(self._buffer, axis=0)
        self._buffer = []
        compressed = _compress_storage(stacked, self.mode)
        # Wrap in chunk envelope: magic + chunk_len + compressed
        chunk = struct.pack("<4sI", b"UMS1", len(compressed)) + compressed
        return chunk

    @staticmethod
    def decode_chunk(chunk: bytes) -> np.ndarray:
        """Decode a single chunk back to windows.

        Args:
            chunk: Compressed chunk from push() or flush().

        Returns:
            (n_windows, window_size, n_features) float32.
        """
        magic = chunk[:4]
        if magic != b"UMS1":
            raise ValueError(f"Invalid chunk magic: {magic!r}")
        length = struct.unpack("<I", chunk[4:8])[0]
        return _decompress_storage(chunk[8:8 + length])

    @staticmethod
    def iter_decode(chunks: List[bytes]) -> Iterator[np.ndarray]:
        """Iterate over chunks, yielding decoded window arrays.

        Args:
            chunks: List of compressed chunk bytes.

        Yields:
            (n_windows, window_size, n_features) float32 per chunk.
        """
        for chunk in chunks:
            yield StreamingStorageCodec.decode_chunk(chunk)

    @staticmethod
    def decode_all(chunks: List[bytes]) -> np.ndarray:
        """Decode all chunks and concatenate into a single array.

        Args:
            chunks: List of compressed chunk bytes.

        Returns:
            (total_windows, window_size, n_features) float32.
        """
        arrays = [StreamingStorageCodec.decode_chunk(c) for c in chunks]
        return np.concatenate(arrays, axis=0)

    def reset(self):
        """Clear the internal buffer."""
        self._buffer = []


def serialize_chunks(chunks: List[bytes]) -> bytes:
    """Serialize a list of chunks into a single byte stream.

    Format:
        magic: b'UMSS' (UMC Streaming Storage)
        n_chunks: uint32
        For each chunk:
            chunk_len: uint32
            chunk_data: bytes
    """
    parts = [struct.pack("<4sI", b"UMSS", len(chunks))]
    for chunk in chunks:
        parts.append(struct.pack("<I", len(chunk)))
        parts.append(chunk)
    return b"".join(parts)


def deserialize_chunks(data: bytes) -> List[bytes]:
    """Deserialize a byte stream into a list of chunks."""
    magic = data[:4]
    if magic != b"UMSS":
        raise ValueError(f"Invalid stream magic: {magic!r}")
    n_chunks = struct.unpack("<I", data[4:8])[0]
    offset = 8
    chunks = []
    for _ in range(n_chunks):
        chunk_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        chunks.append(data[offset:offset + chunk_len])
        offset += chunk_len
    return chunks
