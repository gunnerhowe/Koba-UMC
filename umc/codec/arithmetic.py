"""Arithmetic coding backends for XOR residual compression.

Phase 3 of the UMC compression pipeline. Replaces zlib with arithmetic
coding using learned or adaptive probability models, approaching the
Shannon entropy limit.

Three compression backends, in order of sophistication:

1. StaticByteCompressor:   Per-channel byte frequency tables (2-pass, no training)
2. AdaptiveByteCompressor: Online adaptive model (1-pass, no training)
3. NeuralByteCompressor:   Trained neural model for byte prediction (requires training)

All backends compress byte-transposed XOR residuals. After byte transposition,
float32 data is split into 4 channels (byte 0 of every element, byte 1, etc.).
Channels 0-1 (exponent bytes) are heavily skewed toward zero for similar floats,
compressing extremely well with arithmetic coding.

Usage:
    compressor = StaticByteCompressor()
    compressed = compressor.compress(xor_residual_bytes, element_size=4)
    recovered  = compressor.decompress(compressed, element_size=4)
"""

import struct
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

try:
    import constriction
    HAS_CONSTRICTION = True
except ImportError:
    HAS_CONSTRICTION = False


# ---- Base class ----

class ByteCompressor(ABC):
    """Base class for byte-level arithmetic compressors."""

    @abstractmethod
    def compress(self, data: bytes, element_size: int = 4) -> bytes:
        """Compress byte data using arithmetic coding.

        Args:
            data: Raw bytes (typically byte-transposed XOR residual).
            element_size: Bytes per element (4 for float32). Used to
                define channels in byte-transposed data.

        Returns:
            Compressed bytes with header for decompression.
        """

    @abstractmethod
    def decompress(self, data: bytes, element_size: int = 4) -> bytes:
        """Decompress bytes back to original data.

        Args:
            data: Compressed bytes from compress().
            element_size: Must match the value used in compress().

        Returns:
            Original byte data.
        """


# ---- Static per-channel compressor ----

class StaticByteCompressor(ByteCompressor):
    """Per-channel static arithmetic coding (2-pass, no training).

    First pass: collect byte frequency tables per channel.
    Second pass: encode using these static probabilities.

    For byte-transposed float32 XOR residuals:
        - Channel 0 (MSB/exponent): mostly 0x00 → near-zero entropy
        - Channel 1 (mid exponent): low entropy
        - Channel 2 (high mantissa): moderate entropy
        - Channel 3 (LSB mantissa): high entropy (~7-8 bits)

    The frequency tables are stored in the header for decompression.
    Total overhead: ~256 bytes per channel (4 channels × 256 × 2-byte counts
    = 2048 bytes). Negligible for typical residual sizes (>10KB).

    Compression format:
        magic: bytes[4] = b'AC01'
        n_channels: uint8
        total_symbols: uint32  (per channel)
        per channel:
            freq_table: 256 × uint16  (normalized counts, sum=65535)
        payload: constriction Range-coded symbols
    """

    MAGIC = b"AC01"

    def compress(self, data: bytes, element_size: int = 4) -> bytes:
        if not HAS_CONSTRICTION:
            raise RuntimeError("constriction library required: pip install constriction")

        if len(data) == 0:
            return self.MAGIC + struct.pack("<BI", element_size, 0)

        n_channels = element_size
        symbols_per_channel = len(data) // n_channels

        # Split into channels (already byte-transposed: ch0 ch0 ch0... ch1 ch1 ch1...)
        channels = []
        for ch in range(n_channels):
            start = ch * symbols_per_channel
            end = start + symbols_per_channel
            channels.append(np.frombuffer(data[start:end], dtype=np.uint8).copy())

        # Build frequency tables and encode each channel
        header = self.MAGIC + struct.pack("<BI", n_channels, symbols_per_channel)
        all_compressed = []

        for ch_idx, ch_data in enumerate(channels):
            # Compute frequencies
            freq = np.bincount(ch_data, minlength=256).astype(np.float64)

            # Ensure no zero probabilities (Laplace smoothing)
            freq = freq + 1.0

            # Quantize frequencies for storage (uint16, sum=65535)
            quantized_freq = self._quantize_freqs(freq)
            header += quantized_freq.tobytes()

            # CRITICAL: Use quantized frequencies for encoding too,
            # so encode and decode use exactly the same probabilities.
            probs = self._dequantize_freqs(quantized_freq)

            # Encode with constriction
            symbols = ch_data.astype(np.int32)
            # Broadcast single distribution to all symbols
            probs_broadcast = np.tile(probs, (len(symbols), 1))

            encoder = constriction.stream.queue.RangeEncoder()
            model = constriction.stream.model.Categorical(perfect=False)
            encoder.encode(symbols, model, probs_broadcast)

            compressed_words = encoder.get_compressed()
            compressed_bytes = compressed_words.tobytes()

            # Store length + data
            all_compressed.append(struct.pack("<I", len(compressed_bytes)))
            all_compressed.append(compressed_bytes)

        return header + b"".join(all_compressed)

    def decompress(self, data: bytes, element_size: int = 4) -> bytes:
        if not HAS_CONSTRICTION:
            raise RuntimeError("constriction library required: pip install constriction")

        offset = 0

        # Parse header
        magic = data[offset:offset + 4]
        offset += 4
        if magic != self.MAGIC:
            raise ValueError(f"Invalid arithmetic coding magic: {magic!r}")

        n_channels = struct.unpack_from("<B", data, offset)[0]
        offset += 1
        symbols_per_channel = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        if symbols_per_channel == 0:
            return b""

        # Read frequency tables
        freq_tables = []
        for _ in range(n_channels):
            quantized = np.frombuffer(
                data[offset:offset + 256 * 2], dtype=np.uint16
            ).copy()
            offset += 256 * 2
            freq_tables.append(quantized)

        # Decode each channel
        all_channels = []
        for ch_idx in range(n_channels):
            # Reconstruct probabilities from quantized frequencies
            probs = self._dequantize_freqs(freq_tables[ch_idx])
            probs_broadcast = np.tile(probs, (symbols_per_channel, 1))

            # Read compressed data
            comp_size = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            compressed_bytes = data[offset:offset + comp_size]
            offset += comp_size

            # Decode
            compressed_words = np.frombuffer(compressed_bytes, dtype=np.uint32).copy()
            decoder = constriction.stream.queue.RangeDecoder(compressed_words)
            model = constriction.stream.model.Categorical(perfect=False)
            symbols = decoder.decode(model, probs_broadcast)

            all_channels.append(symbols.astype(np.uint8).tobytes())

        return b"".join(all_channels)

    @staticmethod
    def _quantize_freqs(freq: np.ndarray) -> np.ndarray:
        """Quantize frequency table to uint16 (sum=65535) for storage."""
        total = freq.sum()
        if total == 0:
            result = np.ones(256, dtype=np.uint16)
            result[0] = 65535 - 255
            return result

        # Scale to sum=65535
        scaled = (freq / total * 65535).astype(np.float64)

        # Floor and distribute remainder
        result = np.floor(scaled).astype(np.uint16)
        # Ensure minimum count of 1 for every symbol
        result = np.maximum(result, np.uint16(1))

        # Adjust to hit target sum
        current_sum = result.sum()
        target = 65535
        diff = target - int(current_sum)

        if diff > 0:
            # Add to highest-frequency symbols
            indices = np.argsort(freq)[::-1]
            for i in range(min(diff, 256)):
                result[indices[i % 256]] += 1
        elif diff < 0:
            # Subtract from highest-frequency symbols (but keep >= 1)
            indices = np.argsort(freq)[::-1]
            for i in range(-diff):
                idx = indices[i % 256]
                if result[idx] > 1:
                    result[idx] -= 1

        return result

    @staticmethod
    def _dequantize_freqs(quantized: np.ndarray) -> np.ndarray:
        """Convert quantized uint16 frequencies back to float32 probabilities."""
        freq = quantized.astype(np.float32)
        # Ensure no zeros
        freq = np.maximum(freq, 1e-10)
        return freq / freq.sum()


# ---- Adaptive compressor ----

class AdaptiveByteCompressor(ByteCompressor):
    """Adaptive arithmetic coding with online probability updates.

    Uses a simple order-0 adaptive model: byte frequencies are updated
    as each symbol is encoded/decoded. No training required, single-pass
    over the data.

    The initial distribution is uniform. As symbols are processed, the
    model adapts to the actual data distribution. This handles non-stationary
    data better than the static model.

    Since constriction requires all probabilities up-front for batch encoding,
    we pre-compute the adaptive distributions by simulating the update process,
    then encode in a single batch call.

    Format:
        magic: bytes[4] = b'AC02'
        n_channels: uint8
        total_symbols: uint32 (per channel)
        per channel:
            compressed_size: uint32
            compressed_bytes: Range-coded data
    """

    MAGIC = b"AC02"

    def __init__(self, adaptation_rate: float = 1.0):
        """
        Args:
            adaptation_rate: How fast to adapt (1.0 = standard counting,
                higher = faster adaptation). Multiplier on count updates.
        """
        self.adaptation_rate = adaptation_rate

    def _build_adaptive_probs(
        self, symbols: np.ndarray, alphabet_size: int = 256
    ) -> np.ndarray:
        """Build per-symbol probability arrays using adaptive counting.

        Simulates the adaptive model: for symbol i, the probability is
        based on the frequency table after processing symbols 0..i-1.

        Returns:
            (n_symbols, alphabet_size) float32 array of probabilities.
        """
        n = len(symbols)
        # Start with Laplace-smoothed uniform
        counts = np.ones(alphabet_size, dtype=np.float64)
        probs = np.empty((n, alphabet_size), dtype=np.float32)

        for i in range(n):
            total = counts.sum()
            probs[i] = (counts / total).astype(np.float32)
            # Update counts after this symbol
            counts[symbols[i]] += self.adaptation_rate

        return probs

    def compress(self, data: bytes, element_size: int = 4) -> bytes:
        if not HAS_CONSTRICTION:
            raise RuntimeError("constriction library required: pip install constriction")

        if len(data) == 0:
            return self.MAGIC + struct.pack("<BI", element_size, 0)

        n_channels = element_size
        symbols_per_channel = len(data) // n_channels

        header = self.MAGIC + struct.pack("<BI", n_channels, symbols_per_channel)
        all_compressed = []

        for ch in range(n_channels):
            start = ch * symbols_per_channel
            end = start + symbols_per_channel
            ch_data = np.frombuffer(data[start:end], dtype=np.uint8).copy()
            symbols = ch_data.astype(np.int32)

            # Build adaptive probability tables
            probs = self._build_adaptive_probs(symbols)

            # Encode
            encoder = constriction.stream.queue.RangeEncoder()
            model = constriction.stream.model.Categorical(perfect=False)
            encoder.encode(symbols, model, probs)

            compressed_words = encoder.get_compressed()
            compressed_bytes = compressed_words.tobytes()

            all_compressed.append(struct.pack("<I", len(compressed_bytes)))
            all_compressed.append(compressed_bytes)

        return header + b"".join(all_compressed)

    def decompress(self, data: bytes, element_size: int = 4) -> bytes:
        if not HAS_CONSTRICTION:
            raise RuntimeError("constriction library required: pip install constriction")

        offset = 0
        magic = data[offset:offset + 4]
        offset += 4
        if magic != self.MAGIC:
            raise ValueError(f"Invalid adaptive coding magic: {magic!r}")

        n_channels = struct.unpack_from("<B", data, offset)[0]
        offset += 1
        symbols_per_channel = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        if symbols_per_channel == 0:
            return b""

        all_channels = []
        for ch in range(n_channels):
            comp_size = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            compressed_bytes = data[offset:offset + comp_size]
            offset += comp_size

            # Decode symbol-by-symbol (must match encoder's adaptive state)
            compressed_words = np.frombuffer(compressed_bytes, dtype=np.uint32).copy()
            decoder = constriction.stream.queue.RangeDecoder(compressed_words)
            model = constriction.stream.model.Categorical(perfect=False)

            counts = np.ones(256, dtype=np.float64)
            symbols = np.empty(symbols_per_channel, dtype=np.int32)

            for i in range(symbols_per_channel):
                total = counts.sum()
                probs = (counts / total).astype(np.float32).reshape(1, 256)
                sym = decoder.decode(model, probs)
                symbols[i] = sym[0]
                counts[sym[0]] += self.adaptation_rate

            all_channels.append(symbols.astype(np.uint8).tobytes())

        return b"".join(all_channels)


# ---- Neural compressor ----

class NeuralByteCompressor(ByteCompressor):
    """Neural arithmetic coding using a trained byte prediction model.

    Uses a small neural network (BytePredictor from byte_model.py) to
    predict P(next_byte | context) for each byte in the stream. The
    model must be pre-trained on representative XOR residual data.

    This achieves the best compression but requires:
    1. A trained BytePredictor model
    2. The same model at encode and decode time

    Format:
        magic: bytes[4] = b'AC03'
        n_channels: uint8
        total_symbols: uint32 (per channel)
        model_hash: bytes[16]  (MD5 of model weights, for verification)
        per channel:
            compressed_size: uint32
            compressed_bytes: Range-coded data
    """

    MAGIC = b"AC03"

    def __init__(self, model=None, device: str = "cpu"):
        """
        Args:
            model: Trained BytePredictor instance. If None, compress/decompress
                will raise an error.
            device: Device for model inference.
        """
        self.model = model
        self.device = device

    def compress(self, data: bytes, element_size: int = 4) -> bytes:
        if self.model is None:
            raise RuntimeError("NeuralByteCompressor requires a trained BytePredictor model")
        if not HAS_CONSTRICTION:
            raise RuntimeError("constriction library required: pip install constriction")

        import hashlib
        import torch

        if len(data) == 0:
            return self.MAGIC + struct.pack("<BI", element_size, 0) + b"\x00" * 16

        n_channels = element_size
        symbols_per_channel = len(data) // n_channels

        model_hash = self._model_hash()
        header = (
            self.MAGIC
            + struct.pack("<BI", n_channels, symbols_per_channel)
            + model_hash
        )
        all_compressed = []

        self.model.eval()
        self.model.to(self.device)

        for ch in range(n_channels):
            start = ch * symbols_per_channel
            end = start + symbols_per_channel
            ch_data = np.frombuffer(data[start:end], dtype=np.uint8).copy()
            symbols = ch_data.astype(np.int32)

            # Build probabilities using predict_next (sequential), matching
            # decode path exactly to avoid float precision mismatches
            with torch.no_grad():
                probs = self._sequential_predict(
                    ch_data, channel_id=ch
                )  # (N, 256) float32 numpy

            # Encode
            encoder = constriction.stream.queue.RangeEncoder()
            model_family = constriction.stream.model.Categorical(perfect=False)
            encoder.encode(symbols, model_family, probs)

            compressed_words = encoder.get_compressed()
            compressed_bytes = compressed_words.tobytes()

            all_compressed.append(struct.pack("<I", len(compressed_bytes)))
            all_compressed.append(compressed_bytes)

        return header + b"".join(all_compressed)

    def decompress(self, data: bytes, element_size: int = 4) -> bytes:
        if self.model is None:
            raise RuntimeError("NeuralByteCompressor requires a trained BytePredictor model")
        if not HAS_CONSTRICTION:
            raise RuntimeError("constriction library required: pip install constriction")

        import torch

        offset = 0
        magic = data[offset:offset + 4]
        offset += 4
        if magic != self.MAGIC:
            raise ValueError(f"Invalid neural coding magic: {magic!r}")

        n_channels = struct.unpack_from("<B", data, offset)[0]
        offset += 1
        symbols_per_channel = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        stored_hash = data[offset:offset + 16]
        offset += 16

        if symbols_per_channel == 0:
            return b""

        # Verify model hash matches
        current_hash = self._model_hash()
        if stored_hash != current_hash:
            raise ValueError(
                "Model hash mismatch: data was encoded with a different model. "
                "Ensure the same trained BytePredictor is used for encode and decode."
            )

        self.model.eval()
        self.model.to(self.device)

        all_channels = []
        for ch in range(n_channels):
            comp_size = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            compressed_bytes = data[offset:offset + comp_size]
            offset += comp_size

            compressed_words = np.frombuffer(compressed_bytes, dtype=np.uint32).copy()
            decoder = constriction.stream.queue.RangeDecoder(compressed_words)
            model_family = constriction.stream.model.Categorical(perfect=False)

            # Autoregressive decode: predict one symbol at a time
            decoded_so_far = np.zeros(symbols_per_channel, dtype=np.uint8)

            with torch.no_grad():
                for i in range(symbols_per_channel):
                    probs = self.model.predict_next(
                        decoded_so_far[:i], channel_id=ch, device=self.device
                    )  # (256,) float32
                    probs = probs.reshape(1, 256)

                    sym = decoder.decode(model_family, probs)
                    decoded_so_far[i] = sym[0]

            all_channels.append(decoded_so_far.tobytes())

        return b"".join(all_channels)

    def _sequential_predict(
        self, symbols: np.ndarray, channel_id: int
    ) -> np.ndarray:
        """Build probability table sequentially using predict_next.

        This ensures encode and decode use identical probabilities,
        avoiding float precision mismatches from batch vs sequential
        model evaluation.

        Args:
            symbols: (N,) uint8 array of byte values.
            channel_id: Which byte channel (0-3).

        Returns:
            (N, 256) float32 probability array.
        """
        import torch
        n = len(symbols)
        probs = np.empty((n, 256), dtype=np.float32)
        with torch.no_grad():
            for i in range(n):
                probs[i] = self.model.predict_next(
                    symbols[:i], channel_id=channel_id, device=self.device
                )
        return probs

    def _model_hash(self) -> bytes:
        """Compute MD5 hash of model weights for verification."""
        import hashlib
        h = hashlib.md5()
        for p in self.model.parameters():
            h.update(p.data.cpu().numpy().tobytes())
        return h.digest()
