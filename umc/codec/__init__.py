"""UMC codec subpackage.

Lightweight imports (residual, gorilla, streaming) are loaded eagerly.
Torch-dependent modules (lossless, tiered TieredCodec, arithmetic, byte_model)
are lazy-imported so that `umc.compress()` works without torch.
"""

# Lightweight — no torch dependency
from .residual import ResidualCoder
from .gorilla import gorilla_compress, gorilla_decompress, chimp_compress, chimp_decompress

# Lazy imports for torch-dependent classes
_LAZY_IMPORTS = {
    "LosslessCodec": "lossless",
    "LosslessEncoding": "lossless",
    "TieredCodec": "tiered",
    "TieredEncoding": "tiered",
    "serialize_tiered": "tiered",
    "deserialize_tiered": "tiered",
    "ByteCompressor": "arithmetic",
    "StaticByteCompressor": "arithmetic",
    "AdaptiveByteCompressor": "arithmetic",
    "NeuralByteCompressor": "arithmetic",
    "BytePredictor": "byte_model",
    "train_byte_predictor": "byte_model",
    "StreamingStorageCodec": "streaming",
    "serialize_chunks": "streaming",
    "deserialize_chunks": "streaming",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(f".{_LAZY_IMPORTS[name]}", __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ResidualCoder",
    "LosslessCodec",
    "LosslessEncoding",
    "TieredCodec",
    "TieredEncoding",
    "serialize_tiered",
    "deserialize_tiered",
    "ByteCompressor",
    "StaticByteCompressor",
    "AdaptiveByteCompressor",
    "NeuralByteCompressor",
    "BytePredictor",
    "train_byte_predictor",
    "gorilla_compress",
    "gorilla_decompress",
    "chimp_compress",
    "chimp_decompress",
    "StreamingStorageCodec",
    "serialize_chunks",
    "deserialize_chunks",
]
