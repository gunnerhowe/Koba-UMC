from .base import BaseEncoder, EncodingResult

__all__ = ["BaseEncoder", "EncodingResult", "VAEEncoder", "ConvEncoder", "HVQVAEEncoder"]


def __getattr__(name):
    """Lazy imports for neural encoder classes (require torch + vector_quantize_pytorch)."""
    if name == "VAEEncoder":
        from .vae import VAEEncoder
        return VAEEncoder
    if name == "ConvEncoder":
        from .conv_encoder import ConvEncoder
        return ConvEncoder
    if name == "HVQVAEEncoder":
        from .hvqvae_encoder import HVQVAEEncoder
        return HVQVAEEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
