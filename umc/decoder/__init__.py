from .base import BaseDecoder

__all__ = ["BaseDecoder", "MLPDecoder", "ConvDecoder", "HVQVAEDecoder"]


def __getattr__(name):
    """Lazy imports for neural decoder classes (require torch)."""
    if name == "MLPDecoder":
        from .mlp_decoder import MLPDecoder
        return MLPDecoder
    if name == "ConvDecoder":
        from .conv_decoder import ConvDecoder
        return ConvDecoder
    if name == "HVQVAEDecoder":
        from .hvqvae_decoder import HVQVAEDecoder
        return HVQVAEDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
