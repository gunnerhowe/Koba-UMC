from .loaders import load_yahoo_finance, load_csv, combine_datasets
from .synthetic import SyntheticManifoldGenerator
from .image_loader import load_images, save_images
from .audio_loader import load_audio, save_audio
from .video_loader import load_video, save_video, get_video_info

__all__ = [
    "load_yahoo_finance",
    "load_csv",
    "combine_datasets",
    "OHLCVPreprocessor",
    "WindowNormalizer",
    "UniversalPreprocessor",
    "create_windows",
    "WindowDataset",
    "SyntheticManifoldGenerator",
    "load_images",
    "save_images",
    "load_audio",
    "save_audio",
    "load_video",
    "save_video",
    "get_video_info",
]


def __getattr__(name):
    """Lazy imports for preprocessor classes (require torch)."""
    _preprocessor_names = {
        "OHLCVPreprocessor", "WindowNormalizer", "UniversalPreprocessor",
        "create_windows", "WindowDataset",
    }
    if name in _preprocessor_names:
        from . import preprocessors
        return getattr(preprocessors, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
