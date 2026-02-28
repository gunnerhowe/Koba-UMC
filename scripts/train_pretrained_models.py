#!/usr/bin/env python3
"""Train the 3 pre-trained hub models: financial-v1, iot-sensors-v1, audio-v1.

Each model is a small HVQ-VAE trained on synthetic (or real) data, saved in the
format expected by TieredManifoldCodec.from_checkpoint().

Usage:
    python scripts/train_pretrained_models.py [--output-dir checkpoints/pretrained]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from umc.config import UMCConfig
from umc.encoder.hvqvae_encoder import HVQVAEEncoder
from umc.decoder.hvqvae_decoder import HVQVAEDecoder
from umc.training.trainer import UMCTrainer
from umc.data.preprocessors import WindowDataset


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

FINANCIAL_CONFIG = UMCConfig(
    encoder_type="hvqvae",
    window_size=32,
    features=("open", "high", "low", "close", "volume"),
    d_model=96,
    n_heads=4,
    n_encoder_layers=3,
    n_decoder_layers=3,
    d_ff=256,
    patch_size=8,
    vq_dim=32,
    vq_top_n_codes=256,
    vq_bottom_n_codes=256,
    vq_bottom_n_levels=1,
    max_latent_dim=64,
    num_charts=8,
    chart_embedding_dim=8,
    batch_size=64,
    epochs=40,
    early_stopping_patience=15,
    learning_rate=1e-3,
    transformer_dropout=0.1,
)

IOT_CONFIG = UMCConfig(
    encoder_type="hvqvae",
    window_size=64,
    features=(
        "temp1", "humid1", "press1", "temp2",
        "humid2", "press2", "temp3", "humid3",
    ),
    d_model=96,
    n_heads=4,
    n_encoder_layers=4,
    n_decoder_layers=4,
    d_ff=256,
    patch_size=8,
    vq_dim=32,
    vq_top_n_codes=256,
    vq_bottom_n_codes=256,
    vq_bottom_n_levels=1,
    max_latent_dim=64,
    num_charts=8,
    chart_embedding_dim=8,
    batch_size=64,
    epochs=40,
    early_stopping_patience=15,
    learning_rate=1e-3,
    transformer_dropout=0.1,
)

AUDIO_CONFIG = UMCConfig(
    encoder_type="hvqvae",
    window_size=1024,
    features=("amplitude",),
    d_model=128,
    n_heads=4,
    n_encoder_layers=3,
    n_decoder_layers=3,
    d_ff=256,
    patch_size=32,
    vq_dim=32,
    vq_top_n_codes=512,
    vq_bottom_n_codes=512,
    vq_bottom_n_levels=1,
    max_latent_dim=64,
    num_charts=8,
    chart_embedding_dim=8,
    batch_size=32,
    epochs=40,
    early_stopping_patience=15,
    learning_rate=1e-3,
    transformer_dropout=0.1,
)

SCIENTIFIC_CONFIG = UMCConfig(
    encoder_type="hvqvae",
    window_size=64,
    features=("x", "y", "z", "pressure", "temperature"),
    d_model=96,
    n_heads=4,
    n_encoder_layers=4,
    n_decoder_layers=4,
    d_ff=256,
    patch_size=8,
    vq_dim=32,
    vq_top_n_codes=512,
    vq_bottom_n_codes=512,
    vq_bottom_n_levels=1,
    max_latent_dim=64,
    num_charts=8,
    chart_embedding_dim=8,
    batch_size=64,
    epochs=40,
    early_stopping_patience=15,
    learning_rate=1e-3,
    transformer_dropout=0.1,
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_financial_data(n_windows: int = 2000) -> np.ndarray:
    """Generate financial training data. Tries yfinance first, falls back to synthetic."""
    try:
        import yfinance as yf
        print("  Downloading SPY 5y daily from Yahoo Finance...")
        df = yf.Ticker("SPY").history(period="5y", interval="1d")
        ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float32)
        w = 32
        n = len(ohlcv) // w
        if n >= 50:
            windows = ohlcv[: n * w].reshape(n, w, 5)
            print(f"  Got {len(windows)} real windows from SPY data")
            return windows
    except Exception as e:
        print(f"  yfinance failed ({e}), using synthetic data")

    from umc.data.synthetic import generate_financial
    return generate_financial(n_windows=n_windows, window_size=32, n_features=5)


def make_iot_data(n_windows: int = 2000) -> np.ndarray:
    """Generate 8-feature IoT sensor data."""
    from umc.data.synthetic import SyntheticManifoldGenerator
    gen = SyntheticManifoldGenerator(seed=42)
    data, _ = gen.generate_financial_synthetic(
        n_samples=n_windows, intrinsic_dim=8,
        window_size=64, n_features=8,
    )
    return data


def make_audio_data(n_windows: int = 800) -> np.ndarray:
    """Generate single-channel audio-like data (ECG waveforms)."""
    from umc.data.synthetic import generate_ecg
    return generate_ecg(n_windows=n_windows, window_size=1024, n_features=1)


def make_scientific_data(n_windows: int = 2000) -> np.ndarray:
    """Generate multi-physics simulation data (5 channels)."""
    from umc.data.synthetic import generate_sine_waves
    return generate_sine_waves(n_windows=n_windows, window_size=64, n_features=5)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_loaders(
    windows: np.ndarray,
    batch_size: int,
    val_fraction: float = 0.15,
) -> tuple[DataLoader, DataLoader]:
    """Z-score normalize, split, and wrap in DataLoaders."""
    n = len(windows)
    n_val = max(1, int(n * val_fraction))
    train_w = windows[:-n_val].copy()
    val_w = windows[-n_val:].copy()

    # Z-score normalize across training set
    mean = train_w.mean(axis=(0, 1), keepdims=True)
    std = train_w.std(axis=(0, 1), keepdims=True) + 1e-8
    train_w = ((train_w - mean) / std).astype(np.float32)
    val_w = ((val_w - mean) / std).astype(np.float32)

    train_loader = DataLoader(
        WindowDataset(train_w),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        WindowDataset(val_w),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def train_one_model(
    name: str,
    config: UMCConfig,
    windows: np.ndarray,
    output_dir: Path,
) -> Path:
    """Instantiate, train, save, and return the checkpoint path."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  windows shape: {windows.shape}")
    print(f"  d_model={config.d_model}, layers={config.n_encoder_layers}enc/"
          f"{config.n_decoder_layers}dec, patch={config.patch_size}")
    print(f"{'='*60}")

    encoder = HVQVAEEncoder(config)
    decoder = HVQVAEDecoder(config)

    n_params = sum(p.numel() for p in encoder.parameters()) + \
               sum(p.numel() for p in decoder.parameters())
    print(f"  Parameters: {n_params:,} ({n_params * 4 / 1e6:.2f} MB)")

    train_loader, val_loader = make_loaders(windows, config.batch_size)
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Point trainer checkpoints to a temp subdirectory
    tmp_ckpt = output_dir / "tmp" / name
    config.checkpoint_dir = str(tmp_ckpt)

    trainer = UMCTrainer(encoder, decoder, config, device="cpu")

    t0 = time.time()
    history = trainer.train(train_loader, val_loader, verbose=True)
    elapsed = time.time() - t0

    best_loss = min(h["val"]["total"] for h in history)
    print(f"  Done: {len(history)} epochs in {elapsed:.1f}s, best val_loss={best_loss:.4f}")

    # Load best weights back into encoder/decoder
    trainer.load_checkpoint("best")

    # Save in hub-compatible format
    checkpoint = {
        "config": config,
        "encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
        "decoder": {k: v.cpu().clone() for k, v in decoder.state_dict().items()},
    }
    out_path = output_dir / f"{name}.pt"
    torch.save(checkpoint, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved: {out_path} ({size_mb:.2f} MB)")

    return out_path


def verify_model(name: str, path: Path, windows: np.ndarray) -> None:
    """Verify the checkpoint loads and can encode/decode."""
    from umc._neural import TieredManifoldCodec
    from umc.hub import ModelRegistry

    info = ModelRegistry.MODELS[name]
    codec = TieredManifoldCodec.from_checkpoint(
        str(path), device="cpu", storage_mode=info["recommended_mode"],
    )

    # Z-score normalize test windows the same way
    test_w = windows[:10].copy()
    mean = windows.mean(axis=(0, 1), keepdims=True)
    std = windows.std(axis=(0, 1), keepdims=True) + 1e-8
    test_w = ((test_w - mean) / std).astype(np.float32)

    encoding = codec.encode(test_w)
    print(f"  Verify {name}: encode OK, n_windows={encoding.n_windows}, "
          f"storage_bytes={len(encoding.storage_compressed)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train UMC pre-trained hub models")
    parser.add_argument(
        "--output-dir", default="checkpoints/pretrained",
        help="Directory to save .pt checkpoints",
    )
    parser.add_argument(
        "--models", nargs="*",
        default=["financial-v1", "iot-sensors-v1", "audio-v1", "scientific-v1"],
        help="Which models to train (default: all 4)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = {
        "financial-v1":   (FINANCIAL_CONFIG, make_financial_data),
        "iot-sensors-v1": (IOT_CONFIG, make_iot_data),
        "audio-v1":       (AUDIO_CONFIG, make_audio_data),
        "scientific-v1":  (SCIENTIFIC_CONFIG, make_scientific_data),
    }

    results = {}
    for name in args.models:
        if name not in model_specs:
            print(f"Unknown model: {name}, skipping")
            continue

        config, data_fn = model_specs[name]
        windows = data_fn()
        path = train_one_model(name, config, windows, output_dir)
        verify_model(name, path, windows)
        results[name] = path

    print(f"\n{'='*60}")
    print("All models trained and verified!")
    print(f"{'='*60}")
    for name, path in results.items():
        size_mb = path.stat().st_size / 1e6
        print(f"  {name}: {path} ({size_mb:.2f} MB)")

    print("\nTo use locally:")
    print("  from umc._neural import TieredManifoldCodec")
    print(f"  codec = TieredManifoldCodec.from_checkpoint('{output_dir}/financial-v1.pt')")


if __name__ == "__main__":
    main()
