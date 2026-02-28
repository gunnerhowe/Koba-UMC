#!/usr/bin/env python3
"""CLI training entrypoint for UMC."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from umc.config import UMCConfig
from umc import ManifoldCodec
from umc.data.loaders import load_yahoo_finance, combine_datasets


def main():
    parser = argparse.ArgumentParser(description="Train a UMC codec on financial data")
    parser.add_argument("--symbols", type=str, default="SPY,AAPL,MSFT,GOOGL,AMZN",
                        help="Comma-separated ticker symbols")
    parser.add_argument("--period", type=str, default="5y", help="Data period")
    parser.add_argument("--interval", type=str, default="1d", help="Candle interval")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--latent-dim", type=int, default=128, help="Max latent dimensions")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--window-size", type=int, default=64, help="Window size (candles)")
    parser.add_argument("--output", type=str, default="checkpoints/financial_v1",
                        help="Output directory for codec")
    parser.add_argument("--encoder-type", type=str, default="vae",
                        choices=["vae", "adaptive"], help="Encoder type")
    parser.add_argument("--config", type=str, default=None, help="YAML config file (not yet supported)")
    args = parser.parse_args()

    config = UMCConfig(
        epochs=args.epochs,
        max_latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        window_size=args.window_size,
        encoder_type=args.encoder_type,
        checkpoint_dir=args.output,
    )

    symbols = [s.strip() for s in args.symbols.split(",")]
    print(f"=== UMC Training ===")
    print(f"Symbols: {symbols}")
    print(f"Period: {args.period}, Interval: {args.interval}")
    print(f"Config: latent_dim={config.max_latent_dim}, epochs={config.epochs}")
    print()

    print("Loading data...")
    datasets = load_yahoo_finance(symbols, period=args.period, interval=args.interval)
    df = combine_datasets(datasets)
    print(f"Total rows: {len(df)} across {len(datasets)} symbols")
    print()

    print("Training codec...")
    codec = ManifoldCodec(config)
    history = codec.fit(df, verbose=True)

    codec.save(args.output)
    print(f"\nCodec saved to {args.output}")

    # Print final metrics
    final = history[-1]
    print(f"\n=== Final Metrics ===")
    print(f"Train Loss: {final['train']['total']:.6f}")
    print(f"Val Reconstruction: {final['val']['reconstruction']:.6f}")
    print(f"Active Dims: {final['train']['active_dims']:.0f}")


if __name__ == "__main__":
    main()
