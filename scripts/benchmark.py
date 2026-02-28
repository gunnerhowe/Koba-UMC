#!/usr/bin/env python3
"""Full benchmark suite for UMC — compares against traditional compression."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from umc.config import UMCConfig
from umc import ManifoldCodec
from umc.data.loaders import load_yahoo_finance, combine_datasets
from umc.data.preprocessors import OHLCVPreprocessor, create_windows
from umc.evaluation.metrics import (
    reconstruction_rmse,
    reconstruction_rmse_per_feature,
    compression_ratio,
    effective_dimensionality,
)
from umc.evaluation.benchmarks import run_all_baselines


def main():
    parser = argparse.ArgumentParser(description="Run UMC benchmark suite")
    parser.add_argument("--symbols", type=str, default="SPY,AAPL,MSFT",
                        help="Comma-separated symbols for benchmark data")
    parser.add_argument("--period", type=str, default="5y")
    parser.add_argument("--codec-path", type=str, default=None,
                        help="Path to pretrained codec (will train if not provided)")
    parser.add_argument("--output", type=str, default="results/benchmark.json")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    print("=== UMC Benchmark Suite ===\n")

    # Load data
    symbols = [s.strip() for s in args.symbols.split(",")]
    print(f"Loading data for {symbols}...")
    datasets = load_yahoo_finance(symbols, period=args.period)
    df = combine_datasets(datasets)
    print(f"  Total rows: {len(df)}\n")

    # Load or train codec
    if args.codec_path and Path(args.codec_path).exists():
        print(f"Loading pretrained codec from {args.codec_path}...")
        codec = ManifoldCodec.from_pretrained(args.codec_path)
    else:
        print("Training codec (50 epochs for benchmark)...")
        config = UMCConfig(epochs=50, max_latent_dim=64)
        codec = ManifoldCodec(config)
        codec.fit(df, verbose=True)
        print()

    # Encode
    print("Encoding data...")
    mnf = codec.encode(df)
    print(f"  Encoded {mnf.n_samples} windows, latent_dim={mnf.latent_dim}")

    # Decode
    print("Decoding data...")
    decoded = codec.decode(mnf)

    # Get original windows for comparison
    preprocessor = OHLCVPreprocessor(codec.config)
    normalized = preprocessor.fit_transform(df)
    original_windows = create_windows(normalized, codec.config.window_size)
    # Trim to match encoded count
    original_windows = original_windows[:mnf.n_samples]

    # === Metrics ===
    print("\n=== Results ===\n")

    # Reconstruction
    rmse = reconstruction_rmse(original_windows, decoded)
    rmse_per_feat = reconstruction_rmse_per_feature(
        original_windows, decoded, list(codec.config.features)
    )
    price_range = original_windows.max() - original_windows.min()
    rmse_pct = (rmse / (price_range + 1e-8)) * 100

    print(f"Reconstruction RMSE: {rmse:.6f} ({rmse_pct:.4f}% of range)")
    for feat, val in rmse_per_feat.items():
        print(f"  {feat}: {val:.6f}")

    # Compression
    raw_csv_bytes = len(df.to_csv().encode())
    coord_bytes = mnf.n_samples * mnf.latent_dim * 2  # float16
    encoder_decoder_bytes = sum(
        p.numel() * 4 for p in codec.encoder.parameters()
    ) + sum(
        p.numel() * 4 for p in codec.decoder.parameters()
    )

    comp = compression_ratio(raw_csv_bytes, coord_bytes, encoder_decoder_bytes)
    print(f"\nCompression:")
    print(f"  Raw CSV: {raw_csv_bytes:,} bytes")
    print(f"  .mnf coords: {coord_bytes:,} bytes")
    print(f"  Decoder size: {encoder_decoder_bytes:,} bytes")
    print(f"  Coord-only ratio: {comp['coordinate_ratio']:.1f}x")
    print(f"  System ratio: {comp['system_ratio']:.1f}x")

    # Dimensionality
    dim_info = effective_dimensionality(mnf.coordinates)
    print(f"\nDimensionality:")
    print(f"  Active dims: {dim_info['active_dims']} / {dim_info['total_dims']}")
    print(f"  Participation ratio: {dim_info['participation_ratio']:.1f}")

    # Throughput
    print(f"\nThroughput (estimated from encoding {mnf.n_samples} windows):")
    n_candles = mnf.n_samples * codec.config.window_size
    # Quick encode timing
    t0 = time.perf_counter()
    with torch.no_grad():
        x = torch.from_numpy(original_windows[:1000]).float().to(codec.device)
        codec.encoder.encode(x)
    t1 = time.perf_counter()
    enc_throughput = 1000 * codec.config.window_size / (t1 - t0 + 1e-8)
    print(f"  Encode: {enc_throughput:,.0f} candles/sec")

    # Baseline comparisons
    print(f"\nBaseline Compression:")
    baselines = run_all_baselines(original_windows)
    for b in baselines:
        if "ratio" in b:
            print(f"  {b['method']}: {b['ratio']:.1f}x")

    # Save JSON
    results = {
        "reconstruction": {
            "rmse": rmse,
            "rmse_pct_of_range": rmse_pct,
            "rmse_per_feature": rmse_per_feat,
        },
        "compression": comp,
        "dimensionality": dim_info,
        "encode_throughput_candles_per_sec": enc_throughput,
        "baselines": baselines,
        "n_windows": mnf.n_samples,
        "n_symbols": len(symbols),
    }
    # Remove non-serializable items
    results["dimensionality"].pop("variance_per_dim", None)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nBenchmark results saved to {args.output}")


if __name__ == "__main__":
    main()
