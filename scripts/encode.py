#!/usr/bin/env python3
"""CLI: Encode a dataset to .mnf format."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from umc import ManifoldCodec


def main():
    parser = argparse.ArgumentParser(description="Encode data to .mnf format")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output .mnf file")
    parser.add_argument("--codec", type=str, required=True, help="Path to pretrained codec")
    args = parser.parse_args()

    print(f"Loading codec from {args.codec}...")
    codec = ManifoldCodec.from_pretrained(args.codec)

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]

    print("Encoding...")
    mnf = codec.encode(df)
    bytes_written = mnf.save(args.output)

    raw_size = Path(args.input).stat().st_size
    print(f"\nResults:")
    print(f"  Windows encoded: {mnf.n_samples}")
    print(f"  Latent dim: {mnf.latent_dim}")
    print(f"  .mnf size: {bytes_written:,} bytes")
    print(f"  Raw CSV size: {raw_size:,} bytes")
    print(f"  Compression ratio: {raw_size / bytes_written:.1f}x")


if __name__ == "__main__":
    main()
