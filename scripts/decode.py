#!/usr/bin/env python3
"""CLI: Decode .mnf back to CSV."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from umc import ManifoldCodec, ManifoldCodecResult


def main():
    parser = argparse.ArgumentParser(description="Decode .mnf to CSV")
    parser.add_argument("--input", type=str, required=True, help="Input .mnf file")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--codec", type=str, required=True, help="Path to pretrained codec")
    args = parser.parse_args()

    print(f"Loading codec from {args.codec}...")
    codec = ManifoldCodec.from_pretrained(args.codec)

    print(f"Loading .mnf from {args.input}...")
    mnf = ManifoldCodecResult.load(args.input, config=codec.config)

    print(f"Decoding {mnf.n_samples} windows...")
    decoded = codec.decode(mnf)

    # Save as CSV
    flat = decoded.reshape(-1, decoded.shape[-1])
    header = ",".join(codec.config.features)
    np.savetxt(args.output, flat, delimiter=",", header=header, comments="")

    print(f"Decoded to {args.output} ({flat.shape[0]} rows x {flat.shape[1]} features)")


if __name__ == "__main__":
    main()
