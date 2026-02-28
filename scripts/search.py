#!/usr/bin/env python3
"""CLI: Search for similar patterns in manifold space."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from umc import ManifoldCodec, ManifoldCodecResult


def main():
    parser = argparse.ArgumentParser(description="Search .mnf for similar patterns")
    parser.add_argument("--index", type=str, required=True, help="Path to .mnf index file")
    parser.add_argument("--query", type=str, required=True, help="Query CSV file")
    parser.add_argument("--k", type=int, default=10, help="Number of results")
    parser.add_argument("--codec", type=str, required=True, help="Path to pretrained codec")
    args = parser.parse_args()

    codec = ManifoldCodec.from_pretrained(args.codec)
    mnf = ManifoldCodecResult.load(args.index, config=codec.config)

    query_df = pd.read_csv(args.query, index_col=0, parse_dates=True)
    query_df.columns = [c.lower() for c in query_df.columns]
    query_mnf = codec.encode(query_df)

    print(f"Searching {mnf.n_samples} windows for {args.k} nearest neighbors...")
    result = mnf.search(query_mnf.coordinates[0], k=args.k)

    print(f"\nTop {args.k} matches:")
    for i in range(min(args.k, result.indices.shape[1])):
        idx = result.indices[0, i]
        dist = result.distances[0, i]
        print(f"  #{i+1}: window_index={idx}, L2_distance={dist:.6f}")


if __name__ == "__main__":
    main()
