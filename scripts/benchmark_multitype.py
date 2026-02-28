"""Benchmark UMC storage compression across multiple data types.

Tests the byte-transposed storage tier (no neural network needed) on:
  - Financial OHLCV
  - Sine waves
  - IoT sensors
  - Weather
  - ECG
  - Audio spectrograms

Also compares against zstd, lzma, and raw zlib baselines.

Usage:
    python scripts/benchmark_multitype.py [--n-windows 1000] [--output results/multitype.json]
"""

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from umc.data.synthetic import generate_all_types
from umc.codec.tiered import _compress_storage, _decompress_storage


def baseline_zlib(data: np.ndarray) -> dict:
    raw = data.astype(np.float32).tobytes()
    start = time.perf_counter()
    compressed = zlib.compress(raw, 9)
    enc_time = time.perf_counter() - start
    start = time.perf_counter()
    zlib.decompress(compressed)
    dec_time = time.perf_counter() - start
    return {
        "method": "zlib-9",
        "compressed_bytes": len(compressed),
        "ratio": len(raw) / len(compressed),
        "encode_sec": enc_time,
        "decode_sec": dec_time,
    }


def baseline_lzma(data: np.ndarray) -> dict:
    import lzma as _lzma
    raw = data.astype(np.float32).tobytes()
    start = time.perf_counter()
    compressed = _lzma.compress(raw, preset=6)
    enc_time = time.perf_counter() - start
    start = time.perf_counter()
    _lzma.decompress(compressed)
    dec_time = time.perf_counter() - start
    return {
        "method": "lzma-6",
        "compressed_bytes": len(compressed),
        "ratio": len(raw) / len(compressed),
        "encode_sec": enc_time,
        "decode_sec": dec_time,
    }


def baseline_zstd(data: np.ndarray) -> dict:
    try:
        import zstandard as zstd
    except ImportError:
        return {"method": "zstd-19", "ratio": 0, "note": "zstandard not installed"}
    raw = data.astype(np.float32).tobytes()
    cctx = zstd.ZstdCompressor(level=19)
    dctx = zstd.ZstdDecompressor()
    start = time.perf_counter()
    compressed = cctx.compress(raw)
    enc_time = time.perf_counter() - start
    start = time.perf_counter()
    dctx.decompress(compressed)
    dec_time = time.perf_counter() - start
    return {
        "method": "zstd-19",
        "compressed_bytes": len(compressed),
        "ratio": len(raw) / len(compressed),
        "encode_sec": enc_time,
        "decode_sec": dec_time,
    }


def umc_storage(data: np.ndarray, mode: str) -> dict:
    raw_bytes = data.nbytes
    start = time.perf_counter()
    compressed = _compress_storage(data, mode)
    enc_time = time.perf_counter() - start
    start = time.perf_counter()
    decoded = _decompress_storage(compressed)
    dec_time = time.perf_counter() - start

    # Verify roundtrip quality
    bit_exact = mode in ("lossless", "lossless_zstd", "lossless_lzma")
    if bit_exact:
        assert np.array_equal(data.astype(np.float32), decoded), f"Lossless verification failed for {mode}"
    elif mode in ("normalized_lossless", "normalized_lossless_zstd"):
        rmse = np.sqrt(np.mean((data.astype(np.float32) - decoded) ** 2))
        assert rmse < 1e-3, f"Normalized mode RMSE too high: {rmse}"
    lossless = bit_exact

    return {
        "method": f"umc-{mode}",
        "compressed_bytes": len(compressed),
        "ratio": raw_bytes / len(compressed),
        "encode_sec": enc_time,
        "decode_sec": dec_time,
        "lossless": lossless,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-type compression benchmark")
    parser.add_argument("--n-windows", type=int, default=1000)
    parser.add_argument("--output", type=str, default="results/benchmark_multitype.json")
    args = parser.parse_args()

    print(f"Generating synthetic data ({args.n_windows} windows per type)...\n")
    all_data = generate_all_types(n_windows=args.n_windows)

    results = {}
    for dtype_name, data in all_data.items():
        print(f"--- {dtype_name} ---")
        print(f"  Shape: {data.shape}, Raw: {data.nbytes:,} bytes")

        type_results = []

        # UMC storage modes
        for mode in ["lossless", "near_lossless", "lossless_zstd", "lossless_lzma",
                      "normalized_lossless", "normalized_lossless_zstd"]:
            try:
                r = umc_storage(data, mode)
                type_results.append(r)
                print(f"  {r['method']:>20s}: {r['ratio']:>6.1f}x  ({r['encode_sec']*1000:.0f}ms enc, {r['decode_sec']*1000:.0f}ms dec)")
            except Exception as e:
                print(f"  {mode:>20s}: FAILED ({e})")

        # Baselines
        for baseline_fn in [baseline_zlib, baseline_zstd, baseline_lzma]:
            r = baseline_fn(data)
            type_results.append(r)
            if r.get("ratio", 0) > 0:
                print(f"  {r['method']:>20s}: {r['ratio']:>6.1f}x  ({r.get('encode_sec', 0)*1000:.0f}ms enc, {r.get('decode_sec', 0)*1000:.0f}ms dec)")

        results[dtype_name] = type_results
        print()

    # Summary table
    print("=" * 110)
    print(f"{'Data Type':<18s} {'lossless':>10s} {'zstd':>10s} {'norm':>10s} {'norm+zstd':>10s} {'near_loss':>10s} {'zlib':>8s} {'zstd-19':>8s} {'lzma':>8s}")
    print("-" * 110)
    for dtype_name, type_results in results.items():
        row = {}
        for r in type_results:
            if r.get("ratio", 0) > 0:
                row[r["method"]] = f"{r['ratio']:.1f}x"
        print(f"{dtype_name:<18s} "
              f"{row.get('umc-lossless', '-'):>10s} "
              f"{row.get('umc-lossless_zstd', '-'):>10s} "
              f"{row.get('umc-normalized_lossless', '-'):>10s} "
              f"{row.get('umc-normalized_lossless_zstd', '-'):>10s} "
              f"{row.get('umc-near_lossless', '-'):>10s} "
              f"{row.get('zlib-9', '-'):>8s} "
              f"{row.get('zstd-19', '-'):>8s} "
              f"{row.get('lzma-6', '-'):>8s}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
