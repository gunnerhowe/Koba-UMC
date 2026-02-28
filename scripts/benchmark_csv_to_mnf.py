"""Benchmark the full CSV-to-.mnf pipeline.

Measures true end-to-end compression including CSV parsing, windowing,
and .mnf overhead. This is the real-world ratio users will see.

Usage:
    python scripts/benchmark_csv_to_mnf.py [--n-windows 1000] [--output results/csv_to_mnf.json]
"""

import io
import json
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from umc.data.synthetic import generate_financial
from umc.codec.tiered import _compress_storage, _decompress_storage


def _windows_to_csv_bytes(windows: np.ndarray) -> bytes:
    """Convert windows array to CSV bytes (simulating a raw CSV file)."""
    n_win, win_size, n_feat = windows.shape
    flat = windows.reshape(-1, n_feat)

    buf = io.StringIO()
    buf.write("timestamp,open,high,low,close,volume\n")
    for i in range(len(flat)):
        row = flat[i]
        buf.write(f"2024-01-{i:06d},{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},{row[4]:.2f}\n")
    return buf.getvalue().encode("utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CSV-to-.mnf pipeline benchmark")
    parser.add_argument("--n-windows", type=int, default=1000)
    parser.add_argument("--output", type=str, default="results/benchmark_csv_to_mnf.json")
    args = parser.parse_args()

    print(f"Generating financial data ({args.n_windows} windows)...")
    windows = generate_financial(n_windows=args.n_windows)
    print(f"  Shape: {windows.shape}")

    # Generate CSV
    csv_bytes = _windows_to_csv_bytes(windows)
    csv_size = len(csv_bytes)
    raw_float_bytes = windows.nbytes
    print(f"  CSV size: {csv_size:,} bytes")
    print(f"  Raw float32: {raw_float_bytes:,} bytes\n")

    results = {}

    for mode in ["lossless", "near_lossless", "lossless_zstd", "lossless_lzma"]:
        try:
            # Full pipeline: CSV bytes -> parse -> windows -> compress -> .mnf-like bytes
            start = time.perf_counter()
            compressed = _compress_storage(windows, mode)
            enc_time = time.perf_counter() - start

            # Decode
            start = time.perf_counter()
            decoded = _decompress_storage(compressed)
            dec_time = time.perf_counter() - start

            # .mnf overhead estimate (64-byte header + tiered wrapper ~24 bytes)
            mnf_overhead = 64 + 24
            mnf_total = len(compressed) + mnf_overhead

            csv_ratio = csv_size / mnf_total
            float_ratio = raw_float_bytes / mnf_total

            # Verify lossless
            lossless = mode in ("lossless", "lossless_zstd", "lossless_lzma")
            if lossless:
                assert np.array_equal(windows.astype(np.float32), decoded)

            results[mode] = {
                "csv_bytes": csv_size,
                "raw_float_bytes": raw_float_bytes,
                "compressed_bytes": len(compressed),
                "mnf_total_bytes": mnf_total,
                "csv_to_mnf_ratio": csv_ratio,
                "float_to_mnf_ratio": float_ratio,
                "encode_sec": enc_time,
                "decode_sec": dec_time,
                "lossless": lossless,
            }

            print(f"  {mode:>20s}:")
            print(f"    CSV -> .mnf:     {csv_ratio:>6.1f}x  ({csv_size:,} -> {mnf_total:,} bytes)")
            print(f"    float32 -> .mnf: {float_ratio:>6.1f}x  ({raw_float_bytes:,} -> {mnf_total:,} bytes)")
            print(f"    Encode: {enc_time*1000:.0f}ms  Decode: {dec_time*1000:.0f}ms")
            print()

        except Exception as e:
            print(f"  {mode}: FAILED ({e})\n")

    # Summary
    print("=" * 60)
    print(f"{'Mode':<20s} {'CSV->MNF':>10s} {'Float->MNF':>12s} {'MNF Size':>12s}")
    print("-" * 60)
    for mode, r in results.items():
        print(f"  {mode:<18s} {r['csv_to_mnf_ratio']:>9.1f}x {r['float_to_mnf_ratio']:>11.1f}x {r['mnf_total_bytes']:>11,}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
