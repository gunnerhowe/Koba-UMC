"""Head-to-head benchmark: UMC vs Gorilla/Chimp on financial data.

Compares UMC's byte-transposed storage tier against Gorilla (Facebook) and
Chimp (improved XOR) compression on identical financial time series data.

Usage:
    python scripts/benchmark_gorilla.py [--n-windows 1000] [--output results/gorilla.json]
"""

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from umc.data.synthetic import generate_financial
from umc.codec.tiered import _compress_storage, _decompress_storage
from umc.codec.gorilla import (
    gorilla_compress, gorilla_decompress,
    chimp_compress, chimp_decompress,
    compress_array, decompress_array,
)


def bench_gorilla(data: np.ndarray) -> dict:
    flat = data.ravel().astype(np.float32)
    raw_bytes = flat.nbytes

    start = time.perf_counter()
    compressed = gorilla_compress(flat)
    enc_time = time.perf_counter() - start

    start = time.perf_counter()
    decoded = gorilla_decompress(compressed)
    dec_time = time.perf_counter() - start

    lossless = np.array_equal(flat, decoded)
    return {
        "method": "gorilla",
        "compressed_bytes": len(compressed),
        "ratio": raw_bytes / len(compressed),
        "encode_sec": enc_time,
        "decode_sec": dec_time,
        "lossless": lossless,
    }


def bench_chimp(data: np.ndarray) -> dict:
    flat = data.ravel().astype(np.float32)
    raw_bytes = flat.nbytes

    start = time.perf_counter()
    compressed = chimp_compress(flat)
    enc_time = time.perf_counter() - start

    start = time.perf_counter()
    decoded = chimp_decompress(compressed)
    dec_time = time.perf_counter() - start

    lossless = np.array_equal(flat, decoded)
    return {
        "method": "chimp",
        "compressed_bytes": len(compressed),
        "ratio": raw_bytes / len(compressed),
        "encode_sec": enc_time,
        "decode_sec": dec_time,
        "lossless": lossless,
    }


def bench_umc(data: np.ndarray, mode: str) -> dict:
    raw_bytes = data.nbytes
    start = time.perf_counter()
    compressed = _compress_storage(data, mode)
    enc_time = time.perf_counter() - start

    start = time.perf_counter()
    decoded = _decompress_storage(compressed)
    dec_time = time.perf_counter() - start

    lossless = mode != "near_lossless"
    if lossless:
        assert np.array_equal(data.astype(np.float32), decoded)

    return {
        "method": f"umc-{mode}",
        "compressed_bytes": len(compressed),
        "ratio": raw_bytes / len(compressed),
        "encode_sec": enc_time,
        "decode_sec": dec_time,
        "lossless": lossless,
    }


def bench_zstd(data: np.ndarray) -> dict:
    try:
        import zstandard as zstd
    except ImportError:
        return {"method": "zstd", "ratio": 0, "note": "not installed"}
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
        "lossless": True,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="UMC vs Gorilla/Chimp benchmark")
    parser.add_argument("--n-windows", type=int, default=1000)
    parser.add_argument("--output", type=str, default="results/benchmark_gorilla.json")
    args = parser.parse_args()

    print(f"Generating financial data ({args.n_windows} windows)...")
    data = generate_financial(n_windows=args.n_windows)
    print(f"  Shape: {data.shape}, Raw: {data.nbytes:,} bytes\n")

    results = []

    # UMC storage modes
    for mode in ["lossless", "near_lossless", "lossless_zstd", "lossless_lzma"]:
        try:
            r = bench_umc(data, mode)
            results.append(r)
            print(f"  {r['method']:>20s}: {r['ratio']:>6.1f}x  "
                  f"({'lossless' if r['lossless'] else 'lossy':>8s})  "
                  f"{r['encode_sec']*1000:.0f}ms enc  {r['decode_sec']*1000:.0f}ms dec")
        except Exception as e:
            print(f"  umc-{mode:>16s}: FAILED ({e})")

    # Gorilla
    r = bench_gorilla(data)
    results.append(r)
    print(f"  {r['method']:>20s}: {r['ratio']:>6.1f}x  "
          f"({'lossless' if r['lossless'] else 'LOSSY':>8s})  "
          f"{r['encode_sec']*1000:.0f}ms enc  {r['decode_sec']*1000:.0f}ms dec")

    # Chimp
    r = bench_chimp(data)
    results.append(r)
    print(f"  {r['method']:>20s}: {r['ratio']:>6.1f}x  "
          f"({'lossless' if r['lossless'] else 'LOSSY':>8s})  "
          f"{r['encode_sec']*1000:.0f}ms enc  {r['decode_sec']*1000:.0f}ms dec")

    # zstd baseline
    r = bench_zstd(data)
    results.append(r)
    if r.get("ratio", 0) > 0:
        print(f"  {r['method']:>20s}: {r['ratio']:>6.1f}x  "
              f"({'lossless' if r['lossless'] else 'lossy':>8s})  "
              f"{r['encode_sec']*1000:.0f}ms enc  {r['decode_sec']*1000:.0f}ms dec")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Method':>20s} {'Ratio':>8s} {'Lossless':>10s} {'Speed':>10s}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x.get("ratio", 0), reverse=True):
        if r.get("ratio", 0) > 0:
            speed = r.get("encode_sec", 0)
            speed_str = f"{speed*1000:.0f}ms" if speed > 0 else "N/A"
            print(f"  {r['method']:>18s} {r['ratio']:>7.1f}x {'Yes' if r.get('lossless') else 'No':>10s} {speed_str:>10s}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
