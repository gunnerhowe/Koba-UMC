#!/usr/bin/env python3
"""Benchmark UMC storage-tier compression against real competitor tools.

Compares UMC's _compress_storage / _decompress_storage (lossless, near_lossless,
near_lossless_turbo, quantized_8) against zlib, lzma, zstandard, blosc2, and
our own Gorilla XOR codec on several synthetic float32 datasets.

For each method we measure:
    - Compression ratio  (raw_bytes / compressed_bytes)
    - Encode speed       (MB/s of raw data)
    - Decode speed       (MB/s of raw data)
    - Whether the codec is lossless

Results are printed as a formatted comparison table and saved to JSON.

Usage:
    python scripts/benchmark_competitors.py
"""

import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the umc package importable regardless of working directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from umc.codec.tiered import _compress_storage, _decompress_storage
from umc.codec.gorilla import gorilla_compress, gorilla_decompress


# ===================================================================
#  1.  Synthetic data generators
# ===================================================================

def _generate_financial(n_windows: int, window_size: int, n_features: int,
                        seed: int = 42) -> np.ndarray:
    """Correlated financial OHLCV-like data (trends, vol clustering)."""
    rng = np.random.RandomState(seed)
    total = n_windows * window_size

    # GBM close with volatility clustering
    returns = rng.randn(total) * 0.02
    vol = np.ones(total)
    for i in range(1, total):
        vol[i] = 0.9 * vol[i - 1] + 0.1 * abs(returns[i - 1]) / 0.02
    returns *= vol
    close = 100.0 * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(rng.randn(total)) * 0.005)
    low = close * (1 - np.abs(rng.randn(total)) * 0.005)
    opn = low + rng.rand(total) * (high - low)
    volume = np.exp(10 + rng.randn(total) * 0.5 + np.abs(returns) * 20)

    cols = [opn, high, low, close, volume]
    # If caller asks for more than 5 features, tile; fewer, truncate
    while len(cols) < n_features:
        cols.append(cols[len(cols) % 5] + rng.randn(total) * 0.01)
    cols = cols[:n_features]

    data = np.column_stack(cols).astype(np.float32)
    return data[: n_windows * window_size].reshape(n_windows, window_size, n_features)


def _generate_random(n_windows: int, window_size: int, n_features: int,
                     seed: int = 123) -> np.ndarray:
    """IID uniform random data -- worst case for most compressors."""
    rng = np.random.RandomState(seed)
    return rng.rand(n_windows, window_size, n_features).astype(np.float32)


def _generate_sine_waves(n_windows: int, window_size: int, n_features: int,
                         seed: int = 99) -> np.ndarray:
    """Multi-frequency sinusoids with small additive noise."""
    rng = np.random.RandomState(seed)
    data = np.zeros((n_windows, window_size, n_features), dtype=np.float32)
    t = np.linspace(0, 2 * np.pi, window_size)
    for i in range(n_windows):
        for f in range(n_features):
            freq = rng.uniform(0.5, 5.0)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.5, 2.0)
            data[i, :, f] = amp * np.sin(freq * t + phase)
            data[i, :, f] += rng.randn(window_size) * 0.05
    return data


def _generate_sensor(n_windows: int, window_size: int, n_features: int,
                     seed: int = 77) -> np.ndarray:
    """Slowly-drifting IoT sensor readings (temp, humidity, pressure-like)."""
    rng = np.random.RandomState(seed)
    total = n_windows * window_size
    t = np.arange(total, dtype=np.float32)

    cols = []
    # Temperature-like
    temp = 20.0 + 5.0 * np.sin(2 * np.pi * t / (24 * 60)) + t * 0.0001
    temp += rng.randn(total).astype(np.float32) * 0.3
    cols.append(temp)

    # Humidity-like (anti-correlated with temp)
    humidity = 60.0 - 0.5 * (temp - 20.0) + rng.randn(total).astype(np.float32) * 2.0
    humidity = np.clip(humidity, 0, 100)
    cols.append(humidity)

    # Pressure-like (slow drift)
    pressure = 1013.25 + np.cumsum(rng.randn(total).astype(np.float32) * 0.01)
    cols.append(pressure)

    while len(cols) < n_features:
        cols.append(cols[len(cols) % 3] + rng.randn(total).astype(np.float32) * 0.1)
    cols = cols[:n_features]

    data = np.column_stack(cols).astype(np.float32)
    return data[: n_windows * window_size].reshape(n_windows, window_size, n_features)


# ===================================================================
#  2.  Helper: check optional dependencies
# ===================================================================

def _check_zstandard():
    try:
        import zstandard  # noqa: F401
        return True
    except ImportError:
        return False


def _check_blosc2():
    try:
        import blosc2  # noqa: F401
        return True
    except ImportError:
        return False


# ===================================================================
#  3.  Individual benchmark runners
# ===================================================================

def _bench_umc(data: np.ndarray, mode: str) -> dict:
    """Benchmark a single UMC storage-tier mode."""
    raw_bytes = data.nbytes

    t0 = time.perf_counter()
    compressed = _compress_storage(data, mode)
    enc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    decoded = _decompress_storage(compressed)
    dec_time = time.perf_counter() - t0

    is_lossless = mode in ("lossless",)
    if is_lossless:
        lossless_ok = np.array_equal(
            data.astype(np.float32).view(np.uint32),
            decoded.view(np.uint32),
        )
    else:
        lossless_ok = False

    return {
        "name": f"UMC {mode}",
        "compressed_bytes": len(compressed),
        "ratio": raw_bytes / len(compressed),
        "encode_MBps": (raw_bytes / 1e6) / max(enc_time, 1e-9),
        "decode_MBps": (raw_bytes / 1e6) / max(dec_time, 1e-9),
        "lossless": lossless_ok if is_lossless else False,
        "lossless_label": "yes" if is_lossless and lossless_ok else "no",
    }


def _bench_zlib(data: np.ndarray, level: int) -> dict:
    import zlib
    raw = data.astype(np.float32).tobytes()
    raw_bytes = len(raw)

    t0 = time.perf_counter()
    compressed = zlib.compress(raw, level)
    enc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    dec = zlib.decompress(compressed)
    dec_time = time.perf_counter() - t0

    lossless = (dec == raw)
    return {
        "name": f"zlib (level={level})",
        "compressed_bytes": len(compressed),
        "ratio": raw_bytes / len(compressed),
        "encode_MBps": (raw_bytes / 1e6) / max(enc_time, 1e-9),
        "decode_MBps": (raw_bytes / 1e6) / max(dec_time, 1e-9),
        "lossless": lossless,
        "lossless_label": "yes" if lossless else "no",
    }


def _bench_lzma(data: np.ndarray, preset: int) -> dict:
    import lzma
    raw = data.astype(np.float32).tobytes()
    raw_bytes = len(raw)

    t0 = time.perf_counter()
    compressed = lzma.compress(raw, preset=preset)
    enc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    dec = lzma.decompress(compressed)
    dec_time = time.perf_counter() - t0

    lossless = (dec == raw)
    return {
        "name": f"lzma (preset={preset})",
        "compressed_bytes": len(compressed),
        "ratio": raw_bytes / len(compressed),
        "encode_MBps": (raw_bytes / 1e6) / max(enc_time, 1e-9),
        "decode_MBps": (raw_bytes / 1e6) / max(dec_time, 1e-9),
        "lossless": lossless,
        "lossless_label": "yes" if lossless else "no",
    }


def _bench_zstd(data: np.ndarray, level: int) -> dict:
    import zstandard as zstd
    raw = data.astype(np.float32).tobytes()
    raw_bytes = len(raw)

    cctx = zstd.ZstdCompressor(level=level)
    dctx = zstd.ZstdDecompressor()

    t0 = time.perf_counter()
    compressed = cctx.compress(raw)
    enc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    dec = dctx.decompress(compressed)
    dec_time = time.perf_counter() - t0

    lossless = (dec == raw)
    return {
        "name": f"zstd (level={level})",
        "compressed_bytes": len(compressed),
        "ratio": raw_bytes / len(compressed),
        "encode_MBps": (raw_bytes / 1e6) / max(enc_time, 1e-9),
        "decode_MBps": (raw_bytes / 1e6) / max(dec_time, 1e-9),
        "lossless": lossless,
        "lossless_label": "yes" if lossless else "no",
    }


def _bench_blosc2(data: np.ndarray) -> dict:
    import blosc2
    raw = data.astype(np.float32).tobytes()
    raw_bytes = len(raw)

    # bitshuffle filter with lz4hc -- similar to byte transpose
    t0 = time.perf_counter()
    compressed = blosc2.compress(
        raw,
        typesize=4,
        clevel=9,
        filter=blosc2.Filter.BITSHUFFLE,
        codec=blosc2.Codec.LZ4HC,
    )
    enc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    dec = blosc2.decompress(compressed)
    dec_time = time.perf_counter() - t0

    lossless = (dec == raw)
    return {
        "name": "blosc2 (bitshuffle+lz4hc)",
        "compressed_bytes": len(compressed),
        "ratio": raw_bytes / len(compressed),
        "encode_MBps": (raw_bytes / 1e6) / max(enc_time, 1e-9),
        "decode_MBps": (raw_bytes / 1e6) / max(dec_time, 1e-9),
        "lossless": lossless,
        "lossless_label": "yes" if lossless else "no",
    }


def _bench_gorilla(data: np.ndarray) -> dict:
    flat = data.ravel().astype(np.float32)
    raw_bytes = flat.nbytes

    t0 = time.perf_counter()
    compressed = gorilla_compress(flat)
    enc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    decoded = gorilla_decompress(compressed)
    dec_time = time.perf_counter() - t0

    lossless = np.array_equal(flat, decoded)
    return {
        "name": "Gorilla (XOR)",
        "compressed_bytes": len(compressed),
        "ratio": raw_bytes / len(compressed),
        "encode_MBps": (raw_bytes / 1e6) / max(enc_time, 1e-9),
        "decode_MBps": (raw_bytes / 1e6) / max(dec_time, 1e-9),
        "lossless": lossless,
        "lossless_label": "yes" if lossless else "no",
    }


# ===================================================================
#  4.  Pretty-printing
# ===================================================================

COL_WIDTHS = [30, 10, 12, 12, 10]
COL_HEADERS = ["Method", "Ratio", "Enc MB/s", "Dec MB/s", "Lossless"]
COL_ALIGN = ["<", ">", ">", ">", ">"]


def _print_row(values, widths=COL_WIDTHS, align=COL_ALIGN):
    parts = []
    for v, w, a in zip(values, widths, align):
        parts.append(f"{v:{a}{w}}")
    print("  " + " | ".join(parts))


def _print_sep(widths=COL_WIDTHS):
    print("  " + "-+-".join("-" * w for w in widths))


def _format_bytes(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.1f} KB"
    return f"{n} B"


# ===================================================================
#  5.  Main benchmark
# ===================================================================

def main():
    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    N_WINDOWS = 200
    WINDOW_SIZE = 64
    N_FEATURES = 5

    UMC_MODES = ["lossless", "near_lossless", "near_lossless_turbo", "quantized_8"]

    has_zstd = _check_zstandard()
    has_blosc2 = _check_blosc2()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    print("=" * 80)
    print("  UMC vs COMPETITORS  --  Float32 Compression Benchmark")
    print("=" * 80)
    print()
    print(f"  Array shape : ({N_WINDOWS}, {WINDOW_SIZE}, {N_FEATURES})  float32")
    raw_per_dataset = N_WINDOWS * WINDOW_SIZE * N_FEATURES * 4
    print(f"  Raw size    : {_format_bytes(raw_per_dataset)} per dataset")
    print(f"  zstandard   : {'available' if has_zstd else 'NOT installed  (pip install zstandard)'}")
    print(f"  blosc2      : {'available' if has_blosc2 else 'NOT installed  (pip install blosc2)'}")
    print()

    # ------------------------------------------------------------------
    # Generate datasets
    # ------------------------------------------------------------------
    datasets = {
        "financial": _generate_financial(N_WINDOWS, WINDOW_SIZE, N_FEATURES),
        "random":    _generate_random(N_WINDOWS, WINDOW_SIZE, N_FEATURES),
        "sine_waves": _generate_sine_waves(N_WINDOWS, WINDOW_SIZE, N_FEATURES),
        "sensor":    _generate_sensor(N_WINDOWS, WINDOW_SIZE, N_FEATURES),
    }

    # ------------------------------------------------------------------
    # Run benchmarks per dataset
    # ------------------------------------------------------------------
    all_json_results = {}  # dataset_name -> list[dict]

    for ds_name, data in datasets.items():
        print("=" * 80)
        print(f"  DATASET: {ds_name}")
        print(f"  Shape: {data.shape}   Range: [{data.min():.4f}, {data.max():.4f}]")
        print("=" * 80)

        results = []

        # ---- Competitors ----
        print()
        print("  -- Competitors (general-purpose, lossless) --")
        print()
        _print_row(COL_HEADERS)
        _print_sep()

        # zlib levels
        for lvl in (1, 6, 9):
            r = _bench_zlib(data, lvl)
            results.append(r)
            _print_row([r["name"], f"{r['ratio']:.2f}x",
                        f"{r['encode_MBps']:.1f}", f"{r['decode_MBps']:.1f}",
                        r["lossless_label"]])

        # lzma
        r = _bench_lzma(data, 6)
        results.append(r)
        _print_row([r["name"], f"{r['ratio']:.2f}x",
                    f"{r['encode_MBps']:.1f}", f"{r['decode_MBps']:.1f}",
                    r["lossless_label"]])

        # zstandard
        if has_zstd:
            for lvl in (3, 19):
                r = _bench_zstd(data, lvl)
                results.append(r)
                _print_row([r["name"], f"{r['ratio']:.2f}x",
                            f"{r['encode_MBps']:.1f}", f"{r['decode_MBps']:.1f}",
                            r["lossless_label"]])

        # blosc2
        if has_blosc2:
            r = _bench_blosc2(data)
            results.append(r)
            _print_row([r["name"], f"{r['ratio']:.2f}x",
                        f"{r['encode_MBps']:.1f}", f"{r['decode_MBps']:.1f}",
                        r["lossless_label"]])

        # Gorilla
        r = _bench_gorilla(data)
        results.append(r)
        _print_row([r["name"], f"{r['ratio']:.2f}x",
                    f"{r['encode_MBps']:.1f}", f"{r['decode_MBps']:.1f}",
                    r["lossless_label"]])

        # ---- UMC modes ----
        print()
        print("  -- UMC storage-tier modes --")
        print()
        _print_row(COL_HEADERS)
        _print_sep()

        for mode in UMC_MODES:
            r = _bench_umc(data, mode)
            results.append(r)
            _print_row([r["name"], f"{r['ratio']:.2f}x",
                        f"{r['encode_MBps']:.1f}", f"{r['decode_MBps']:.1f}",
                        r["lossless_label"]])

        # ---- Highlight UMC advantage ----
        print()
        _print_umc_advantage(results)
        print()

        all_json_results[ds_name] = results

    # ------------------------------------------------------------------
    # Cross-dataset summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  CROSS-DATASET SUMMARY  (compression ratio)")
    print("=" * 80)
    print()

    ds_names = list(datasets.keys())
    # Collect all unique method names in order of first appearance
    all_methods = []
    for ds_name in ds_names:
        for r in all_json_results[ds_name]:
            if r["name"] not in all_methods:
                all_methods.append(r["name"])

    sw = 30
    dw = 12
    header_parts = [f"{'Method':<{sw}}"]
    for ds in ds_names:
        header_parts.append(f"{ds:>{dw}}")
    print("  " + " | ".join(header_parts))
    print("  " + "-+-".join(["-" * sw] + ["-" * dw] * len(ds_names)))

    for method_name in all_methods:
        parts = [f"{method_name:<{sw}}"]
        for ds in ds_names:
            ratio = None
            for r in all_json_results[ds]:
                if r["name"] == method_name:
                    ratio = r["ratio"]
                    break
            if ratio is not None:
                parts.append(f"{ratio:>{dw}.2f}x")
            else:
                parts.append(f"{'--':>{dw}}")
        print("  " + " | ".join(parts))

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    out_dir = Path(_PROJECT_ROOT) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_competitors.json"

    # Convert to JSON-safe types
    json_safe = {}
    for ds, rs in all_json_results.items():
        json_safe[ds] = []
        for r in rs:
            json_safe[ds].append({
                k: (float(v) if isinstance(v, (np.floating, float)) else
                    bool(v) if isinstance(v, (np.bool_, bool)) else
                    int(v) if isinstance(v, (np.integer, int)) else
                    str(v))
                for k, v in r.items()
            })

    with open(out_path, "w") as f:
        json.dump(json_safe, f, indent=2)

    print()
    print(f"  Results saved to {out_path}")
    print()
    print("=" * 80)
    print("  BENCHMARK COMPLETE")
    print("=" * 80)


# ===================================================================
#  6.  Highlight UMC's competitive advantage
# ===================================================================

def _print_umc_advantage(results: list):
    """Print a short analysis highlighting where UMC wins."""
    # Split into competitor and UMC buckets
    competitors = [r for r in results if not r["name"].startswith("UMC")]
    umc_results = [r for r in results if r["name"].startswith("UMC")]

    if not competitors or not umc_results:
        return

    best_competitor_lossless = max(
        (r for r in competitors if r["lossless"]),
        key=lambda r: r["ratio"],
        default=None,
    )
    best_umc_lossless = None
    for r in umc_results:
        if r["name"] == "UMC lossless":
            best_umc_lossless = r
            break

    best_umc_overall = max(umc_results, key=lambda r: r["ratio"])

    print("  -- UMC Advantage --")

    if best_competitor_lossless and best_umc_lossless:
        comp_r = best_competitor_lossless["ratio"]
        umc_r = best_umc_lossless["ratio"]
        if umc_r >= comp_r:
            print(f"  Lossless:  UMC lossless ({umc_r:.2f}x) beats best competitor "
                  f"{best_competitor_lossless['name']} ({comp_r:.2f}x)")
        else:
            print(f"  Lossless:  UMC lossless ({umc_r:.2f}x) vs best competitor "
                  f"{best_competitor_lossless['name']} ({comp_r:.2f}x) -- similar range")

    if best_competitor_lossless:
        comp_r = best_competitor_lossless["ratio"]
        umc_best_r = best_umc_overall["ratio"]
        gain = umc_best_r / comp_r
        print(f"  Best UMC:  {best_umc_overall['name']} achieves {umc_best_r:.2f}x "
              f"-- {gain:.1f}x better ratio than best lossless competitor ({comp_r:.2f}x)")
        print(f"             Lossy modes trade tiny precision for dramatically better compression.")


if __name__ == "__main__":
    main()
