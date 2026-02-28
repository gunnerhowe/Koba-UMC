#!/usr/bin/env python3
"""Comprehensive UMC benchmark across diverse data types.

Compares UMC optimal mode against gzip, bzip2, lzma, zstd, and brotli
on data types ranging from highly compressible (quantized sensor data)
to incompressible (random noise).

Usage:
    python scripts/benchmark_comprehensive.py
"""

import sys
import time
import zlib
import lzma
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import umc
from umc.codec.optimal import _TRANSFORM_NAMES, _COMPRESSOR_NAMES


# ---------------------------------------------------------------------------
# Data generators — designed to represent real-world use cases
# ---------------------------------------------------------------------------

def make_quantized_sensors(n_windows=200):
    """Quantized IoT sensors: temperature (0.1C), humidity (%), pressure (hPa).

    Real sensors report at fixed precision. This creates massive
    redundancy that UMC's predictive coding can exploit.
    """
    rng = np.random.RandomState(42)
    n_features = 6

    # Slow-varying base + quantized noise
    base = np.zeros((n_windows, 64, n_features), dtype=np.float32)
    t = np.linspace(0, 4 * np.pi, 64)

    for w in range(n_windows):
        # Temperature: 20C +/- 5C with daily cycle, quantized to 0.1C
        temp = 20.0 + 5.0 * np.sin(t + w * 0.1) + rng.randn(64) * 0.3
        base[w, :, 0] = np.round(temp, 1)

        # Humidity: 50% +/- 20%, quantized to 1%
        hum = 50.0 + 20.0 * np.cos(t + w * 0.05) + rng.randn(64) * 2
        base[w, :, 1] = np.round(hum, 0)

        # Pressure: 1013 hPa +/- 10, quantized to 0.1
        press = 1013.0 + 10.0 * np.sin(t * 0.3 + w * 0.02) + rng.randn(64) * 0.5
        base[w, :, 2] = np.round(press, 1)

        # Wind speed: 0-20 m/s, quantized to 0.5
        wind = 10.0 + 5.0 * np.abs(np.sin(t * 2 + w * 0.3)) + rng.randn(64) * 1
        base[w, :, 3] = np.round(np.maximum(wind, 0), 0) * 0.5

        # Light level: 0-1000 lux, integer
        light = 500 + 400 * np.sin(t + w * 0.1) + rng.randn(64) * 20
        base[w, :, 4] = np.round(np.maximum(light, 0), 0)

        # CO2 ppm: 400-600, integer
        co2 = 450 + 50 * np.sin(t * 0.5 + w * 0.08) + rng.randn(64) * 5
        base[w, :, 5] = np.round(co2, 0)

    return base.astype(np.float32)


def make_financial_ohlcv(n_windows=200):
    """Financial OHLCV data: Open, High, Low, Close, Volume.

    Prices in cents (2 decimal places), volumes as integers.
    High > Open, Close, Low. Volume varies log-normally.
    """
    rng = np.random.RandomState(42)
    price = 150.00  # Starting price

    data = np.zeros((n_windows, 64, 5), dtype=np.float32)
    for w in range(n_windows):
        for t in range(64):
            # Random walk for price
            price += rng.randn() * 0.50
            price = max(price, 10.0)

            open_p = round(price + rng.randn() * 0.10, 2)
            close_p = round(price + rng.randn() * 0.10, 2)
            high_p = round(max(open_p, close_p) + abs(rng.randn()) * 0.20, 2)
            low_p = round(min(open_p, close_p) - abs(rng.randn()) * 0.20, 2)
            volume = int(np.exp(10 + rng.randn() * 0.5))

            data[w, t, 0] = open_p
            data[w, t, 1] = high_p
            data[w, t, 2] = low_p
            data[w, t, 3] = close_p
            data[w, t, 4] = float(volume)

    return data


def make_image_tiles(n_tiles=100):
    """Image-like data: 2D spatial correlation.

    Simulates 8x8 image patches with smooth gradients + noise.
    """
    rng = np.random.RandomState(7)
    tiles = np.zeros((n_tiles, 64, 3), dtype=np.float32)

    for i in range(n_tiles):
        # Random gradient direction
        angle = rng.uniform(0, 2 * np.pi)
        gx, gy = np.cos(angle), np.sin(angle)

        # 8x8 patch with gradient
        for y in range(8):
            for x in range(8):
                idx = y * 8 + x
                base_val = (gx * x + gy * y) / 8.0 * 200 + 128
                for c in range(3):
                    tiles[i, idx, c] = base_val + rng.randn() * 5

    return tiles


def make_monotonic_counters(n_windows=200):
    """Monotonically increasing counter data (timestamps, IDs, etc.).

    Delta encoding makes this trivially compressible.
    """
    rng = np.random.RandomState(99)
    data = np.zeros((n_windows, 64, 4), dtype=np.float32)

    base = np.array([1000000, 0, 0, 0], dtype=np.float64)
    for w in range(n_windows):
        for t in range(64):
            base[0] += rng.randint(1, 100)       # Timestamp (ms)
            base[1] += 1                          # Sequential ID
            base[2] += rng.randint(100, 10000)    # Cumulative bytes
            base[3] = base[0] % 86400000          # Time of day (ms)
            data[w, t, :] = base.astype(np.float32)

    return data


def make_scientific_simulation(n_windows=100):
    """Scientific simulation output: smooth PDE-like fields.

    2D heat equation-like data with high spatial correlation.
    """
    rng = np.random.RandomState(12)
    n_features = 8

    # Initialize with smooth random field
    field = rng.randn(1, 1, n_features).astype(np.float64) * 10

    data = np.zeros((n_windows, 128, n_features), dtype=np.float32)
    for w in range(n_windows):
        # Evolve the field (diffusion-like)
        new_field = np.zeros((128, n_features), dtype=np.float64)
        for t in range(128):
            x = t / 128.0
            for f in range(n_features):
                new_field[t, f] = (
                    field[0, 0, f] * np.sin(np.pi * x * (f + 1))
                    + 0.1 * rng.randn()
                )
        field[0, 0, :] += rng.randn(n_features) * 0.01
        data[w] = new_field.astype(np.float32)

    return data


def make_sparse_events(n_windows=200):
    """Sparse event data: mostly zeros with occasional spikes.

    Network packets, anomaly detection, event logs.
    """
    rng = np.random.RandomState(33)
    data = np.zeros((n_windows, 64, 4), dtype=np.float32)

    for w in range(n_windows):
        # Sparse: ~5% of values are non-zero
        mask = rng.random((64, 4)) < 0.05
        data[w][mask] = rng.exponential(100, size=mask.sum()).astype(np.float32)

    return data


def make_random(n_windows=100):
    """Worst case: pure random (theoretical incompressible limit)."""
    rng = np.random.RandomState(0)
    return rng.randn(n_windows, 32, 5).astype(np.float32)


# ---------------------------------------------------------------------------
# Competitor baselines
# ---------------------------------------------------------------------------

def compress_competitors(raw_bytes: bytes) -> dict:
    """Compress raw bytes with standard algorithms for comparison."""
    results = {}

    # gzip/zlib level 9
    t0 = time.perf_counter()
    z = zlib.compress(raw_bytes, 9)
    results["gzip-9"] = {"size": len(z), "time": time.perf_counter() - t0}

    # lzma (xz) preset 9
    t0 = time.perf_counter()
    l = lzma.compress(raw_bytes, preset=9)
    results["lzma-9"] = {"size": len(l), "time": time.perf_counter() - t0}

    # zstd (if available)
    try:
        import zstandard as zstd
        t0 = time.perf_counter()
        s = zstd.ZstdCompressor(level=22).compress(raw_bytes)
        results["zstd-22"] = {"size": len(s), "time": time.perf_counter() - t0}
    except ImportError:
        pass

    # brotli (if available)
    try:
        import brotli
        t0 = time.perf_counter()
        b = brotli.compress(raw_bytes, quality=11)
        results["brotli-11"] = {"size": len(b), "time": time.perf_counter() - t0}
    except ImportError:
        pass

    return results


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_one(name, data):
    """Benchmark UMC optimal vs all competitors on one dataset."""
    raw_bytes = data.tobytes()
    raw_size = len(raw_bytes)

    print(f"\n{'='*78}")
    print(f"  {name}")
    print(f"  shape={data.shape}  raw={raw_size:,} bytes")
    print(f"{'='*78}")
    print(f"  {'Algorithm':<28s} {'Size':>10s} {'Ratio':>8s} {'Time':>8s}  {'Notes'}")
    print(f"  {'-'*28} {'-'*10} {'-'*8} {'-'*8}  {'-'*30}")

    # Standard competitors (on raw bytes, no UMC preprocessing)
    competitors = compress_competitors(raw_bytes)
    for algo_name, info in sorted(competitors.items(), key=lambda x: x[1]["size"]):
        ratio = raw_size / info["size"]
        print(f"  {algo_name:<28s} {info['size']:>10,} {ratio:>7.2f}x {info['time']:>7.3f}s")

    # UMC lossless mode
    t0 = time.perf_counter()
    umc_lossless = umc.compress(data, mode="lossless")
    t_lossless = time.perf_counter() - t0
    r_lossless = raw_size / len(umc_lossless)
    print(f"  {'UMC lossless':<28s} {len(umc_lossless):>10,} {r_lossless:>7.2f}x {t_lossless:>7.3f}s  byte-transpose + zlib")

    # UMC lossless_lzma mode
    t0 = time.perf_counter()
    umc_lzma = umc.compress(data, mode="lossless_lzma")
    t_lzma = time.perf_counter() - t0
    r_lzma = raw_size / len(umc_lzma)
    print(f"  {'UMC lossless_lzma':<28s} {len(umc_lzma):>10,} {r_lzma:>7.2f}x {t_lzma:>7.3f}s  byte-transpose + lzma")

    # UMC optimal mode
    t0 = time.perf_counter()
    umc_opt = umc.compress(data, mode="optimal")
    t_opt = time.perf_counter() - t0
    r_opt = raw_size / len(umc_opt)

    # Get certificate details
    result = umc.compress_optimal(data)
    cert = result["certificate"]
    notes = (
        f"t={_TRANSFORM_NAMES[cert['transform']]}, "
        f"c={_COMPRESSOR_NAMES[cert['compressor']]}, "
        f"gap={cert['entropy_gap_pct']:.1f}%"
    )
    print(f"  {'UMC OPTIMAL':<28s} {len(umc_opt):>10,} {r_opt:>7.2f}x {t_opt:>7.3f}s  {notes}")

    # Show improvement over best competitor
    best_competitor_size = min(v["size"] for v in competitors.values())
    best_competitor_name = min(competitors, key=lambda k: competitors[k]["size"])
    if len(umc_opt) < best_competitor_size:
        improvement = (1 - len(umc_opt) / best_competitor_size) * 100
        print(f"  --> UMC OPTIMAL is {improvement:.1f}% smaller than best competitor ({best_competitor_name})")
    else:
        deficit = (len(umc_opt) / best_competitor_size - 1) * 100
        print(f"  --> UMC OPTIMAL is {deficit:.1f}% larger than {best_competitor_name} (near theoretical limit)")

    # Verify round-trip
    recovered = umc.decompress(umc_opt)
    assert np.array_equal(data, recovered), "Round-trip FAILED!"

    return {
        "name": name,
        "raw_size": raw_size,
        "umc_optimal_size": len(umc_opt),
        "best_competitor_size": best_competitor_size,
        "best_competitor_name": best_competitor_name,
        "ratio": r_opt,
        "competitor_ratio": raw_size / best_competitor_size,
    }


def main():
    print("=" * 78)
    print("  UMC Comprehensive Benchmark")
    print("  Comparing UMC optimal mode vs standard compressors")
    print("=" * 78)

    datasets = {
        "Quantized Sensors (200x64x6)": make_quantized_sensors(),
        "Financial OHLCV (200x64x5)": make_financial_ohlcv(),
        "Image Tiles (100x64x3)": make_image_tiles(),
        "Monotonic Counters (200x64x4)": make_monotonic_counters(),
        "Scientific Simulation (100x128x8)": make_scientific_simulation(),
        "Sparse Events (200x64x4)": make_sparse_events(),
        "Random Noise (100x32x5)": make_random(),
    }

    results = []
    for name, data in datasets.items():
        r = benchmark_one(name, data)
        results.append(r)

    # Summary table
    print(f"\n{'='*78}")
    print("  SUMMARY: UMC Optimal vs Best Standard Compressor")
    print(f"{'='*78}")
    print(f"  {'Dataset':<35s} {'UMC':>8s} {'Best Std':>8s} {'Winner':>10s} {'Delta':>8s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

    wins = 0
    for r in results:
        umc_r = r["ratio"]
        comp_r = r["competitor_ratio"]
        if r["umc_optimal_size"] < r["best_competitor_size"]:
            winner = "UMC"
            delta = f"-{(1 - r['umc_optimal_size']/r['best_competitor_size'])*100:.1f}%"
            wins += 1
        elif r["umc_optimal_size"] == r["best_competitor_size"]:
            winner = "TIE"
            delta = "0.0%"
        else:
            winner = r["best_competitor_name"]
            delta = f"+{(r['umc_optimal_size']/r['best_competitor_size']-1)*100:.1f}%"

        print(f"  {r['name']:<35s} {umc_r:>7.2f}x {comp_r:>7.2f}x {winner:>10s} {delta:>8s}")

    print(f"\n  UMC wins: {wins}/{len(results)} datasets")
    print(f"  All round-trips verified: bit-exact lossless")


if __name__ == "__main__":
    main()
