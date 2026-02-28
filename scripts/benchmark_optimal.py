#!/usr/bin/env python3
"""Benchmark the optimal compression mode vs all other modes.

Shows compression ratios, speeds, and the optimality certificate
(entropy gap, randomness test) for each data type.

Usage:
    python scripts/benchmark_optimal.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import umc
from umc.codec.optimal import _TRANSFORM_NAMES, _COMPRESSOR_NAMES


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def make_financial(n_windows=100):
    """Financial time series: slow-varying with occasional jumps."""
    rng = np.random.RandomState(42)
    base = 100 + np.cumsum(rng.randn(n_windows, 1, 1) * 0.3, axis=0)
    noise = rng.randn(n_windows, 32, 5) * 0.01
    return (base + noise).astype(np.float32)


def make_iot(n_windows=100):
    """IoT sensor data: smooth sinusoids + noise."""
    rng = np.random.RandomState(123)
    t = np.linspace(0, 10, 64).reshape(1, 64, 1)
    signals = np.sin(t * np.arange(1, 9).reshape(1, 1, 8))
    return (signals + rng.randn(n_windows, 64, 8) * 0.05).astype(np.float32)


def make_audio(n_windows=50):
    """Audio-like: harmonic series + noise."""
    rng = np.random.RandomState(7)
    t = np.linspace(0, 2 * np.pi, 1024).reshape(1, 1024, 1)
    waves = np.sin(440 * t) + 0.3 * np.sin(880 * t)
    return (waves + rng.randn(n_windows, 1024, 1) * 0.01).astype(np.float32)


def make_random(n_windows=100):
    """Worst case: pure random (incompressible)."""
    rng = np.random.RandomState(0)
    return rng.randn(n_windows, 32, 5).astype(np.float32)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

MODES = [
    "lossless", "lossless_zstd", "lossless_lzma",
    "near_lossless", "near_lossless_turbo", "quantized_8",
    "optimal",
]


def benchmark_one(name, data):
    """Benchmark all modes on one dataset."""
    raw_bytes = data.nbytes
    print(f"\n{'='*70}")
    print(f"  {name}:  shape={data.shape}  raw={raw_bytes:,} bytes")
    print(f"{'='*70}")
    print(f"  {'Mode':<28s} {'Size':>10s} {'Ratio':>8s} {'Time':>8s}  Notes")
    print(f"  {'-'*28} {'-'*10} {'-'*8} {'-'*8}  {'-'*20}")

    for mode in MODES:
        t0 = time.perf_counter()
        compressed = umc.compress(data, mode=mode)
        elapsed = time.perf_counter() - t0
        ratio = raw_bytes / len(compressed)

        notes = ""
        if mode == "optimal":
            # Get the certificate
            result = umc.compress_optimal(data)
            cert = result["certificate"]
            notes = (
                f"H0={cert['entropy_h0']:.2f} bpb, "
                f"gap={cert['entropy_gap_pct']:.1f}%, "
                f"p={cert['randomness_p_value']:.3f}, "
                f"t={_TRANSFORM_NAMES[cert['transform']]}, "
                f"c={_COMPRESSOR_NAMES[cert['compressor']]}"
            )

        print(
            f"  {mode:<28s} {len(compressed):>10,} {ratio:>7.2f}x {elapsed:>7.3f}s  {notes}"
        )

        # Verify round-trip for lossless modes
        if mode in ("lossless", "lossless_zstd", "lossless_lzma", "optimal"):
            recovered = umc.decompress(compressed)
            assert np.array_equal(data, recovered), f"Round-trip failed for {mode}!"


def main():
    print("UMC Optimal Mode Benchmark")
    print("=" * 70)
    print("Comparing all compression modes with optimality certificates")

    datasets = {
        "Financial (100x32x5)": make_financial(),
        "IoT Sensors (100x64x8)": make_iot(),
        "Audio (50x1024x1)": make_audio(),
        "Random (100x32x5)": make_random(),
    }

    for name, data in datasets.items():
        benchmark_one(name, data)

    # Print detailed certificate for financial data
    print(f"\n{'='*70}")
    print("  DETAILED OPTIMALITY CERTIFICATE -- Financial Data")
    print(f"{'='*70}")
    data = datasets["Financial (100x32x5)"]
    result = umc.compress_optimal(data)
    cert = result["certificate"]
    print(f"  Original size:      {cert['original_size']:>10,} bytes")
    print(f"  Compressed size:    {cert['compressed_size']:>10,} bytes")
    print(f"  Compression ratio:  {cert['ratio']:>10.2f}x")
    print()
    print(f"  Shannon entropy H0: {cert['entropy_h0']:>10.4f} bits/byte")
    print(f"  Achieved:           {cert['achieved_bpb']:>10.4f} bits/byte")
    print(f"  Entropy gap:        {cert['entropy_gap_bpb']:>10.4f} bits/byte ({cert['entropy_gap_pct']:.1f}%)")
    print()
    print(f"  Randomness test:    p = {cert['randomness_p_value']:.4f}", end="")
    if cert['randomness_p_value'] > 0.05:
        print("  <- PASS (output is statistically random)")
    else:
        print("  <- output has detectable structure")
    print()
    print(f"  Best transform:     {_TRANSFORM_NAMES[cert['transform']]}")
    print(f"  Best compressor:    {_COMPRESSOR_NAMES[cert['compressor']]}")
    print()
    if cert['entropy_gap_pct'] < 5.0:
        print("  VERDICT: Within 5% of Shannon limit -- near-optimal compression!")
    elif cert['entropy_gap_pct'] < 15.0:
        print("  VERDICT: Within 15% of Shannon limit -- good compression.")
    else:
        print(f"  VERDICT: {cert['entropy_gap_pct']:.0f}% from Shannon limit -- room for improvement.")


if __name__ == "__main__":
    main()
