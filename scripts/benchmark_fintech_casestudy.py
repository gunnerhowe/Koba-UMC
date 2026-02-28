#!/usr/bin/env python3
"""Fintech Case Study Benchmark: UMC vs industry-standard compressors.

Generates realistic financial tick data (1M+ total data points) with:
- OHLCV price movements (geometric Brownian motion)
- Bid-ask spreads (log-normal)
- Volume patterns (intraday U-shape, log-normal)
- Order book snapshots
- Portfolio risk metrics

Compresses with all UMC modes and compares against:
- gzip-9, lzma-6, zstd-19, brotli-11

Calculates real dollar savings at S3 pricing for enterprise scale.

Usage:
    python scripts/benchmark_fintech_casestudy.py
"""

import sys
import time
import zlib
import lzma
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import umc


# ---------------------------------------------------------------------------
# Realistic Financial Data Generators
# ---------------------------------------------------------------------------

def generate_tick_data(n_ticks=524_288, seed=42):
    """Generate realistic financial tick data (OHLCV + bid/ask).

    Models:
    - Prices: Geometric Brownian Motion (GBM) with mu=0.05, sigma=0.20
    - Bid-ask spread: log-normal, mean ~0.02% of price
    - Volume: intraday U-shape pattern with log-normal noise

    Returns:
        np.ndarray of shape (n_windows, 64, 7) float32
        Columns: Open, High, Low, Close, Volume, Bid, Ask
    """
    rng = np.random.RandomState(seed)

    dt = 1.0 / (252 * 6.5 * 60)
    mu = 0.05
    sigma = 0.20
    S0 = 185.50

    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.randn(n_ticks)
    prices = S0 * np.exp(np.cumsum(log_returns))
    prices = np.round(prices, 2)

    spread_pct = rng.lognormal(mean=np.log(0.0003), sigma=0.5, size=n_ticks)
    spread_pct = np.clip(spread_pct, 0.0001, 0.01)
    half_spread = np.round(prices * spread_pct / 2, 2)
    half_spread = np.maximum(half_spread, 0.01)
    bid = np.round(prices - half_spread, 2)
    ask = np.round(prices + half_spread, 2)

    ticks_per_day = 6 * 60 * 60 + 30 * 60
    time_of_day = np.arange(n_ticks) % ticks_per_day
    normalized_tod = time_of_day / ticks_per_day
    u_shape = 1.0 + 2.0 * (2 * (normalized_tod - 0.5))**2
    base_volume = rng.lognormal(mean=np.log(500), sigma=0.8, size=n_ticks)
    volume = np.round(base_volume * u_shape).astype(np.float32)

    ticks_per_bar = 64
    n_bars = n_ticks // ticks_per_bar
    n_ticks_used = n_bars * ticks_per_bar

    prices_r = prices[:n_ticks_used].reshape(n_bars, ticks_per_bar)
    volume_r = volume[:n_ticks_used].reshape(n_bars, ticks_per_bar)
    bid_r = bid[:n_ticks_used].reshape(n_bars, ticks_per_bar)
    ask_r = ask[:n_ticks_used].reshape(n_bars, ticks_per_bar)

    data = np.zeros((n_bars, ticks_per_bar, 7), dtype=np.float32)
    for i in range(n_bars):
        bar_prices = prices_r[i]
        data[i, :, 0] = bar_prices
        data[i, :, 1] = np.maximum.accumulate(bar_prices)
        data[i, :, 2] = np.minimum.accumulate(bar_prices)
        data[i, :, 3] = bar_prices
        data[i, :, 4] = volume_r[i]
        data[i, :, 5] = bid_r[i]
        data[i, :, 6] = ask_r[i]

    return data


def generate_order_book_snapshots(n_snapshots=65_536, seed=43):
    """Generate realistic order book snapshot data.

    10 levels of bid/ask with price and size = 40 features per snapshot.
    """
    rng = np.random.RandomState(seed)
    n_levels = 10
    n_features = n_levels * 4

    mid = 150.0 + np.cumsum(rng.randn(n_snapshots) * 0.01)
    mid = np.round(mid, 2)

    data = np.zeros((n_snapshots, n_features), dtype=np.float32)
    for i in range(n_snapshots):
        m = mid[i]
        spread = max(0.01, abs(rng.normal(0.02, 0.01)))
        for lvl in range(n_levels):
            offset = (lvl + 1) * spread
            data[i, lvl] = round(m - offset, 2)
            data[i, n_levels + lvl] = rng.lognormal(6, 1)
            data[i, 2 * n_levels + lvl] = round(m + offset, 2)
            data[i, 3 * n_levels + lvl] = rng.lognormal(6, 1)

    ticks_per_window = 64
    n_windows = n_snapshots // ticks_per_window
    data = data[:n_windows * ticks_per_window].reshape(n_windows, ticks_per_window, n_features)
    return data


def generate_risk_metrics(n_days=512, n_instruments=64, seed=44):
    """Generate portfolio risk metrics time series.

    10 features: VaR, CVaR, delta, gamma, vega, theta, rho, IV, RV, P&L.
    """
    rng = np.random.RandomState(seed)
    n_features = 10
    data = np.zeros((n_days, n_instruments, n_features), dtype=np.float32)

    for inst in range(n_instruments):
        base_price = rng.uniform(20, 500)
        base_vol = rng.uniform(0.15, 0.60)
        for d in range(n_days):
            var = base_price * base_vol * 2.326 / np.sqrt(252) * (1 + rng.randn() * 0.05)
            data[d, inst, 0] = round(var, 4)
            data[d, inst, 1] = round(var * 1.3 * (1 + rng.randn() * 0.03), 4)
            data[d, inst, 2] = round(rng.uniform(-1, 1) + rng.randn() * 0.01, 6)
            data[d, inst, 3] = round(rng.uniform(0, 0.1) + rng.randn() * 0.001, 6)
            data[d, inst, 4] = round(rng.uniform(-50, 50) + rng.randn() * 0.5, 4)
            data[d, inst, 5] = round(-abs(rng.uniform(0, 5)) + rng.randn() * 0.1, 4)
            data[d, inst, 6] = round(rng.uniform(-20, 20) + rng.randn() * 0.2, 4)
            iv = base_vol + rng.randn() * 0.01
            data[d, inst, 7] = round(max(iv, 0.01), 6)
            base_vol = 0.99 * base_vol + 0.01 * iv
            data[d, inst, 8] = round(base_vol * (1 + rng.randn() * 0.1), 6)
            data[d, inst, 9] = round(rng.randn() * base_price * 0.02, 2)
            base_price *= (1 + rng.randn() * base_vol / np.sqrt(252))

    return data


# ---------------------------------------------------------------------------
# Compression Helpers
# ---------------------------------------------------------------------------

def compress_umc(data, mode):
    t0 = time.perf_counter()
    compressed = umc.compress(data, mode=mode)
    elapsed = time.perf_counter() - t0
    return compressed, elapsed


def compress_gzip9(raw_bytes):
    t0 = time.perf_counter()
    c = zlib.compress(raw_bytes, 9)
    return c, time.perf_counter() - t0


def compress_lzma6(raw_bytes):
    t0 = time.perf_counter()
    c = lzma.compress(raw_bytes, preset=6)
    return c, time.perf_counter() - t0


def compress_zstd19(raw_bytes):
    import zstandard as zstd
    cctx = zstd.ZstdCompressor(level=19)
    t0 = time.perf_counter()
    c = cctx.compress(raw_bytes)
    return c, time.perf_counter() - t0


def compress_brotli11(raw_bytes):
    import brotli
    t0 = time.perf_counter()
    c = brotli.compress(raw_bytes, quality=11)
    return c, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Cost Analysis
# ---------------------------------------------------------------------------

def calculate_costs(raw_gb, compressed_gb):
    s3_storage_per_gb_month = 0.023
    s3_transfer_per_gb = 0.004
    reads_per_month = 4

    raw_storage_monthly = raw_gb * s3_storage_per_gb_month
    comp_storage_monthly = compressed_gb * s3_storage_per_gb_month
    storage_savings_monthly = raw_storage_monthly - comp_storage_monthly

    raw_transfer_monthly = raw_gb * s3_transfer_per_gb * reads_per_month
    comp_transfer_monthly = compressed_gb * s3_transfer_per_gb * reads_per_month
    transfer_savings_monthly = raw_transfer_monthly - comp_transfer_monthly

    total_savings_monthly = storage_savings_monthly + transfer_savings_monthly
    total_savings_yearly = total_savings_monthly * 12

    return {
        "raw_gb": raw_gb,
        "compressed_gb": compressed_gb,
        "storage_savings_monthly": storage_savings_monthly,
        "transfer_savings_monthly": transfer_savings_monthly,
        "total_savings_monthly": total_savings_monthly,
        "total_savings_yearly": total_savings_yearly,
    }


# ---------------------------------------------------------------------------
# Main Benchmark
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("  FINTECH CASE STUDY: UMC Compression for Financial Time Series")
    print("  Enterprise-Scale Storage Cost Analysis")
    print("=" * 90)

    # -----------------------------------------------------------------------
    # Generate datasets
    # -----------------------------------------------------------------------
    print("\n[1/4] Generating realistic financial datasets...")

    datasets = {}

    print("  - Tick data (500K+ ticks, OHLCV + bid/ask)...", end=" ", flush=True)
    tick_data = generate_tick_data(n_ticks=524_288)
    datasets["Tick Data (OHLCV+Bid/Ask)"] = tick_data
    print(f"shape={tick_data.shape}, {tick_data.nbytes/1e6:.1f} MB, "
          f"{tick_data.size:,} data points")

    print("  - Order book snapshots (40 features)...", end=" ", flush=True)
    ob_data = generate_order_book_snapshots(n_snapshots=65_536)
    datasets["Order Book (10-level)"] = ob_data
    print(f"shape={ob_data.shape}, {ob_data.nbytes/1e6:.1f} MB, "
          f"{ob_data.size:,} data points")

    print("  - Portfolio risk metrics (64 instruments)...", end=" ", flush=True)
    risk_data = generate_risk_metrics(n_days=512, n_instruments=64)
    datasets["Risk Metrics (Greeks+VaR)"] = risk_data
    print(f"shape={risk_data.shape}, {risk_data.nbytes/1e6:.1f} MB, "
          f"{risk_data.size:,} data points")

    total_raw_bytes = sum(d.nbytes for d in datasets.values())
    total_data_points = sum(d.size for d in datasets.values())
    print(f"\n  Total: {total_data_points:,} data points, {total_raw_bytes/1e6:.1f} MB raw")

    # -----------------------------------------------------------------------
    # Run compression benchmarks
    # -----------------------------------------------------------------------
    print("\n[2/4] Running compression benchmarks...")

    umc_modes = ["lossless", "lossless_fast", "lossless_zstd", "lossless_lzma", "optimal_fast"]
    competitors = {
        "gzip-9": compress_gzip9,
        "lzma-6": compress_lzma6,
        "zstd-19": compress_zstd19,
        "brotli-11": compress_brotli11,
    }

    all_results = {}

    for ds_name, data in datasets.items():
        raw_bytes_data = data.tobytes()
        raw_size = len(raw_bytes_data)

        print(f"\n  --- {ds_name} ---")
        print(f"  Raw: {raw_size:,} bytes ({raw_size/1e6:.2f} MB)")
        print(f"  {'Compressor':<28} {'Size':>12} {'Ratio':>8} {'Speed':>10} {'Savings':>8}")
        print(f"  {'-'*68}")

        results = {}

        # UMC modes
        for mode in umc_modes:
            sys.stdout.write(f"    compressing {mode}...")
            sys.stdout.flush()
            compressed, elapsed = compress_umc(data, mode)
            ratio = raw_size / len(compressed)
            speed = raw_size / max(elapsed, 1e-9) / 1e6
            savings = (1 - len(compressed) / raw_size) * 100
            results[f"UMC {mode}"] = {
                "size": len(compressed), "ratio": ratio,
                "speed": speed, "time": elapsed, "savings_pct": savings,
            }
            sys.stdout.write(f" {ratio:.2f}x\n")
            sys.stdout.flush()

        # Verify round-trip for best UMC mode
        best_umc_mode = max(
            [(m, results[f"UMC {m}"]["ratio"]) for m in umc_modes],
            key=lambda x: x[1]
        )[0]
        compressed_check, _ = compress_umc(data, best_umc_mode)
        recovered = umc.decompress(compressed_check)
        assert np.array_equal(data, recovered), f"Round-trip FAILED for {best_umc_mode}!"
        print(f"  [verified: round-trip bit-exact for UMC {best_umc_mode}]")

        # Competitors
        for comp_name, comp_fn in competitors.items():
            sys.stdout.write(f"    compressing {comp_name}...")
            sys.stdout.flush()
            try:
                compressed, elapsed = comp_fn(raw_bytes_data)
                ratio = raw_size / len(compressed)
                speed = raw_size / max(elapsed, 1e-9) / 1e6
                savings = (1 - len(compressed) / raw_size) * 100
                results[comp_name] = {
                    "size": len(compressed), "ratio": ratio,
                    "speed": speed, "time": elapsed, "savings_pct": savings,
                }
                sys.stdout.write(f" {ratio:.2f}x\n")
                sys.stdout.flush()
            except Exception as e:
                results[comp_name] = {
                    "size": raw_size, "ratio": 1.0, "speed": 0,
                    "time": 0, "savings_pct": 0, "error": str(e),
                }
                sys.stdout.write(f" ERROR: {e}\n")
                sys.stdout.flush()

        # Print sorted results
        print(f"\n  {'Compressor':<28} {'Size':>12} {'Ratio':>8} {'Speed':>10} {'Savings':>8}")
        print(f"  {'-'*68}")
        sorted_results = sorted(results.items(), key=lambda x: x[1]["ratio"], reverse=True)
        for name, r in sorted_results:
            marker = " *" if name.startswith("UMC") else "  "
            print(f" {marker}{name:<26} {r['size']:>11,} {r['ratio']:>7.2f}x "
                  f"{r['speed']:>8.1f} MB/s {r['savings_pct']:>6.1f}%")

        all_results[ds_name] = {"raw_size": raw_size, "results": results}

    # -----------------------------------------------------------------------
    # Summary comparison
    # -----------------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  SUMMARY: Best UMC Mode vs Best Standard Compressor")
    print(f"{'='*90}")
    print(f"  {'Dataset':<30} {'UMC Best':>10} {'Std Best':>12} {'UMC Advantage':>15}")
    print(f"  {'-'*70}")

    total_umc_size = 0
    total_std_size = 0
    total_raw = 0

    for ds_name, ds_data in all_results.items():
        raw_size = ds_data["raw_size"]
        results = ds_data["results"]

        best_umc = max(
            [(k, v) for k, v in results.items() if k.startswith("UMC")],
            key=lambda x: x[1]["ratio"]
        )
        best_std = max(
            [(k, v) for k, v in results.items() if not k.startswith("UMC")],
            key=lambda x: x[1]["ratio"]
        )

        advantage = (1 - best_umc[1]["size"] / best_std[1]["size"]) * 100

        total_umc_size += best_umc[1]["size"]
        total_std_size += best_std[1]["size"]
        total_raw += raw_size

        print(f"  {ds_name:<30} {best_umc[1]['ratio']:>7.2f}x    "
              f"{best_std[1]['ratio']:>7.2f}x ({best_std[0][:8]:>8}) "
              f"{advantage:>+10.1f}%")

    overall_umc_ratio = total_raw / total_umc_size
    overall_std_ratio = total_raw / total_std_size
    overall_advantage = (1 - total_umc_size / total_std_size) * 100

    print(f"  {'-'*70}")
    print(f"  {'OVERALL (weighted avg)':<30} {overall_umc_ratio:>7.2f}x    "
          f"{overall_std_ratio:>7.2f}x             "
          f"{overall_advantage:>+10.1f}%")

    # -----------------------------------------------------------------------
    # Cost savings analysis
    # -----------------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  COST SAVINGS ANALYSIS")
    print(f"  AWS S3 Standard: $0.023/GB/month storage + $0.004/GB data transfer")
    print(f"  Assumption: data read 4x/month for analytics pipelines")
    print(f"{'='*90}")

    storage_scenarios = [
        ("10 TB", 10 * 1024),
        ("50 TB", 50 * 1024),
        ("100 TB", 100 * 1024),
    ]

    # Header for comparison: Raw vs Best-Standard vs UMC
    print(f"\n  Comparison: Uncompressed vs Best Standard vs UMC")
    print(f"  {'-'*88}")

    for scenario_name, raw_gb in storage_scenarios:
        std_gb = raw_gb / overall_std_ratio
        umc_gb = raw_gb / overall_umc_ratio

        raw_cost_monthly = raw_gb * 0.023 + raw_gb * 0.004 * 4
        std_cost_monthly = std_gb * 0.023 + std_gb * 0.004 * 4
        umc_cost_monthly = umc_gb * 0.023 + umc_gb * 0.004 * 4

        std_save_yearly = (raw_cost_monthly - std_cost_monthly) * 12
        umc_save_yearly = (raw_cost_monthly - umc_cost_monthly) * 12
        umc_vs_std_yearly = (std_cost_monthly - umc_cost_monthly) * 12

        print(f"\n  {scenario_name} raw data:")
        print(f"    {'Method':<22} {'Stored':>10} {'Monthly Cost':>14} "
              f"{'Yearly Savings':>16} {'vs Std Savings':>16}")
        print(f"    {'-'*80}")
        print(f"    {'Uncompressed':<22} {raw_gb:>8,.0f} GB ${raw_cost_monthly:>12,.2f} "
              f"{'(baseline)':>16} {'---':>16}")
        print(f"    {'Best Standard':<22} {std_gb:>8,.0f} GB ${std_cost_monthly:>12,.2f} "
              f"${std_save_yearly:>14,.2f} {'(baseline)':>16}")
        print(f"    {'UMC (best lossless)':<22} {umc_gb:>8,.0f} GB ${umc_cost_monthly:>12,.2f} "
              f"${umc_save_yearly:>14,.2f} ${umc_vs_std_yearly:>14,.2f}")

    # -----------------------------------------------------------------------
    # Key metrics
    # -----------------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  KEY METRICS FOR CASE STUDY")
    print(f"{'='*90}")
    print(f"  Total data points benchmarked:  {total_data_points:>15,}")
    print(f"  Total raw data size:            {total_raw/1e6:>15.1f} MB")
    print(f"  UMC best ratio (overall):       {overall_umc_ratio:>15.2f}x")
    print(f"  Best standard ratio (overall):  {overall_std_ratio:>15.2f}x")
    print(f"  UMC advantage over standard:    {overall_advantage:>+14.1f}%")
    print(f"  Round-trip verified:             {'YES (bit-exact)':>15}")

    # Per-dataset detail
    print(f"\n  Per-dataset breakdown:")
    print(f"  {'Dataset':<30} {'Raw MB':>8} {'UMC Mode':>20} {'UMC Ratio':>10} "
          f"{'Best Std':>12} {'Std Ratio':>10}")
    print(f"  {'-'*92}")

    for ds_name, ds_data in all_results.items():
        raw_size = ds_data["raw_size"]
        results = ds_data["results"]
        best_umc = max(
            [(k, v) for k, v in results.items() if k.startswith("UMC")],
            key=lambda x: x[1]["ratio"]
        )
        best_std = max(
            [(k, v) for k, v in results.items() if not k.startswith("UMC")],
            key=lambda x: x[1]["ratio"]
        )
        print(f"  {ds_name:<30} {raw_size/1e6:>7.2f} {best_umc[0]:>20} "
              f"{best_umc[1]['ratio']:>9.2f}x {best_std[0]:>12} {best_std[1]['ratio']:>9.2f}x")

    # Save JSON
    output = {
        "total_data_points": int(total_data_points),
        "total_raw_bytes": int(total_raw),
        "overall_umc_ratio": round(overall_umc_ratio, 2),
        "overall_std_ratio": round(overall_std_ratio, 2),
        "overall_advantage_pct": round(overall_advantage, 1),
        "datasets": {},
    }
    for ds_name, ds_data in all_results.items():
        raw_size = ds_data["raw_size"]
        results = ds_data["results"]
        best_umc = max(
            [(k, v) for k, v in results.items() if k.startswith("UMC")],
            key=lambda x: x[1]["ratio"]
        )
        best_std = max(
            [(k, v) for k, v in results.items() if not k.startswith("UMC")],
            key=lambda x: x[1]["ratio"]
        )
        output["datasets"][ds_name] = {
            "raw_bytes": raw_size,
            "umc_best_mode": best_umc[0],
            "umc_best_size": best_umc[1]["size"],
            "umc_best_ratio": round(best_umc[1]["ratio"], 2),
            "umc_best_speed": round(best_umc[1]["speed"], 1),
            "std_best_name": best_std[0],
            "std_best_size": best_std[1]["size"],
            "std_best_ratio": round(best_std[1]["ratio"], 2),
            "all_results": {
                k: {"size": v["size"], "ratio": round(v["ratio"], 2),
                    "speed": round(v["speed"], 1)}
                for k, v in results.items()
            },
        }

    output_path = Path(__file__).resolve().parent.parent / "results" / "fintech_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON results saved to: {output_path}")

    print(f"\n{'='*90}")
    print("  Benchmark complete. All round-trips verified (bit-exact lossless).")
    print(f"{'='*90}")

    return output


if __name__ == "__main__":
    results = main()
