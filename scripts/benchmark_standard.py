#!/usr/bin/env python3
"""Benchmark UMC on standard/reproducible datasets.

Runs compression benchmarks on well-known data patterns that anyone can
reproduce. Results can be included in papers, README, or sales materials.

Usage:
    python scripts/benchmark_standard.py [--html report.html]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import umc


# ---------------------------------------------------------------------------
# Standard Datasets (deterministic, reproducible)
# ---------------------------------------------------------------------------

def make_datasets() -> list[dict]:
    """Generate standard benchmark datasets. All deterministic."""
    datasets = []

    # 1. Financial OHLCV — correlated random walk with volume
    rng = np.random.RandomState(42)
    base = 100.0 + rng.randn(200, 1, 5).cumsum(axis=0) * 0.3
    noise = rng.randn(200, 64, 5) * 0.02
    data = (base + noise).astype(np.float32)
    data[:, :, 4] = np.abs(data[:, :, 4]) * 1e6  # volume
    datasets.append({
        "name": "Financial OHLCV",
        "description": "Correlated random walk (200 windows x 64 x 5)",
        "data": data,
    })

    # 2. IoT Sensors — periodic signals with noise
    rng = np.random.RandomState(123)
    t = np.linspace(0, 20, 128).reshape(1, 128, 1)
    signals = np.broadcast_to(
        np.sin(t * np.arange(1, 9).reshape(1, 1, 8)), (100, 128, 8)
    ).copy()
    signals += rng.randn(100, 128, 8) * 0.05
    datasets.append({
        "name": "IoT Sensors (8ch)",
        "description": "Periodic sinusoids + noise (100 x 128 x 8)",
        "data": signals.astype(np.float32),
    })

    # 3. Monotonic Counters — highly compressible
    rng = np.random.RandomState(7)
    counters = np.arange(0, 50 * 64, dtype=np.float32).reshape(50, 64, 1)
    counters = np.broadcast_to(counters, (50, 64, 4)).copy()
    counters += rng.randn(50, 64, 4).astype(np.float32) * 0.001
    datasets.append({
        "name": "Monotonic Counters",
        "description": "Incrementing counters (50 x 64 x 4)",
        "data": counters,
    })

    # 4. Image Tiles — spatial correlation
    rng = np.random.RandomState(99)
    img = np.zeros((100, 32, 32), dtype=np.float32)
    for i in range(100):
        cx, cy = rng.randint(4, 28, 2)
        r = rng.randint(3, 10)
        y, x = np.ogrid[:32, :32]
        mask = ((x - cx) ** 2 + (y - cy) ** 2) < r ** 2
        img[i] = mask.astype(np.float32) * rng.uniform(0.5, 1.0)
        img[i] += rng.randn(32, 32).astype(np.float32) * 0.02
    datasets.append({
        "name": "Image Tiles (32x32)",
        "description": "Synthetic blob images (100 x 32 x 32)",
        "data": img,
    })

    # 5. Audio Waveforms — sinusoidal with harmonics
    rng = np.random.RandomState(11)
    t = np.linspace(0, 2 * np.pi, 1024).reshape(1, 1024, 1)
    audio = (np.sin(440 * t) + 0.3 * np.sin(880 * t) + 0.1 * np.sin(1320 * t))
    audio = np.broadcast_to(audio, (30, 1024, 1)).copy()
    audio += rng.randn(30, 1024, 1).astype(np.float32) * 0.01
    datasets.append({
        "name": "Audio Waveforms",
        "description": "Sinusoidal + harmonics (30 x 1024 x 1)",
        "data": audio.astype(np.float32),
    })

    # 6. Sparse Events — mostly zeros with spikes
    rng = np.random.RandomState(55)
    sparse = np.zeros((80, 64, 4), dtype=np.float32)
    for i in range(80):
        n_events = rng.randint(1, 6)
        for _ in range(n_events):
            t_idx = rng.randint(0, 64)
            f_idx = rng.randint(0, 4)
            sparse[i, t_idx, f_idx] = rng.exponential(10.0)
    datasets.append({
        "name": "Sparse Events",
        "description": "Mostly zeros with random spikes (80 x 64 x 4)",
        "data": sparse,
    })

    # 7. Random Noise — incompressible baseline
    rng = np.random.RandomState(0)
    noise = rng.randn(50, 64, 8).astype(np.float32)
    datasets.append({
        "name": "Random Noise",
        "description": "IID Gaussian — theoretical limit (50 x 64 x 8)",
        "data": noise,
    })

    # 8. Scientific Simulation — smooth with occasional discontinuities
    rng = np.random.RandomState(77)
    t = np.linspace(0, 10, 128).reshape(1, 128, 1)
    sim = np.exp(-0.3 * t) * np.sin(5 * t)
    sim = np.broadcast_to(sim, (60, 128, 3)).copy()
    sim += rng.randn(60, 128, 3).astype(np.float32) * 0.005
    # Add discontinuities
    for i in range(0, 60, 10):
        t_jump = rng.randint(20, 100)
        sim[i, t_jump:, :] += 2.0
    datasets.append({
        "name": "Scientific Sim",
        "description": "Damped oscillation with jumps (60 x 128 x 3)",
        "data": sim.astype(np.float32),
    })

    return datasets


# ---------------------------------------------------------------------------
# Benchmark Engine
# ---------------------------------------------------------------------------

def benchmark_one(data: np.ndarray, mode: str) -> dict:
    """Compress with a single mode, return stats."""
    raw_size = data.nbytes
    try:
        t0 = time.perf_counter()
        compressed = umc.compress(data, mode=mode)
        elapsed = time.perf_counter() - t0
        ratio = raw_size / max(len(compressed), 1)
        speed = raw_size / max(elapsed, 1e-9) / 1e6
        return {"ratio": ratio, "speed_mbs": speed, "size": len(compressed), "time": elapsed}
    except Exception as e:
        return {"ratio": 0, "speed_mbs": 0, "size": 0, "time": 0, "error": str(e)}


def benchmark_competitor(data_bytes: bytes, name: str) -> dict:
    """Compress raw bytes with a standard compressor."""
    raw_size = len(data_bytes)
    try:
        if name == "gzip-9":
            import zlib
            t0 = time.perf_counter()
            c = zlib.compress(data_bytes, 9)
            elapsed = time.perf_counter() - t0
        elif name == "lzma-6":
            import lzma
            t0 = time.perf_counter()
            c = lzma.compress(data_bytes, preset=6)
            elapsed = time.perf_counter() - t0
        elif name == "zstd-19":
            import zstandard as zstd
            cctx = zstd.ZstdCompressor(level=19)
            t0 = time.perf_counter()
            c = cctx.compress(data_bytes)
            elapsed = time.perf_counter() - t0
        elif name == "brotli-11":
            import brotli
            t0 = time.perf_counter()
            c = brotli.compress(data_bytes, quality=11)
            elapsed = time.perf_counter() - t0
        else:
            return {"ratio": 0, "speed_mbs": 0, "size": 0, "time": 0}

        ratio = raw_size / max(len(c), 1)
        speed = raw_size / max(elapsed, 1e-9) / 1e6
        return {"ratio": ratio, "speed_mbs": speed, "size": len(c), "time": elapsed}
    except ImportError:
        return {"ratio": 0, "speed_mbs": 0, "size": 0, "time": 0, "error": "not installed"}


def run_benchmarks():
    """Run full benchmark suite."""
    datasets = make_datasets()

    umc_modes = ["lossless", "lossless_fast", "lossless_zstd", "lossless_lzma", "optimal_fast"]
    competitors = ["gzip-9", "lzma-6", "zstd-19", "brotli-11"]

    results = []

    for ds in datasets:
        data = ds["data"]
        raw_bytes = data.tobytes()
        entry = {
            "name": ds["name"],
            "description": ds["description"],
            "shape": data.shape,
            "raw_bytes": data.nbytes,
            "umc": {},
            "competitors": {},
        }

        for mode in umc_modes:
            entry["umc"][mode] = benchmark_one(data, mode)

        for comp in competitors:
            entry["competitors"][comp] = benchmark_competitor(raw_bytes, comp)

        results.append(entry)

    return results


def print_results(results: list[dict]):
    """Print results as a formatted table."""
    # Header
    print(f"\n{'='*100}")
    print(f"UMC Standard Benchmark Suite")
    print(f"{'='*100}")

    for r in results:
        print(f"\n  {r['name']} — {r['description']}")
        print(f"  Raw: {r['raw_bytes']:,} bytes ({r['raw_bytes']/1e6:.2f} MB)")
        print(f"  {'Compressor':<22} {'Ratio':>8} {'Speed':>10} {'Size':>12}")
        print(f"  {'-'*54}")

        all_entries = []

        for mode, stats in r["umc"].items():
            if stats.get("error"):
                continue
            label = f"UMC {mode}"
            all_entries.append((label, stats["ratio"], stats["speed_mbs"], stats["size"]))

        for comp, stats in r["competitors"].items():
            if stats.get("error") or stats["ratio"] == 0:
                continue
            all_entries.append((comp, stats["ratio"], stats["speed_mbs"], stats["size"]))

        # Sort by ratio descending
        all_entries.sort(key=lambda x: x[1], reverse=True)

        for label, ratio, speed, size in all_entries:
            marker = " *" if label.startswith("UMC") else "  "
            print(f" {marker}{label:<20} {ratio:>7.2f}x {speed:>8.1f} MB/s {size:>11,}")

    # Summary table: best UMC vs best competitor per dataset
    print(f"\n{'='*100}")
    print(f"{'SUMMARY: UMC optimal_fast vs Best Standard Compressor':^100}")
    print(f"{'='*100}")
    print(f"  {'Dataset':<25} {'UMC best':>10} {'Std best':>12} {'Winner':>10} {'Delta':>8}")
    print(f"  {'-'*67}")

    umc_wins = 0
    for r in results:
        # Best UMC lossless ratio
        best_umc = 0
        best_umc_name = ""
        for mode, stats in r["umc"].items():
            if not stats.get("error") and stats["ratio"] > best_umc:
                best_umc = stats["ratio"]
                best_umc_name = mode

        # Best competitor ratio
        best_comp = 0
        best_comp_name = ""
        for comp, stats in r["competitors"].items():
            if not stats.get("error") and stats["ratio"] > best_comp:
                best_comp = stats["ratio"]
                best_comp_name = comp

        if best_umc >= best_comp:
            winner = "UMC"
            delta = -(1 - best_comp / max(best_umc, 0.01)) * 100
            umc_wins += 1
        else:
            winner = best_comp_name
            delta = (1 - best_umc / max(best_comp, 0.01)) * 100

        print(f"  {r['name']:<25} {best_umc:>9.2f}x {best_comp:>9.2f}x ({best_comp_name})"
              f"  {winner:>6} {delta:>+6.1f}%")

    print(f"\n  UMC wins: {umc_wins}/{len(results)} datasets")


def main():
    parser = argparse.ArgumentParser(description="UMC Standard Benchmark Suite")
    parser.add_argument("--html", type=str, default=None, help="Save HTML report")
    args = parser.parse_args()

    from umc.cext import HAS_C_EXT
    print(f"C extension: {'active' if HAS_C_EXT else 'not available (pure Python)'}")

    results = run_benchmarks()
    print_results(results)

    if args.html:
        _generate_html(results, args.html)
        print(f"\nHTML report saved to: {args.html}")


def _generate_html(results: list[dict], output_path: str):
    """Generate an HTML report from benchmark results."""
    from umc.benchmark_report import _fmt_size
    import html as html_mod
    from datetime import datetime

    rows = ""
    for r in results:
        all_entries = []
        for mode, stats in r["umc"].items():
            if not stats.get("error"):
                all_entries.append((f"UMC {mode}", stats["ratio"], stats["speed_mbs"], True))
        for comp, stats in r["competitors"].items():
            if not stats.get("error") and stats["ratio"] > 0:
                all_entries.append((comp, stats["ratio"], stats["speed_mbs"], False))

        all_entries.sort(key=lambda x: x[1], reverse=True)
        best_ratio = all_entries[0][1] if all_entries else 1

        for i, (name, ratio, speed, is_umc) in enumerate(all_entries):
            bar_pct = (ratio / max(best_ratio, 0.01)) * 100
            cls = "umc-row" if is_umc else "std-row"
            rows += f"""<tr class="{cls}">
                <td>{html_mod.escape(r['name'])}</td>
                <td>{html_mod.escape(name)}</td>
                <td><strong>{ratio:.2f}x</strong></td>
                <td><div class="bar-container"><div class="bar {'bar-umc' if is_umc else 'bar-std'}" style="width:{bar_pct:.1f}%"></div></div></td>
                <td>{speed:.1f} MB/s</td>
            </tr>\n"""

    report = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>UMC Standard Benchmark</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;padding:2rem}}
.container{{max-width:1000px;margin:0 auto}}
h1{{color:#58a6ff;margin-bottom:0.5rem}}
.subtitle{{color:#8b949e;margin-bottom:2rem}}
table{{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}}
th{{background:#21262d;color:#8b949e;padding:0.75rem 1rem;text-align:left;font-size:0.85rem;text-transform:uppercase}}
td{{padding:0.5rem 1rem;border-top:1px solid #21262d}}
.umc-row td:nth-child(2){{color:#58a6ff;font-weight:600}}
.std-row td:nth-child(2){{color:#8b949e}}
.bar-container{{width:100%;height:18px;background:#21262d;border-radius:3px;overflow:hidden}}
.bar{{height:100%;border-radius:3px}}
.bar-umc{{background:linear-gradient(90deg,#1f6feb,#58a6ff)}}
.bar-std{{background:linear-gradient(90deg,#30363d,#484f58)}}
.footer{{margin-top:2rem;color:#484f58;font-size:0.8rem;text-align:center}}
</style></head><body>
<div class="container">
<h1>UMC Standard Benchmark Suite</h1>
<p class="subtitle">Reproducible compression benchmarks on 8 standard datasets — {datetime.now().strftime('%Y-%m-%d')}</p>
<table><thead><tr><th>Dataset</th><th>Compressor</th><th>Ratio</th><th>Visual</th><th>Speed</th></tr></thead>
<tbody>{rows}</tbody></table>
<div class="footer">Generated by UMC (Universal Manifold Codec)</div>
</div></body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
