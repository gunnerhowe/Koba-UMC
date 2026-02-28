"""Real-world data benchmark for UMC compression.

Tests UMC on realistic data from multiple domains:
  - Financial: Yahoo Finance OHLCV data
  - Images: generated test images
  - Audio: generated test WAV
  - Video: generated test video frames
  - Scientific: random float arrays (simulating sensor/experiment data)

Usage:
    python scripts/benchmark_real_world.py [--output results/real_world.json]
"""

import json
import sys
import time
import tempfile
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from umc.codec.tiered import _compress_storage, _decompress_storage


def _timed_compress(data, mode):
    """Compress and return stats."""
    raw = data.nbytes
    start = time.perf_counter()
    compressed = _compress_storage(data, mode)
    enc_time = time.perf_counter() - start
    start = time.perf_counter()
    decoded = _decompress_storage(compressed)
    dec_time = time.perf_counter() - start
    return {
        "mode": mode,
        "raw_bytes": raw,
        "compressed_bytes": len(compressed),
        "ratio": raw / len(compressed),
        "encode_ms": enc_time * 1000,
        "decode_ms": dec_time * 1000,
    }


def benchmark_financial():
    """Real financial data from Yahoo Finance."""
    print("--- Financial Data (Yahoo Finance) ---")
    try:
        import yfinance as yf
        ticker = yf.Ticker("SPY")
        df = ticker.history(period="5y", interval="1d")
        ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float32)
        # Window it
        win_size = 32
        n_windows = len(ohlcv) // win_size
        data = ohlcv[:n_windows * win_size].reshape(n_windows, win_size, 5)
        print(f"  SPY 5y daily: {data.shape}, {data.nbytes:,} bytes")
        return data
    except Exception as e:
        print(f"  Skipped: {e}")
        # Fallback: generate realistic financial data
        from umc.data.synthetic import generate_financial
        data = generate_financial(n_windows=200)
        print(f"  Synthetic financial: {data.shape}, {data.nbytes:,} bytes")
        return data


def benchmark_images():
    """Realistic image data (gradient + noise patterns)."""
    print("--- Image Data ---")
    rng = np.random.RandomState(42)
    n_images = 20
    h, w, c = 64, 64, 3
    frames = []
    for i in range(n_images):
        # Create structured image (gradient + noise)
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        r = (np.sin(xx * 6 + i * 0.5) * 0.5 + 0.5).astype(np.float32)
        g = (np.cos(yy * 4 + i * 0.3) * 0.5 + 0.5).astype(np.float32)
        b = ((xx + yy) / 2).astype(np.float32)
        img = np.stack([r, g, b], axis=-1)
        img += rng.randn(h, w, c).astype(np.float32) * 0.05
        img = np.clip(img, 0, 1)
        frames.append(img.reshape(h * w, c))

    data = np.stack(frames).astype(np.float32)
    print(f"  {n_images} images @ {h}x{w}x{c}: {data.shape}, {data.nbytes:,} bytes")
    return data


def benchmark_audio():
    """Realistic audio data (mixed tones)."""
    print("--- Audio Data ---")
    sr = 16000
    duration = 5.0
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples)

    # Mix of tones + noise
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
        + 0.1 * np.sin(2 * np.pi * 1320 * t)
        + 0.05 * np.random.randn(n_samples)
    ).astype(np.float32)

    win_size = 1024
    n_windows = n_samples // win_size
    data = audio[:n_windows * win_size].reshape(n_windows, win_size, 1)
    print(f"  {duration}s audio @ {sr}Hz: {data.shape}, {data.nbytes:,} bytes")
    return data


def benchmark_video():
    """Realistic video data (animated gradient)."""
    print("--- Video Data ---")
    rng = np.random.RandomState(42)
    n_frames = 30
    h, w, c = 32, 32, 3
    frames = []
    for i in range(n_frames):
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        phase = i / n_frames * 2 * np.pi
        r = (np.sin(xx * 4 + phase) * 0.5 + 0.5).astype(np.float32)
        g = (np.cos(yy * 3 + phase) * 0.5 + 0.5).astype(np.float32)
        b = ((xx * yy) + i / n_frames * 0.3).astype(np.float32)
        img = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
        img += rng.randn(h, w, c).astype(np.float32) * 0.02
        img = np.clip(img, 0, 1)
        frames.append(img.reshape(h * w, c))

    data = np.stack(frames).astype(np.float32)
    print(f"  {n_frames} frames @ {h}x{w}x{c}: {data.shape}, {data.nbytes:,} bytes")
    return data


def benchmark_scientific():
    """Scientific/sensor data (correlated measurements)."""
    print("--- Scientific Data ---")
    rng = np.random.RandomState(42)
    # Simulate multi-channel sensor with correlations
    n_windows, win_size, n_chan = 200, 64, 8
    base = np.cumsum(rng.randn(n_windows, win_size, 1) * 0.1, axis=1)
    data = np.concatenate([
        base + rng.randn(n_windows, win_size, 1) * 0.01 * i
        for i in range(n_chan)
    ], axis=2).astype(np.float32)
    print(f"  Sensor data: {data.shape}, {data.nbytes:,} bytes")
    return data


def run_baselines(data):
    """Run zlib and zstd baselines."""
    raw = data.astype(np.float32).tobytes()
    raw_size = len(raw)

    results = []

    # zlib
    start = time.perf_counter()
    c = zlib.compress(raw, 9)
    t = time.perf_counter() - start
    results.append({"mode": "zlib-9", "ratio": raw_size / len(c), "encode_ms": t * 1000})

    # zstd
    try:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=19)
        start = time.perf_counter()
        c = cctx.compress(raw)
        t = time.perf_counter() - start
        results.append({"mode": "zstd-19", "ratio": raw_size / len(c), "encode_ms": t * 1000})
    except ImportError:
        pass

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Real-world UMC benchmark")
    parser.add_argument("--output", type=str, default="results/benchmark_real_world.json")
    args = parser.parse_args()

    modes = ["lossless", "near_lossless", "near_lossless_turbo", "quantized_8"]
    all_results = {}

    datasets = {
        "financial": benchmark_financial,
        "images": benchmark_images,
        "audio": benchmark_audio,
        "video": benchmark_video,
        "scientific": benchmark_scientific,
    }

    for ds_name, gen_fn in datasets.items():
        data = gen_fn()
        results = []

        for mode in modes:
            try:
                r = _timed_compress(data, mode)
                results.append(r)
                print(f"  {r['mode']:>25s}: {r['ratio']:>6.1f}x  "
                      f"({r['encode_ms']:.0f}ms enc, {r['decode_ms']:.0f}ms dec)")
            except Exception as e:
                print(f"  {mode:>25s}: FAILED ({e})")

        # Baselines
        baselines = run_baselines(data)
        for b in baselines:
            print(f"  {b['mode']:>25s}: {b['ratio']:>6.1f}x  ({b['encode_ms']:.0f}ms enc)")
            results.append(b)

        all_results[ds_name] = results
        print()

    # Summary table
    print("=" * 100)
    print(f"{'Dataset':<15s} {'lossless':>10s} {'near_loss':>10s} {'turbo':>10s} {'q8':>10s} {'zlib':>10s} {'zstd':>10s}")
    print("-" * 100)
    for ds_name, results in all_results.items():
        row = {}
        for r in results:
            row[r["mode"]] = f"{r['ratio']:.1f}x"
        print(f"{ds_name:<15s} "
              f"{row.get('lossless', '-'):>10s} "
              f"{row.get('near_lossless', '-'):>10s} "
              f"{row.get('near_lossless_turbo', '-'):>10s} "
              f"{row.get('quantized_8', '-'):>10s} "
              f"{row.get('zlib-9', '-'):>10s} "
              f"{row.get('zstd-19', '-'):>10s}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
