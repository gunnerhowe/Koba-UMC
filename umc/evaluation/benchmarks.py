"""Benchmark UMC against traditional compression methods."""

import gzip
import io
import time
from typing import Optional

import numpy as np


def _to_csv_bytes(data: np.ndarray) -> bytes:
    """Convert numpy array to CSV bytes (simulating raw storage)."""
    buf = io.BytesIO()
    np.savetxt(buf, data.reshape(-1, data.shape[-1]) if data.ndim == 3 else data,
               delimiter=",", fmt="%.8f")
    return buf.getvalue()


def benchmark_gzip(data: np.ndarray) -> dict:
    """Compress data with gzip and report ratio + timing."""
    raw_bytes = _to_csv_bytes(data)
    t0 = time.perf_counter()
    compressed = gzip.compress(raw_bytes, compresslevel=9)
    t1 = time.perf_counter()

    return {
        "method": "gzip",
        "raw_bytes": len(raw_bytes),
        "compressed_bytes": len(compressed),
        "ratio": len(raw_bytes) / len(compressed),
        "compress_time_sec": t1 - t0,
    }


def benchmark_zstd(data: np.ndarray) -> Optional[dict]:
    """Compress data with zstandard (if available)."""
    try:
        import zstandard as zstd
    except ImportError:
        return None

    raw_bytes = _to_csv_bytes(data)
    compressor = zstd.ZstdCompressor(level=22)
    t0 = time.perf_counter()
    compressed = compressor.compress(raw_bytes)
    t1 = time.perf_counter()

    return {
        "method": "zstd",
        "raw_bytes": len(raw_bytes),
        "compressed_bytes": len(compressed),
        "ratio": len(raw_bytes) / len(compressed),
        "compress_time_sec": t1 - t0,
    }


def benchmark_numpy_raw(data: np.ndarray) -> dict:
    """Raw numpy binary storage baseline."""
    raw_csv = _to_csv_bytes(data)
    flat = data.astype(np.float32)
    binary_bytes = flat.tobytes()

    return {
        "method": "numpy_binary",
        "csv_bytes": len(raw_csv),
        "binary_bytes": len(binary_bytes),
        "csv_to_binary_ratio": len(raw_csv) / len(binary_bytes),
    }


def run_all_baselines(data: np.ndarray) -> list[dict]:
    """Run all baseline compression benchmarks."""
    results = [
        benchmark_gzip(data),
        benchmark_numpy_raw(data),
    ]
    zstd_result = benchmark_zstd(data)
    if zstd_result is not None:
        results.append(zstd_result)
    return results
