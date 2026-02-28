"""Evaluation metrics for UMC: compression ratio, RMSE, throughput, etc."""

import time
from typing import Optional

import numpy as np


def reconstruction_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Root mean squared error between original and reconstructed data."""
    return float(np.sqrt(np.mean((original - reconstructed) ** 2)))


def reconstruction_rmse_per_feature(
    original: np.ndarray,
    reconstructed: np.ndarray,
    feature_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """Per-feature RMSE.

    Args:
        original: (n_samples, window_size, n_features) or (n_samples, n_features).
        reconstructed: Same shape.
        feature_names: Names for each feature dimension.

    Returns:
        Dict mapping feature name -> RMSE.
    """
    if original.ndim == 3:
        # (n, window, features) -> flatten to (n * window, features)
        n_features = original.shape[-1]
        orig_flat = original.reshape(-1, n_features)
        recon_flat = reconstructed.reshape(-1, n_features)
    else:
        n_features = original.shape[-1]
        orig_flat = original
        recon_flat = reconstructed

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    result = {}
    for i, name in enumerate(feature_names):
        result[name] = float(np.sqrt(np.mean((orig_flat[:, i] - recon_flat[:, i]) ** 2)))
    return result


def compression_ratio(
    raw_bytes: int,
    manifold_bytes: int,
    decoder_bytes: int = 0,
) -> dict[str, float]:
    """Compute compression ratios.

    Args:
        raw_bytes: Size of raw data in bytes.
        manifold_bytes: Size of manifold coordinates in bytes.
        decoder_bytes: Size of decoder model weights in bytes.

    Returns:
        Dict with coordinate-only and system compression ratios.
    """
    coord_ratio = raw_bytes / max(manifold_bytes, 1)
    system_ratio = raw_bytes / max(manifold_bytes + decoder_bytes, 1)
    return {
        "coordinate_ratio": float(coord_ratio),
        "system_ratio": float(system_ratio),
        "raw_bytes": raw_bytes,
        "manifold_bytes": manifold_bytes,
        "decoder_bytes": decoder_bytes,
    }


def effective_dimensionality(
    latents: np.ndarray,
    threshold: float = 0.01,
) -> dict[str, object]:
    """Compute the effective dimensionality of the latent space.

    Args:
        latents: (n_samples, latent_dim) array of latent coordinates.
        threshold: Variance threshold below which a dim is considered collapsed.

    Returns:
        Dict with active_dims, variance_per_dim, etc.
    """
    var_per_dim = np.var(latents, axis=0)
    active_mask = var_per_dim > threshold
    active_dims = int(np.sum(active_mask))

    # Participation ratio: a soft measure of effective dimensions
    # PR = (sum of variances)^2 / sum of variances^2
    total_var = np.sum(var_per_dim)
    if total_var > 0:
        participation_ratio = float(total_var ** 2 / np.sum(var_per_dim ** 2))
    else:
        participation_ratio = 0.0

    return {
        "active_dims": active_dims,
        "total_dims": latents.shape[1],
        "participation_ratio": participation_ratio,
        "variance_per_dim": var_per_dim.tolist(),
    }


def throughput_benchmark(
    encode_fn,
    decode_fn,
    data: np.ndarray,
    n_runs: int = 5,
    window_size: int = 64,
) -> dict[str, float]:
    """Benchmark encoding and decoding throughput.

    Args:
        encode_fn: Callable that takes data and returns latents.
        decode_fn: Callable that takes latents and returns reconstructed data.
        data: Input data array.
        n_runs: Number of runs for averaging.
        window_size: Candles per window (for candles/sec calculation).

    Returns:
        Dict with throughput metrics.
    """
    n_samples = data.shape[0]

    # Encode benchmark
    encode_times = []
    latents = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        latents = encode_fn(data)
        t1 = time.perf_counter()
        encode_times.append(t1 - t0)

    # Decode benchmark
    decode_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        decode_fn(latents)
        t1 = time.perf_counter()
        decode_times.append(t1 - t0)

    avg_encode = np.mean(encode_times)
    avg_decode = np.mean(decode_times)

    return {
        "encode_time_sec": float(avg_encode),
        "decode_time_sec": float(avg_decode),
        "encode_samples_per_sec": float(n_samples / (avg_encode + 1e-8)),
        "decode_samples_per_sec": float(n_samples / (avg_decode + 1e-8)),
        "encode_candles_per_sec": float(n_samples * window_size / (avg_encode + 1e-8)),
        "decode_candles_per_sec": float(n_samples * window_size / (avg_decode + 1e-8)),
    }


def compute_all_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    latents: np.ndarray,
    raw_bytes: int,
    manifold_bytes: int,
    decoder_bytes: int = 0,
    feature_names: Optional[list[str]] = None,
) -> dict:
    """Compute all evaluation metrics in one call.

    Returns:
        Comprehensive dict of all metrics.
    """
    return {
        "rmse": reconstruction_rmse(original, reconstructed),
        "rmse_per_feature": reconstruction_rmse_per_feature(
            original, reconstructed, feature_names
        ),
        "compression": compression_ratio(raw_bytes, manifold_bytes, decoder_bytes),
        "dimensionality": effective_dimensionality(latents),
    }
