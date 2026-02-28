#!/usr/bin/env python3
"""Two-phase experiment:
  Phase 1: Train pure autoencoder (no KL) for best reconstruction
  Phase 2: PCA on latent codes to discover intrinsic dimensionality
  Then evaluate compression, search speedup, and reconstruction.
"""

import sys
import time
import json
import gzip
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from tqdm import tqdm

from umc.config import UMCConfig
from umc.encoder.vae import VAEEncoder
from umc.decoder.mlp_decoder import MLPDecoder
from umc.data.loaders import load_yahoo_finance, combine_datasets
from umc.data.preprocessors import OHLCVPreprocessor, create_windows, WindowDataset
from umc.storage.mnf_format import MNFWriter, MNFReader
from umc.storage.manifest import DecoderManifest
from umc.processor.search import ManifoldSearch
from umc.processor.cluster import ManifoldCluster
from umc.evaluation.benchmarks import run_all_baselines


def train_autoencoder(encoder, decoder, train_loader, val_loader, config, device, epochs=300):
    """Phase 1: Pure autoencoder training (no KL, no sparsity)."""
    encoder.to(device)
    decoder.to(device)

    optimizer = Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15,
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        encoder.train()
        decoder.train()
        train_loss_sum = 0
        n_batches = 0

        for batch in train_loader:
            x = batch.to(device)
            enc = encoder.encode(x)
            x_hat = decoder.decode(enc.z, enc.chart_id)

            # Pure MSE reconstruction loss — no KL, no sparsity
            loss = nn.functional.mse_loss(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0
            )
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        avg_train = train_loss_sum / n_batches

        # Validate
        encoder.eval()
        decoder.eval()
        val_loss_sum = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                enc = encoder.encode(x)
                x_hat = decoder.decode(enc.z, enc.chart_id)
                val_loss_sum += nn.functional.mse_loss(x_hat, x).item()
                n_val += 1
        avg_val = val_loss_sum / max(n_val, 1)

        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = {
                'encoder': {k: v.clone() for k, v in encoder.state_dict().items()},
                'decoder': {k: v.clone() for k, v in decoder.state_dict().items()},
            }
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:4d} | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f} | LR: {current_lr:.2e}")

        if patience_counter >= 40:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best
    if best_state:
        encoder.load_state_dict(best_state['encoder'])
        decoder.load_state_dict(best_state['decoder'])

    return best_val_loss


def encode_all(encoder, loader, device):
    """Encode all data through the encoder."""
    encoder.eval()
    all_z, all_x, all_chart, all_conf = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            enc = encoder.encode(x)
            all_z.append(enc.z.cpu().numpy())
            all_x.append(x.cpu().numpy())
            all_chart.append(enc.chart_id.cpu().numpy())
            all_conf.append(enc.confidence.cpu().numpy())
    return (
        np.concatenate(all_z),
        np.concatenate(all_x),
        np.concatenate(all_chart),
        np.concatenate(all_conf),
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== UMC Experiment v2: Autoencoder + PCA ===")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print()

    # === Config ===
    config = UMCConfig(
        window_size=64,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=128,
        encoder_hidden=(512, 256, 128),
        decoder_hidden=(128, 256, 512),
        num_charts=16,
        chart_embedding_dim=16,
        batch_size=512,
        learning_rate=1e-3,
        # These are ignored in phase 1 but kept for config compatibility
        beta_start=0.0, beta_end=0.0,
        sparsity_weight=0.0, smoothness_weight=0.0,
        epochs=300,
    )

    # === Download Data ===
    symbols = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "BTC-USD", "ETH-USD", "GC=F", "CL=F"]
    print(f"Downloading data: {symbols}")
    datasets = load_yahoo_finance(symbols, period="5y", interval="1d")
    df = combine_datasets(datasets)
    print(f"  Total rows: {len(df):,}")
    print()

    # === Preprocess ===
    preprocessor = OHLCVPreprocessor(config)
    normalized = preprocessor.fit_transform(df)
    windows = create_windows(normalized, config.window_size, stride=1)
    print(f"Windows: {windows.shape}")
    print(f"Normalized: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
    print(f"  range=[{normalized.min():.2f}, {normalized.max():.2f}]")

    # Split 80/10/10
    n = len(windows)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)
    n_train = n - n_val - n_test

    train_loader = DataLoader(WindowDataset(windows[:n_train]), batch_size=config.batch_size,
                              shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(WindowDataset(windows[n_train:n_train + n_val]),
                            batch_size=config.batch_size, pin_memory=True)
    test_loader = DataLoader(WindowDataset(windows[n_train + n_val:]),
                             batch_size=config.batch_size, pin_memory=True)
    all_loader = DataLoader(WindowDataset(windows[:n_train + n_val]),
                            batch_size=config.batch_size, pin_memory=True)

    print(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")

    # === Phase 1: Pure Autoencoder ===
    print(f"\n=== Phase 1: Autoencoder Training ===")
    encoder = VAEEncoder(config)
    decoder = MLPDecoder(config)

    n_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
    print(f"  Parameters: {n_params:,}")

    t0 = time.perf_counter()
    best_val = train_autoencoder(encoder, decoder, train_loader, val_loader, config, device, epochs=300)
    t_train = time.perf_counter() - t0
    print(f"  Best val MSE: {best_val:.6f}")
    print(f"  Training time: {t_train:.1f}s")

    # === Encode Everything ===
    print(f"\n=== Encoding Full Dataset ===")
    # Encode train+val for PCA fitting
    z_trainval, x_trainval, _, _ = encode_all(encoder, all_loader, device)
    z_test, x_test, chart_test, conf_test = encode_all(encoder, test_loader, device)

    print(f"  Latent shape: {z_trainval.shape}")
    print(f"  z stats: mean={z_trainval.mean():.4f}, std={z_trainval.std():.4f}")

    # === Phase 2: PCA Dimensionality Reduction ===
    print(f"\n=== Phase 2: PCA Dimensionality Analysis ===")
    pca_full = PCA(n_components=config.max_latent_dim)
    pca_full.fit(z_trainval)

    explained_var = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # Find dims needed for various thresholds
    for threshold in [0.90, 0.95, 0.99, 0.999]:
        n_dims = int(np.searchsorted(cumulative_var, threshold)) + 1
        print(f"  Dims for {threshold*100:.1f}% variance: {n_dims}")

    # Use 99% variance as our target
    target_dims = int(np.searchsorted(cumulative_var, 0.99)) + 1
    print(f"\n  Selected: {target_dims} dims (99% variance explained)")

    # Project latent codes to reduced space
    pca_reduced = PCA(n_components=target_dims)
    pca_reduced.fit(z_trainval)
    z_reduced_trainval = pca_reduced.transform(z_trainval)
    z_reduced_test = pca_reduced.transform(z_test)

    # === Reconstruction with reduced dims ===
    # Inverse PCA -> decode
    z_reconstructed_test = pca_reduced.inverse_transform(z_reduced_test)
    decoder.eval()
    with torch.no_grad():
        z_tensor = torch.from_numpy(z_reconstructed_test.astype(np.float32)).to(device)
        chart_tensor = torch.from_numpy(chart_test.astype(np.int64)).to(device)
        x_hat_reduced = decoder.decode(z_tensor, chart_tensor).cpu().numpy()

    # Also get full-dim reconstruction for comparison
    with torch.no_grad():
        z_tensor_full = torch.from_numpy(z_test.astype(np.float32)).to(device)
        x_hat_full = decoder.decode(z_tensor_full, chart_tensor).cpu().numpy()

    # === Metrics ===
    print(f"\n=== Results ===")

    # Reconstruction
    rmse_full = np.sqrt(np.mean((x_test - x_hat_full) ** 2))
    rmse_reduced = np.sqrt(np.mean((x_test - x_hat_reduced) ** 2))
    data_range = x_test.max() - x_test.min()
    rmse_full_pct = rmse_full / data_range * 100
    rmse_reduced_pct = rmse_reduced / data_range * 100

    print(f"\nReconstruction (full {config.max_latent_dim} dims):")
    print(f"  RMSE: {rmse_full:.6f} ({rmse_full_pct:.4f}% of range)")
    print(f"Reconstruction (PCA {target_dims} dims):")
    print(f"  RMSE: {rmse_reduced:.6f} ({rmse_reduced_pct:.4f}% of range)")

    # Per-feature
    for i, feat in enumerate(config.features):
        feat_rmse = np.sqrt(np.mean((x_test[:,:,i] - x_hat_full[:,:,i]) ** 2))
        print(f"  {feat}: {feat_rmse:.6f}")

    # Compression
    bytes_per_window_raw = config.window_size * config.n_features * 4  # float32
    bytes_per_window_manifold = target_dims * 2 + 1 + 2  # float16 coords + chart + conf
    ratio_per_sample = bytes_per_window_raw / bytes_per_window_manifold

    full_raw = n * bytes_per_window_raw
    full_mnf = n * (target_dims * 2 + 3) + 64  # coords + chart + conf + header
    decoder_bytes = n_params * 4
    ratio_full = full_raw / full_mnf
    ratio_system = full_raw / (full_mnf + decoder_bytes)

    print(f"\nCompression:")
    print(f"  Raw bytes/window: {bytes_per_window_raw}")
    print(f"  Manifold bytes/window: {bytes_per_window_manifold} ({target_dims} dims × 2 bytes)")
    print(f"  Per-sample ratio: {ratio_per_sample:.1f}x")
    print(f"  Full dataset: {full_raw:,} bytes -> {full_mnf:,} bytes = {ratio_full:.1f}x")
    print(f"  Decoder: {decoder_bytes:,} bytes ({decoder_bytes/1024/1024:.1f} MB)")
    print(f"  System ratio: {ratio_system:.1f}x")

    # Baselines
    raw_binary = x_test.astype(np.float32).tobytes()
    gzip_binary = gzip.compress(raw_binary, compresslevel=9)
    gzip_ratio = len(raw_binary) / len(gzip_binary)
    print(f"  gzip on float32 binary: {gzip_ratio:.1f}x")

    # Throughput
    n_candles_test = len(z_test) * config.window_size

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            x = batch.to(device)
            enc = encoder.encode(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        z_t = torch.from_numpy(z_test.astype(np.float32)).to(device)
        c_t = torch.from_numpy(chart_test.astype(np.int64)).to(device)
        decoder.decode(z_t, c_t)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_decode = time.perf_counter() - t0

    enc_throughput = n_candles_test / t_encode
    dec_throughput = n_candles_test / t_decode

    print(f"\nThroughput:")
    print(f"  Encode: {enc_throughput:,.0f} candles/sec")
    print(f"  Decode: {dec_throughput:,.0f} candles/sec")

    # Search speedup (using reduced PCA coordinates)
    print(f"\nSearch (using PCA-reduced {target_dims}-dim coords):")

    z_reduced_all = pca_reduced.transform(np.vstack([z_trainval, z_test]))
    search = ManifoldSearch(z_reduced_all)

    ambient_all = windows[:len(z_reduced_all)].reshape(len(z_reduced_all), -1)
    ambient_dim = ambient_all.shape[1]

    n_iters = 1000
    query_m = z_reduced_all[0]
    t0 = time.perf_counter()
    for _ in range(n_iters):
        search.query(query_m, k=10)
    t_manifold = (time.perf_counter() - t0) / n_iters

    query_a = ambient_all[0]
    t0 = time.perf_counter()
    for _ in range(n_iters):
        dists = np.sum((ambient_all - query_a) ** 2, axis=1)
        np.argsort(dists)[:10]
    t_ambient = (time.perf_counter() - t0) / n_iters

    speedup = t_ambient / max(t_manifold, 1e-10)
    dim_ratio = ambient_dim / target_dims

    print(f"  Manifold ({target_dims}d): {t_manifold*1000:.3f} ms/query")
    print(f"  Ambient ({ambient_dim}d): {t_ambient*1000:.3f} ms/query")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Dimension ratio: {dim_ratio:.1f}x")

    # Clustering
    clusterer = ManifoldCluster()
    clusters = clusterer.cluster(z_reduced_all, n_clusters=5)
    regimes = clusterer.find_regimes(z_reduced_all, n_clusters=5)
    print(f"\nClustering: {clusters.n_clusters} clusters, {len(regimes)} regimes")

    # === Final Summary ===
    print(f"\n{'='*65}")
    print(f"{'METRIC':<35} {'VALUE':>15} {'TARGET':>8} {'':>6}")
    print(f"{'='*65}")
    print(f"{'Intrinsic dims (PCA 99%)':<35} {target_dims:>15}")
    print(f"{'Per-sample compression':<35} {ratio_per_sample:>14.1f}x {'>50x':>8} {'PASS' if ratio_per_sample > 50 else 'MISS':>6}")
    print(f"{'Full dataset ratio':<35} {ratio_full:>14.1f}x")
    print(f"{'System ratio (+ decoder)':<35} {ratio_system:>14.1f}x")
    print(f"{'gzip (binary) baseline':<35} {gzip_ratio:>14.1f}x")
    print(f"{'RMSE % (full dims)':<35} {rmse_full_pct:>14.4f}% {'<0.1%':>8} {'PASS' if rmse_full_pct < 0.1 else 'MISS':>6}")
    print(f"{'RMSE % (PCA reduced)':<35} {rmse_reduced_pct:>14.4f}% {'<0.1%':>8} {'PASS' if rmse_reduced_pct < 0.1 else 'MISS':>6}")
    print(f"{'Encode (candles/s)':<35} {enc_throughput:>15,.0f} {'>10K':>8} {'PASS' if enc_throughput > 10000 else 'MISS':>6}")
    print(f"{'Decode (candles/s)':<35} {dec_throughput:>15,.0f} {'>50K':>8} {'PASS' if dec_throughput > 50000 else 'MISS':>6}")
    print(f"{'NN search speedup':<35} {speedup:>14.1f}x {'>100x':>8} {'PASS' if speedup > 100 else 'MISS':>6}")
    print(f"{'Training time':<35} {t_train:>14.1f}s")
    print(f"{'='*65}")

    # Save
    Path("results").mkdir(exist_ok=True)
    results = {
        "intrinsic_dims_pca99": target_dims,
        "pca_explained_variance": cumulative_var[:20].tolist(),
        "rmse_full": float(rmse_full),
        "rmse_full_pct": float(rmse_full_pct),
        "rmse_reduced": float(rmse_reduced),
        "rmse_reduced_pct": float(rmse_reduced_pct),
        "per_sample_ratio": float(ratio_per_sample),
        "full_dataset_ratio": float(ratio_full),
        "system_ratio": float(ratio_system),
        "gzip_binary_ratio": float(gzip_ratio),
        "encode_throughput": float(enc_throughput),
        "decode_throughput": float(dec_throughput),
        "search_speedup": float(speedup),
        "training_time_sec": float(t_train),
        "n_total_windows": n,
    }
    with open("results/experiment_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/experiment_v2.json")


if __name__ == "__main__":
    main()
