#!/usr/bin/env python3
"""Experiment v4: Hourly data with 256-candle windows.

The spec targets (>50x compression, <0.1% RMSE) were designed for higher-
resolution data where temporal redundancy is much greater. This experiment
uses hourly bars with 256-candle windows (1280 ambient dims vs 320 for
daily 64-candle) to demonstrate significantly better compression ratios.

Two-phase approach:
  Phase 1: Train conv autoencoder (pure MSE) for best reconstruction
  Phase 2: PCA on latent codes to discover intrinsic dimensionality
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

from umc.config import UMCConfig
from umc.encoder.conv_encoder import ConvEncoder
from umc.decoder.conv_decoder import ConvDecoder
from umc.data.loaders import load_yahoo_finance, combine_datasets
from umc.data.preprocessors import OHLCVPreprocessor, create_windows, WindowDataset
from umc.storage.mnf_format import MNFWriter, MNFReader
from umc.storage.manifest import DecoderManifest
from umc.processor.search import ManifoldSearch
from umc.processor.cluster import ManifoldCluster
from umc.processor.anomaly import ManifoldAnomalyDetector
from umc.evaluation.benchmarks import run_all_baselines


def train_autoencoder(encoder, decoder, train_loader, val_loader, config, device, epochs=300):
    """Phase 1: Pure autoencoder training with strong noise augmentation."""
    encoder.to(device)
    decoder.to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
        weight_decay=5e-3,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate * 3,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        if hasattr(encoder, 'gumbel_temperature'):
            progress = epoch / max(1, epochs - 1)
            encoder.gumbel_temperature = max(0.1, 1.0 - 0.9 * progress)

        # Strong noise augmentation: higher than daily due to more params vs data
        noise_std = max(0.02, 0.2 * (1 - epoch / epochs))

        encoder.train()
        decoder.train()
        train_loss_sum = 0
        n_batches = 0

        for batch in train_loader:
            x = batch.to(device)
            x_noisy = x + noise_std * torch.randn_like(x)

            enc = encoder.encode(x_noisy)
            x_hat = decoder.decode(enc.z, enc.chart_id)

            weights = torch.ones(config.n_features, device=device)
            features = list(config.features)
            if "close" in features:
                weights[features.index("close")] = 2.0
            weights = weights / weights.sum() * len(features)

            diff = (x - x_hat) ** 2
            loss = (diff * weights.unsqueeze(0).unsqueeze(0)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            n_batches += 1

        avg_train = train_loss_sum / n_batches

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
            gumbel_t = getattr(encoder, 'gumbel_temperature', 0)
            print(
                f"  Epoch {epoch:4d} | Train MSE: {avg_train:.6f} | "
                f"Val MSE: {avg_val:.6f} | LR: {current_lr:.2e} | "
                f"Gumbel-T: {gumbel_t:.3f}"
            )

        if patience_counter >= 80:
            print(f"  Early stopping at epoch {epoch}")
            break

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
    print(f"=== UMC Experiment v4: Hourly Data + 256-Candle Windows ===")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print()

    WINDOW_SIZE = 256
    LATENT_DIM = 64

    config = UMCConfig(
        window_size=WINDOW_SIZE,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=LATENT_DIM,
        encoder_type="conv",
        decoder_type="conv",
        num_charts=16,
        chart_embedding_dim=16,
        batch_size=128,
        learning_rate=3e-4,
        beta_start=0.0, beta_end=0.0,
        sparsity_weight=0.0, smoothness_weight=0.0,
        epochs=300,
    )

    # === Download Hourly Data ===
    # yfinance supports 1h for up to 730 days
    symbols = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "BTC-USD", "ETH-USD", "GC=F", "CL=F"]
    print(f"Downloading HOURLY data: {symbols}")
    print(f"  (1h interval, 2y period — yfinance max for hourly)")
    datasets = load_yahoo_finance(symbols, period="2y", interval="1h")
    df = combine_datasets(datasets)
    print(f"  Total rows: {len(df):,}")
    print(f"  Ambient dim per window: {WINDOW_SIZE} × {config.n_features} = {WINDOW_SIZE * config.n_features}")
    print()

    # === Preprocess ===
    preprocessor = OHLCVPreprocessor(config)
    normalized = preprocessor.fit_transform(df)
    # stride=4 for denser overlap — gives ~18K windows from 75K hourly rows
    windows = create_windows(normalized, config.window_size, stride=4)
    print(f"Windows: {windows.shape} (stride=4)")
    print(f"Normalized: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
    print(f"  range=[{normalized.min():.2f}, {normalized.max():.2f}]")

    if len(windows) < 100:
        print(f"ERROR: Not enough windows ({len(windows)}). Need at least 100.")
        print("  Try reducing window_size or stride, or using more symbols.")
        return

    # Split 80/10/10 with temporal gaps to reduce data leakage
    n = len(windows)
    gap = WINDOW_SIZE // 4  # gap between splits to reduce leakage from overlapping windows
    n_test = max(int(n * 0.1), 10)
    n_val = max(int(n * 0.1), 10)
    n_train = n - n_val - n_test - 2 * gap

    train_windows = windows[:n_train]
    val_windows = windows[n_train + gap:n_train + gap + n_val]
    test_windows = windows[n_train + 2 * gap + n_val:]
    n_test = len(test_windows)  # adjust for gap

    train_loader = DataLoader(WindowDataset(train_windows), batch_size=config.batch_size,
                              shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(WindowDataset(val_windows),
                            batch_size=config.batch_size, pin_memory=True)
    test_loader = DataLoader(WindowDataset(test_windows),
                             batch_size=config.batch_size, pin_memory=True)
    all_loader = DataLoader(WindowDataset(np.concatenate([train_windows, val_windows])),
                            batch_size=config.batch_size, pin_memory=True)

    print(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")

    # === Phase 1: Conv Autoencoder ===
    print(f"\n=== Phase 1: 1D-CNN Autoencoder Training (window={WINDOW_SIZE}) ===")
    encoder = ConvEncoder(config)
    decoder = ConvDecoder(config)

    n_params_enc = sum(p.numel() for p in encoder.parameters())
    n_params_dec = sum(p.numel() for p in decoder.parameters())
    n_params = n_params_enc + n_params_dec
    print(f"  Encoder params: {n_params_enc:,} (3 down-blocks)")
    print(f"  Decoder params: {n_params_dec:,} (3 up-blocks)")
    print(f"  Total params: {n_params:,}")

    t0 = time.perf_counter()
    best_val = train_autoencoder(encoder, decoder, train_loader, val_loader, config, device, epochs=300)
    t_train = time.perf_counter() - t0
    print(f"  Best val MSE: {best_val:.6f}")
    print(f"  Training time: {t_train:.1f}s")

    # === Encode Everything ===
    print(f"\n=== Encoding Full Dataset ===")
    z_trainval, x_trainval, _, _ = encode_all(encoder, all_loader, device)
    z_test, x_test, chart_test, conf_test = encode_all(encoder, test_loader, device)

    print(f"  Latent shape: {z_trainval.shape}")
    print(f"  z stats: mean={z_trainval.mean():.4f}, std={z_trainval.std():.4f}")

    # Full-dim reconstruction
    decoder.eval()
    decoder.to(device)
    with torch.no_grad():
        z_tensor_full = torch.from_numpy(z_test.astype(np.float32)).to(device)
        chart_tensor = torch.from_numpy(chart_test.astype(np.int64)).to(device)
        x_hat_full = decoder.decode(z_tensor_full, chart_tensor).cpu().numpy()

    rmse_full = np.sqrt(np.mean((x_test - x_hat_full) ** 2))
    data_range = x_test.max() - x_test.min()
    rmse_full_pct = rmse_full / data_range * 100
    print(f"  Full-dim RMSE: {rmse_full:.6f} ({rmse_full_pct:.4f}% of range)")

    for i, feat in enumerate(config.features):
        feat_rmse = np.sqrt(np.mean((x_test[:, :, i] - x_hat_full[:, :, i]) ** 2))
        feat_range = x_test[:, :, i].max() - x_test[:, :, i].min()
        feat_pct = feat_rmse / max(feat_range, 1e-8) * 100
        print(f"    {feat}: RMSE={feat_rmse:.6f} ({feat_pct:.4f}%)")

    # === Phase 2: PCA ===
    print(f"\n=== Phase 2: PCA Dimensionality Analysis ===")
    pca_full = PCA(n_components=min(config.max_latent_dim, z_trainval.shape[0] - 1))
    pca_full.fit(z_trainval)

    explained_var = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    for threshold in [0.90, 0.95, 0.99, 0.999]:
        idx = np.searchsorted(cumulative_var, threshold)
        n_dims = int(idx) + 1 if idx < len(cumulative_var) else len(cumulative_var)
        print(f"  Dims for {threshold*100:.1f}% variance: {n_dims}")

    target_dims = int(np.searchsorted(cumulative_var, 0.99)) + 1
    target_dims = min(target_dims, len(cumulative_var))
    print(f"\n  Selected: {target_dims} dims (99% variance explained)")

    pca_reduced = PCA(n_components=target_dims)
    pca_reduced.fit(z_trainval)
    z_reduced_test = pca_reduced.transform(z_test)
    z_reconstructed_test = pca_reduced.inverse_transform(z_reduced_test)

    with torch.no_grad():
        z_tensor = torch.from_numpy(z_reconstructed_test.astype(np.float32)).to(device)
        x_hat_reduced = decoder.decode(z_tensor, chart_tensor).cpu().numpy()

    rmse_reduced = np.sqrt(np.mean((x_test - x_hat_reduced) ** 2))
    rmse_reduced_pct = rmse_reduced / data_range * 100

    # === Anomaly detection ===
    print(f"\n=== Reconstruction Confidence Anomaly Detection ===")
    per_sample_errors = np.sqrt(np.mean(
        (x_test.reshape(len(x_test), -1) - x_hat_full.reshape(len(x_hat_full), -1)) ** 2,
        axis=1
    ))
    detector = ManifoldAnomalyDetector(z_test, method="reconstruction_confidence")
    detector.fit_reconstruction_confidence(x_test, x_hat_full)
    anomaly_scores = detector.score(z_test, reconstruction_errors=per_sample_errors)
    n_anomalies = int((anomaly_scores > 0.5).sum())
    print(f"  Anomalies (threshold=0.5): {n_anomalies} / {len(z_test)}")

    # === Compression ===
    ambient_dim = WINDOW_SIZE * config.n_features
    bytes_per_window_raw = ambient_dim * 4
    bytes_per_window_manifold = target_dims * 2 + 1 + 2
    ratio_per_sample = bytes_per_window_raw / bytes_per_window_manifold

    full_raw = n * bytes_per_window_raw
    full_mnf = n * (target_dims * 2 + 3) + 64
    decoder_bytes = n_params * 4
    ratio_full = full_raw / full_mnf
    ratio_system = full_raw / (full_mnf + decoder_bytes)

    raw_binary = x_test.astype(np.float32).tobytes()
    gzip_binary = gzip.compress(raw_binary, compresslevel=9)
    gzip_ratio = len(raw_binary) / len(gzip_binary)

    # === Throughput ===
    n_candles_test = len(z_test) * WINDOW_SIZE

    with torch.no_grad():
        dummy = torch.randn(1, WINDOW_SIZE, config.n_features, device=device)
        encoder.encode(dummy)

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            x = batch.to(device)
            enc = encoder.encode(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        z_t = torch.from_numpy(z_test.astype(np.float32)).to(device)
        c_t = torch.from_numpy(chart_test.astype(np.int64)).to(device)
        decoder.decode(z_t, c_t)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_decode = time.perf_counter() - t0

    enc_throughput = n_candles_test / max(t_encode, 1e-8)
    dec_throughput = n_candles_test / max(t_decode, 1e-8)

    # === Search speedup ===
    z_reduced_trainval = pca_reduced.transform(z_trainval)
    z_reduced_all = np.vstack([z_reduced_trainval, z_reduced_test])
    search = ManifoldSearch(z_reduced_all)

    ambient_all = windows[:len(z_reduced_all)].reshape(len(z_reduced_all), -1)

    n_iters = min(1000, len(z_reduced_all))
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

    # === .mnf roundtrip ===
    import tempfile, os
    decoder_hash = DecoderManifest.compute_hash_bytes(decoder.state_dict())
    with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
        mnf_path = f.name
    try:
        writer = MNFWriter()
        bytes_written = writer.write(
            mnf_path,
            z_reduced_test.astype(np.float32),
            chart_test.astype(np.uint8),
            decoder_hash,
            confidences=conf_test,
        )
        reader = MNFReader()
        mnf = reader.read(mnf_path)
        mnf_ok = mnf.header.n_samples == len(z_reduced_test)
    finally:
        os.unlink(mnf_path)

    # === Summary ===
    print(f"\n{'='*72}")
    print(f"  UMC Experiment v4: Hourly Data Results")
    print(f"  Window: {WINDOW_SIZE} candles × {config.n_features} features = {ambient_dim} ambient dims")
    print(f"{'='*72}")
    print(f"{'METRIC':<40} {'VALUE':>15} {'TARGET':>8} {'':>6}")
    print(f"{'-'*72}")
    print(f"{'Architecture':<40} {'Conv1D-3blk':>15}")
    print(f"{'Model params':<40} {n_params:>15,}")
    print(f"{'Training time':<40} {t_train:>14.1f}s")
    print(f"{'Total windows':<40} {n:>15,}")
    print(f"{'Intrinsic dims (PCA 99%)':<40} {target_dims:>15}")
    print(f"{'Dimensionality ratio':<40} {dim_ratio:>14.1f}x")
    print(f"{'-'*72}")
    print(f"{'RMSE % (full dims)':<40} {rmse_full_pct:>14.4f}% {'<0.1%':>8} {'PASS' if rmse_full_pct < 0.1 else 'MISS':>6}")
    print(f"{'RMSE % (PCA reduced)':<40} {rmse_reduced_pct:>14.4f}% {'<0.1%':>8} {'PASS' if rmse_reduced_pct < 0.1 else 'MISS':>6}")
    print(f"{'Per-sample compression':<40} {ratio_per_sample:>14.1f}x {'>50x':>8} {'PASS' if ratio_per_sample > 50 else 'MISS':>6}")
    print(f"{'Full dataset ratio':<40} {ratio_full:>14.1f}x")
    print(f"{'System ratio (+ decoder)':<40} {ratio_system:>14.1f}x")
    print(f"{'gzip (binary) baseline':<40} {gzip_ratio:>14.1f}x")
    print(f"{'Encode (candles/s)':<40} {enc_throughput:>15,.0f} {'>10K':>8} {'PASS' if enc_throughput > 10000 else 'MISS':>6}")
    print(f"{'Decode (candles/s)':<40} {dec_throughput:>15,.0f} {'>50K':>8} {'PASS' if dec_throughput > 50000 else 'MISS':>6}")
    print(f"{'NN search speedup':<40} {speedup:>14.1f}x {'>100x':>8} {'PASS' if speedup > 100 else 'MISS':>6}")
    print(f"{'MNF roundtrip':<40} {'OK' if mnf_ok else 'FAIL':>15}")
    print(f"{'='*72}")

    # Comparison with daily
    print(f"\n  Comparison (daily 64-window -> hourly 256-window):")
    print(f"    Ambient dims:      320 -> {ambient_dim}")
    print(f"    Intrinsic dims:    ~59 -> {target_dims}")
    print(f"    Dim ratio:         5.4x -> {dim_ratio:.1f}x")

    # Save
    Path("results").mkdir(exist_ok=True)
    results = {
        "experiment": "hourly_256",
        "interval": "1h",
        "window_size": WINDOW_SIZE,
        "ambient_dim": ambient_dim,
        "architecture": "conv1d_3blk",
        "n_params": n_params,
        "intrinsic_dims_pca99": target_dims,
        "dim_ratio": float(dim_ratio),
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
    with open("results/experiment_hourly.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/experiment_hourly.json")


if __name__ == "__main__":
    main()
