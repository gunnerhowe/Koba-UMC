#!/usr/bin/env python3
"""Real-world experiment: train UMC on financial data and evaluate all targets."""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from umc.config import UMCConfig
from umc.encoder.vae import VAEEncoder
from umc.decoder.mlp_decoder import MLPDecoder
from umc.training.trainer import UMCTrainer
from umc.data.loaders import load_yahoo_finance, combine_datasets
from umc.data.preprocessors import OHLCVPreprocessor, create_windows, WindowDataset
from umc.storage.mnf_format import MNFWriter, MNFReader
from umc.storage.manifest import DecoderManifest
from umc.processor.search import ManifoldSearch
from umc.processor.cluster import ManifoldCluster
from umc.processor.anomaly import ManifoldAnomalyDetector
from umc.evaluation.benchmarks import run_all_baselines


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== UMC Real-World Experiment ===")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # === Configuration ===
    config = UMCConfig(
        window_size=64,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=128,
        min_latent_dim=8,
        encoder_hidden=(512, 256, 128),
        decoder_hidden=(128, 256, 512),
        num_charts=16,
        chart_embedding_dim=16,
        batch_size=512,
        learning_rate=1e-3,
        epochs=300,
        beta_start=0.0,
        beta_end=0.05,            # Very low KL — prioritize reconstruction
        beta_warmup_epochs=100,   # Slow warmup
        sparsity_weight=0.001,
        smoothness_weight=0.0,
        early_stopping_patience=50,
        close_weight=2.0,
    )

    # === Download Data ===
    symbols = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "BTC-USD", "ETH-USD", "GC=F", "CL=F"]
    print(f"Downloading data for {symbols}...")
    datasets = load_yahoo_finance(symbols, period="5y", interval="1d")
    df = combine_datasets(datasets)
    print(f"  Total rows: {len(df):,}")
    for sym, sdf in datasets.items():
        print(f"  {sym}: {len(sdf):,} rows")
    print()

    # === Preprocess ===
    print("Preprocessing...")
    preprocessor = OHLCVPreprocessor(config)
    normalized = preprocessor.fit_transform(df)
    windows = create_windows(normalized, config.window_size, stride=1)
    print(f"  Windows: {windows.shape}")
    print(f"  Normalized stats: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
    print()

    # Split: 80/10/10
    n = len(windows)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)
    n_train = n - n_val - n_test

    train_windows = windows[:n_train]
    val_windows = windows[n_train:n_train + n_val]
    test_windows = windows[n_train + n_val:]

    print(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")

    train_loader = DataLoader(
        WindowDataset(train_windows), batch_size=config.batch_size,
        shuffle=True, drop_last=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        WindowDataset(val_windows), batch_size=config.batch_size,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        WindowDataset(test_windows), batch_size=config.batch_size,
        num_workers=0, pin_memory=True,
    )

    # === Train ===
    print("\n=== Training ===")
    encoder = VAEEncoder(config)
    decoder = MLPDecoder(config)

    n_params_enc = sum(p.numel() for p in encoder.parameters())
    n_params_dec = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder params: {n_params_enc:,}")
    print(f"Decoder params: {n_params_dec:,}")
    print(f"Total params: {n_params_enc + n_params_dec:,}")
    print()

    trainer = UMCTrainer(encoder, decoder, config, device=str(device))

    t_start = time.perf_counter()
    history = trainer.train(train_loader, val_loader, verbose=True)
    t_train = time.perf_counter() - t_start
    print(f"\nTraining time: {t_train:.1f}s ({t_train/60:.1f} min)")

    # Load best checkpoint
    trainer.load_checkpoint("best")

    # === Evaluate on Test Set ===
    print("\n=== Evaluation on Test Set ===")
    encoder.eval()
    decoder.eval()

    all_z = []
    all_x = []
    all_x_hat = []
    all_chart_ids = []
    all_confidences = []
    encode_times = []
    decode_times = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch.to(device)

            t0 = time.perf_counter()
            enc = encoder.encode(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            encode_times.append(t1 - t0)

            t0 = time.perf_counter()
            x_hat = decoder.decode(enc.z, enc.chart_id)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            decode_times.append(t1 - t0)

            all_z.append(enc.z.cpu().numpy())
            all_x.append(x.cpu().numpy())
            all_x_hat.append(x_hat.cpu().numpy())
            all_chart_ids.append(enc.chart_id.cpu().numpy())
            all_confidences.append(enc.confidence.cpu().numpy())

    z = np.concatenate(all_z)
    original = np.concatenate(all_x)
    reconstructed = np.concatenate(all_x_hat)
    chart_ids = np.concatenate(all_chart_ids)
    confidences = np.concatenate(all_confidences)

    # === Reconstruction Quality ===
    rmse = np.sqrt(np.mean((original - reconstructed) ** 2))
    data_range = original.max() - original.min()
    rmse_pct = rmse / data_range * 100

    # Per-feature RMSE
    feature_names = list(config.features)
    per_feat_rmse = {}
    for i, feat in enumerate(feature_names):
        feat_rmse = np.sqrt(np.mean((original[:, :, i] - reconstructed[:, :, i]) ** 2))
        per_feat_rmse[feat] = feat_rmse

    print(f"\nReconstruction:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  RMSE as % of range: {rmse_pct:.4f}%")
    print(f"  Target: <0.1% | {'PASS' if rmse_pct < 0.1 else 'MISS' if rmse_pct < 1.0 else 'FAIL'}")
    for feat, val in per_feat_rmse.items():
        print(f"  {feat}: {val:.6f}")

    # === Effective Dimensionality ===
    var_per_dim = np.var(z, axis=0)
    sorted_var = np.sort(var_per_dim)[::-1]
    var_threshold = sorted_var[0] * 0.01
    active_dims = int(np.sum(sorted_var > var_threshold))

    print(f"\nDimensionality:")
    print(f"  Active dims: {active_dims} / {config.max_latent_dim}")
    print(f"  Top 10 variances: {np.round(sorted_var[:10], 6)}")
    print(f"  Dim 50 variance: {sorted_var[min(49, len(sorted_var)-1)]:.8f}")

    # === Compression Ratio ===
    raw_csv_bytes = len(df.to_csv().encode())
    raw_binary_bytes = original.nbytes
    # Full dataset binary size (all windows, not just test)
    full_raw_binary = n * config.window_size * config.n_features * 4  # float32

    # Coords using only active dims
    coord_bytes_active = z.shape[0] * active_dims * 2  # float16
    coord_bytes_full = z.shape[0] * config.max_latent_dim * 2  # float16
    decoder_bytes = (n_params_enc + n_params_dec) * 4  # float32 weights
    mnf_overhead = z.shape[0] * 1 + z.shape[0] * 2 + 64  # charts + conf + header

    # Per-sample cost
    bytes_per_window_raw = config.window_size * config.n_features * 4
    bytes_per_window_manifold = active_dims * 2 + 1 + 2  # coords + chart + conf

    ratio_per_sample = bytes_per_window_raw / bytes_per_window_manifold

    # Full dataset ratio (amortized decoder)
    full_coord_bytes = n * active_dims * 2
    full_mnf = full_coord_bytes + n * 3 + 64
    ratio_full_dataset = full_raw_binary / max(full_mnf, 1)
    ratio_full_system = full_raw_binary / max(full_mnf + decoder_bytes, 1)

    print(f"\nCompression:")
    print(f"  Bytes per window (raw float32): {bytes_per_window_raw:,}")
    print(f"  Bytes per window (manifold): {bytes_per_window_manifold:,}")
    print(f"  Per-sample ratio: {ratio_per_sample:.1f}x")
    print(f"  Full dataset raw (float32): {full_raw_binary:,} bytes ({full_raw_binary/1024/1024:.1f} MB)")
    print(f"  Full dataset .mnf: {full_mnf:,} bytes ({full_mnf/1024:.1f} KB)")
    print(f"  Full dataset ratio: {ratio_full_dataset:.1f}x")
    print(f"  Decoder size: {decoder_bytes:,} bytes ({decoder_bytes/1024/1024:.1f} MB)")
    print(f"  System ratio (incl decoder): {ratio_full_system:.1f}x")
    print(f"  Target: >50x per-sample | {'PASS' if ratio_per_sample > 50 else 'MISS'}")

    # === Throughput ===
    total_encode_time = sum(encode_times)
    total_decode_time = sum(decode_times)
    n_candles = len(z) * config.window_size

    enc_throughput = n_candles / total_encode_time
    dec_throughput = n_candles / total_decode_time

    print(f"\nThroughput:")
    print(f"  Encode: {enc_throughput:,.0f} candles/sec")
    print(f"  Decode: {dec_throughput:,.0f} candles/sec")
    print(f"  Target encode: >10,000 | {'PASS' if enc_throughput > 10000 else 'MISS'}")
    print(f"  Target decode: >50,000 | {'PASS' if dec_throughput > 50000 else 'MISS'}")

    # === Search Speedup ===
    print(f"\nSearch Speedup:")

    # Use ALL encoded data (not just test) for realistic search benchmark
    print("  Encoding full dataset for search benchmark...")
    all_z_full = []
    encoder.eval()
    with torch.no_grad():
        for loader in [train_loader, val_loader, test_loader]:
            for batch in loader:
                x_b = batch.to(device)
                enc_b = encoder.encode(x_b)
                all_z_full.append(enc_b.z.cpu().numpy())
    z_full = np.concatenate(all_z_full)
    ambient_full = windows[:len(z_full)].reshape(len(z_full), -1)
    print(f"  Search dataset: {len(z_full)} windows")

    # Manifold search (FAISS)
    t0 = time.perf_counter()
    search = ManifoldSearch(z_full)
    t_build_manifold = time.perf_counter() - t0

    n_search_iters = 1000
    query = z_full[0]
    t0 = time.perf_counter()
    for _ in range(n_search_iters):
        result = search.query(query, k=10)
    t_manifold_search = (time.perf_counter() - t0) / n_search_iters

    # Ambient (brute-force numpy) search
    query_ambient = ambient_full[0]
    t0 = time.perf_counter()
    for _ in range(n_search_iters):
        dists = np.sum((ambient_full - query_ambient) ** 2, axis=1)
        top_k = np.argsort(dists)[:10]
    t_ambient_search = (time.perf_counter() - t0) / n_search_iters

    speedup = t_ambient_search / max(t_manifold_search, 1e-10)

    print(f"  Manifold search: {t_manifold_search*1000:.3f} ms/query")
    print(f"  Ambient search: {t_ambient_search*1000:.3f} ms/query")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Target: >100x | {'PASS' if speedup > 100 else 'MISS'}")

    # === Clustering ===
    print(f"\nClustering:")
    clusterer = ManifoldCluster()
    clusters = clusterer.cluster(z, n_clusters=5)
    regimes = clusterer.find_regimes(z, n_clusters=5)
    print(f"  5 clusters, inertia={clusters.inertia:.2f}")
    print(f"  {len(regimes)} regimes detected")

    # === Baseline Comparisons ===
    print(f"\nBaseline Compression (on test windows as CSV text):")
    baselines = run_all_baselines(original)
    for b in baselines:
        if "ratio" in b:
            print(f"  {b['method']}: {b['ratio']:.1f}x (on CSV text)")

    # Fair comparison: compress raw binary float32
    import gzip
    raw_binary = original.astype(np.float32).tobytes()
    gzip_binary = gzip.compress(raw_binary, compresslevel=9)
    gzip_binary_ratio = len(raw_binary) / len(gzip_binary)
    print(f"  gzip (on float32 binary): {gzip_binary_ratio:.1f}x")
    print(f"  UMC per-sample: {ratio_per_sample:.1f}x")
    print(f"  UMC full dataset: {ratio_full_dataset:.1f}x")

    # === Save .mnf ===
    decoder_hash = DecoderManifest.compute_hash_bytes(decoder.state_dict())
    writer = MNFWriter()
    mnf_path = "results/test_data.mnf"
    Path("results").mkdir(exist_ok=True)
    bytes_written = writer.write(
        mnf_path, z, chart_ids.astype(np.uint8), decoder_hash,
        confidences=confidences,
    )
    print(f"\n.mnf file: {mnf_path} ({bytes_written:,} bytes)")

    # Read back and verify
    reader = MNFReader()
    mnf = reader.read(mnf_path)
    print(f"Readback: {mnf.header.n_samples} samples, dim={mnf.header.latent_dim}")

    # === Summary ===
    print(f"\n{'='*65}")
    print(f"{'METRIC':<35} {'VALUE':>15} {'TARGET':>8} {'STATUS':>6}")
    print(f"{'='*65}")
    print(f"{'Per-sample compression':<35} {ratio_per_sample:>14.1f}x {'>50x':>8} {'PASS' if ratio_per_sample > 50 else 'MISS':>6}")
    print(f"{'Full dataset compression':<35} {ratio_full_dataset:>14.1f}x {'':>8} {'':>6}")
    print(f"{'System ratio (+ decoder)':<35} {ratio_full_system:>14.1f}x {'':>8} {'':>6}")
    print(f"{'gzip on binary baseline':<35} {gzip_binary_ratio:>14.1f}x {'':>8} {'':>6}")
    print(f"{'RMSE % of range':<35} {rmse_pct:>14.4f}% {'<0.1%':>8} {'PASS' if rmse_pct < 0.1 else 'MISS':>6}")
    print(f"{'Active dims discovered':<35} {active_dims:>15d} {'':>8} {'':>6}")
    print(f"{'Encode (candles/s)':<35} {enc_throughput:>15,.0f} {'>10K':>8} {'PASS' if enc_throughput > 10000 else 'MISS':>6}")
    print(f"{'Decode (candles/s)':<35} {dec_throughput:>15,.0f} {'>50K':>8} {'PASS' if dec_throughput > 50000 else 'MISS':>6}")
    print(f"{'NN search speedup':<35} {speedup:>14.1f}x {'>100x':>8} {'PASS' if speedup > 100 else 'MISS':>6}")
    print(f"{'Training time':<35} {t_train:>14.1f}s {'':>8} {'':>6}")
    print(f"{'='*65}")

    # Save results JSON
    results = {
        "reconstruction_rmse": float(rmse),
        "rmse_pct_of_range": float(rmse_pct),
        "per_feature_rmse": {k: float(v) for k, v in per_feat_rmse.items()},
        "active_dims": active_dims,
        "max_latent_dim": config.max_latent_dim,
        "per_sample_ratio": float(ratio_per_sample),
        "full_dataset_ratio": float(ratio_full_dataset),
        "system_ratio": float(ratio_full_system),
        "gzip_binary_ratio": float(gzip_binary_ratio),
        "encode_throughput_candles_per_sec": float(enc_throughput),
        "decode_throughput_candles_per_sec": float(dec_throughput),
        "search_speedup": float(speedup),
        "manifold_search_ms": float(t_manifold_search * 1000),
        "ambient_search_ms": float(t_ambient_search * 1000),
        "training_time_sec": float(t_train),
        "n_test_windows": len(z),
        "n_total_windows": n,
        "baselines": baselines,
    }
    with open("results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to results/experiment_results.json")


if __name__ == "__main__":
    main()
