#!/usr/bin/env python3
"""Experiment v6b: Conv-bottleneck architecture for parameter-efficient reconstruction.

Key insight from v6: Widening the FC bottleneck added 13M params and caused massive
overfitting (gap grew to 2.08x by epoch 120, val loss WORSE than v5).

The FC bottleneck is fundamentally wrong — it creates a full-rank matrix between
flat conv features and latent dims, with millions of independent parameters.

Conv-bottleneck approach:
  - 1x1 conv reduces 256 channels → C_red channels (~1K params vs ~8M for FC)
  - Preserves ALL temporal positions (no adaptive pooling, no info loss)
  - Flatten: C_red * compressed_len = latent dim (or small FC to latent)
  - 1.2M total params vs 5.4M FC / 18M wide-FC

The model is lean enough to generalize, while preserving all spatial info.
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from umc.config import UMCConfig
from umc.encoder.conv_encoder import ConvEncoder
from umc.decoder.conv_decoder import ConvDecoder
from umc.data.loaders import load_yahoo_finance, combine_datasets
from umc.data.preprocessors import OHLCVPreprocessor, WindowNormalizer, create_windows, WindowDataset
from umc.storage.mnf_format import MNFWriter, MNFReader
from umc.storage.manifest import DecoderManifest
from umc.processor.search import ManifoldSearch


def train_autoencoder(encoder, decoder, train_loader, val_loader, config, device, epochs=500):
    """Train conv-bottleneck autoencoder."""
    encoder.to(device)
    decoder.to(device)

    optimizer = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
        weight_decay=1e-3,
    )
    # Warmup + cosine decay
    warmup_epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6,
    )

    # Feature weights for loss (close gets 2x)
    weights = torch.ones(config.n_features, device=device)
    features = list(config.features)
    if "close" in features:
        weights[features.index("close")] = 2.0
    weights = weights / weights.sum() * len(features)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # LR warmup
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = config.learning_rate * lr_scale
        else:
            scheduler.step()

        if hasattr(encoder, 'gumbel_temperature'):
            progress = epoch / max(1, epochs - 1)
            encoder.gumbel_temperature = max(0.1, 1.0 - 0.9 * progress)

        # Gentle noise — decays to near-zero
        noise_std = max(0.002, 0.03 * (1 - epoch / epochs))

        encoder.train()
        decoder.train()
        train_loss_sum = 0
        n_batches = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            x_noisy = x + noise_std * torch.randn_like(x)

            enc = encoder.encode(x_noisy)
            x_hat = decoder.decode(enc.z, enc.chart_id)

            # Weighted MSE
            diff = (x - x_hat) ** 2
            loss = (diff * weights.unsqueeze(0).unsqueeze(0)).mean()

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
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)
                enc = encoder.encode(x)
                x_hat = decoder.decode(enc.z, enc.chart_id)
                val_loss_sum += F.mse_loss(x_hat, x).item()
                n_val += 1
        avg_val = val_loss_sum / max(n_val, 1)

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
            current_lr = optimizer.param_groups[0]['lr']
            gumbel_t = getattr(encoder, 'gumbel_temperature', 0)
            print(
                f"  Epoch {epoch:4d} | Train: {avg_train:.6f} | "
                f"Val: {avg_val:.6f} | LR: {current_lr:.2e} | "
                f"Gumbel-T: {gumbel_t:.3f} | Gap: {avg_val/max(avg_train, 1e-8):.2f}x"
            )

        if patience_counter >= 100:
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
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
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
    print(f"=== UMC Experiment v6b: Conv-Bottleneck Architecture ===")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print()

    WINDOW_SIZE = 256
    LATENT_DIM = 128
    C_RED = 16  # Conv bottleneck channels (flat_dim = C_RED * 32 = 512)
    STRIDE = 2

    config = UMCConfig(
        window_size=WINDOW_SIZE,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=LATENT_DIM,
        encoder_type="conv",
        decoder_type="conv",
        num_charts=16,
        chart_embedding_dim=16,
        batch_size=256,
        learning_rate=5e-4,
        per_window_normalize=True,
        # Conv-bottleneck — the key change
        conv_bottleneck=True,
        conv_bottleneck_channels=C_RED,
        # Pure autoencoder — no KL, no sparsity
        beta_start=0.0, beta_end=0.0,
        sparsity_weight=0.0, smoothness_weight=0.0,
        multiscale_weight=0.0, spectral_weight=0.0,
        epochs=500,
    )

    # === Download Hourly Data ===
    symbols = [
        "SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "BTC-USD", "ETH-USD", "SOL-USD",
        "GC=F", "CL=F", "SI=F",
        "EURUSD=X", "GBPUSD=X",
        "TLT", "IEF",
    ]
    print(f"Downloading HOURLY data: {len(symbols)} symbols")
    datasets = load_yahoo_finance(symbols, period="2y", interval="1h")
    df = combine_datasets(datasets)
    print(f"  Total rows: {len(df):,}")
    print()

    # === Preprocess ===
    preprocessor = OHLCVPreprocessor(config)
    normalized = preprocessor.fit_transform(df)
    windows_raw = create_windows(normalized, config.window_size, stride=STRIDE)
    print(f"Windows (stride={STRIDE}): {windows_raw.shape}")

    # === Per-Window Normalization ===
    normalizer = WindowNormalizer()
    windows_norm, win_means, win_stds = normalizer.normalize(windows_raw)
    print(f"After per-window normalization: mean={windows_norm.mean():.4f}, std={windows_norm.std():.4f}")

    if len(windows_norm) < 100:
        print(f"ERROR: Not enough windows ({len(windows_norm)}).")
        return

    # Split with temporal gaps
    n = len(windows_norm)
    gap = WINDOW_SIZE
    n_test = max(int(n * 0.1), 10)
    n_val = max(int(n * 0.1), 10)
    n_train = n - n_val - n_test - 2 * gap

    train_w = windows_norm[:n_train]
    train_m = win_means[:n_train]
    train_s = win_stds[:n_train]

    val_start = n_train + gap
    val_w = windows_norm[val_start:val_start + n_val]
    val_m = win_means[val_start:val_start + n_val]
    val_s = win_stds[val_start:val_start + n_val]

    test_start = val_start + n_val + gap
    test_w = windows_norm[test_start:]
    test_m = win_means[test_start:]
    test_s = win_stds[test_start:]
    n_test = len(test_w)

    test_raw = windows_raw[test_start:]

    train_loader = DataLoader(
        WindowDataset(train_w, train_m, train_s),
        batch_size=config.batch_size, shuffle=True, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        WindowDataset(val_w, val_m, val_s),
        batch_size=config.batch_size, pin_memory=True
    )
    test_loader = DataLoader(
        WindowDataset(test_w, test_m, test_s),
        batch_size=config.batch_size, pin_memory=True
    )
    all_w = np.concatenate([train_w, val_w])
    all_loader = DataLoader(
        WindowDataset(all_w), batch_size=config.batch_size, pin_memory=True
    )

    print(f"Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,} (gap={gap})")

    # === Phase 1: Conv-Bottleneck Autoencoder ===
    print(f"\n=== Phase 1: Conv-Bottleneck AE (c_red={C_RED}, latent={LATENT_DIM}) ===")
    encoder = ConvEncoder(config)
    decoder = ConvDecoder(config)

    n_params_enc = sum(p.numel() for p in encoder.parameters())
    n_params_dec = sum(p.numel() for p in decoder.parameters())
    n_params = n_params_enc + n_params_dec
    print(f"  Encoder: {n_params_enc:,}  Decoder: {n_params_dec:,}  Total: {n_params:,}")
    print(f"  Params/window: {n_params / n_train:.0f} (low = less overfitting)")

    compressed = config.window_size // (2 ** len(encoder.down_blocks))
    flat_dim = C_RED * compressed
    print(f"  Info path: ({256}ch, {compressed}t) -> conv1x1 -> ({C_RED}ch, {compressed}t) -> flat({flat_dim}) -> latent({LATENT_DIM})")

    t0 = time.perf_counter()
    best_val = train_autoencoder(encoder, decoder, train_loader, val_loader, config, device, epochs=500)
    t_train = time.perf_counter() - t0
    print(f"  Best val MSE: {best_val:.6f}")
    print(f"  Training time: {t_train:.1f}s")

    # === Encode Everything ===
    print(f"\n=== Encoding ===")
    z_trainval, x_trainval, _, _ = encode_all(encoder, all_loader, device)
    z_test, x_test_norm, chart_test, conf_test = encode_all(encoder, test_loader, device)

    # Decode in normalized space (full latent dims)
    decoder.eval()
    with torch.no_grad():
        z_tensor = torch.from_numpy(z_test.astype(np.float32)).to(device)
        chart_tensor = torch.from_numpy(chart_test.astype(np.int64)).to(device)
        x_hat_norm = decoder.decode(z_tensor, chart_tensor).cpu().numpy()

    # RMSE in normalized space
    rmse_norm = np.sqrt(np.mean((x_test_norm - x_hat_norm) ** 2))
    norm_range = x_test_norm.max() - x_test_norm.min()
    rmse_norm_pct = rmse_norm / norm_range * 100

    # Denormalize and compute RMSE in original scale
    x_hat_original = normalizer.denormalize(x_hat_norm, test_m, test_s)
    rmse_original = np.sqrt(np.mean((test_raw - x_hat_original) ** 2))
    original_range = test_raw.max() - test_raw.min()
    rmse_original_pct = rmse_original / original_range * 100

    print(f"  Normalized RMSE: {rmse_norm:.6f} ({rmse_norm_pct:.4f}%)")
    print(f"  Original-scale RMSE: {rmse_original:.6f} ({rmse_original_pct:.4f}%)")

    # Per-feature RMSE
    for i, feat in enumerate(config.features):
        feat_rmse = np.sqrt(np.mean((test_raw[:, :, i] - x_hat_original[:, :, i]) ** 2))
        feat_range = test_raw[:, :, i].max() - test_raw[:, :, i].min()
        feat_pct = feat_rmse / max(feat_range, 1e-8) * 100
        print(f"    {feat}: RMSE={feat_rmse:.6f} ({feat_pct:.4f}%)")

    # === Phase 2: PCA ===
    print(f"\n=== Phase 2: PCA ===")
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
    print(f"  Selected: {target_dims} dims (99% variance)")

    pca_reduced = PCA(n_components=target_dims)
    pca_reduced.fit(z_trainval)
    z_reduced_test = pca_reduced.transform(z_test)
    z_reconstructed_test = pca_reduced.inverse_transform(z_reduced_test)

    with torch.no_grad():
        z_pca_tensor = torch.from_numpy(z_reconstructed_test.astype(np.float32)).to(device)
        x_hat_pca_norm = decoder.decode(z_pca_tensor, chart_tensor).cpu().numpy()

    x_hat_pca_original = normalizer.denormalize(x_hat_pca_norm, test_m, test_s)
    rmse_pca = np.sqrt(np.mean((test_raw - x_hat_pca_original) ** 2))
    rmse_pca_pct = rmse_pca / original_range * 100

    # === Metrics ===
    ambient_dim = WINDOW_SIZE * config.n_features
    bytes_per_window_raw = ambient_dim * 4
    scale_factor_bytes = config.n_features * 2 * 2
    bytes_per_window_manifold = target_dims * 2 + 1 + 2 + scale_factor_bytes
    ratio_per_sample = bytes_per_window_raw / bytes_per_window_manifold

    full_raw = n * bytes_per_window_raw
    full_mnf = n * bytes_per_window_manifold + 64
    decoder_bytes = n_params * 4
    ratio_full = full_raw / full_mnf
    ratio_system = full_raw / (full_mnf + decoder_bytes)

    raw_binary = test_raw.astype(np.float32).tobytes()
    gzip_binary = gzip.compress(raw_binary, compresslevel=9)
    gzip_ratio = len(raw_binary) / len(gzip_binary)

    # Throughput
    n_candles_test = len(z_test) * WINDOW_SIZE
    with torch.no_grad():
        dummy = torch.randn(1, WINDOW_SIZE, config.n_features, device=device)
        encoder.encode(dummy)

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            encoder.encode(x)
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

    # Search speedup
    z_reduced_trainval = pca_reduced.transform(z_trainval)
    z_reduced_all = np.vstack([z_reduced_trainval, z_reduced_test])
    search = ManifoldSearch(z_reduced_all)

    all_raw_for_search = np.concatenate([
        windows_raw[:n_train],
        windows_raw[val_start:val_start + n_val],
        windows_raw[test_start:]
    ])[:len(z_reduced_all)]
    ambient_all = all_raw_for_search.reshape(len(z_reduced_all), -1)

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

    # .mnf roundtrip
    import tempfile, os
    decoder_hash = DecoderManifest.compute_hash_bytes(decoder.state_dict())
    with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
        mnf_path = f.name
    try:
        writer = MNFWriter()
        writer.write(
            mnf_path,
            z_reduced_test.astype(np.float32),
            chart_test.astype(np.uint8),
            decoder_hash,
            confidences=conf_test,
            scale_means=test_m,
            scale_stds=test_s,
        )
        reader = MNFReader()
        mnf = reader.read(mnf_path)
        mnf_ok = (
            mnf.header.n_samples == len(z_reduced_test)
            and mnf.header.has_scale_factors
            and mnf.scale_means is not None
        )
    finally:
        os.unlink(mnf_path)

    # === Summary ===
    print(f"\n{'='*76}")
    print(f"  UMC Experiment v6b: Conv-Bottleneck Results")
    print(f"  Window: {WINDOW_SIZE} x {config.n_features} = {ambient_dim} ambient dims")
    print(f"  Conv-bottleneck: c_red={C_RED}, latent={LATENT_DIM}")
    print(f"{'='*76}")
    print(f"{'METRIC':<44} {'VALUE':>15} {'TARGET':>8} {'':>6}")
    print(f"{'-'*76}")
    print(f"{'Architecture':<44} {'ConvBot-'+str(C_RED):>15}")
    print(f"{'Model params':<44} {n_params:>15,}")
    print(f"{'Params/window':<44} {n_params / n_train:>15.0f}")
    print(f"{'Training time':<44} {t_train:>14.1f}s")
    print(f"{'Total windows':<44} {n:>15,}")
    print(f"{'Intrinsic dims (PCA 99%)':<44} {target_dims:>15}")
    print(f"{'Dimensionality ratio':<44} {dim_ratio:>14.1f}x")
    print(f"{'-'*76}")
    print(f"{'RMSE % (normalized space)':<44} {rmse_norm_pct:>14.4f}%")
    print(f"{'RMSE % (original, full dims)':<44} {rmse_original_pct:>14.4f}% {'<0.1%':>8} {'PASS' if rmse_original_pct < 0.1 else 'MISS':>6}")
    print(f"{'RMSE % (original, PCA reduced)':<44} {rmse_pca_pct:>14.4f}% {'<0.1%':>8} {'PASS' if rmse_pca_pct < 0.1 else 'MISS':>6}")
    print(f"{'Per-sample compression':<44} {ratio_per_sample:>14.1f}x {'>50x':>8} {'PASS' if ratio_per_sample > 50 else 'MISS':>6}")
    print(f"{'Full dataset ratio':<44} {ratio_full:>14.1f}x")
    print(f"{'System ratio (+ decoder)':<44} {ratio_system:>14.1f}x")
    print(f"{'gzip baseline':<44} {gzip_ratio:>14.1f}x")
    print(f"{'Encode (candles/s)':<44} {enc_throughput:>15,.0f} {'>10K':>8} {'PASS' if enc_throughput > 10000 else 'MISS':>6}")
    print(f"{'Decode (candles/s)':<44} {dec_throughput:>15,.0f} {'>50K':>8} {'PASS' if dec_throughput > 50000 else 'MISS':>6}")
    print(f"{'NN search speedup':<44} {speedup:>14.1f}x {'>100x':>8} {'PASS' if speedup > 100 else 'MISS':>6}")
    print(f"{'MNF roundtrip':<44} {'OK' if mnf_ok else 'FAIL':>15}")
    print(f"{'='*76}")

    print(f"\n  Comparison:")
    print(f"    v4 (FC-bot, no norm):    17.3% RMSE, 66.5x compression, 5.4M params")
    print(f"    v5 (FC-bot, per-win):    13.5% RMSE, 35.8x compression, 5.4M params")
    print(f"    v6b (Conv-bot, per-win): {rmse_original_pct:.1f}% RMSE, {ratio_per_sample:.1f}x compression, {n_params/1e6:.1f}M params")

    # Save
    Path("results").mkdir(exist_ok=True)
    results = {
        "experiment": "v6b_conv_bottleneck",
        "interval": "1h",
        "window_size": WINDOW_SIZE,
        "ambient_dim": ambient_dim,
        "architecture": f"conv_bottleneck_c{C_RED}",
        "conv_bottleneck_channels": C_RED,
        "per_window_normalize": True,
        "n_params": n_params,
        "n_symbols": len(symbols),
        "latent_dim": LATENT_DIM,
        "stride": STRIDE,
        "intrinsic_dims_pca99": target_dims,
        "dim_ratio": float(dim_ratio),
        "rmse_normalized_pct": float(rmse_norm_pct),
        "rmse_original_pct": float(rmse_original_pct),
        "rmse_pca_pct": float(rmse_pca_pct),
        "per_sample_ratio": float(ratio_per_sample),
        "full_dataset_ratio": float(ratio_full),
        "system_ratio": float(ratio_system),
        "gzip_binary_ratio": float(gzip_ratio),
        "encode_throughput": float(enc_throughput),
        "decode_throughput": float(dec_throughput),
        "search_speedup": float(speedup),
        "training_time_sec": float(t_train),
        "n_total_windows": n,
        "n_train_windows": n_train,
    }
    with open("results/experiment_v6b.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/experiment_v6b.json")


if __name__ == "__main__":
    main()
