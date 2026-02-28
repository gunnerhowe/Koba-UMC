#!/usr/bin/env python3
"""Experiment v7: No pool cap — targeted test of temporal information preservation.

Analysis from v5/v6/v6b:
  v5 (FC-bot, pool_cap=16, 5.4M):   Best val=0.065, RMSE=13.5% (BEST)
  v6 (FC-bot, pool_cap=0, 18M):     Overfit badly, val=0.200 (WORSE)
  v6b (Conv-bot, no pool, 1.2M):    Underfit, val=0.228 (WORSE)

Hypothesis: The pool_cap=16 in v5 discards half the temporal info (32→16).
Removing it should help reconstruction IF we control param count.

Strategy: pool_cap=0 + bottleneck=512 (same as v5). The FC bottleneck goes
from 4096→512 to 8192→512, adding ~2.1M params. With stride=2 data (57K windows),
the params/window ratio is still manageable.

Expected: ~9.5M params total, params/window ≈ 165 (vs v5's 360).
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
    """Focused training — same as v5 but with pool_cap=0."""
    encoder.to(device)
    decoder.to(device)

    optimizer = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
        weight_decay=1e-3,
    )
    warmup_epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6,
    )

    weights = torch.ones(config.n_features, device=device)
    features = list(config.features)
    if "close" in features:
        weights[features.index("close")] = 2.0
    weights = weights / weights.sum() * len(features)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = config.learning_rate * lr_scale
        else:
            scheduler.step()

        if hasattr(encoder, 'gumbel_temperature'):
            progress = epoch / max(1, epochs - 1)
            encoder.gumbel_temperature = max(0.1, 1.0 - 0.9 * progress)

        noise_std = max(0.002, 0.05 * (1 - epoch / epochs))

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
    print(f"=== UMC Experiment v7: No Pool Cap (Targeted) ===")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print()

    WINDOW_SIZE = 256
    LATENT_DIM = 64  # Same as v5
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
        learning_rate=1e-4,  # Same as v5
        per_window_normalize=True,
        # KEY CHANGE: no pool cap, but keep same bottleneck width as v5
        pool_size_cap=0,  # Full compressed_len=32 preserved
        bottleneck_dim=512,  # Same as v5
        bottleneck_dropout=0.2,  # Same as v5
        # Pure autoencoder
        beta_start=0.0, beta_end=0.0,
        sparsity_weight=0.0, smoothness_weight=0.0,
        multiscale_weight=0.0, spectral_weight=0.0,
        epochs=500,
    )

    # === Download Data ===
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

    # Preprocess
    preprocessor = OHLCVPreprocessor(config)
    normalized = preprocessor.fit_transform(df)
    windows_raw = create_windows(normalized, config.window_size, stride=STRIDE)
    print(f"Windows (stride={STRIDE}): {windows_raw.shape}")

    # Per-window normalization
    normalizer = WindowNormalizer()
    windows_norm, win_means, win_stds = normalizer.normalize(windows_raw)
    print(f"After per-window norm: mean={windows_norm.mean():.4f}, std={windows_norm.std():.4f}")

    # Split
    n = len(windows_norm)
    gap = WINDOW_SIZE
    n_test = max(int(n * 0.1), 10)
    n_val = max(int(n * 0.1), 10)
    n_train = n - n_val - n_test - 2 * gap

    train_w, train_m, train_s = windows_norm[:n_train], win_means[:n_train], win_stds[:n_train]
    val_start = n_train + gap
    val_w = windows_norm[val_start:val_start + n_val]
    val_m, val_s = win_means[val_start:val_start + n_val], win_stds[val_start:val_start + n_val]
    test_start = val_start + n_val + gap
    test_w = windows_norm[test_start:]
    test_m, test_s = win_means[test_start:], win_stds[test_start:]
    n_test = len(test_w)
    test_raw = windows_raw[test_start:]

    train_loader = DataLoader(WindowDataset(train_w, train_m, train_s), batch_size=256, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(WindowDataset(val_w, val_m, val_s), batch_size=256, pin_memory=True)
    test_loader = DataLoader(WindowDataset(test_w, test_m, test_s), batch_size=256, pin_memory=True)
    all_w = np.concatenate([train_w, val_w])
    all_loader = DataLoader(WindowDataset(all_w), batch_size=256, pin_memory=True)

    print(f"Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")

    # === Train ===
    print(f"\n=== Phase 1: FC-Bottleneck AE, pool_cap=0, bottleneck=512, latent=64 ===")
    encoder = ConvEncoder(config)
    decoder = ConvDecoder(config)

    n_params_enc = sum(p.numel() for p in encoder.parameters())
    n_params_dec = sum(p.numel() for p in decoder.parameters())
    n_params = n_params_enc + n_params_dec
    print(f"  Encoder: {n_params_enc:,}  Decoder: {n_params_dec:,}  Total: {n_params:,}")
    print(f"  Params/window: {n_params / n_train:.0f}")

    compressed = config.window_size // (2 ** len(encoder.down_blocks))
    flat_dim = 256 * compressed
    print(f"  Info path: flat({flat_dim}) -> FC({config.bottleneck_dim}) -> latent({LATENT_DIM})")
    print(f"  vs v5: flat(4096) -> FC(512) -> latent(64)")

    t0 = time.perf_counter()
    best_val = train_autoencoder(encoder, decoder, train_loader, val_loader, config, device, epochs=500)
    t_train = time.perf_counter() - t0
    print(f"  Best val MSE: {best_val:.6f}")
    print(f"  Training time: {t_train:.1f}s")

    # === Encode ===
    z_trainval, x_trainval, _, _ = encode_all(encoder, all_loader, device)
    z_test, x_test_norm, chart_test, conf_test = encode_all(encoder, test_loader, device)

    decoder.eval()
    with torch.no_grad():
        z_tensor = torch.from_numpy(z_test.astype(np.float32)).to(device)
        chart_tensor = torch.from_numpy(chart_test.astype(np.int64)).to(device)
        x_hat_norm = decoder.decode(z_tensor, chart_tensor).cpu().numpy()

    rmse_norm = np.sqrt(np.mean((x_test_norm - x_hat_norm) ** 2))
    norm_range = x_test_norm.max() - x_test_norm.min()
    rmse_norm_pct = rmse_norm / norm_range * 100

    x_hat_original = normalizer.denormalize(x_hat_norm, test_m, test_s)
    rmse_original = np.sqrt(np.mean((test_raw - x_hat_original) ** 2))
    original_range = test_raw.max() - test_raw.min()
    rmse_original_pct = rmse_original / original_range * 100

    print(f"\n  Normalized RMSE: {rmse_norm_pct:.4f}%")
    print(f"  Original-scale RMSE: {rmse_original_pct:.4f}%")

    for i, feat in enumerate(config.features):
        feat_rmse = np.sqrt(np.mean((test_raw[:, :, i] - x_hat_original[:, :, i]) ** 2))
        feat_range = test_raw[:, :, i].max() - test_raw[:, :, i].min()
        feat_pct = feat_rmse / max(feat_range, 1e-8) * 100
        print(f"    {feat}: {feat_pct:.4f}%")

    # === PCA ===
    print(f"\n=== Phase 2: PCA ===")
    pca_full = PCA(n_components=min(config.max_latent_dim, z_trainval.shape[0] - 1))
    pca_full.fit(z_trainval)
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

    for threshold in [0.90, 0.95, 0.99, 0.999]:
        idx = np.searchsorted(cumulative_var, threshold)
        n_dims = int(idx) + 1 if idx < len(cumulative_var) else len(cumulative_var)
        print(f"  Dims for {threshold*100:.1f}%: {n_dims}")

    target_dims = int(np.searchsorted(cumulative_var, 0.99)) + 1
    target_dims = min(target_dims, len(cumulative_var))
    print(f"  Selected: {target_dims} dims (99%)")

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
    bytes_raw = ambient_dim * 4
    scale_bytes = config.n_features * 2 * 2
    bytes_mnf = target_dims * 2 + 1 + 2 + scale_bytes
    ratio = bytes_raw / bytes_mnf

    full_raw = n * bytes_raw
    full_mnf = n * bytes_mnf + 64
    ratio_full = full_raw / full_mnf
    ratio_system = full_raw / (full_mnf + n_params * 4)

    raw_binary = test_raw.astype(np.float32).tobytes()
    gzip_binary = gzip.compress(raw_binary, compresslevel=9)
    gzip_ratio = len(raw_binary) / len(gzip_binary)

    n_candles = len(z_test) * WINDOW_SIZE
    with torch.no_grad():
        encoder.encode(torch.randn(1, WINDOW_SIZE, config.n_features, device=device))

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            encoder.encode(x)
    if device.type == 'cuda': torch.cuda.synchronize()
    enc_tp = n_candles / (time.perf_counter() - t0 + 1e-8)

    t0 = time.perf_counter()
    with torch.no_grad():
        decoder.decode(z_tensor, chart_tensor)
    if device.type == 'cuda': torch.cuda.synchronize()
    dec_tp = n_candles / (time.perf_counter() - t0 + 1e-8)

    # Search
    z_rv = pca_reduced.transform(z_trainval)
    z_ra = np.vstack([z_rv, z_reduced_test])
    search = ManifoldSearch(z_ra)
    raw_all = np.concatenate([windows_raw[:n_train], windows_raw[val_start:val_start+n_val], windows_raw[test_start:]])[:len(z_ra)]
    amb = raw_all.reshape(len(z_ra), -1)

    ni = min(1000, len(z_ra))
    t0 = time.perf_counter()
    for _ in range(ni): search.query(z_ra[0], k=10)
    t_m = (time.perf_counter() - t0) / ni
    t0 = time.perf_counter()
    for _ in range(ni):
        dists = np.sum((amb - amb[0]) ** 2, axis=1)
        np.argsort(dists)[:10]
    t_a = (time.perf_counter() - t0) / ni
    speedup = t_a / max(t_m, 1e-10)

    # .mnf roundtrip
    import tempfile, os
    dh = DecoderManifest.compute_hash_bytes(decoder.state_dict())
    with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
        mp = f.name
    try:
        MNFWriter().write(mp, z_reduced_test.astype(np.float32), chart_test.astype(np.uint8), dh, confidences=conf_test, scale_means=test_m, scale_stds=test_s)
        mnf = MNFReader().read(mp)
        mnf_ok = mnf.header.n_samples == len(z_reduced_test) and mnf.header.has_scale_factors
    finally:
        os.unlink(mp)

    dim_ratio = ambient_dim / target_dims

    # === Summary ===
    print(f"\n{'='*76}")
    print(f"  UMC v7: No Pool Cap (pool_cap=0, bottleneck=512, latent=64)")
    print(f"  {WINDOW_SIZE}x{config.n_features} = {ambient_dim} dims, per-window norm, stride={STRIDE}")
    print(f"{'='*76}")
    print(f"{'METRIC':<44} {'VALUE':>15} {'TARGET':>8} {'':>6}")
    print(f"{'-'*76}")
    print(f"{'Model params':<44} {n_params:>15,}")
    print(f"{'Params/window':<44} {n_params / n_train:>15.0f}")
    print(f"{'Training time':<44} {t_train:>14.1f}s")
    print(f"{'Intrinsic dims (PCA 99%)':<44} {target_dims:>15}")
    print(f"{'Dim ratio':<44} {dim_ratio:>14.1f}x")
    print(f"{'-'*76}")
    print(f"{'RMSE % (normalized)':<44} {rmse_norm_pct:>14.4f}%")
    print(f"{'RMSE % (original, full dims)':<44} {rmse_original_pct:>14.4f}% {'<0.1%':>8} {'PASS' if rmse_original_pct < 0.1 else 'MISS':>6}")
    print(f"{'RMSE % (original, PCA)':<44} {rmse_pca_pct:>14.4f}% {'<0.1%':>8} {'PASS' if rmse_pca_pct < 0.1 else 'MISS':>6}")
    print(f"{'Compression':<44} {ratio:>14.1f}x {'>50x':>8} {'PASS' if ratio > 50 else 'MISS':>6}")
    print(f"{'System ratio (+ decoder)':<44} {ratio_system:>14.1f}x")
    print(f"{'gzip baseline':<44} {gzip_ratio:>14.1f}x")
    print(f"{'Encode (candles/s)':<44} {enc_tp:>15,.0f} {'>10K':>8} {'PASS' if enc_tp > 10000 else 'MISS':>6}")
    print(f"{'Decode (candles/s)':<44} {dec_tp:>15,.0f} {'>50K':>8} {'PASS' if dec_tp > 50000 else 'MISS':>6}")
    print(f"{'Search speedup':<44} {speedup:>14.1f}x {'>100x':>8} {'PASS' if speedup > 100 else 'MISS':>6}")
    print(f"{'MNF roundtrip':<44} {'OK' if mnf_ok else 'FAIL':>15}")
    print(f"{'='*76}")

    print(f"\n  Comparison:")
    print(f"    v5 (pool_cap=16, 5.4M):   13.5% RMSE, 35.8x compr")
    print(f"    v7 (pool_cap=0,  {n_params/1e6:.1f}M):   {rmse_original_pct:.1f}% RMSE, {ratio:.1f}x compr")
    print(f"    Delta: pool_cap=0 {'helps' if rmse_original_pct < 13.5 else 'hurts'} RMSE by {abs(rmse_original_pct - 13.5):.1f}pp")

    Path("results").mkdir(exist_ok=True)
    results = {
        "experiment": "v7_nopool",
        "window_size": WINDOW_SIZE, "latent_dim": LATENT_DIM,
        "bottleneck_dim": 512, "pool_size_cap": 0,
        "n_params": n_params, "stride": STRIDE,
        "intrinsic_dims": target_dims,
        "rmse_normalized_pct": float(rmse_norm_pct),
        "rmse_original_pct": float(rmse_original_pct),
        "rmse_pca_pct": float(rmse_pca_pct),
        "compression_ratio": float(ratio),
        "system_ratio": float(ratio_system),
        "encode_throughput": float(enc_tp),
        "decode_throughput": float(dec_tp),
        "search_speedup": float(speedup),
        "training_time_sec": float(t_train),
    }
    with open("results/experiment_v7.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/experiment_v7.json")


if __name__ == "__main__":
    main()
