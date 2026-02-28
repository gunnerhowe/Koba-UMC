#!/usr/bin/env python3
"""Experiment v8: Optimized v5 architecture with more data + wider bottleneck.

Lessons from v5-v7:
  - pool_cap=16 is optimal (v7 pool_cap=0 was WORSE: bottleneck starved)
  - FC bottleneck works best (conv-bottleneck underfits)
  - v5's ~5M params is the sweet spot for ~15K windows
  - With stride=2 (57K windows), we can afford a slightly wider bottleneck

v8 changes vs v5:
  - stride=2 → 57K windows (4x more training data)
  - bottleneck=768 (vs 512) → ~7.5M params (130 params/window, less than v5's 360)
  - latent=96 (vs 64) → more capacity for PCA to work with
  - Longer training: 800 epochs, patience=150
  - Higher initial LR with longer warmup
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


def train_autoencoder(encoder, decoder, train_loader, val_loader, config, device, epochs=800):
    encoder.to(device)
    decoder.to(device)

    optimizer = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
        weight_decay=5e-4,
    )
    warmup_epochs = 20
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

        noise_std = max(0.002, 0.04 * (1 - epoch / epochs))

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

        if patience_counter >= 150:
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
    print(f"=== UMC Experiment v8: Optimized Architecture ===")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print()

    WINDOW_SIZE = 256
    LATENT_DIM = 96
    BOTTLENECK = 768
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
        learning_rate=2e-4,
        per_window_normalize=True,
        pool_size_cap=16,  # PROVEN: better than 0
        bottleneck_dim=BOTTLENECK,
        bottleneck_dropout=0.15,
        beta_start=0.0, beta_end=0.0,
        sparsity_weight=0.0, smoothness_weight=0.0,
        multiscale_weight=0.0, spectral_weight=0.0,
        epochs=800,
    )

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

    preprocessor = OHLCVPreprocessor(config)
    normalized = preprocessor.fit_transform(df)
    windows_raw = create_windows(normalized, config.window_size, stride=STRIDE)
    print(f"Windows (stride={STRIDE}): {windows_raw.shape}")

    normalizer = WindowNormalizer()
    windows_norm, win_means, win_stds = normalizer.normalize(windows_raw)
    print(f"Per-window norm: mean={windows_norm.mean():.4f}, std={windows_norm.std():.4f}")

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
    print(f"\n=== Phase 1: Optimized Conv AE (pool=16, bot={BOTTLENECK}, lat={LATENT_DIM}) ===")
    encoder = ConvEncoder(config)
    decoder = ConvDecoder(config)

    n_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
    print(f"  Total params: {n_params:,} | Params/window: {n_params/n_train:.0f}")

    t0 = time.perf_counter()
    best_val = train_autoencoder(encoder, decoder, train_loader, val_loader, config, device, epochs=800)
    t_train = time.perf_counter() - t0
    print(f"  Best val MSE: {best_val:.6f}")
    print(f"  Training time: {t_train:.1f}s")

    # === Encode + Evaluate ===
    z_trainval, x_trainval, _, _ = encode_all(encoder, all_loader, device)
    z_test, x_test_norm, chart_test, conf_test = encode_all(encoder, test_loader, device)

    decoder.eval()
    with torch.no_grad():
        z_t = torch.from_numpy(z_test.astype(np.float32)).to(device)
        c_t = torch.from_numpy(chart_test.astype(np.int64)).to(device)
        x_hat_norm = decoder.decode(z_t, c_t).cpu().numpy()

    rmse_norm = np.sqrt(np.mean((x_test_norm - x_hat_norm) ** 2))
    rmse_norm_pct = rmse_norm / (x_test_norm.max() - x_test_norm.min()) * 100

    x_hat_orig = normalizer.denormalize(x_hat_norm, test_m, test_s)
    rmse_orig = np.sqrt(np.mean((test_raw - x_hat_orig) ** 2))
    orig_range = test_raw.max() - test_raw.min()
    rmse_orig_pct = rmse_orig / orig_range * 100

    print(f"\n  Normalized RMSE: {rmse_norm_pct:.4f}%")
    print(f"  Original RMSE: {rmse_orig_pct:.4f}%")
    for i, feat in enumerate(config.features):
        fr = np.sqrt(np.mean((test_raw[:, :, i] - x_hat_orig[:, :, i]) ** 2))
        fp = fr / max(test_raw[:, :, i].max() - test_raw[:, :, i].min(), 1e-8) * 100
        print(f"    {feat}: {fp:.4f}%")

    # PCA
    print(f"\n=== Phase 2: PCA ===")
    pca_full = PCA(n_components=min(LATENT_DIM, z_trainval.shape[0] - 1))
    pca_full.fit(z_trainval)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    for th in [0.90, 0.95, 0.99, 0.999]:
        idx = np.searchsorted(cumvar, th)
        nd = int(idx) + 1 if idx < len(cumvar) else len(cumvar)
        print(f"  {th*100:.1f}%: {nd} dims")

    td = int(np.searchsorted(cumvar, 0.99)) + 1
    td = min(td, len(cumvar))
    print(f"  Selected: {td} dims (99%)")

    pca_r = PCA(n_components=td)
    pca_r.fit(z_trainval)
    zrt = pca_r.transform(z_test)
    zrr = pca_r.inverse_transform(zrt)

    with torch.no_grad():
        xpn = decoder.decode(torch.from_numpy(zrr.astype(np.float32)).to(device), c_t).cpu().numpy()
    xpo = normalizer.denormalize(xpn, test_m, test_s)
    rmse_pca = np.sqrt(np.mean((test_raw - xpo) ** 2))
    rmse_pca_pct = rmse_pca / orig_range * 100

    # Metrics
    ad = WINDOW_SIZE * config.n_features
    br = ad * 4
    sf = config.n_features * 4
    bm = td * 2 + 1 + 2 + sf
    cr = br / bm
    rs = (n * br) / (n * bm + n_params * 4 + 64)
    gz = len(test_raw.astype(np.float32).tobytes()) / len(gzip.compress(test_raw.astype(np.float32).tobytes(), 9))

    nc = len(z_test) * WINDOW_SIZE
    with torch.no_grad():
        encoder.encode(torch.randn(1, WINDOW_SIZE, config.n_features, device=device))
    t0 = time.perf_counter()
    with torch.no_grad():
        for b in test_loader:
            x = b[0].to(device) if isinstance(b, (list, tuple)) else b.to(device)
            encoder.encode(x)
    if device.type == 'cuda': torch.cuda.synchronize()
    et = nc / (time.perf_counter() - t0 + 1e-8)
    t0 = time.perf_counter()
    with torch.no_grad():
        decoder.decode(z_t, c_t)
    if device.type == 'cuda': torch.cuda.synchronize()
    dt = nc / (time.perf_counter() - t0 + 1e-8)

    # Search
    zrv = pca_r.transform(z_trainval)
    zra = np.vstack([zrv, zrt])
    srch = ManifoldSearch(zra)
    raw_all = np.concatenate([windows_raw[:n_train], windows_raw[val_start:val_start+n_val], windows_raw[test_start:]])[:len(zra)]
    amb = raw_all.reshape(len(zra), -1)
    ni = min(1000, len(zra))
    t0 = time.perf_counter()
    for _ in range(ni): srch.query(zra[0], k=10)
    tm = (time.perf_counter() - t0) / ni
    t0 = time.perf_counter()
    for _ in range(ni):
        d = np.sum((amb - amb[0]) ** 2, axis=1)
        np.argsort(d)[:10]
    ta = (time.perf_counter() - t0) / ni
    sp = ta / max(tm, 1e-10)

    # MNF
    import tempfile, os
    dh = DecoderManifest.compute_hash_bytes(decoder.state_dict())
    with tempfile.NamedTemporaryFile(suffix=".mnf", delete=False) as f:
        mp = f.name
    try:
        MNFWriter().write(mp, zrt.astype(np.float32), chart_test.astype(np.uint8), dh, confidences=conf_test, scale_means=test_m, scale_stds=test_s)
        mnf = MNFReader().read(mp)
        ok = mnf.header.n_samples == len(zrt) and mnf.header.has_scale_factors
    finally:
        os.unlink(mp)

    dr = ad / td

    print(f"\n{'='*76}")
    print(f"  UMC v8: Optimized Architecture Results")
    print(f"  {WINDOW_SIZE}x{config.n_features}={ad} dims | bot={BOTTLENECK} lat={LATENT_DIM} pool=16")
    print(f"{'='*76}")
    print(f"{'METRIC':<44} {'VALUE':>15} {'TARGET':>8} {'':>6}")
    print(f"{'-'*76}")
    print(f"{'Model params':<44} {n_params:>15,}")
    print(f"{'Params/window':<44} {n_params/n_train:>15.0f}")
    print(f"{'Training time':<44} {t_train:>14.1f}s")
    print(f"{'Intrinsic dims (PCA 99%)':<44} {td:>15}")
    print(f"{'Dim ratio':<44} {dr:>14.1f}x")
    print(f"{'-'*76}")
    print(f"{'RMSE % (normalized)':<44} {rmse_norm_pct:>14.4f}%")
    print(f"{'RMSE % (original, full)':<44} {rmse_orig_pct:>14.4f}% {'<0.1%':>8} {'PASS' if rmse_orig_pct < 0.1 else 'MISS':>6}")
    print(f"{'RMSE % (original, PCA)':<44} {rmse_pca_pct:>14.4f}% {'<0.1%':>8} {'PASS' if rmse_pca_pct < 0.1 else 'MISS':>6}")
    print(f"{'Compression':<44} {cr:>14.1f}x {'>50x':>8} {'PASS' if cr > 50 else 'MISS':>6}")
    print(f"{'System ratio':<44} {rs:>14.1f}x")
    print(f"{'gzip baseline':<44} {gz:>14.1f}x")
    print(f"{'Encode (candles/s)':<44} {et:>15,.0f} {'>10K':>8} {'PASS' if et > 10000 else 'MISS':>6}")
    print(f"{'Decode (candles/s)':<44} {dt:>15,.0f} {'>50K':>8} {'PASS' if dt > 50000 else 'MISS':>6}")
    print(f"{'Search speedup':<44} {sp:>14.1f}x {'>100x':>8} {'PASS' if sp > 100 else 'MISS':>6}")
    print(f"{'MNF roundtrip':<44} {'OK' if ok else 'FAIL':>15}")
    print(f"{'='*76}")

    print(f"\n  Version comparison:")
    print(f"    v4 (no norm, FC512):    17.3% RMSE, 66.5x compr, 5.4M")
    print(f"    v5 (pw norm, FC512):    13.5% RMSE, 35.8x compr, 5.4M")
    print(f"    v8 (pw norm, FC{BOTTLENECK}):    {rmse_orig_pct:.1f}% RMSE, {cr:.1f}x compr, {n_params/1e6:.1f}M")

    Path("results").mkdir(exist_ok=True)
    with open("results/experiment_v8.json", "w") as f:
        json.dump({
            "experiment": "v8_optimal", "window_size": WINDOW_SIZE,
            "bottleneck_dim": BOTTLENECK, "latent_dim": LATENT_DIM, "pool_cap": 16,
            "n_params": n_params, "stride": STRIDE,
            "intrinsic_dims": td,
            "rmse_norm_pct": float(rmse_norm_pct),
            "rmse_orig_pct": float(rmse_orig_pct),
            "rmse_pca_pct": float(rmse_pca_pct),
            "compression": float(cr), "system_ratio": float(rs),
            "encode_throughput": float(et), "decode_throughput": float(dt),
            "search_speedup": float(sp), "training_time": float(t_train),
        }, f, indent=2)
    print(f"\nSaved to results/experiment_v8.json")


if __name__ == "__main__":
    main()
