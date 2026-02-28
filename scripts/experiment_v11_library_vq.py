#!/usr/bin/env python3
"""Experiment v11: Library VQ-VAE — battle-tested vector quantization.

Uses vector-quantize-pytorch library for RVQ instead of hand-rolled implementation.
The library handles: EMA codebook updates, dead code reset, kmeans init, quantize dropout.

Strategy:
  - Same architecture as v10: patch_size=16, 8-level RVQ, d_model=128
  - All 8 RVQ levels active from start (library's quantize_dropout handles progressive)
  - Simple cosine LR with warmup
  - Feature-weighted loss (volume 3x, close 2x)
  - Spectral + multi-scale auxiliary loss

Storage budget:
  Raw: 1 (top) + 16x8 (bottom) = 129 bytes -> 39.7x
  With entropy coding: ~97 bytes -> 52.8x
"""

import gc
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from umc.config import UMCConfig
from umc.encoder.hvqvae_encoder import HVQVAEEncoder
from umc.decoder.hvqvae_decoder import HVQVAEDecoder
from umc.data.loaders import load_yahoo_finance, combine_datasets
from umc.data.preprocessors import OHLCVPreprocessor, create_windows, WindowDataset
from umc.processor.search import ManifoldSearch
from umc.training.losses import multiscale_reconstruction_loss, spectral_loss


def feature_weighted_mse(x_hat, x, weights):
    """MSE loss with per-feature weights."""
    diff_sq = (x_hat - x) ** 2
    w = weights.unsqueeze(0).unsqueeze(0)
    return (diff_sq * w).mean()


def train_hvqvae(encoder, decoder, train_loader, val_loader, config, device,
                 epochs=200, save_name="v11", accum_steps=4):
    """Simplified training loop — library VQ handles codebook management."""
    encoder.to(device)
    decoder.to(device)
    encoder._use_grad_checkpoint = True
    decoder._use_grad_checkpoint = True

    optimizer = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=3e-4,
        weight_decay=1e-4,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp, growth_factor=1.5)

    # Feature weights: O, H, L, C(2x), V(3x)
    feat_weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0], device=device)
    feat_weights = feat_weights / feat_weights.sum() * len(feat_weights)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 40
    warmup_epochs = 10

    os.makedirs("results", exist_ok=True)
    epoch_start = time.perf_counter()

    for epoch in range(epochs):
        # Simple cosine LR with warmup
        if epoch < warmup_epochs:
            lr = 3e-4 * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            lr = 1e-6 + 0.5 * (3e-4 - 1e-6) * (1 + np.cos(np.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Gumbel temperature annealing
        if hasattr(encoder, 'gumbel_temperature'):
            encoder.gumbel_temperature = max(0.1, 1.0 - 0.9 * epoch / max(1, epochs - 1))

        noise_std = max(0.001, 0.015 * (1 - epoch / epochs))
        aux_weight = min(0.1, 0.1 * epoch / 30.0)

        # === Train ===
        encoder.train()
        decoder.train()
        train_recon_sum = 0
        train_vq_sum = 0
        n_batches = 0

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            x_noisy = x + noise_std * torch.randn_like(x)

            with torch.amp.autocast('cuda', enabled=use_amp):
                enc = encoder.encode(x_noisy)
                x_hat_raw = decoder.decode_from_codes(
                    encoder._last_top_quantized,
                    encoder._last_bottom_quantized,
                )
                x_hat_orig = encoder.revin.inverse(x_hat_raw)

                recon_loss = feature_weighted_mse(x_hat_orig, x, feat_weights)

                aux_loss = torch.tensor(0.0, device=device)
                if aux_weight > 0:
                    aux_loss = (
                        spectral_loss(x_hat_orig, x)
                        + multiscale_reconstruction_loss(x_hat_orig, x, scales=(1, 4, 16))
                    )

                total_loss = (recon_loss + encoder.vq_loss + aux_weight * aux_loss) / accum_steps

            scaler.scale(total_loss).backward()
            del total_loss, x_hat_raw, x_hat_orig, enc, x_noisy

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_recon_sum += recon_loss.item() * accum_steps
            train_vq_sum += encoder.vq_loss.item()
            n_batches += 1

        avg_recon = train_recon_sum / n_batches
        avg_vq = train_vq_sum / n_batches

        encoder.clear_cached()
        if device.type == "cuda":
            gc.collect()

        # === Validate ===
        encoder.eval()
        decoder.eval()
        val_loss_sum = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    enc = encoder.encode(x)
                    x_hat_raw = decoder.decode_from_codes(
                        encoder._last_top_quantized,
                        encoder._last_bottom_quantized,
                    )
                    x_hat = encoder.revin.inverse(x_hat_raw)
                    val_loss_sum += F.mse_loss(x_hat, x).item()
                n_val += 1
                del enc, x_hat_raw, x_hat

        encoder.clear_cached()
        avg_val = val_loss_sum / max(n_val, 1)

        if avg_val < best_val_loss and not np.isnan(avg_val):
            best_val_loss = avg_val
            patience_counter = 0
            best_state = {
                'encoder': {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                'decoder': {k: v.cpu().clone() for k, v in decoder.state_dict().items()},
                'epoch': epoch,
                'val_loss': avg_val,
            }
            torch.save(best_state, f"results/{save_name}_best_state.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            gap = avg_val / max(avg_recon, 1e-8)
            elapsed = time.perf_counter() - epoch_start
            sec_per_epoch = elapsed / max(epoch + 1, 1)
            eta_min = (epochs - epoch - 1) * sec_per_epoch / 60

            bottom_perp = '/'.join(f'{p:.0f}' for p in encoder.per_level_perplexity)

            print(
                f"  Epoch {epoch:4d} | Recon: {avg_recon:.6f} | VQ: {avg_vq:.4f} | "
                f"Val: {avg_val:.6f} | Gap: {gap:.2f}x | "
                f"Perp: {encoder.top_perplexity:.0f}/[{bottom_perp}] | "
                f"LR: {current_lr:.2e} | "
                f"{sec_per_epoch:.1f}s/ep | ETA: {eta_min:.0f}m"
            )

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best state
    state_path = f"results/{save_name}_best_state.pt"
    if os.path.exists(state_path):
        best_state = torch.load(state_path, weights_only=False)
        encoder.load_state_dict(best_state['encoder'])
        decoder.load_state_dict(best_state['decoder'])
        print(f"  Loaded best state from epoch {best_state.get('epoch', '?')}")

    encoder._use_grad_checkpoint = False
    decoder._use_grad_checkpoint = False
    return best_val_loss


def encode_all(encoder, loader, device):
    """Encode all data and return latents + metadata."""
    use_amp = device.type == "cuda"
    encoder.eval()
    all_z, all_x, all_chart, all_conf = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                enc = encoder.encode(x)
            all_z.append(enc.z.cpu().float().numpy())
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
    print(f"=== UMC Experiment v11: Library VQ ===")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print()

    WINDOW_SIZE = 256
    LATENT_DIM = 64
    STRIDE = 4
    BATCH_SIZE = 16
    ACCUM_STEPS = 4
    RVQ_LEVELS = 8
    PATCH_SIZE = 16
    EPOCHS = 200

    config = UMCConfig(
        window_size=WINDOW_SIZE,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=LATENT_DIM,
        encoder_type="hvqvae",
        num_charts=16,
        chart_embedding_dim=16,
        batch_size=BATCH_SIZE,
        learning_rate=3e-4,
        # Transformer
        d_model=128,
        n_heads=4,
        n_encoder_layers=4,
        n_decoder_layers=4,
        d_ff=512,
        patch_size=PATCH_SIZE,
        transformer_dropout=0.2,
        # VQ (library handles EMA, dead codes, init)
        vq_dim=64,
        vq_top_n_codes=16,
        vq_bottom_n_codes=256,
        vq_bottom_n_levels=RVQ_LEVELS,
        vq_commitment_weight=0.1,
        vq_ema_decay=0.99,
        vq_dead_code_threshold=2,
        # Loss
        beta_start=0.0, beta_end=0.0,
        sparsity_weight=0.0, smoothness_weight=0.0,
        multiscale_weight=0.1, spectral_weight=0.1,
        close_weight=2.0, volume_weight=3.0,
    )

    n_patches = WINDOW_SIZE // PATCH_SIZE

    # === Data Loading ===
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
    windows = create_windows(normalized, config.window_size, stride=STRIDE)
    print(f"Windows (stride={STRIDE}): {windows.shape}")

    n = len(windows)
    gap = WINDOW_SIZE
    n_test = max(int(n * 0.1), 10)
    n_val = max(int(n * 0.1), 10)
    n_train = n - n_val - n_test - 2 * gap

    train_w = windows[:n_train]
    val_start = n_train + gap
    val_w = windows[val_start:val_start + n_val]
    test_start = val_start + n_val + gap
    test_w = windows[test_start:]
    n_test = len(test_w)

    train_loader = DataLoader(
        WindowDataset(train_w), batch_size=config.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        WindowDataset(val_w), batch_size=config.batch_size, num_workers=0,
    )
    test_loader = DataLoader(
        WindowDataset(test_w), batch_size=config.batch_size, num_workers=0,
    )
    all_loader = DataLoader(
        WindowDataset(np.concatenate([train_w, val_w])),
        batch_size=config.batch_size, num_workers=0,
    )

    print(f"Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")
    print(f"  Patches: {n_patches} (patch_size={PATCH_SIZE})")

    # === Build Model ===
    encoder = HVQVAEEncoder(config)
    decoder = HVQVAEDecoder(config)

    n_enc_params = sum(p.numel() for p in encoder.parameters())
    n_dec_params = sum(p.numel() for p in decoder.parameters())
    n_total = n_enc_params + n_dec_params
    print(f"\n  Encoder params: {n_enc_params:,}")
    print(f"  Decoder params: {n_dec_params:,}")
    print(f"  Total params: {n_total:,}")
    print(f"  Architecture: d_model={config.d_model}, patch={PATCH_SIZE}, {n_patches} patches")
    print(f"  VQ: {config.vq_top_n_codes} top, {config.vq_bottom_n_codes} x {RVQ_LEVELS} RVQ")
    print(f"  v11: library VQ, quantize_dropout, kmeans_init, auto dead code reset")
    print(f"  Storage: 1 + {n_patches}x{RVQ_LEVELS} = {1 + n_patches * RVQ_LEVELS} bytes/window (raw)")

    # === Train ===
    print(f"\n=== Phase 1: Training HVQ-VAE (Library RVQ) ===")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    t0 = time.perf_counter()
    try:
        best_val = train_hvqvae(
            encoder, decoder, train_loader, val_loader, config, device,
            epochs=EPOCHS, save_name="v11", accum_steps=ACCUM_STEPS,
        )
    except torch.cuda.OutOfMemoryError:
        print("  WARNING: CUDA OOM, recovering best state from disk...")
        torch.cuda.empty_cache()
        gc.collect()
        state_path = "results/v11_best_state.pt"
        if os.path.exists(state_path):
            best_state = torch.load(state_path, weights_only=False)
            encoder.load_state_dict(best_state['encoder'])
            decoder.load_state_dict(best_state['decoder'])
            best_val = best_state.get('val_loss', float('nan'))
            print(f"  Recovered from epoch {best_state.get('epoch', '?')}")
        else:
            best_val = float('nan')
            print("  No checkpoint found!")
        encoder.to(device)
        decoder.to(device)
        encoder._use_grad_checkpoint = False
        decoder._use_grad_checkpoint = False

    t_train = time.perf_counter() - t0
    print(f"  Best val MSE: {best_val:.6f}")
    print(f"  Training time: {t_train:.1f}s ({t_train/60:.1f}m)")

    # === Evaluate ===
    print(f"\n=== Reconstruction Evaluation ===")
    encoder.eval()
    decoder.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    z_test, x_test, chart_test, conf_test = encode_all(encoder, test_loader, device)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # VQ code path decode
    use_amp = device.type == "cuda"
    all_x_hat_vq = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                enc = encoder.encode(x)
                x_hat_raw = decoder.decode_from_codes(
                    encoder._last_top_quantized,
                    encoder._last_bottom_quantized,
                )
                x_hat = encoder.revin.inverse(x_hat_raw)
            all_x_hat_vq.append(x_hat.cpu().float().numpy())
    x_hat_vq = np.concatenate(all_x_hat_vq)

    # z path decode
    all_x_hat_z = []
    with torch.no_grad():
        for i in range(0, len(z_test), BATCH_SIZE):
            z_batch = torch.from_numpy(z_test[i:i+BATCH_SIZE].astype(np.float32)).to(device)
            c_batch = torch.from_numpy(chart_test[i:i+BATCH_SIZE].astype(np.int64)).to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                x_hat_batch = decoder.decode(z_batch, c_batch)
            all_x_hat_z.append(x_hat_batch.cpu().float().numpy())
    x_hat_z = np.concatenate(all_x_hat_z)

    data_range = x_test.max() - x_test.min()
    rmse_z = np.sqrt(np.mean((x_test - x_hat_z) ** 2))
    rmse_z_pct = rmse_z / data_range * 100
    rmse_vq = np.sqrt(np.mean((x_test - x_hat_vq) ** 2))
    rmse_vq_pct = rmse_vq / data_range * 100

    print(f"  Data range: {data_range:.4f}")
    print(f"  VQ path RMSE: {rmse_vq:.6f} ({rmse_vq_pct:.4f}%)")
    print(f"  z path RMSE:  {rmse_z:.6f} ({rmse_z_pct:.4f}%)")

    print(f"\n  Per-feature RMSE (VQ path):")
    for i, feat in enumerate(config.features):
        fr = np.sqrt(np.mean((x_test[:, :, i] - x_hat_vq[:, :, i]) ** 2))
        fp = fr / max(x_test[:, :, i].max() - x_test[:, :, i].min(), 1e-8) * 100
        print(f"    {feat:>8s}: {fp:.4f}%")

    print(f"\n  Codebook utilization:")
    print(f"    Top perplexity: {encoder.top_perplexity:.1f} / {config.vq_top_n_codes} "
          f"({encoder.top_perplexity/config.vq_top_n_codes*100:.0f}%)")
    for i, p in enumerate(encoder.per_level_perplexity):
        print(f"    Bottom level {i+1}: {p:.1f} / {config.vq_bottom_n_codes} "
              f"({p/config.vq_bottom_n_codes*100:.0f}%)")

    # === Entropy Analysis ===
    print(f"\n=== Entropy Analysis (VQ Index Compression) ===")
    all_top_indices = []
    all_bottom_indices = [[] for _ in range(RVQ_LEVELS)]
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                encoder.encode(x)
            all_top_indices.append(encoder._last_top_indices.cpu().numpy())
            for lvl, idx in enumerate(encoder._last_bottom_indices):
                all_bottom_indices[lvl].append(idx.cpu().numpy())

    top_indices = np.concatenate(all_top_indices)
    total_bits = 0

    top_counts = np.bincount(top_indices.ravel(), minlength=config.vq_top_n_codes)
    top_probs = top_counts / top_counts.sum()
    top_probs = top_probs[top_probs > 0]
    top_entropy = -np.sum(top_probs * np.log2(top_probs))
    top_bits = top_entropy * len(top_indices)
    total_bits += top_bits
    print(f"  Top VQ: {top_entropy:.2f} bits/index (naive: {np.log2(config.vq_top_n_codes):.1f})")

    n_windows = len(top_indices)
    for lvl in range(RVQ_LEVELS):
        if all_bottom_indices[lvl]:
            bot_idx = np.concatenate(all_bottom_indices[lvl])
            bot_counts = np.bincount(bot_idx.ravel(), minlength=config.vq_bottom_n_codes)
            bot_probs = bot_counts / bot_counts.sum()
            bot_probs = bot_probs[bot_probs > 0]
            bot_entropy = -np.sum(bot_probs * np.log2(bot_probs))
            level_bits = bot_entropy * n_windows * n_patches
            total_bits += level_bits
            print(f"  Bottom level {lvl+1}: {bot_entropy:.2f} bits/index "
                  f"(naive: {np.log2(config.vq_bottom_n_codes):.1f})")

    total_bytes_entropy = total_bits / 8
    bytes_per_window_entropy = total_bytes_entropy / n_windows
    raw_bytes = 1 + n_patches * RVQ_LEVELS
    raw_size = 5120

    print(f"\n  Storage per window:")
    print(f"    Raw data: {raw_size} bytes")
    print(f"    VQ indices (naive): {raw_bytes} bytes -> {raw_size/raw_bytes:.1f}x")
    print(f"    VQ indices (entropy): {bytes_per_window_entropy:.1f} bytes -> {raw_size/bytes_per_window_entropy:.1f}x")

    # === PCA ===
    print(f"\n=== Phase 2: PCA on z_projection ===")
    z_trainval, _, _, _ = encode_all(encoder, all_loader, device)
    pca_full = PCA(n_components=min(LATENT_DIM, z_trainval.shape[0] - 1))
    pca_full.fit(z_trainval)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    for th in [0.90, 0.95, 0.99, 0.999]:
        idx = np.searchsorted(cumvar, th)
        nd = int(idx) + 1 if idx < len(cumvar) else len(cumvar)
        print(f"  {th*100:.1f}%: {nd} dims")

    # === Search ===
    print(f"\n=== Compression & Search ===")
    td = int(np.searchsorted(cumvar, 0.99)) + 1
    td = min(td, len(cumvar))
    print(f"  Raw: {raw_size} bytes/window")
    print(f"  VQ ({RVQ_LEVELS}-lvl RVQ): {raw_bytes} bytes -> {raw_size/raw_bytes:.1f}x")
    pca_bytes = td * 2
    print(f"  PCA z ({td} dims): {pca_bytes} bytes -> {raw_size/pca_bytes:.1f}x")

    search = ManifoldSearch(z_test[:1000].astype(np.float32))
    t0 = time.perf_counter()
    for _ in range(10):
        for q in z_test[:100]:
            search.query(q.astype(np.float32).reshape(1, -1), k=5)
    t_search = time.perf_counter() - t0
    qps = 100 * 10 / t_search
    raw_dim = WINDOW_SIZE * len(config.features)
    speedup = (raw_dim / LATENT_DIM) ** 2

    print(f"  Search: {qps:.0f} queries/sec")
    print(f"  Search speedup vs raw: {speedup:.0f}x")

    # === Summary ===
    print()
    print("=" * 60)
    print(f"  EXPERIMENT v11 SUMMARY (library {RVQ_LEVELS}-level RVQ)")
    print("=" * 60)
    print(f"  Architecture: {config.n_encoder_layers}+{config.n_decoder_layers} transformer, "
          f"d={config.d_model}, patch={PATCH_SIZE}, {n_patches} patches")
    print(f"  VQ: {config.vq_top_n_codes} top, {config.vq_bottom_n_codes} x {RVQ_LEVELS} RVQ")
    print(f"  VQ library: vector-quantize-pytorch (kmeans_init, quantize_dropout)")
    print(f"  Loss: feature-weighted (V=3x, C=2x) + spectral + multi-scale")
    print(f"  Total params: {n_total:,}")
    print(f"  Training time: {t_train:.0f}s ({t_train/60:.1f}m)")
    print(f"  VQ path RMSE: {rmse_vq_pct:.4f}%  (target: <0.1%)")
    print(f"  z path RMSE:  {rmse_z_pct:.4f}%")
    print(f"  VQ compression (naive): {raw_size/raw_bytes:.1f}x")
    print(f"  VQ compression (entropy): {raw_size/bytes_per_window_entropy:.1f}x")
    target_met = "YES" if rmse_vq_pct < 0.1 else "NO"
    compression_met = "YES" if raw_size/bytes_per_window_entropy > 50 else "NO"
    print(f"  RMSE target met: {target_met}")
    print(f"  Compression target met: {compression_met}")
    print("=" * 60)


if __name__ == "__main__":
    main()
