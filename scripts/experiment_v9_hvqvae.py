#!/usr/bin/env python3
"""Experiment v9h: Gradient checkpointing + smaller batch for max epochs.

v9g achieved 2.98% RMSE (new best) but OOM'd at epoch ~27.
Val loss was still decreasing → more epochs = better results.

Changes from v9g:
1. Gradient checkpointing in transformer blocks (saves ~40% activation VRAM)
2. Batch_size=16, accum_steps=3 (effective batch=48, less peak VRAM)
3. 120 epochs target (previously only reached 25-27)
4. Mid-epoch cache clearing every 50 batches

Config: v9d base = 32 patches, 3-level RVQ, d_model=128
"""

import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
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


def train_hvqvae(encoder, decoder, train_loader, val_loader, config, device,
                 epochs=120, entropy_weight=0.1, save_name="v9h",
                 accum_steps=3, norm_loss_weight=0.2):
    """Training loop with AMP + gradient checkpointing + hybrid loss."""
    encoder.to(device)
    decoder.to(device)

    # Enable gradient checkpointing
    encoder._use_grad_checkpoint = True
    decoder._use_grad_checkpoint = True

    optimizer = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
        weight_decay=5e-4,
    )
    warmup_epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6,
    )

    # AMP
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 35
    best_state = None

    epoch_start = time.perf_counter()
    for epoch in range(epochs):
        # LR warmup
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = config.learning_rate * lr_scale
        else:
            scheduler.step()

        # EMA decay annealing: 0.9 -> 0.99
        ema_progress = min(1.0, epoch / 30.0)
        ema_decay = 0.9 + 0.09 * ema_progress
        encoder.set_ema_decay(ema_decay)

        # Entropy weight annealing
        ent_progress = min(1.0, epoch / 40.0)
        current_entropy_weight = entropy_weight * (1.0 - 0.5 * ent_progress)

        # Gumbel temperature
        if hasattr(encoder, 'gumbel_temperature'):
            progress = epoch / max(1, epochs - 1)
            encoder.gumbel_temperature = max(0.1, 1.0 - 0.9 * progress)

        noise_std = max(0.001, 0.02 * (1 - epoch / epochs))

        # === Train ===
        encoder.train()
        decoder.train()
        train_recon_sum = 0
        train_vq_sum = 0
        train_ent_sum = 0
        n_batches = 0

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            x_noisy = x + noise_std * torch.randn_like(x)

            with torch.amp.autocast('cuda', enabled=use_amp):
                enc = encoder.encode(x_noisy)

                x_hat_raw = decoder.decode_from_codes(
                    encoder._last_top_quantized,
                    encoder._last_bottom_quantized,
                )

                # Hybrid loss: original-space + normalized-space
                x_hat_orig = encoder.revin.inverse(x_hat_raw)
                recon_loss_orig = F.mse_loss(x_hat_orig, x)

                revin = encoder.revin
                x_target_norm = (x - revin._mean) / revin._std * revin.affine_weight + revin.affine_bias
                recon_loss_norm = F.mse_loss(x_hat_raw, x_target_norm)

                recon_loss = (1 - norm_loss_weight) * recon_loss_orig + norm_loss_weight * recon_loss_norm

                ent_loss = encoder.entropy_loss
                total_loss = (recon_loss + encoder.vq_loss + current_entropy_weight * ent_loss) / accum_steps

            scaler.scale(total_loss).backward()

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
            train_ent_sum += ent_loss.item()
            n_batches += 1

            # Mid-epoch cache clearing to reduce fragmentation
            if device.type == "cuda" and batch_idx % 100 == 99:
                torch.cuda.empty_cache()

        avg_recon = train_recon_sum / n_batches
        avg_vq = train_vq_sum / n_batches
        avg_ent = train_ent_sum / n_batches

        # Dead code reset
        if epoch < 15:
            reset_interval = 2
        else:
            reset_interval = 10
        if epoch > 0 and epoch % reset_interval == 0:
            n_top, n_bottom = encoder.reset_dead_codes()
            if n_top > 0 or n_bottom > 0:
                print(f"  Dead code reset: {n_top} top, {n_bottom} bottom codes reinitialized")

        # Aggressive cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        # === Validate (in original space) ===
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
                with torch.amp.autocast('cuda', enabled=use_amp):
                    enc = encoder.encode(x)
                    x_hat_raw = decoder.decode_from_codes(
                        encoder._last_top_quantized,
                        encoder._last_bottom_quantized,
                    )
                    x_hat = encoder.revin.inverse(x_hat_raw)
                    val_loss_sum += F.mse_loss(x_hat, x).item()
                n_val += 1
        avg_val = val_loss_sum / max(n_val, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = {
                'encoder': {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                'decoder': {k: v.cpu().clone() for k, v in decoder.state_dict().items()},
            }
            os.makedirs("results", exist_ok=True)
            torch.save(best_state, f"results/{save_name}_best_state.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            gap = avg_val / max(avg_recon, 1e-8)
            elapsed = time.perf_counter() - epoch_start
            sec_per_epoch = elapsed / max(epoch + 1, 1)
            eta_min = (epochs - epoch - 1) * sec_per_epoch / 60

            if hasattr(encoder.vq_bottom, 'per_level_perplexity'):
                bottom_perp = '/'.join(f'{p:.0f}' for p in encoder.vq_bottom.per_level_perplexity)
            else:
                bottom_perp = f'{encoder.bottom_perplexity:.0f}'

            print(
                f"  Epoch {epoch:4d} | Recon: {avg_recon:.6f} | VQ: {avg_vq:.4f} | "
                f"Ent: {avg_ent:.3f} | Val: {avg_val:.6f} | Gap: {gap:.2f}x | "
                f"Perp: {encoder.top_perplexity:.0f}/[{bottom_perp}] | "
                f"EMA: {ema_decay:.3f} | LR: {current_lr:.2e} | {sec_per_epoch:.1f}s/ep | ETA: {eta_min:.0f}m"
            )

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state:
        encoder.load_state_dict(best_state['encoder'])
        decoder.load_state_dict(best_state['decoder'])

    # Disable gradient checkpointing for eval
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
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
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
    print(f"=== UMC Experiment v9h: Grad Checkpoint + Smaller Batch ===")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print()

    WINDOW_SIZE = 256
    LATENT_DIM = 64
    STRIDE = 4
    BATCH_SIZE = 16
    ACCUM_STEPS = 3
    RVQ_LEVELS = 3

    config = UMCConfig(
        window_size=WINDOW_SIZE,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=LATENT_DIM,
        encoder_type="hvqvae",
        num_charts=16,
        chart_embedding_dim=16,
        batch_size=BATCH_SIZE,
        learning_rate=1e-4,
        # Transformer
        d_model=128,
        n_heads=4,
        n_encoder_layers=4,
        n_decoder_layers=4,
        d_ff=512,
        patch_size=8,
        transformer_dropout=0.2,
        # VQ
        vq_dim=64,
        vq_top_n_codes=64,
        vq_bottom_n_codes=256,
        vq_bottom_n_levels=RVQ_LEVELS,
        vq_commitment_weight=0.1,
        vq_ema_decay=0.9,
        vq_dead_code_threshold=2,
        beta_start=0.0, beta_end=0.0,
        sparsity_weight=0.0, smoothness_weight=0.0,
        multiscale_weight=0.0, spectral_weight=0.0,
    )

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
        shuffle=True, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        WindowDataset(val_w), batch_size=config.batch_size, pin_memory=True,
    )
    test_loader = DataLoader(
        WindowDataset(test_w), batch_size=config.batch_size, pin_memory=True,
    )
    all_loader = DataLoader(
        WindowDataset(np.concatenate([train_w, val_w])),
        batch_size=config.batch_size, pin_memory=True,
    )

    print(f"Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")
    n_patches = WINDOW_SIZE // config.patch_size
    print(f"  Patches: {n_patches} (patch_size={config.patch_size})")

    # === Build Model ===
    encoder = HVQVAEEncoder(config)
    decoder = HVQVAEDecoder(config)

    n_enc_params = sum(p.numel() for p in encoder.parameters())
    n_dec_params = sum(p.numel() for p in decoder.parameters())
    n_total = n_enc_params + n_dec_params
    print(f"\n  Encoder params: {n_enc_params:,}")
    print(f"  Decoder params: {n_dec_params:,}")
    print(f"  Total params: {n_total:,}")
    print(f"  Architecture: d_model={config.d_model}, patch=8, {n_patches} patches")
    print(f"  VQ: {config.vq_top_n_codes} top, {config.vq_bottom_n_codes} x {RVQ_LEVELS} RVQ")
    print(f"  v9h: grad checkpoint + batch={BATCH_SIZE} x accum={ACCUM_STEPS} + AMP + hybrid loss")

    # === Train ===
    print(f"\n=== Phase 1: Training HVQ-VAE (GradCheckpoint + AMP) ===")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    t0 = time.perf_counter()
    try:
        best_val = train_hvqvae(
            encoder, decoder, train_loader, val_loader, config, device,
            epochs=120, entropy_weight=0.1, save_name="v9h",
            accum_steps=ACCUM_STEPS, norm_loss_weight=0.2,
        )
    except torch.cuda.OutOfMemoryError:
        print("  WARNING: CUDA OOM, recovering best state from disk...")
        torch.cuda.empty_cache()
        gc.collect()
        best_state = torch.load("results/v9h_best_state.pt", weights_only=True)
        encoder.load_state_dict(best_state['encoder'])
        decoder.load_state_dict(best_state['decoder'])
        encoder.to(device)
        decoder.to(device)
        encoder._use_grad_checkpoint = False
        decoder._use_grad_checkpoint = False
        best_val = float('nan')
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

    # z path decode (batched)
    use_amp = device.type == "cuda"
    all_x_hat_z = []
    with torch.no_grad():
        for i in range(0, len(z_test), BATCH_SIZE):
            z_batch = torch.from_numpy(z_test[i:i+BATCH_SIZE].astype(np.float32)).to(device)
            c_batch = torch.from_numpy(chart_test[i:i+BATCH_SIZE].astype(np.int64)).to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                x_hat_batch = decoder.decode(z_batch, c_batch)
            all_x_hat_z.append(x_hat_batch.cpu().float().numpy())
    x_hat_z = np.concatenate(all_x_hat_z)

    # VQ code path decode
    all_x_hat_vq = []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                enc = encoder.encode(x)
                x_hat_raw = decoder.decode_from_codes(
                    encoder._last_top_quantized,
                    encoder._last_bottom_quantized,
                )
                x_hat = encoder.revin.inverse(x_hat_raw)
            all_x_hat_vq.append(x_hat.cpu().float().numpy())
    x_hat_vq = np.concatenate(all_x_hat_vq)

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
    if hasattr(encoder.vq_bottom, 'per_level_perplexity'):
        for i, p in enumerate(encoder.vq_bottom.per_level_perplexity):
            print(f"    Bottom level {i+1}: {p:.1f} / {config.vq_bottom_n_codes} "
                  f"({p/config.vq_bottom_n_codes*100:.0f}%)")

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

    td = int(np.searchsorted(cumvar, 0.99)) + 1
    td = min(td, len(cumvar))
    print(f"  Selected: {td} dims (99%)")

    pca_r = PCA(n_components=td)
    pca_r.fit(z_trainval)
    zrt = pca_r.transform(z_test)
    zrr = pca_r.inverse_transform(zrt)

    all_x_hat_pca = []
    with torch.no_grad():
        for i in range(0, len(zrr), BATCH_SIZE):
            z_batch = torch.from_numpy(zrr[i:i+BATCH_SIZE].astype(np.float32)).to(device)
            c_batch = torch.from_numpy(chart_test[i:i+BATCH_SIZE].astype(np.int64)).to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                x_hat_batch = decoder.decode(z_batch, c_batch)
            all_x_hat_pca.append(x_hat_batch.cpu().float().numpy())
    x_hat_pca = np.concatenate(all_x_hat_pca)
    rmse_pca = np.sqrt(np.mean((x_test - x_hat_pca) ** 2))
    rmse_pca_pct = rmse_pca / data_range * 100
    print(f"  PCA RMSE ({td} dims): {rmse_pca_pct:.4f}%")

    # === Compression & Search ===
    print(f"\n=== Compression & Search ===")
    raw_bytes = config.window_size * config.n_features * 4
    vq_bytes = 1 + n_patches * RVQ_LEVELS
    z_bytes = td * 2

    vq_ratio = raw_bytes / vq_bytes
    z_ratio = raw_bytes / z_bytes
    print(f"  Raw: {raw_bytes} bytes/window")
    print(f"  VQ ({RVQ_LEVELS}-lvl RVQ): {vq_bytes} bytes -> {vq_ratio:.1f}x")
    print(f"  PCA z ({td} dims): {z_bytes} bytes -> {z_ratio:.1f}x")

    search_idx = ManifoldSearch(z_test[:1000].astype(np.float32))
    query = z_test[:10].astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(100):
        search_idx.query(query, k=10)
    t_search = (time.perf_counter() - t0) / 100
    search_per_sec = 10 / t_search
    raw_search_time = 1000 * config.window_size * config.n_features * 4 / 1e9 * 10
    speedup = raw_search_time / t_search if t_search > 0 else 0

    print(f"  Search: {search_per_sec:.0f} queries/sec")
    print(f"  Search speedup vs raw: {speedup:.0f}x")

    # === Summary ===
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT v9h SUMMARY (grad ckpt + smaller batch)")
    print(f"{'='*60}")
    print(f"  Architecture: 4+4 transformer, d=128, patch=8, {n_patches} patches")
    print(f"  VQ: {config.vq_top_n_codes} top, {config.vq_bottom_n_codes} x {RVQ_LEVELS} RVQ")
    print(f"  Loss: 80% orig + 20% norm + AMP fp16 + grad ckpt")
    print(f"  Batch: {BATCH_SIZE} x accum {ACCUM_STEPS} = effective {BATCH_SIZE*ACCUM_STEPS}")
    print(f"  Total params: {n_total:,}")
    print(f"  Training time: {t_train:.0f}s ({t_train/60:.1f}m)")
    print(f"  VQ path RMSE: {rmse_vq_pct:.4f}%  (v9g: 2.98%, v9d: 3.1%)")
    print(f"  z path RMSE:  {rmse_z_pct:.4f}%")
    print(f"  PCA RMSE ({td}d): {rmse_pca_pct:.4f}%")
    print(f"  VQ compression: {vq_ratio:.1f}x")
    print(f"  z compression:  {z_ratio:.1f}x")
    print(f"  Top perplexity: {encoder.top_perplexity:.0f}/{config.vq_top_n_codes}")
    if hasattr(encoder.vq_bottom, 'per_level_perplexity'):
        perps = encoder.vq_bottom.per_level_perplexity
        print(f"  Bottom perplexity: {'/'.join(f'{p:.0f}' for p in perps)} / {config.vq_bottom_n_codes}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
