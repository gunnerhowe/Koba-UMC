#!/usr/bin/env python3
"""Experiment v14b: Low-dimensional VQ (vq_dim=16) to fix codebook collapse.

Root cause analysis from v2-v14:
  - All VQ variants collapse in 64D space (standard, DAC, FSQ)
  - The curse of dimensionality makes 128-256 codebooks too sparse in 64D
  - FSQ fails differently: its 3D quantization bottleneck loses 95% of info

Solution: Reduce vq_dim from 64 to 16.
  - VQ in 16D is much more tractable (less curse of dimensionality)
  - 128 codes cover 16D space ~4x better than 64D (volume scaling)
  - The decoder works harder (16-dim input per patch), but financial data
    is low-dimensional (OHLC correlated, V mostly independent)

Two configurations tested:
  A) Standard VQ, vq_dim=16, 8 RVQ levels
  B) FSQ([4,4,4,4]=256 codes, 4D), vq_dim=16, 8 RVQ levels

Both have same storage: 1 + 16×8 = 129 bytes = 39.7x compression.
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
                 epochs=150, save_name="v14b", accum_steps=4):
    """Training loop."""
    encoder.to(device)
    decoder.to(device)
    encoder._use_grad_checkpoint = True
    decoder._use_grad_checkpoint = True

    base_lr = 3e-4

    optimizer = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=base_lr,
        weight_decay=1e-4,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp, growth_factor=1.5)

    # Feature weights: O, H, L, C(2x), V(5x)
    feat_weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 5.0], device=device)
    feat_weights = feat_weights / feat_weights.sum() * len(feat_weights)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30
    warmup_epochs = 8

    os.makedirs("results", exist_ok=True)
    epoch_start = time.perf_counter()

    for epoch in range(epochs):
        # Cosine LR with warmup
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            lr = 1e-6 + 0.5 * (base_lr - 1e-6) * (1 + np.cos(np.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if hasattr(encoder, 'gumbel_temperature'):
            encoder.gumbel_temperature = max(0.1, 1.0 - 0.9 * epoch / max(1, epochs - 1))

        noise_std = max(0.001, 0.015 * (1 - epoch / epochs))
        aux_weight = min(0.1, 0.1 * epoch / 20.0)

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


def evaluate_model(encoder, decoder, test_loader, config, device, z_test, x_test,
                   chart_test, n_codes_top, n_codes_bottom, rvq_levels, n_patches,
                   batch_size, label=""):
    """Evaluate model and print results."""
    use_amp = device.type == "cuda"

    # VQ code path decode
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
        for i in range(0, len(z_test), batch_size):
            z_batch = torch.from_numpy(z_test[i:i+batch_size].astype(np.float32)).to(device)
            c_batch = torch.from_numpy(chart_test[i:i+batch_size].astype(np.int64)).to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                x_hat_batch = decoder.decode(z_batch, c_batch)
            all_x_hat_z.append(x_hat_batch.cpu().float().numpy())
    x_hat_z = np.concatenate(all_x_hat_z)

    data_range = x_test.max() - x_test.min()
    rmse_z = np.sqrt(np.mean((x_test - x_hat_z) ** 2))
    rmse_z_pct = rmse_z / data_range * 100
    rmse_vq = np.sqrt(np.mean((x_test - x_hat_vq) ** 2))
    rmse_vq_pct = rmse_vq / data_range * 100

    print(f"\n=== {label} Reconstruction ===")
    print(f"  Data range: {data_range:.4f}")
    print(f"  VQ path RMSE: {rmse_vq:.6f} ({rmse_vq_pct:.4f}%)")
    print(f"  z path RMSE:  {rmse_z:.6f} ({rmse_z_pct:.4f}%)")

    print(f"\n  Per-feature RMSE (VQ path):")
    for i, feat in enumerate(config.features):
        fr = np.sqrt(np.mean((x_test[:, :, i] - x_hat_vq[:, :, i]) ** 2))
        fp = fr / max(x_test[:, :, i].max() - x_test[:, :, i].min(), 1e-8) * 100
        print(f"    {feat:>8s}: {fp:.4f}%")

    print(f"\n  Codebook utilization:")
    print(f"    Top: {encoder.top_perplexity:.1f} / {n_codes_top} "
          f"({encoder.top_perplexity/n_codes_top*100:.0f}%)")
    for i, p in enumerate(encoder.per_level_perplexity):
        print(f"    Lvl {i+1}: {p:.1f} / {n_codes_bottom} "
              f"({p/n_codes_bottom*100:.0f}%)")

    # Entropy analysis
    all_top_indices = []
    all_bottom_indices = [[] for _ in range(rvq_levels)]
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
    n_windows = len(top_indices)

    top_counts = np.bincount(top_indices.ravel(), minlength=n_codes_top)
    top_probs = top_counts / top_counts.sum()
    top_probs = top_probs[top_probs > 0]
    top_entropy = -np.sum(top_probs * np.log2(top_probs))
    total_bits += top_entropy * n_windows

    for lvl in range(rvq_levels):
        if all_bottom_indices[lvl]:
            bot_idx = np.concatenate(all_bottom_indices[lvl])
            bot_counts = np.bincount(bot_idx.ravel(), minlength=n_codes_bottom)
            bot_probs = bot_counts / bot_counts.sum()
            bot_probs = bot_probs[bot_probs > 0]
            bot_entropy = -np.sum(bot_probs * np.log2(bot_probs))
            total_bits += bot_entropy * n_windows * n_patches

    bytes_per_window = total_bits / 8 / n_windows
    raw_bytes = 1 + n_patches * rvq_levels
    raw_size = 5120

    print(f"\n  Storage:")
    print(f"    Raw: {raw_size} bytes/window")
    print(f"    VQ naive: {raw_bytes} bytes -> {raw_size/raw_bytes:.1f}x")
    print(f"    VQ entropy: {bytes_per_window:.1f} bytes -> {raw_size/bytes_per_window:.1f}x")

    return rmse_vq_pct, bytes_per_window


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== UMC Experiment v14b: Low-Dim VQ (vq_dim=16) ===")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print()

    WINDOW_SIZE = 256
    LATENT_DIM = 64
    STRIDE = 4
    BATCH_SIZE = 16
    ACCUM_STEPS = 4
    RVQ_LEVELS = 8
    PATCH_SIZE = 16
    EPOCHS = 150

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

    preprocessor_config = UMCConfig(window_size=WINDOW_SIZE)
    preprocessor = OHLCVPreprocessor(preprocessor_config)
    normalized = preprocessor.fit_transform(df)
    windows = create_windows(normalized, WINDOW_SIZE, stride=STRIDE)
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
    n_patches = WINDOW_SIZE // PATCH_SIZE

    train_loader = DataLoader(
        WindowDataset(train_w), batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        WindowDataset(val_w), batch_size=BATCH_SIZE, num_workers=0,
    )
    test_loader = DataLoader(
        WindowDataset(test_w), batch_size=BATCH_SIZE, num_workers=0,
    )

    print(f"Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")
    print(f"  Patches: {n_patches} (patch_size={PATCH_SIZE})")

    # ============================================================
    # Config A: Standard VQ, vq_dim=16
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  CONFIG A: Standard VQ, vq_dim=16")
    print(f"{'='*60}")

    config_a = UMCConfig(
        window_size=WINDOW_SIZE,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=LATENT_DIM,
        encoder_type="hvqvae",
        num_charts=16,
        chart_embedding_dim=16,
        batch_size=BATCH_SIZE,
        learning_rate=3e-4,
        d_model=128,
        n_heads=4,
        n_encoder_layers=4,
        n_decoder_layers=4,
        d_ff=512,
        patch_size=PATCH_SIZE,
        transformer_dropout=0.1,
        # Standard VQ in 16D
        vq_type="vq",
        vq_dim=16,
        vq_top_n_codes=16,       # Fewer codes for 16D (was 4 for 64D)
        vq_bottom_n_codes=128,
        vq_bottom_n_levels=RVQ_LEVELS,
        vq_commitment_weight=0.25,
        vq_ema_decay=0.99,
        vq_dead_code_threshold=2,
        # Loss
        beta_start=0.0, beta_end=0.0,
        sparsity_weight=0.0, smoothness_weight=0.0,
        multiscale_weight=0.1, spectral_weight=0.1,
        close_weight=2.0, volume_weight=5.0,
    )

    encoder_a = HVQVAEEncoder(config_a)
    decoder_a = HVQVAEDecoder(config_a)
    n_params_a = sum(p.numel() for p in encoder_a.parameters()) + sum(p.numel() for p in decoder_a.parameters())
    print(f"  Total params: {n_params_a:,}")
    print(f"  VQ: {config_a.vq_top_n_codes} top × {config_a.vq_bottom_n_codes} × {RVQ_LEVELS} RVQ, dim={config_a.vq_dim}")
    print(f"  Storage: {1 + n_patches * RVQ_LEVELS} bytes/window")

    print(f"\n  Training Config A...")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    t0 = time.perf_counter()
    try:
        best_val_a = train_hvqvae(
            encoder_a, decoder_a, train_loader, val_loader, config_a, device,
            epochs=EPOCHS, save_name="v14b_vq16", accum_steps=ACCUM_STEPS,
        )
    except torch.cuda.OutOfMemoryError:
        print("  WARNING: CUDA OOM for Config A")
        torch.cuda.empty_cache()
        gc.collect()
        state_path = "results/v14b_vq16_best_state.pt"
        if os.path.exists(state_path):
            best_state = torch.load(state_path, weights_only=False)
            encoder_a.load_state_dict(best_state['encoder'])
            decoder_a.load_state_dict(best_state['decoder'])
            best_val_a = best_state.get('val_loss', float('nan'))
        else:
            best_val_a = float('nan')
        encoder_a.to(device)
        decoder_a.to(device)
        encoder_a._use_grad_checkpoint = False
        decoder_a._use_grad_checkpoint = False

    t_a = time.perf_counter() - t0
    print(f"  Config A training time: {t_a:.0f}s ({t_a/60:.1f}m)")
    print(f"  Best val MSE: {best_val_a:.6f}")

    # Evaluate A
    encoder_a.eval()
    decoder_a.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    z_test, x_test, chart_test, _ = encode_all(encoder_a, test_loader, device)

    rmse_a, entropy_bytes_a = evaluate_model(
        encoder_a, decoder_a, test_loader, config_a, device,
        z_test, x_test, chart_test,
        n_codes_top=config_a.vq_top_n_codes,
        n_codes_bottom=config_a.vq_bottom_n_codes,
        rvq_levels=RVQ_LEVELS, n_patches=n_patches,
        batch_size=BATCH_SIZE, label="Config A (VQ dim=16)",
    )

    # Free GPU for next config
    del encoder_a, decoder_a
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    # ============================================================
    # Config B: FSQ, vq_dim=16
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  CONFIG B: FSQ, vq_dim=16")
    print(f"{'='*60}")

    FSQ_TOP_LEVELS = (4, 4)         # 16 codes in 2D (2D of 16 = 12.5%)
    FSQ_BOTTOM_LEVELS = (4, 4, 4, 4)  # 256 codes in 4D (4D of 16 = 25%)

    config_b = UMCConfig(
        window_size=WINDOW_SIZE,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=LATENT_DIM,
        encoder_type="hvqvae",
        num_charts=16,
        chart_embedding_dim=16,
        batch_size=BATCH_SIZE,
        learning_rate=3e-4,
        d_model=128,
        n_heads=4,
        n_encoder_layers=4,
        n_decoder_layers=4,
        d_ff=512,
        patch_size=PATCH_SIZE,
        transformer_dropout=0.1,
        # FSQ in 16D
        vq_type="fsq",
        vq_dim=16,
        fsq_top_levels=FSQ_TOP_LEVELS,
        fsq_bottom_levels=FSQ_BOTTOM_LEVELS,
        vq_bottom_n_levels=RVQ_LEVELS,
        vq_top_n_codes=int(np.prod(FSQ_TOP_LEVELS)),
        vq_bottom_n_codes=int(np.prod(FSQ_BOTTOM_LEVELS)),
        vq_commitment_weight=0.0,
        vq_ema_decay=0.99,
        vq_dead_code_threshold=0,
        # Loss
        beta_start=0.0, beta_end=0.0,
        sparsity_weight=0.0, smoothness_weight=0.0,
        multiscale_weight=0.1, spectral_weight=0.1,
        close_weight=2.0, volume_weight=5.0,
    )

    top_n_codes_b = int(np.prod(FSQ_TOP_LEVELS))
    bottom_n_codes_b = int(np.prod(FSQ_BOTTOM_LEVELS))

    encoder_b = HVQVAEEncoder(config_b)
    decoder_b = HVQVAEDecoder(config_b)
    n_params_b = sum(p.numel() for p in encoder_b.parameters()) + sum(p.numel() for p in decoder_b.parameters())
    print(f"  Total params: {n_params_b:,}")
    print(f"  FSQ Top: {list(FSQ_TOP_LEVELS)} = {top_n_codes_b} codes")
    print(f"  FSQ Bottom: {list(FSQ_BOTTOM_LEVELS)} = {bottom_n_codes_b} codes × {RVQ_LEVELS} levels")
    print(f"  Storage: {1 + n_patches * RVQ_LEVELS} bytes/window")

    print(f"\n  Training Config B...")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    t0 = time.perf_counter()
    try:
        best_val_b = train_hvqvae(
            encoder_b, decoder_b, train_loader, val_loader, config_b, device,
            epochs=EPOCHS, save_name="v14b_fsq16", accum_steps=ACCUM_STEPS,
        )
    except torch.cuda.OutOfMemoryError:
        print("  WARNING: CUDA OOM for Config B")
        torch.cuda.empty_cache()
        gc.collect()
        state_path = "results/v14b_fsq16_best_state.pt"
        if os.path.exists(state_path):
            best_state = torch.load(state_path, weights_only=False)
            encoder_b.load_state_dict(best_state['encoder'])
            decoder_b.load_state_dict(best_state['decoder'])
            best_val_b = best_state.get('val_loss', float('nan'))
        else:
            best_val_b = float('nan')
        encoder_b.to(device)
        decoder_b.to(device)
        encoder_b._use_grad_checkpoint = False
        decoder_b._use_grad_checkpoint = False

    t_b = time.perf_counter() - t0
    print(f"  Config B training time: {t_b:.0f}s ({t_b/60:.1f}m)")
    print(f"  Best val MSE: {best_val_b:.6f}")

    # Evaluate B
    encoder_b.eval()
    decoder_b.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    z_test_b, x_test_b, chart_test_b, _ = encode_all(encoder_b, test_loader, device)

    rmse_b, entropy_bytes_b = evaluate_model(
        encoder_b, decoder_b, test_loader, config_b, device,
        z_test_b, x_test_b, chart_test_b,
        n_codes_top=top_n_codes_b,
        n_codes_bottom=bottom_n_codes_b,
        rvq_levels=RVQ_LEVELS, n_patches=n_patches,
        batch_size=BATCH_SIZE, label="Config B (FSQ dim=16)",
    )

    # ============================================================
    # Summary
    # ============================================================
    print()
    print("=" * 60)
    print(f"  EXPERIMENT v14b COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<30s} {'VQ dim=16':>15s} {'FSQ dim=16':>15s}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'VQ path RMSE (%)':<30s} {rmse_a:>14.4f}% {rmse_b:>14.4f}%")
    print(f"  {'Best val MSE':<30s} {best_val_a:>15.6f} {best_val_b:>15.6f}")
    print(f"  {'Entropy bytes/window':<30s} {entropy_bytes_a:>15.1f} {entropy_bytes_b:>15.1f}")
    print(f"  {'Compression (entropy)':<30s} {5120/entropy_bytes_a:>14.1f}x {5120/entropy_bytes_b:>14.1f}x")
    print(f"  {'Training time (min)':<30s} {t_a/60:>15.1f} {t_b/60:>15.1f}")
    print(f"  {'Total params':<30s} {n_params_a:>15,} {n_params_b:>15,}")

    winner = "Config A (VQ)" if rmse_a < rmse_b else "Config B (FSQ)"
    print(f"\n  Winner: {winner}")
    print("=" * 60)


if __name__ == "__main__":
    main()
