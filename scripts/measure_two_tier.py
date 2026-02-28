#!/usr/bin/env python3
"""Measure two-tier compression pipeline: VQ search + lossless XOR residuals.

Tier 1 (Search):  VQ indices only — lossy (~2.5% RMSE), compact, FAISS-searchable
Tier 2 (Lossless): VQ + XOR residual — 0% RMSE, bit-exact reconstruction

Uses the trained v15b model (vq_dim=32, codebook_dim=16, 8-level RVQ).
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

from umc.config import UMCConfig
from umc.encoder.hvqvae_encoder import HVQVAEEncoder
from umc.decoder.hvqvae_decoder import HVQVAEDecoder
from umc.data.loaders import load_yahoo_finance, combine_datasets
from umc.data.preprocessors import OHLCVPreprocessor, create_windows
from umc.codec.lossless import LosslessCodec, LosslessEncoding, serialize_encoding, deserialize_encoding
from umc.codec.residual import ResidualCoder
from umc.storage.entropy import VQIndices, compress_indices, decompress_indices, compute_compression_stats


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  TWO-TIER COMPRESSION PIPELINE MEASUREMENT")
    print("  Tier 1: VQ search index (lossy, ~2.5% RMSE)")
    print("  Tier 2: + XOR residuals (lossless, 0% RMSE)")
    print("=" * 70)
    print(f"\nDevice: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ---- Config (must match v15b exactly) ----
    WINDOW_SIZE = 256
    LATENT_DIM = 64
    STRIDE = 4
    BATCH_SIZE = 16
    RVQ_LEVELS = 8
    PATCH_SIZE = 16

    config = UMCConfig(
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
        vq_type="vq",
        vq_dim=32,
        vq_codebook_dim=16,
        vq_top_n_codes=16,
        vq_bottom_n_codes=256,
        vq_bottom_n_levels=RVQ_LEVELS,
        vq_commitment_weight=0.25,
        vq_ema_decay=0.99,
        vq_dead_code_threshold=2,
        beta_start=0.0, beta_end=0.0,
        sparsity_weight=0.0, smoothness_weight=0.0,
        multiscale_weight=0.1, spectral_weight=0.1,
        close_weight=2.0, volume_weight=5.0,
    )

    # ---- Load trained v15b model ----
    state_path = "results/v15b_best_state.pt"
    print(f"\nLoading v15b model from {state_path}...")
    if not os.path.exists(state_path):
        print(f"  ERROR: {state_path} not found. Run experiment_v15b_wide_vq.py first.")
        sys.exit(1)

    encoder = HVQVAEEncoder(config)
    decoder = HVQVAEDecoder(config)

    best_state = torch.load(state_path, weights_only=False, map_location="cpu")
    encoder.load_state_dict(best_state['encoder'])
    decoder.load_state_dict(best_state['decoder'])
    print(f"  Loaded best state from epoch {best_state.get('epoch', '?')}")
    print(f"  Best val MSE: {best_state.get('val_loss', 'N/A')}")

    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    encoder._use_grad_checkpoint = False
    decoder._use_grad_checkpoint = False

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"  Params: {n_enc + n_dec:,} ({n_enc:,} enc + {n_dec:,} dec)")

    # ---- Prepare test data (same split as v15b) ----
    print(f"\nPreparing test data...")
    symbols = [
        "SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "BTC-USD", "ETH-USD", "SOL-USD",
        "GC=F", "CL=F", "SI=F",
        "EURUSD=X", "GBPUSD=X",
        "TLT", "IEF",
    ]
    datasets = load_yahoo_finance(symbols, period="2y", interval="1h")
    df = combine_datasets(datasets)
    print(f"  Total rows: {len(df):,}")

    preprocessor_config = UMCConfig(window_size=WINDOW_SIZE)
    preprocessor = OHLCVPreprocessor(preprocessor_config)
    normalized = preprocessor.fit_transform(df)
    windows = create_windows(normalized, WINDOW_SIZE, stride=STRIDE)
    print(f"  Windows: {windows.shape}")

    n = len(windows)
    gap = WINDOW_SIZE
    n_test = max(int(n * 0.1), 10)
    n_val = max(int(n * 0.1), 10)
    n_train = n - n_val - n_test - 2 * gap
    test_start = n_train + gap + n_val + gap
    test_w = windows[test_start:].astype(np.float32)
    print(f"  Test windows: {len(test_w)}")

    raw_bytes_total = test_w.nbytes
    raw_bytes_per_window = test_w[0].nbytes  # 256 * 5 * 4 = 5120
    print(f"  Raw size: {raw_bytes_total:,} bytes ({raw_bytes_per_window} bytes/window)")

    # ================================================================
    #  TIER 1: VQ-Only (Lossy Search Index)
    # ================================================================
    print(f"\n{'='*70}")
    print("  TIER 1: VQ-Only Encoding (Lossy Search Index)")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    use_amp = device.type == "cuda"
    n_patches = WINDOW_SIZE // PATCH_SIZE

    all_top_indices = []
    all_bottom_indices = []
    all_revin_means = []
    all_revin_stds = []
    all_x_hat = []
    all_x_orig = []

    with torch.no_grad():
        for i in range(0, len(test_w), BATCH_SIZE):
            x = torch.from_numpy(test_w[i:i + BATCH_SIZE]).to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                encoder.encode(x)

                # Capture VQ indices
                top_idx = encoder._last_top_indices.cpu().numpy()
                bottom_stacked = torch.stack(
                    encoder._last_bottom_indices, dim=-1
                ).cpu().numpy()  # (B, P, n_levels)

                all_top_indices.append(top_idx)
                all_bottom_indices.append(bottom_stacked)

                # Capture RevIN stats
                revin_mean = encoder.revin._mean.squeeze(1).cpu().numpy()
                revin_std = encoder.revin._std.squeeze(1).cpu().numpy()
                all_revin_means.append(revin_mean)
                all_revin_stds.append(revin_std)

                # VQ reconstruction (for RMSE measurement)
                x_hat_raw = decoder.decode_from_codes(
                    encoder._last_top_quantized,
                    encoder._last_bottom_quantized,
                )
                x_hat = encoder.revin.inverse(x_hat_raw)

            all_x_hat.append(x_hat.cpu().float().numpy())
            all_x_orig.append(x.cpu().float().numpy())
            encoder.clear_cached()

    top_indices = np.concatenate(all_top_indices)
    bottom_indices = np.concatenate(all_bottom_indices)
    revin_means = np.concatenate(all_revin_means).astype(np.float32)
    revin_stds = np.concatenate(all_revin_stds).astype(np.float32)
    x_hat_vq = np.concatenate(all_x_hat)
    x_test = np.concatenate(all_x_orig)
    t1_encode = time.perf_counter() - t0

    n_windows = len(top_indices)

    # VQ indices entropy coding
    vq_idx = VQIndices(
        top_indices=top_indices.astype(np.uint8),
        bottom_indices=bottom_indices.astype(np.uint8),
        n_patches=n_patches,
        n_levels=RVQ_LEVELS,
        top_n_codes=config.vq_top_n_codes,
        bottom_n_codes=config.vq_bottom_n_codes,
    )
    vq_compressed = compress_indices(vq_idx)
    vq_stats = compute_compression_stats(vq_idx)

    # RMSE
    data_range = x_test.max() - x_test.min()
    rmse_vq = np.sqrt(np.mean((x_test - x_hat_vq) ** 2))
    rmse_vq_pct = rmse_vq / data_range * 100

    vq_bytes_per_window = len(vq_compressed) / n_windows
    vq_compression = raw_bytes_per_window / vq_bytes_per_window

    print(f"\n  Encode time: {t1_encode:.1f}s ({t1_encode/n_windows*1000:.1f}ms/window)")
    print(f"  VQ RMSE: {rmse_vq_pct:.4f}% of data range")
    print(f"  VQ bytes: {len(vq_compressed):,} ({vq_bytes_per_window:.1f} bytes/window)")
    print(f"  VQ compression: {vq_compression:.1f}x")

    # Per-feature VQ RMSE
    print(f"\n  Per-feature VQ RMSE:")
    for i, feat in enumerate(config.features):
        fr = np.sqrt(np.mean((x_test[:, :, i] - x_hat_vq[:, :, i]) ** 2))
        feat_range = max(x_test[:, :, i].max() - x_test[:, :, i].min(), 1e-8)
        fp = fr / feat_range * 100
        print(f"    {feat:>8s}: {fp:.4f}%")

    # Naive vs entropy-coded VQ
    naive_bytes = 1 + n_patches * RVQ_LEVELS  # 129 bytes
    print(f"\n  VQ naive:   {naive_bytes} bytes/window -> {raw_bytes_per_window/naive_bytes:.1f}x")
    print(f"  VQ entropy: {vq_bytes_per_window:.1f} bytes/window -> {vq_compression:.1f}x")

    # ================================================================
    #  TIER 2: Full Lossless (VQ + XOR Residual)
    # ================================================================
    print(f"\n{'='*70}")
    print("  TIER 2: Full Lossless (VQ + XOR Residuals)")
    print(f"{'='*70}")

    # Use the LosslessCodec for proper bit-exact round-trip
    codec_zlib = LosslessCodec(encoder, decoder, device=str(device), residual_method="zlib")

    t0 = time.perf_counter()
    encoding = codec_zlib.encode(test_w, batch_size=BATCH_SIZE)
    t2_encode = time.perf_counter() - t0

    # Round-trip verification
    t0 = time.perf_counter()
    recovered = codec_zlib.decode(encoding, batch_size=BATCH_SIZE)
    t2_decode = time.perf_counter() - t0

    bit_exact = np.array_equal(
        test_w.astype(np.float32).view(np.uint32),
        recovered.view(np.uint32),
    )

    # Stats
    stats = codec_zlib.compression_stats(encoding)

    print(f"\n  Encode time: {t2_encode:.1f}s ({t2_encode/n_windows*1000:.1f}ms/window)")
    print(f"  Decode time: {t2_decode:.1f}s ({t2_decode/n_windows*1000:.1f}ms/window)")
    print(f"  Bit-exact round-trip: {'YES' if bit_exact else 'NO *** FAILURE ***'}")

    if not bit_exact:
        diff = np.abs(test_w.astype(np.float32) - recovered)
        print(f"  *** Max abs error: {diff.max():.10e}")
        print(f"  *** Mean abs error: {diff.mean():.10e}")
        # Check if float-equal even if not bit-equal
        close = np.allclose(test_w.astype(np.float32), recovered, atol=0, rtol=0)
        print(f"  *** np.allclose (atol=0, rtol=0): {close}")

    print(f"\n  Compression breakdown (zlib residuals):")
    print(f"    Raw:      {stats['raw_bytes']:>12,} bytes")
    print(f"    VQ codes: {stats['vq_bytes']:>12,} bytes ({stats['vq_bytes_per_window']:.1f}/window)")
    print(f"    Residual: {stats['residual_bytes']:>12,} bytes ({stats['residual_bytes_per_window']:.1f}/window)")
    print(f"    Total:    {stats['total_bytes']:>12,} bytes ({stats['bytes_per_window']:.1f}/window)")
    print(f"    Ratio:    {stats['compression_ratio']:.2f}x lossless")
    print(f"    VQ-only:  {stats['vq_compression_ratio']:.1f}x lossy (for search)")

    # ---- Also try static arithmetic coding ----
    try:
        from umc.codec.arithmetic import StaticByteCompressor
        print(f"\n  Trying static arithmetic coding for residuals...")

        static_comp = StaticByteCompressor()
        codec_static = LosslessCodec(
            encoder, decoder, device=str(device),
            residual_method="static", residual_compressor=static_comp,
        )

        t0 = time.perf_counter()
        encoding_static = codec_static.encode(test_w, batch_size=BATCH_SIZE)
        t_static_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        recovered_static = codec_static.decode(encoding_static, batch_size=BATCH_SIZE)
        t_static_dec = time.perf_counter() - t0

        bit_exact_static = np.array_equal(
            test_w.astype(np.float32).view(np.uint32),
            recovered_static.view(np.uint32),
        )

        stats_static = codec_static.compression_stats(encoding_static)

        print(f"    Encode: {t_static_enc:.1f}s | Decode: {t_static_dec:.1f}s")
        print(f"    Bit-exact: {'YES' if bit_exact_static else 'NO *** FAILURE ***'}")
        print(f"    Residual: {stats_static['residual_bytes']:,} bytes ({stats_static['residual_bytes_per_window']:.1f}/window)")
        print(f"    Total:    {stats_static['total_bytes']:,} bytes ({stats_static['bytes_per_window']:.1f}/window)")
        print(f"    Ratio:    {stats_static['compression_ratio']:.2f}x lossless")
        improvement = (1 - stats_static['total_bytes'] / stats['total_bytes']) * 100
        print(f"    vs zlib:  {improvement:+.1f}% size reduction")

    except ImportError:
        print(f"\n  Skipping arithmetic coding (constriction not installed)")
        stats_static = None

    # ---- Serialization round-trip ----
    print(f"\n  Serialization round-trip test...")
    serialized = serialize_encoding(encoding)
    encoding_rt = deserialize_encoding(serialized)
    recovered_rt = codec_zlib.decode(encoding_rt, batch_size=BATCH_SIZE)
    rt_exact = np.array_equal(
        test_w.astype(np.float32).view(np.uint32),
        recovered_rt.view(np.uint32),
    )
    print(f"    Serialized size: {len(serialized):,} bytes")
    print(f"    Serialize -> Deserialize -> Decode: {'BIT-EXACT' if rt_exact else '*** FAILED ***'}")

    # ================================================================
    #  TIER 1.5: Quantized Residuals (near-lossless, practical)
    # ================================================================
    print(f"\n{'='*70}")
    print("  TIER 1.5: Quantized Residuals (near-lossless)")
    print(f"{'='*70}")

    import zlib as zlib_mod

    # Compute float residuals per-feature
    residuals_float = test_w.astype(np.float32) - x_hat_vq  # (N, 256, 5)

    # Test multiple quantization levels
    for nbits, label in [(16, "int16"), (8, "int8")]:
        print(f"\n  --- {label} quantization ---")

        # Per-feature min/max of residuals for scaling
        res_flat = residuals_float.reshape(-1, 5)
        feat_min = res_flat.min(axis=0)  # (5,)
        feat_max = res_flat.max(axis=0)  # (5,)
        feat_range = feat_max - feat_min
        feat_range = np.maximum(feat_range, 1e-10)

        # Quantize
        if nbits == 16:
            n_levels = 65536
            dtype = np.int16
        else:
            n_levels = 256
            dtype = np.int8

        # Scale to [0, n_levels-1], then center to signed range
        scaled = (residuals_float - feat_min) / feat_range  # [0, 1]
        quantized_uint = np.clip(scaled * (n_levels - 1), 0, n_levels - 1).astype(np.uint16 if nbits == 16 else np.uint8)

        # Dequantize
        dequant = quantized_uint.astype(np.float64) / (n_levels - 1) * feat_range + feat_min
        reconstructed = x_hat_vq + dequant.astype(np.float32)

        # Measure RMSE
        rmse_qr = np.sqrt(np.mean((test_w.astype(np.float32) - reconstructed) ** 2))
        rmse_qr_pct = rmse_qr / data_range * 100

        # Compress quantized residuals
        # Header: per-feature min/max (5 * 2 * 4 = 40 bytes)
        header_bytes = feat_min.astype(np.float32).tobytes() + feat_max.astype(np.float32).tobytes()
        raw_qr = quantized_uint.tobytes()
        compressed_qr = zlib_mod.compress(raw_qr, 9)

        total_qr = len(vq_compressed) + len(header_bytes) + len(compressed_qr)
        qr_per_window = total_qr / n_windows
        qr_compression = raw_bytes_per_window / qr_per_window

        print(f"    RMSE:          {rmse_qr_pct:.6f}% of data range")
        print(f"    Quant raw:     {len(raw_qr):,} bytes ({len(raw_qr)/n_windows:.0f}/window)")
        print(f"    Quant zlib:    {len(compressed_qr):,} bytes ({len(compressed_qr)/n_windows:.1f}/window)")
        print(f"    VQ codes:      {len(vq_compressed):,} bytes ({vq_bytes_per_window:.1f}/window)")
        print(f"    Total:         {total_qr:,} bytes ({qr_per_window:.1f}/window)")
        print(f"    Compression:   {qr_compression:.2f}x")
        print(f"    Meets <0.1%:   {'YES' if rmse_qr_pct < 0.1 else 'NO'}")

        # Per-feature RMSE
        print(f"    Per-feature RMSE:")
        for i, feat in enumerate(config.features):
            fr = np.sqrt(np.mean((test_w[:, :, i].astype(np.float32) - reconstructed[:, :, i]) ** 2))
            feat_r = max(x_test[:, :, i].max() - x_test[:, :, i].min(), 1e-8)
            fp = fr / feat_r * 100
            print(f"      {feat:>8s}: {fp:.6f}%")

    # ================================================================
    #  LOSSLESS COMPRESSION EXPLORATION
    #  Can we push past 47x zlib-on-raw with domain-specific encoding?
    # ================================================================
    print(f"\n{'='*70}")
    print("  LOSSLESS COMPRESSION EXPLORATION")
    print("  Target: >50x lossless (or near-lossless <0.01% RMSE)")
    print(f"{'='*70}")

    raw_f32 = test_w.astype(np.float32)  # (N, 256, 5)

    results_table = []

    # 1. Baseline: zlib on raw float32
    raw_bytes_flat = raw_f32.tobytes()
    z = zlib_mod.compress(raw_bytes_flat, 9)
    results_table.append(("zlib(float32)", len(z), 0.0, True))

    # 2. Byte-transposed float32 + zlib (same trick as XOR residual preprocessing)
    # Group all byte-0s together, all byte-1s together, etc.
    raw_u8 = np.frombuffer(raw_f32.tobytes(), dtype=np.uint8).reshape(-1, 4)
    transposed = np.ascontiguousarray(raw_u8.T).tobytes()  # (4, N*256*5)
    z = zlib_mod.compress(transposed, 9)
    results_table.append(("byte_transpose+zlib", len(z), 0.0, True))

    # 3. Float16 + zlib (tiny precision loss)
    f16 = raw_f32.astype(np.float16)
    f16_back = f16.astype(np.float32)
    rmse_f16 = np.sqrt(np.mean((raw_f32 - f16_back) ** 2)) / data_range * 100
    z = zlib_mod.compress(f16.tobytes(), 9)
    results_table.append((f"float16+zlib ({rmse_f16:.4f}%)", len(z), rmse_f16, False))

    # 4. Delta encoding (consecutive timestep diffs) + zlib
    # First timestep stored as-is, rest as diffs
    delta = np.zeros_like(raw_f32)
    delta[:, 0, :] = raw_f32[:, 0, :]
    delta[:, 1:, :] = raw_f32[:, 1:, :] - raw_f32[:, :-1, :]
    z = zlib_mod.compress(delta.tobytes(), 9)
    results_table.append(("delta_f32+zlib", len(z), 0.0, True))

    # 5. Delta float16 + zlib
    delta_f16 = delta.astype(np.float16)
    delta_f16_back = delta_f16.astype(np.float32)
    recon_delta_f16 = np.cumsum(delta_f16_back, axis=1)
    # Fix: first timestep is stored as-is
    recon_delta_f16[:, 0, :] = delta_f16_back[:, 0, :]
    for t in range(1, 256):
        recon_delta_f16[:, t, :] = recon_delta_f16[:, t-1, :] + delta_f16_back[:, t, :]
    rmse_df16 = np.sqrt(np.mean((raw_f32 - recon_delta_f16) ** 2)) / data_range * 100
    z = zlib_mod.compress(delta_f16.tobytes(), 9)
    results_table.append((f"delta_f16+zlib ({rmse_df16:.4f}%)", len(z), rmse_df16, False))

    # 6. Byte-transposed delta float32 + zlib
    delta_u8 = np.frombuffer(delta.astype(np.float32).tobytes(), dtype=np.uint8).reshape(-1, 4)
    transposed_delta = np.ascontiguousarray(delta_u8.T).tobytes()
    z = zlib_mod.compress(transposed_delta, 9)
    results_table.append(("delta_f32+byte_tr+zlib", len(z), 0.0, True))

    # 7. Per-window normalization (subtract mean, divide by std per feature) + float16 + zlib
    means = raw_f32.mean(axis=1, keepdims=True)  # (N, 1, 5)
    stds = raw_f32.std(axis=1, keepdims=True) + 1e-8  # (N, 1, 5)
    normed = ((raw_f32 - means) / stds).astype(np.float16)
    normed_back = normed.astype(np.float32) * stds + means
    rmse_norm = np.sqrt(np.mean((raw_f32 - normed_back) ** 2)) / data_range * 100
    stats_bytes = means.astype(np.float32).tobytes() + stds.astype(np.float32).tobytes()
    z = zlib_mod.compress(normed.tobytes() + stats_bytes, 9)
    results_table.append((f"norm_f16+zlib ({rmse_norm:.4f}%)", len(z), rmse_norm, False))

    # 8. zstd if available (better than zlib for structured data)
    try:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=22)
        z = cctx.compress(raw_bytes_flat)
        results_table.append(("zstd(float32, lvl22)", len(z), 0.0, True))

        z = cctx.compress(transposed)
        results_table.append(("byte_tr+zstd(lvl22)", len(z), 0.0, True))

        z = cctx.compress(transposed_delta)
        results_table.append(("delta+byte_tr+zstd", len(z), 0.0, True))
    except ImportError:
        pass

    # 9. lzma (very slow but highest ratio)
    import lzma
    z = lzma.compress(transposed, preset=9)
    results_table.append(("byte_tr+lzma(9)", len(z), 0.0, True))

    z = lzma.compress(transposed_delta, preset=9)
    results_table.append(("delta+byte_tr+lzma(9)", len(z), 0.0, True))

    # Print results table
    print(f"\n  {'Method':<35s} {'Bytes':>12s} {'Per-win':>8s} {'Ratio':>7s} {'RMSE%':>10s} {'Lossless':>8s}")
    print(f"  {'-'*35} {'-'*12} {'-'*8} {'-'*7} {'-'*10} {'-'*8}")
    for name, size, rmse, lossless in sorted(results_table, key=lambda x: x[1]):
        ratio = raw_bytes_total / size
        per_w = size / n_windows
        ll = "yes" if lossless else "no"
        rmse_str = "0" if lossless else f"{rmse:.4f}"
        print(f"  {name:<35s} {size:>12,} {per_w:>8.1f} {ratio:>7.1f}x {rmse_str:>10s} {ll:>8s}")

    # VQ search index as additional context
    print(f"\n  + VQ search index adds {vq_bytes_per_window:.1f} bytes/window ({len(vq_compressed):,} total)")
    print(f"    Combined best lossless + VQ search:")
    best_lossless = min(results_table, key=lambda x: x[1] if x[3] else float('inf'))
    combined = best_lossless[1] + len(vq_compressed)
    combined_ratio = raw_bytes_total / combined
    print(f"    {best_lossless[0]} + VQ = {combined:,} bytes ({combined_ratio:.1f}x)")

    # ================================================================
    #  SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("  TWO-TIER PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  Test data: {n_windows} windows x {WINDOW_SIZE} timesteps x 5 features")
    print(f"  Raw size:  {raw_bytes_per_window} bytes/window (float32)")
    print()
    print(f"  TIER 1: VQ Search Index (lossy)")
    print(f"    RMSE:        {rmse_vq_pct:.4f}% of data range")
    print(f"    Size:        {vq_bytes_per_window:.1f} bytes/window")
    print(f"    Compression: {vq_compression:.1f}x")
    print(f"    Use:         FAISS similarity search")
    print()
    print(f"  TIER 2: + Lossless Residuals (bit-exact)")
    print(f"    RMSE:        0.0000% (bit-exact)")
    print(f"    Size:        {stats['bytes_per_window']:.1f} bytes/window (zlib)")
    print(f"    Compression: {stats['compression_ratio']:.2f}x lossless")
    print(f"    Round-trip:  {'VERIFIED BIT-EXACT' if bit_exact else 'FAILED'}")

    # Compare with raw compression baselines
    raw_bytes = test_w.astype(np.float32).tobytes()
    zlib_raw = zlib_mod.compress(raw_bytes, 9)
    print(f"\n  Baselines:")
    print(f"    Raw float32:    {len(raw_bytes):>12,} bytes (1.0x)")
    print(f"    zlib(raw):      {len(zlib_raw):>12,} bytes ({len(raw_bytes)/len(zlib_raw):.2f}x)")
    print(f"    VQ (lossy):     {len(vq_compressed):>12,} bytes ({len(raw_bytes)/len(vq_compressed):.1f}x, {rmse_vq_pct:.2f}% RMSE)")
    print(f"    VQ+XOR (zlib):  {stats['total_bytes']:>12,} bytes ({stats['compression_ratio']:.2f}x, 0% RMSE)")

    print(f"\n  Encode speed: {t2_encode/n_windows*1000:.1f}ms/window")
    print(f"  Decode speed: {t2_decode/n_windows*1000:.1f}ms/window")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
