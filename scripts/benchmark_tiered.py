#!/usr/bin/env python3
"""Comprehensive benchmark for UMC tiered compression system.

Tests all 4 storage modes against raw compression baselines.
Measures compression ratio, encode/decode speed, RMSE, and per-tier stats.

Uses the trained v15b model (window_size=256, patch_size=16, vq_dim=32).
If the model is not found, creates a small test model for benchmarking.
"""

import gc
import os
import sys
import time
import zlib
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from umc.config import UMCConfig
from umc.encoder.hvqvae_encoder import HVQVAEEncoder
from umc.decoder.hvqvae_decoder import HVQVAEDecoder
from umc.codec.tiered import TieredCodec, TieredEncoding, serialize_tiered, deserialize_tiered
from umc.storage.entropy import VQIndices, compress_indices, compute_compression_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_financial_data(n_windows, window_size, n_features=5, seed=42):
    """Generate realistic financial-like OHLCV data with correlated trends.

    Creates correlated prices with:
    - Mean-reverting trends
    - Volatility clustering
    - OHLC consistency (low <= open,close <= high)
    - Log-normal volume
    """
    rng = np.random.RandomState(seed)
    data = np.zeros((n_windows, window_size, n_features), dtype=np.float32)

    for w in range(n_windows):
        # Base price with mean-reverting trend
        base_price = rng.uniform(10, 500)
        drift = rng.uniform(-0.001, 0.001)
        volatility = rng.uniform(0.005, 0.03)

        # Volatility clustering (GARCH-like)
        vol_state = volatility
        prices = np.zeros(window_size)
        prices[0] = base_price

        for t in range(1, window_size):
            # Update vol state
            shock = rng.randn()
            vol_state = 0.9 * vol_state + 0.1 * volatility * (1 + 0.5 * abs(shock))
            # Price update
            ret = drift + vol_state * shock
            prices[t] = prices[t - 1] * (1 + ret)

        # Generate OHLC from close prices
        close = prices
        # Intrabar volatility
        intra_vol = rng.uniform(0.002, 0.015, size=window_size)
        high = close * (1 + abs(rng.randn(window_size)) * intra_vol)
        low = close * (1 - abs(rng.randn(window_size)) * intra_vol)
        open_price = low + rng.rand(window_size) * (high - low)

        # Ensure OHLC consistency
        high = np.maximum(high, np.maximum(open_price, close))
        low = np.minimum(low, np.minimum(open_price, close))

        # Volume: log-normal with autocorrelation
        base_vol = rng.uniform(1e4, 1e7)
        log_vol = np.log(base_vol) + np.cumsum(rng.randn(window_size) * 0.3)
        volume = np.exp(log_vol)

        data[w, :, 0] = open_price
        data[w, :, 1] = high
        data[w, :, 2] = low
        data[w, :, 3] = close
        data[w, :, 4] = volume

    return data


def get_v15b_config():
    """Return the exact config used for v15b training."""
    return UMCConfig(
        window_size=256,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=64,
        encoder_type="hvqvae",
        num_charts=16,
        chart_embedding_dim=16,
        batch_size=16,
        learning_rate=3e-4,
        d_model=128,
        n_heads=4,
        n_encoder_layers=4,
        n_decoder_layers=4,
        d_ff=512,
        patch_size=16,
        transformer_dropout=0.1,
        vq_type="vq",
        vq_dim=32,
        vq_codebook_dim=16,
        vq_top_n_codes=16,
        vq_bottom_n_codes=256,
        vq_bottom_n_levels=8,
        vq_commitment_weight=0.25,
        vq_ema_decay=0.99,
        vq_dead_code_threshold=2,
        beta_start=0.0,
        beta_end=0.0,
        sparsity_weight=0.0,
        smoothness_weight=0.0,
        multiscale_weight=0.1,
        spectral_weight=0.1,
        close_weight=2.0,
        volume_weight=5.0,
    )


def get_small_test_config():
    """Return a small config for testing when no trained model is available."""
    return UMCConfig(
        window_size=32,
        features=("open", "high", "low", "close", "volume"),
        max_latent_dim=16,
        encoder_type="hvqvae",
        num_charts=4,
        chart_embedding_dim=8,
        batch_size=32,
        d_model=32,
        n_heads=2,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=64,
        patch_size=8,
        transformer_dropout=0.1,
        vq_type="vq",
        vq_dim=16,
        vq_codebook_dim=8,
        vq_top_n_codes=8,
        vq_bottom_n_codes=64,
        vq_bottom_n_levels=2,
        vq_commitment_weight=0.25,
        vq_ema_decay=0.99,
        vq_dead_code_threshold=2,
    )


def format_bytes(n):
    """Format byte count for display."""
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    elif n >= 1e6:
        return f"{n / 1e6:.2f} MB"
    elif n >= 1e3:
        return f"{n / 1e3:.1f} KB"
    return f"{n} B"


def print_table_row(cols, widths, align=None):
    """Print a formatted table row."""
    parts = []
    for i, (col, w) in enumerate(zip(cols, widths)):
        a = align[i] if align else "<"
        parts.append(f"{col:{a}{w}}")
    print("  " + " | ".join(parts))


def print_separator(widths):
    print("  " + "-+-".join("-" * w for w in widths))


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("  UMC TIERED COMPRESSION BENCHMARK")
    print("=" * 78)
    print(f"  Device: {device}", end="")
    if device.type == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()

    # ---- Load or create model ----
    state_path = Path(__file__).parent.parent / "results" / "v15b_best_state.pt"
    use_trained = state_path.exists()

    if use_trained:
        print(f"\n  Loading trained v15b model from {state_path}")
        config = get_v15b_config()
        encoder = HVQVAEEncoder(config)
        decoder = HVQVAEDecoder(config)
        best_state = torch.load(str(state_path), weights_only=False, map_location="cpu")
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])
        print(f"  Loaded best state from epoch {best_state.get('epoch', '?')}")
        print(f"  Val MSE: {best_state.get('val_loss', 'N/A')}")
    else:
        print(f"\n  v15b model not found at {state_path}")
        print("  Creating small untrained model for testing...")
        config = get_small_test_config()
        encoder = HVQVAEEncoder(config)
        decoder = HVQVAEDecoder(config)

    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"  Model params: {n_enc + n_dec:,} ({n_enc:,} enc + {n_dec:,} dec)")
    print(f"  Window: {config.window_size} x {config.n_features} = {config.window_size * config.n_features} values")
    print(f"  Patches: {config.window_size // config.patch_size} (patch_size={config.patch_size})")
    print(f"  VQ: dim={config.vq_dim}, cb_dim={config.vq_codebook_dim}, "
          f"top={config.vq_top_n_codes}, bottom={config.vq_bottom_n_codes}x{config.vq_bottom_n_levels}RVQ")

    window_size = config.window_size
    n_features = config.n_features
    raw_bytes_per_window = window_size * n_features * 4  # float32

    # ---- Generate test data ----
    test_sizes = [100, 1000, 10000]
    storage_modes = ["lossless", "near_lossless", "lossless_zstd", "lossless_lzma"]

    print(f"\n  Generating financial test data (window_size={window_size})...")
    datasets = {}
    for n in test_sizes:
        data = generate_financial_data(n, window_size, n_features)
        datasets[n] = data
        print(f"    {n:>6d} windows: {format_bytes(data.nbytes):>10s} raw")

    # ==================================================================
    #  SECTION 1: Per-mode compression benchmarks
    # ==================================================================
    print(f"\n{'=' * 78}")
    print("  SECTION 1: TIERED COMPRESSION — ALL MODES")
    print(f"{'=' * 78}")

    # Results storage
    all_results = {}

    for n_win in test_sizes:
        data = datasets[n_win]
        raw_total = data.nbytes

        print(f"\n  --- {n_win} windows ({format_bytes(raw_total)}) ---")

        widths = [18, 10, 10, 12, 12, 12, 12, 12, 10]
        headers = ["Mode", "Ratio", "RMSE%", "Enc win/s", "Enc MB/s",
                    "Dec(stor)/s", "Dec(srch)/s", "VQ B/win", "Stor B/win"]
        align = ["<", ">", ">", ">", ">", ">", ">", ">", ">"]

        print()
        print_table_row(headers, widths, align)
        print_separator(widths)

        for mode in storage_modes:
            codec = TieredCodec(encoder, decoder, device=str(device), storage_mode=mode)
            batch_size = 32

            # ---- Encode ----
            t0 = time.perf_counter()
            encoding = codec.encode(data, batch_size=batch_size)
            t_encode = time.perf_counter() - t0

            # ---- Serialize (to measure total compressed size) ----
            serialized = serialize_tiered(encoding)
            total_compressed = len(serialized)

            # ---- Compression stats ----
            stats = codec.compression_stats(encoding)
            ratio = raw_total / total_compressed

            # ---- Decode (storage tier) ----
            t0 = time.perf_counter()
            decoded_storage = codec.decode_storage(encoding)
            t_decode_storage = time.perf_counter() - t0

            # ---- Decode (search/VQ tier) ----
            t0 = time.perf_counter()
            decoded_search = codec.decode_search(encoding, batch_size=batch_size)
            t_decode_search = time.perf_counter() - t0

            # ---- RMSE ----
            # Storage RMSE
            if mode == "lossless" or mode == "lossless_zstd" or mode == "lossless_lzma":
                storage_rmse_pct = 0.0
                # Verify bit-exact
                bit_exact = np.array_equal(
                    data.astype(np.float32).view(np.uint32),
                    decoded_storage.view(np.uint32),
                )
                if not bit_exact:
                    diff = np.abs(data.astype(np.float32) - decoded_storage)
                    storage_rmse_pct = np.sqrt(np.mean(diff ** 2)) / max(
                        data.max() - data.min(), 1e-10
                    ) * 100
            else:
                data_range = data.max() - data.min()
                rmse = np.sqrt(np.mean((data.astype(np.float32) - decoded_storage) ** 2))
                storage_rmse_pct = rmse / max(data_range, 1e-10) * 100

            # Search RMSE
            data_range = data.max() - data.min()
            search_rmse = np.sqrt(np.mean((data.astype(np.float32) - decoded_search) ** 2))
            search_rmse_pct = search_rmse / max(data_range, 1e-10) * 100

            # ---- Speeds ----
            enc_win_per_sec = n_win / max(t_encode, 1e-9)
            enc_mb_per_sec = (raw_total / 1e6) / max(t_encode, 1e-9)
            dec_stor_win_per_sec = n_win / max(t_decode_storage, 1e-9)
            dec_srch_win_per_sec = n_win / max(t_decode_search, 1e-9)

            # ---- Per-tier bytes/window ----
            vq_bpw = stats["vq_bytes_per_window"]
            stor_bpw = stats["storage_bytes_per_window"]

            rmse_str = "0.0000" if storage_rmse_pct == 0 else f"{storage_rmse_pct:.4f}"

            row = [
                mode,
                f"{ratio:.1f}x",
                rmse_str,
                f"{enc_win_per_sec:.0f}",
                f"{enc_mb_per_sec:.1f}",
                f"{dec_stor_win_per_sec:.0f}",
                f"{dec_srch_win_per_sec:.0f}",
                f"{vq_bpw:.1f}",
                f"{stor_bpw:.1f}",
            ]
            print_table_row(row, widths, align)

            # Store results
            key = (n_win, mode)
            all_results[key] = {
                "ratio": ratio,
                "storage_rmse_pct": storage_rmse_pct,
                "search_rmse_pct": search_rmse_pct,
                "t_encode": t_encode,
                "t_decode_storage": t_decode_storage,
                "t_decode_search": t_decode_search,
                "enc_win_per_sec": enc_win_per_sec,
                "enc_mb_per_sec": enc_mb_per_sec,
                "dec_stor_win_per_sec": dec_stor_win_per_sec,
                "dec_srch_win_per_sec": dec_srch_win_per_sec,
                "vq_bpw": vq_bpw,
                "stor_bpw": stor_bpw,
                "total_compressed": total_compressed,
                "raw_total": raw_total,
            }

            del codec, encoding, decoded_storage, decoded_search
            if device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

    # ==================================================================
    #  SECTION 2: VQ search tier RMSE detail (largest dataset)
    # ==================================================================
    print(f"\n{'=' * 78}")
    print("  SECTION 2: VQ SEARCH TIER QUALITY (RMSE)")
    print(f"{'=' * 78}")

    n_win_detail = test_sizes[-1]
    data_detail = datasets[n_win_detail]
    codec_tmp = TieredCodec(encoder, decoder, device=str(device), storage_mode="lossless")
    encoding_tmp = codec_tmp.encode(data_detail, batch_size=32)
    decoded_search_detail = codec_tmp.decode_search(encoding_tmp, batch_size=32)

    data_range = data_detail.max() - data_detail.min()
    overall_rmse = np.sqrt(np.mean((data_detail - decoded_search_detail) ** 2))
    print(f"\n  VQ search RMSE ({n_win_detail} windows): {overall_rmse / data_range * 100:.4f}% of data range")
    print(f"  Data range: {data_range:.4f}")
    print()

    feat_names = list(config.features)
    widths2 = [10, 12, 12, 12]
    headers2 = ["Feature", "RMSE", "RMSE%", "Feat Range"]
    align2 = ["<", ">", ">", ">"]
    print_table_row(headers2, widths2, align2)
    print_separator(widths2)
    for i, feat in enumerate(feat_names):
        fr = np.sqrt(np.mean((data_detail[:, :, i] - decoded_search_detail[:, :, i]) ** 2))
        feat_range = data_detail[:, :, i].max() - data_detail[:, :, i].min()
        fp = fr / max(feat_range, 1e-10) * 100
        print_table_row([feat, f"{fr:.6f}", f"{fp:.4f}%", f"{feat_range:.4f}"], widths2, align2)

    del codec_tmp, encoding_tmp, decoded_search_detail
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ==================================================================
    #  SECTION 3: Baseline comparisons
    # ==================================================================
    print(f"\n{'=' * 78}")
    print("  SECTION 3: BASELINE COMPARISONS")
    print(f"{'=' * 78}")

    import lzma
    import zstandard as zstd

    for n_win in test_sizes:
        data = datasets[n_win]
        raw_total = data.nbytes
        raw_bytes = data.astype(np.float32).tobytes()

        print(f"\n  --- {n_win} windows ({format_bytes(raw_total)}) ---")

        widths3 = [25, 12, 10, 12, 12, 10]
        headers3 = ["Baseline", "Compressed", "Ratio", "Enc MB/s", "Dec MB/s", "Lossless"]
        align3 = ["<", ">", ">", ">", ">", ">"]

        print()
        print_table_row(headers3, widths3, align3)
        print_separator(widths3)

        baselines = []

        # --- Raw zlib ---
        t0 = time.perf_counter()
        z = zlib.compress(raw_bytes, 9)
        t_enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = zlib.decompress(z)
        t_dec = time.perf_counter() - t0
        baselines.append(("zlib (level=9)", len(z), t_enc, t_dec, True))

        # --- Raw zstd ---
        cctx = zstd.ZstdCompressor(level=22)
        dctx = zstd.ZstdDecompressor()
        t0 = time.perf_counter()
        z = cctx.compress(raw_bytes)
        t_enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = dctx.decompress(z)
        t_dec = time.perf_counter() - t0
        baselines.append(("zstd (level=22)", len(z), t_enc, t_dec, True))

        # --- Raw lzma ---
        t0 = time.perf_counter()
        z = lzma.compress(raw_bytes, preset=6)
        t_enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = lzma.decompress(z)
        t_dec = time.perf_counter() - t0
        baselines.append(("lzma (preset=6)", len(z), t_enc, t_dec, True))

        # --- numpy savez_compressed ---
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = tmp.name
        t0 = time.perf_counter()
        np.savez_compressed(tmp_path, data=data)
        t_enc = time.perf_counter() - t0
        npz_size = os.path.getsize(tmp_path)
        t0 = time.perf_counter()
        _ = np.load(tmp_path)["data"]
        t_dec = time.perf_counter() - t0
        baselines.append(("numpy savez_compressed", npz_size, t_enc, t_dec, True))
        os.unlink(tmp_path)

        # --- Byte-transpose + zlib (our trick, standalone) ---
        from umc.codec.residual import byte_transpose, byte_untranspose
        transposed = byte_transpose(raw_bytes, element_size=4)
        t0 = time.perf_counter()
        z = zlib.compress(transposed, 9)
        t_enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        dec_t = zlib.decompress(z)
        _ = byte_untranspose(dec_t, element_size=4)
        t_dec = time.perf_counter() - t0
        baselines.append(("byte_transpose+zlib", len(z), t_enc, t_dec, True))

        # --- Byte-transpose + zstd ---
        t0 = time.perf_counter()
        z = cctx.compress(transposed)
        t_enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        dec_t = dctx.decompress(z)
        _ = byte_untranspose(dec_t, element_size=4)
        t_dec = time.perf_counter() - t0
        baselines.append(("byte_transpose+zstd", len(z), t_enc, t_dec, True))

        # --- Byte-transpose + lzma ---
        t0 = time.perf_counter()
        z = lzma.compress(transposed, preset=6)
        t_enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        dec_t = lzma.decompress(z)
        _ = byte_untranspose(dec_t, element_size=4)
        t_dec = time.perf_counter() - t0
        baselines.append(("byte_transpose+lzma", len(z), t_enc, t_dec, True))

        # Print baselines
        for name, comp_size, t_enc, t_dec, lossless in baselines:
            ratio = raw_total / comp_size
            enc_mb = (raw_total / 1e6) / max(t_enc, 1e-9)
            dec_mb = (raw_total / 1e6) / max(t_dec, 1e-9)
            ll = "yes" if lossless else "no"
            row = [name, format_bytes(comp_size), f"{ratio:.1f}x",
                   f"{enc_mb:.1f}", f"{dec_mb:.1f}", ll]
            print_table_row(row, widths3, align3)

        # Now print UMC tiered modes for comparison
        print()
        print(f"  UMC Tiered (same data):")
        print_table_row(headers3, widths3, align3)
        print_separator(widths3)

        for mode in storage_modes:
            key = (n_win, mode)
            r = all_results[key]
            ll = "yes" if r["storage_rmse_pct"] == 0 else f"~{r['storage_rmse_pct']:.4f}%"
            row = [
                f"UMC {mode}",
                format_bytes(r["total_compressed"]),
                f"{r['ratio']:.1f}x",
                f"{r['enc_mb_per_sec']:.1f}",
                f"{(r['raw_total'] / 1e6) / max(r['t_decode_storage'], 1e-9):.1f}",
                ll,
            ]
            print_table_row(row, widths3, align3)

    # ==================================================================
    #  SECTION 4: Entropy coding comparison (grouped vs optimal)
    # ==================================================================
    print(f"\n{'=' * 78}")
    print("  SECTION 4: VQ INDEX ENTROPY CODING (GROUPED vs OPTIMAL)")
    print(f"{'=' * 78}")

    for n_win in test_sizes:
        data = datasets[n_win]
        codec_tmp = TieredCodec(encoder, decoder, device=str(device), storage_mode="lossless")
        encoding_tmp = codec_tmp.encode(data, batch_size=32)

        vq_idx = encoding_tmp.vq_indices

        # Compression stats from entropy module
        entropy_stats = compute_compression_stats(
            vq_idx, raw_window_bytes=raw_bytes_per_window
        )

        # Measure sizes for each mode
        t0 = time.perf_counter()
        flat_bytes = compress_indices(vq_idx, mode="flat")
        t_flat = time.perf_counter() - t0

        t0 = time.perf_counter()
        grouped_bytes = compress_indices(vq_idx, mode="grouped")
        t_grouped = time.perf_counter() - t0

        t0 = time.perf_counter()
        optimal_bytes = compress_indices(vq_idx, mode="optimal")
        t_optimal = time.perf_counter() - t0

        print(f"\n  --- {n_win} windows ---")
        print(f"  Raw VQ indices: {entropy_stats['naive_bytes_per_window']:.0f} bytes/window "
              f"(naive: {entropy_stats['naive_compression']:.1f}x)")
        print(f"  Shannon entropy: {entropy_stats['entropy_bytes_per_window']:.2f} bytes/window "
              f"(theoretical limit: {entropy_stats['entropy_compression']:.1f}x)")
        print(f"  Top entropy: {entropy_stats['top_entropy_bits']:.2f} bits/index")
        for i, e in enumerate(entropy_stats['bottom_entropy_bits']):
            print(f"  Bottom level {i}: {e:.2f} bits/index")

        widths4 = [15, 12, 12, 12, 12]
        headers4 = ["Mode", "Size", "B/window", "Ratio", "Time"]
        align4 = ["<", ">", ">", ">", ">"]
        print()
        print_table_row(headers4, widths4, align4)
        print_separator(widths4)

        for label, size, t in [("flat", len(flat_bytes), t_flat),
                                ("grouped", len(grouped_bytes), t_grouped),
                                ("optimal", len(optimal_bytes), t_optimal)]:
            bpw = size / n_win
            ratio = raw_bytes_per_window / bpw
            row = [label, format_bytes(size), f"{bpw:.2f}", f"{ratio:.1f}x", f"{t * 1000:.1f}ms"]
            print_table_row(row, widths4, align4)

        improvement = (1 - len(optimal_bytes) / len(grouped_bytes)) * 100
        print(f"\n  Optimal vs grouped: {improvement:+.1f}% size reduction")

        del codec_tmp, encoding_tmp
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ==================================================================
    #  SECTION 5: Near-lossless RMSE detail
    # ==================================================================
    print(f"\n{'=' * 78}")
    print("  SECTION 5: NEAR-LOSSLESS MODE RMSE DETAIL")
    print(f"{'=' * 78}")

    n_win_detail = test_sizes[-1]
    data_detail = datasets[n_win_detail]

    codec_nl = TieredCodec(encoder, decoder, device=str(device), storage_mode="near_lossless")
    encoding_nl = codec_nl.encode(data_detail, batch_size=32)
    decoded_nl = codec_nl.decode_storage(encoding_nl)

    data_range = data_detail.max() - data_detail.min()
    overall_rmse = np.sqrt(np.mean((data_detail - decoded_nl) ** 2))
    max_abs_err = np.max(np.abs(data_detail - decoded_nl))
    mean_abs_err = np.mean(np.abs(data_detail - decoded_nl))

    print(f"\n  Near-lossless storage ({n_win_detail} windows):")
    print(f"    Overall RMSE:    {overall_rmse:.8f} ({overall_rmse / data_range * 100:.6f}%)")
    print(f"    Max abs error:   {max_abs_err:.8f}")
    print(f"    Mean abs error:  {mean_abs_err:.8f}")
    print(f"    Data range:      {data_range:.4f}")
    print()

    feat_names = list(config.features)
    widths5 = [10, 14, 10, 14, 14]
    headers5 = ["Feature", "RMSE", "RMSE%", "Max|err|", "Feat Range"]
    align5 = ["<", ">", ">", ">", ">"]
    print_table_row(headers5, widths5, align5)
    print_separator(widths5)
    for i, feat in enumerate(feat_names):
        fr = np.sqrt(np.mean((data_detail[:, :, i] - decoded_nl[:, :, i]) ** 2))
        feat_range = data_detail[:, :, i].max() - data_detail[:, :, i].min()
        fp = fr / max(feat_range, 1e-10) * 100
        max_e = np.max(np.abs(data_detail[:, :, i] - decoded_nl[:, :, i]))
        print_table_row([feat, f"{fr:.8f}", f"{fp:.6f}%", f"{max_e:.8f}", f"{feat_range:.4f}"],
                       widths5, align5)

    del codec_nl, encoding_nl, decoded_nl

    # ==================================================================
    #  SECTION 6: Summary
    # ==================================================================
    print(f"\n{'=' * 78}")
    print("  SUMMARY")
    print(f"{'=' * 78}")

    n_win_summary = test_sizes[-1]
    print(f"\n  Data: {n_win_summary} windows x {window_size} timesteps x {n_features} features (float32)")
    print(f"  Raw: {raw_bytes_per_window} bytes/window ({format_bytes(datasets[n_win_summary].nbytes)})")
    print()

    widths6 = [18, 10, 10, 12, 12, 10, 10]
    headers6 = ["Mode", "Ratio", "RMSE%", "Enc w/s", "Dec(S) w/s", "VQ B/w", "Stor B/w"]
    align6 = ["<", ">", ">", ">", ">", ">", ">"]
    print_table_row(headers6, widths6, align6)
    print_separator(widths6)

    for mode in storage_modes:
        r = all_results[(n_win_summary, mode)]
        rmse_str = "0.0000" if r["storage_rmse_pct"] == 0 else f"{r['storage_rmse_pct']:.4f}"
        row = [
            mode,
            f"{r['ratio']:.1f}x",
            rmse_str,
            f"{r['enc_win_per_sec']:.0f}",
            f"{r['dec_stor_win_per_sec']:.0f}",
            f"{r['vq_bpw']:.1f}",
            f"{r['stor_bpw']:.1f}",
        ]
        print_table_row(row, widths6, align6)

    # VQ search tier (same for all modes)
    r0 = all_results[(n_win_summary, "lossless")]
    print(f"\n  VQ search tier: {r0['vq_bpw']:.1f} bytes/window "
          f"({raw_bytes_per_window / r0['vq_bpw']:.1f}x compression)")

    # Search RMSE
    search_rmse = all_results[(n_win_summary, "lossless")]["search_rmse_pct"]
    print(f"  VQ search RMSE: {search_rmse:.4f}%")

    print(f"\n{'=' * 78}")
    print("  BENCHMARK COMPLETE")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
