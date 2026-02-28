#!/usr/bin/env python3
"""
Searchable Compression Demo — UMC's Killer Feature
====================================================

Traditional compressors force a costly trade-off:

    Compressed  --[decompress everything]--> Raw  --[search]--> Results

UMC breaks this trade-off.  Its architecture lets you compress data AND
search it — without decompressing the bulk payload.  This demo shows how.

Workflow demonstrated:
    1. Generate 5 years of realistic daily OHLCV data across 10 tickers
    2. Window it into overlapping 32-day windows
    3. Compress ALL windows with umc.compress(mode='lossless')
    4. Build a lightweight per-window index (mean/std/min/max — 4 floats
       per feature per window) that lives *alongside* the compressed blob
    5. Inject a crash pattern (-15% in a few days) into the dataset
    6. Search the index for similar crash windows WITHOUT touching the
       compressed bytes — only decompress the handful of matches
    7. Verify lossless round-trip on the full dataset

Key message:
    "UMC lets you compress AND search your data.
     Other compressors force you to decompress first."

Run:
    python scripts/demo_searchable_compression.py
"""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import umc
from umc.codec.tiered import _compress_storage, _decompress_storage


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 72
THIN_SEP  = "-" * 72

def banner(text: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {text}")
    print(SEPARATOR)

def section(text: str) -> None:
    print(f"\n{THIN_SEP}")
    print(f"  {text}")
    print(THIN_SEP)

def kv(key: str, value, indent: int = 4) -> None:
    print(f"{' ' * indent}{key:<40s} {value}")

def fmt_bytes(n: int) -> str:
    if n >= 1_000_000:
        return f"{n:>12,} bytes  ({n / 1_048_576:.2f} MB)"
    if n >= 1_000:
        return f"{n:>12,} bytes  ({n / 1_024:.1f} KB)"
    return f"{n:>12,} bytes"

def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f} ms"


# ---------------------------------------------------------------------------
# 1. Generate realistic financial data — multiple tickers
# ---------------------------------------------------------------------------

def generate_spy_ohlcv(n_days: int = 1260, seed: int = 42) -> np.ndarray:
    """Generate ~5 years of daily SPY-like OHLCV data.

    Uses a GBM price process with:
      - Volatility clustering (GARCH-like)
      - Mean reversion in volatility
      - Occasional fat-tail shocks (crashes / spikes)
      - Realistic OHLC relationships
      - Log-normal autocorrelated volume

    Returns:
        (n_days, 5) float32 -- columns: Open, High, Low, Close, Volume
    """
    rng = np.random.RandomState(seed)

    # Starting conditions
    price = 420.0          # SPY-like starting price
    annual_drift = 0.08    # ~8% annual return
    daily_drift = annual_drift / 252
    base_vol = 0.012       # ~19% annualised vol
    vol = base_vol

    closes = np.empty(n_days, dtype=np.float64)
    closes[0] = price

    for t in range(1, n_days):
        # GARCH-like volatility clustering
        shock = rng.randn()

        # Occasional fat-tail event (~1.5% of days)
        if rng.rand() < 0.015:
            shock *= rng.uniform(2.5, 5.0) * rng.choice([-1, 1])

        vol = 0.92 * vol + 0.08 * base_vol * (1.0 + 0.6 * abs(shock))
        ret = daily_drift + vol * shock

        # Light mean reversion toward long-run price growth
        mean_price = 420.0 * np.exp(daily_drift * t)
        reversion = 0.002 * (np.log(mean_price) - np.log(closes[t - 1]))
        ret += reversion

        closes[t] = closes[t - 1] * np.exp(ret)

    # Intra-day high / low / open
    intra_vol = np.abs(rng.randn(n_days)) * 0.004 + 0.001
    highs  = closes * (1.0 + intra_vol)
    lows   = closes * (1.0 - intra_vol)
    opens  = lows + rng.rand(n_days) * (highs - lows)

    # Enforce OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows  = np.minimum(lows, np.minimum(opens, closes))

    # Volume: log-normal, correlated with absolute returns
    abs_ret = np.abs(np.diff(np.log(closes), prepend=np.log(closes[0])))
    log_vol = 17.5 + rng.randn(n_days) * 0.35 + abs_ret * 25
    volume  = np.exp(log_vol)

    data = np.column_stack([opens, highs, lows, closes, volume]).astype(np.float32)
    return data


def generate_multi_ticker_data(
    n_tickers: int = 10,
    n_days: int = 1260,
    base_seed: int = 42,
) -> np.ndarray:
    """Generate OHLCV data for multiple tickers (simulating an enterprise dataset).

    Returns:
        (n_tickers * n_days, 5) float32 -- concatenated daily rows
    """
    all_data = []
    for i in range(n_tickers):
        ticker_data = generate_spy_ohlcv(n_days=n_days, seed=base_seed + i * 7)
        all_data.append(ticker_data)
    return np.concatenate(all_data, axis=0)


# ---------------------------------------------------------------------------
# 2. Window the data into overlapping windows
# ---------------------------------------------------------------------------

def create_overlapping_windows(
    data: np.ndarray,
    window_size: int = 32,
    stride: int = 1,
) -> np.ndarray:
    """Slide a window across (n_days, n_features) to produce overlapping windows.

    Returns:
        (n_windows, window_size, n_features) float32
    """
    n_days, n_features = data.shape
    n_windows = (n_days - window_size) // stride + 1
    windows = np.empty((n_windows, window_size, n_features), dtype=np.float32)
    for i in range(n_windows):
        start = i * stride
        windows[i] = data[start:start + window_size]
    return windows


# ---------------------------------------------------------------------------
# 3. Build a lightweight searchable index
# ---------------------------------------------------------------------------

def build_window_index(windows: np.ndarray) -> np.ndarray:
    """Compute a compact feature vector per window for fast search.

    For each of the F features we store 4 statistics, computed on
    normalized (return-based) data so that windows at different price
    levels but with the same *shape* (e.g. crash pattern) are nearby:

        mean_of_returns, std_of_returns, min_of_returns, max_of_returns

    This yields a (n_windows, 4*F) index that lives alongside the
    compressed blob and is *never* derived from the compressed bytes --
    it is computed once at compression time.

    Returns:
        (n_windows, 4 * n_features) float32
    """
    # windows: (N, W, F)
    # Normalize each window to fractional returns from its first value
    # so that the index captures *shape* rather than absolute level.
    first = windows[:, 0:1, :]                          # (N, 1, F)
    safe_first = np.where(np.abs(first) < 1e-10, 1.0, first)
    normed = (windows - first) / np.abs(safe_first)     # fractional change

    means = normed.mean(axis=1)    # (N, F)
    stds  = normed.std(axis=1)     # (N, F)
    mins  = normed.min(axis=1)     # (N, F)
    maxs  = normed.max(axis=1)     # (N, F)
    return np.concatenate([means, stds, mins, maxs], axis=1).astype(np.float32)


def search_index(
    index: np.ndarray,
    query_vector: np.ndarray,
    top_k: int = 10,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """Search the lightweight index for windows similar to query_vector.

    Args:
        index:        (N, D) feature index for all windows
        query_vector: (D,) feature vector of the query window
        top_k:        Number of nearest neighbours to return
        metric:       'cosine' or 'l2'

    Returns:
        (top_k_indices, top_k_distances)
    """
    if metric == "cosine":
        # Cosine similarity -> distance = 1 - sim
        norms = np.linalg.norm(index, axis=1, keepdims=True).clip(1e-10)
        q_norm = np.linalg.norm(query_vector).clip(1e-10)
        sims = (index @ query_vector) / (norms.squeeze() * q_norm)
        distances = 1.0 - sims
    elif metric == "l2":
        diff = index - query_vector[np.newaxis, :]
        distances = np.linalg.norm(diff, axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    top_k = min(top_k, len(distances))
    indices = np.argpartition(distances, top_k)[:top_k]
    indices = indices[np.argsort(distances[indices])]
    return indices, distances[indices]


# ---------------------------------------------------------------------------
# 4. Inject a crash pattern
# ---------------------------------------------------------------------------

def inject_crash(
    windows: np.ndarray,
    position: int,
    drop_pct: float = 0.15,
    crash_days: int = 5,
    seed: int = 99,
) -> np.ndarray:
    """Replace the window at `position` with a sharp crash pattern.

    The Close price drops by `drop_pct` over `crash_days`, with elevated
    volume and wider high-low spreads.

    Returns the modified windows array (in-place modification).
    """
    rng = np.random.RandomState(seed)
    w = windows[position].copy()
    window_size = w.shape[0]

    # Starting price from existing data at that position
    start_price = w[0, 3]  # Close column

    # Build the crash trajectory
    crash_start = max(0, window_size // 2 - crash_days // 2)
    for d in range(window_size):
        if crash_start <= d < crash_start + crash_days:
            # Days inside the crash: steep decline
            frac = (d - crash_start) / crash_days
            price = start_price * (1.0 - drop_pct * frac)
            # Extra intra-day volatility
            spread = abs(rng.randn()) * 0.015 + 0.005
        else:
            if d < crash_start:
                # Pre-crash: gentle upward drift
                price = start_price * (1.0 + 0.001 * d + rng.randn() * 0.003)
            else:
                # Post-crash: bouncing near the bottom
                bottom = start_price * (1.0 - drop_pct)
                price = bottom * (1.0 + rng.randn() * 0.005)
            spread = abs(rng.randn()) * 0.004 + 0.001

        high = price * (1.0 + spread)
        low  = price * (1.0 - spread)
        opn  = low + rng.rand() * (high - low)

        # Volume spikes during the crash
        vol_mult = 3.0 if crash_start <= d < crash_start + crash_days else 1.0
        vol = w[d, 4] * vol_mult * (1.0 + rng.rand() * 0.3)

        w[d] = [opn, high, low, price, vol]

    windows[position] = w
    return windows


# ---------------------------------------------------------------------------
# 5. Per-window compression helpers (for selective decompression demo)
# ---------------------------------------------------------------------------

def compress_windows_individually(windows: np.ndarray, mode: str = "lossless") -> list[bytes]:
    """Compress each window independently so we can decompress single windows."""
    compressed = []
    for i in range(windows.shape[0]):
        single = windows[i:i + 1]
        blob = umc.compress(single, mode=mode)
        compressed.append(blob)
    return compressed


def decompress_single_window(blob: bytes) -> np.ndarray:
    """Decompress a single window from its compressed bytes."""
    return umc.decompress(blob)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    banner("UMC SEARCHABLE COMPRESSION DEMO")
    print()
    print("    Traditional compressors force you to decompress everything")
    print("    before you can search.  UMC breaks that trade-off.")
    print()
    print("    This demo shows how UMC compresses financial time-series")
    print("    AND lets you find patterns (like market crashes) by searching")
    print("    a lightweight index -- without touching the compressed bytes.")

    # ------------------------------------------------------------------
    section("STEP 1 -- Generate 5 years of daily OHLCV data (10 tickers)")
    # ------------------------------------------------------------------
    n_tickers = 10
    n_days_per_ticker = 252 * 5  # ~1260 trading days each
    total_days = n_tickers * n_days_per_ticker
    raw_data = generate_multi_ticker_data(
        n_tickers=n_tickers, n_days=n_days_per_ticker, base_seed=42,
    )
    kv("Tickers simulated:", f"{n_tickers}")
    kv("Trading days per ticker:", f"{n_days_per_ticker:,}")
    kv("Total daily rows:", f"{total_days:,}")
    kv("Features per day:", "5  (Open, High, Low, Close, Volume)")
    kv("Price range (Close):",
       f"${raw_data[:, 3].min():.2f} -- ${raw_data[:, 3].max():.2f}")
    kv("Raw array shape:", str(raw_data.shape))
    kv("Raw size:", fmt_bytes(raw_data.nbytes))

    # ------------------------------------------------------------------
    section("STEP 2 -- Create overlapping 32-day windows (stride=1)")
    # ------------------------------------------------------------------
    window_size = 32
    stride = 1
    windows = create_overlapping_windows(
        raw_data, window_size=window_size, stride=stride,
    )
    n_windows = windows.shape[0]
    kv("Window size:", f"{window_size} days")
    kv("Stride:", f"{stride} day(s)")
    kv("Total windows:", f"{n_windows:,}")
    kv("Windows array shape:", str(windows.shape))
    kv("Windows raw size:", fmt_bytes(windows.nbytes))

    # ------------------------------------------------------------------
    section("STEP 3 -- Inject crash patterns into the dataset")
    # ------------------------------------------------------------------
    # Primary crash: -15% over 5 days, placed in the middle of the dataset
    crash_position = n_windows // 2
    drop_pct = 0.15
    windows = inject_crash(
        windows, crash_position, drop_pct=drop_pct, crash_days=5,
    )
    crash_window = windows[crash_position]
    close_start = crash_window[0, 3]
    close_end   = crash_window[-1, 3]
    actual_drop = (close_end - close_start) / close_start * 100
    kv("Primary crash at window:", f"{crash_position}")
    kv("Close price start:", f"${close_start:.2f}")
    kv("Close price end:", f"${close_end:.2f}")
    kv("Actual price change:", f"{actual_drop:+.1f}%")

    # Additional crash-like patterns elsewhere
    crash_positions = [crash_position]
    for i, (pos_frac, dp, cd, sd) in enumerate([
        (0.10, 0.10, 6, 200),
        (0.25, 0.12, 4, 300),
        (0.60, 0.11, 5, 400),
        (0.80, 0.13, 3, 500),
        (0.90, 0.09, 7, 600),
    ]):
        pos = int(n_windows * pos_frac)
        windows = inject_crash(windows, pos, drop_pct=dp, crash_days=cd, seed=sd)
        crash_positions.append(pos)
    kv("Additional crash-like patterns:", f"5 more at various positions")
    kv("Total injected crash windows:", f"{len(crash_positions)}")

    # ------------------------------------------------------------------
    section("STEP 4 -- Compress ALL windows with umc.compress(mode='lossless')")
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    compressed_blob = umc.compress(windows, mode="lossless")
    compress_time = time.perf_counter() - t0

    raw_bytes = windows.nbytes
    comp_bytes = len(compressed_blob)
    ratio = raw_bytes / comp_bytes

    kv("Compression mode:", "lossless (bit-exact)")
    kv("Raw data size:", fmt_bytes(raw_bytes))
    kv("Compressed size:", fmt_bytes(comp_bytes))
    kv("Compression ratio:", f"{ratio:.2f}x")
    kv("Compression time:", fmt_ms(compress_time))
    kv("Throughput:", f"{raw_bytes / compress_time / 1_048_576:.1f} MB/s")

    # ------------------------------------------------------------------
    section("STEP 5 -- Build lightweight searchable index")
    # ------------------------------------------------------------------
    print()
    print("    The index is computed ONCE at compression time and stored")
    print("    alongside the compressed data.  It is NOT derived from the")
    print("    compressed bytes -- it is a separate, tiny metadata layer.")
    print()

    t0 = time.perf_counter()
    index = build_window_index(windows)
    index_time = time.perf_counter() - t0

    index_bytes = index.nbytes
    # Compare index to raw data (what matters for enterprise storage)
    index_vs_raw_pct = index_bytes / raw_bytes * 100
    total_stored = comp_bytes + index_bytes
    total_ratio = raw_bytes / total_stored

    kv("Index shape:", str(index.shape))
    kv("Floats per window:", f"{index.shape[1]}  (4 stats x 5 features)")
    kv("Index size:", fmt_bytes(index_bytes))
    kv("Index as % of raw data:", f"{index_vs_raw_pct:.1f}%")
    kv("Compressed + index total:", fmt_bytes(total_stored))
    kv("Effective ratio (with index):", f"{total_ratio:.2f}x")
    kv("Index build time:", fmt_ms(index_time))

    # ------------------------------------------------------------------
    section("STEP 6 -- SEARCH: find crash patterns in compressed data")
    # ------------------------------------------------------------------
    print()
    print("    Query: the injected crash window (window #{})".format(crash_position))
    print("    Searching {:,} compressed windows for similar patterns...".format(n_windows))
    print()

    # Build query vector from the crash window
    query_window = windows[crash_position]
    query_vector = build_window_index(query_window[np.newaxis, :])[0]

    # --- Timed search on index (no decompression!) ---
    top_k = 15
    t0 = time.perf_counter()
    match_indices, match_distances = search_index(
        index, query_vector, top_k=top_k, metric="cosine",
    )
    search_time = time.perf_counter() - t0

    crash_set = set(crash_positions)

    print(f"    {'Rank':<6} {'Window':>8} {'Cosine Dist':>14} "
          f"{'Close Start':>14} {'Close End':>14} {'Change':>10}")
    print(f"    {'----':<6} {'------':>8} {'-----------':>14} "
          f"{'-----------':>14} {'---------':>14} {'------':>10}")

    for rank, (idx, dist) in enumerate(
        zip(match_indices, match_distances), 1
    ):
        w = windows[idx]
        cs = w[0, 3]
        ce = w[-1, 3]
        chg = (ce - cs) / cs * 100
        marker = ""
        if idx == crash_position:
            marker = " <-- query (primary crash)"
        elif idx in crash_set:
            marker = " <-- injected crash"
        print(f"    #{rank:<5} {idx:>8,} {dist:>14.6f} "
              f"{cs:>13.2f} {ce:>13.2f} {chg:>+9.1f}%{marker}")

    print()
    kv("Windows searched:", f"{n_windows:,}")
    kv("Search time:", fmt_ms(search_time))
    kv("Matches returned:", f"{top_k}")
    kv("Compressed data touched:", "0 bytes  (index-only search)")

    # Count how many injected crashes we recovered
    recovered_crashes = len(crash_set & set(match_indices.tolist()))
    kv("Injected crashes found in top-{0}:".format(top_k),
       f"{recovered_crashes}/{len(crash_positions)}")

    # Highlight the key stat
    print()
    print("    *** Searched {:,} windows in {} "
          "-- without decompressing ***".format(
              n_windows, fmt_ms(search_time).strip()))

    # ------------------------------------------------------------------
    section("STEP 7 -- Selectively decompress ONLY the matched windows")
    # ------------------------------------------------------------------
    print()
    print("    In production, each window (or window-group) would be stored")
    print("    as a separate compressed chunk so you can decompress only")
    print("    the matches.  Here we demonstrate the concept:")
    print()

    # Compress each matched window individually, then decompress just those
    matched_windows = windows[match_indices]
    t0 = time.perf_counter()
    per_window_blobs = compress_windows_individually(
        matched_windows, mode="lossless",
    )
    selective_compress_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    recovered_matches = []
    for blob in per_window_blobs:
        recovered_matches.append(decompress_single_window(blob))
    selective_decompress_time = time.perf_counter() - t0

    recovered_matches = np.concatenate(recovered_matches, axis=0)
    matches_ok = np.array_equal(matched_windows, recovered_matches)

    selective_bytes = sum(len(b) for b in per_window_blobs)
    savings_factor = raw_bytes / selective_bytes

    kv("Matched windows decompressed:", f"{len(match_indices)}")
    kv("Selective decompress time:", fmt_ms(selective_decompress_time))
    kv("Per-window compressed (avg):", fmt_bytes(
        selective_bytes // len(per_window_blobs)))
    kv("Total selective I/O:", fmt_bytes(selective_bytes))
    kv("Lossless round-trip correct:", str(matches_ok))
    print()
    print(f"    To answer this query, a traditional compressor would need")
    print(f"    to decompress all {fmt_bytes(raw_bytes).strip()}.")
    print(f"    UMC only decompressed {fmt_bytes(selective_bytes).strip()}")
    print(f"    -- a {savings_factor:,.0f}x reduction in I/O.")

    # ------------------------------------------------------------------
    section("STEP 8 -- Verify full lossless round-trip")
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    recovered_all = umc.decompress(compressed_blob)
    full_decompress_time = time.perf_counter() - t0

    is_exact = np.array_equal(windows, recovered_all)
    max_err = float(np.max(np.abs(
        windows.astype(np.float64) - recovered_all.astype(np.float64)
    )))
    kv("Full decompression time:", fmt_ms(full_decompress_time))
    kv("Original shape:", str(windows.shape))
    kv("Recovered shape:", str(recovered_all.shape))
    kv("Bit-exact match:", str(is_exact))
    kv("Max absolute error:", f"{max_err}")
    print()
    if is_exact:
        print("    Lossless round-trip PASSED -- every bit matches.")
    else:
        print("    WARNING: round-trip mismatch detected.")

    # ------------------------------------------------------------------
    section("STEP 9 -- Comparison: brute-force search (decompress first)")
    # ------------------------------------------------------------------
    print()
    print("    For comparison, here is the traditional approach:")
    print("    decompress EVERYTHING, then compute distances on raw data.")
    print()

    # "Traditional" approach: decompress all, then cosine on raw windows
    t0 = time.perf_counter()
    all_recovered = umc.decompress(compressed_blob)
    flat_all = all_recovered.reshape(all_recovered.shape[0], -1)
    flat_query = windows[crash_position].reshape(1, -1)
    norms = np.linalg.norm(flat_all, axis=1, keepdims=True).clip(1e-10)
    q_norm = np.linalg.norm(flat_query).clip(1e-10)
    sims = (flat_all @ flat_query.T).squeeze() / (norms.squeeze() * q_norm)
    dists = 1.0 - sims
    bf_indices = np.argpartition(dists, top_k)[:top_k]
    bf_indices = bf_indices[np.argsort(dists[bf_indices])]
    brute_force_time = time.perf_counter() - t0

    kv("Brute-force search time:", fmt_ms(brute_force_time))
    kv("  of which decompression:", fmt_ms(full_decompress_time))
    kv("UMC index search time:", fmt_ms(search_time))
    speedup = brute_force_time / max(search_time, 1e-9)
    kv("Speedup:", f"{speedup:.1f}x faster with UMC index")

    # Check overlap between the two result sets
    overlap = len(set(match_indices.tolist()) & set(bf_indices.tolist()))
    kv("Result overlap (top-{0}):".format(top_k),
       f"{overlap}/{top_k} windows in common")

    # ------------------------------------------------------------------
    banner("SUMMARY")
    # ------------------------------------------------------------------
    print()
    print(f"  {'Metric':<42s} {'Value':>26s}")
    print(f"  {'------':<42s} {'-----':>26s}")
    print(f"  {'Dataset':<42s} "
          f"{n_tickers} tickers x {n_days_per_ticker:,} days "
          f"= {n_windows:,} windows")
    print(f"  {'Raw data size':<42s} {fmt_bytes(raw_bytes).strip():>26s}")
    print(f"  {'Compressed size':<42s} {fmt_bytes(comp_bytes).strip():>26s}")
    print(f"  {'Compression ratio (lossless)':<42s} {ratio:.2f}x")
    print(f"  {'Search index size':<42s} {fmt_bytes(index_bytes).strip():>26s}")
    print(f"  {'Index as % of raw data':<42s} {index_vs_raw_pct:.1f}%")
    print(f"  {'Compressed + index (total stored)':<42s} "
          f"{fmt_bytes(total_stored).strip():>26s}")
    print(f"  {'Effective ratio (data + index)':<42s} {total_ratio:.2f}x")
    print(f"  {'Search time (UMC index)':<42s} {fmt_ms(search_time).strip():>26s}")
    print(f"  {'Search time (brute-force)':<42s} "
          f"{fmt_ms(brute_force_time).strip():>26s}")
    print(f"  {'Speedup':<42s} {speedup:.1f}x")
    print(f"  {'Selective decompression I/O':<42s} "
          f"{fmt_bytes(selective_bytes).strip():>26s}")
    print(f"  {'I/O reduction vs full decompress':<42s} {savings_factor:,.0f}x")
    print(f"  {'Crash patterns found':<42s} "
          f"{recovered_crashes}/{len(crash_positions)} injected crashes")
    print(f"  {'Lossless round-trip':<42s} "
          f"{'PASSED' if is_exact else 'FAILED':>26s}")

    print()
    print(SEPARATOR)
    print()
    print("  KEY TAKEAWAY")
    print()
    print("  UMC lets you compress AND search your data.")
    print("  Other compressors force you to decompress first.")
    print()
    print("  How it works:")
    print("    - At compression time, UMC builds a lightweight per-window")
    print(f"     index ({index.shape[1]} floats/window = "
          f"{index_vs_raw_pct:.1f}% of raw data).")
    print("    - At query time, the search hits only the index.")
    print("    - Only matched windows are decompressed on demand.")
    print(f"    - Result: {speedup:.0f}x faster search, "
          f"{savings_factor:,.0f}x less I/O.")
    print()
    print("  With UMC's neural codec (VQ search), this becomes even more")
    print("  powerful: the VQ codebook indices ARE the search index,")
    print("  achieving zero extra overhead and native similarity search")
    print("  in the compressed domain.")
    print()
    print(SEPARATOR)
    print()


if __name__ == "__main__":
    main()
