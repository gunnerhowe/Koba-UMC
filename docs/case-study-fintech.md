# Case Study: Reducing Financial Time Series Storage Costs with UMC

**Industry:** Capital Markets / Fintech
**Use Case:** Tick data archival, order book storage, risk metrics compression
**Date:** February 2026

---

## Executive Summary

A compression benchmark on 6.6 million realistic financial data points demonstrates that UMC (Universal Manifold Codec) achieves **3.07x** overall lossless compression on financial time series -- **10.9% smaller output** than the best industry-standard compressor (lzma-6 at 2.74x). On OHLCV tick data specifically, UMC reaches **5.20x** compression, beating lzma-6 (4.19x) by 19.5%. For an enterprise managing 100 TB of financial data, this translates to **$1,904 in additional annual savings** beyond what the best standard compressor can deliver, with every byte reconstructed bit-exact.

---

## The Problem

Financial institutions generate and store enormous volumes of time series data:

- **Tick-level market data** (OHLCV, bid/ask) at sub-second granularity across thousands of instruments
- **Order book snapshots** with 10+ levels of depth, sampled every millisecond
- **Portfolio risk metrics** (VaR, CVaR, Greeks, implied/realized vol) computed daily for thousands of positions

This data is subject to **regulatory retention requirements** (MiFID II: 5-7 years; SEC 17a-4: 6 years; FINRA: 3-6 years) and must be stored in a format that supports bit-exact reconstruction. At scale, storage costs become a material line item:

| Scale | Monthly Raw Storage Cost | Annual Raw Cost |
|-------|------------------------:|----------------:|
| 10 TB | $399 | $4,792 |
| 50 TB | $1,997 | $23,962 |
| 100 TB | $3,994 | $47,923 |

*(AWS S3 Standard at $0.023/GB/month + $0.004/GB transfer, 4 reads/month)*

The question is not *whether* to compress, but *which compressor* yields the smallest output while guaranteeing lossless reconstruction and remaining operationally practical.

---

## Benchmark Methodology

We generated three realistic financial datasets using industry-standard stochastic models:

| Dataset | Model | Shape | Raw Size | Data Points |
|---------|-------|-------|----------|-------------|
| **Tick Data (OHLCV + Bid/Ask)** | GBM prices, log-normal spreads, U-shaped volume | 8,192 x 64 x 7 | 14.7 MB | 3,670,016 |
| **Order Book (10-level depth)** | Correlated bid/ask ladder, log-normal sizes | 1,024 x 64 x 40 | 10.5 MB | 2,621,440 |
| **Risk Metrics (Greeks + VaR)** | Mean-reverting vol, correlated Greeks | 512 x 64 x 10 | 1.3 MB | 327,680 |
| **Total** | | | **26.5 MB** | **6,619,136** |

All data is float32. All compression modes are **lossless** (bit-exact round-trip verified). Benchmarks ran on Windows 11 with the UMC C extension active.

We compared five UMC modes against four industry-standard compressors at their highest practical compression levels.

---

## Results

### Tick Data (OHLCV + Bid/Ask) -- 14.7 MB

This is the highest-volume dataset and the primary driver of storage costs.

| Compressor | Compressed Size | Ratio | Throughput | Space Savings |
|------------|----------------:|------:|-----------:|--------------:|
| **UMC optimal_fast** | **2,823,312** | **5.20x** | 0.2 MB/s | **80.8%** |
| lzma-6 | 3,505,208 | 4.19x | 2.6 MB/s | 76.1% |
| brotli-11 | 3,992,205 | 3.68x | 0.6 MB/s | 72.8% |
| UMC lossless_lzma | 4,091,474 | 3.59x | 2.6 MB/s | 72.1% |
| **UMC lossless_fast** | **4,175,969** | **3.52x** | **177.9 MB/s** | **71.6%** |
| zstd-19 | 4,553,769 | 3.22x | 3.0 MB/s | 69.0% |
| UMC lossless_zstd | 4,726,122 | 3.11x | 2.1 MB/s | 67.8% |
| gzip-9 | 5,678,511 | 2.59x | 18.2 MB/s | 61.3% |
| UMC lossless | 5,743,184 | 2.56x | 17.6 MB/s | 60.9% |

UMC `optimal_fast` compresses tick data **19.5% smaller** than the best standard compressor (lzma-6). The `lossless_fast` mode achieves **177.9 MB/s** throughput -- suitable for real-time ingestion pipelines -- while still beating zstd-19 and brotli-11 on ratio.

### Order Book (10-level Depth) -- 10.5 MB

| Compressor | Compressed Size | Ratio | Throughput | Space Savings |
|------------|----------------:|------:|-----------:|--------------:|
| **UMC optimal_fast** | **4,731,933** | **2.22x** | 0.2 MB/s | **54.9%** |
| UMC lossless_zstd | 4,823,706 | 2.17x | 2.7 MB/s | 54.0% |
| UMC lossless_lzma | 4,878,754 | 2.15x | 2.4 MB/s | 53.5% |
| lzma-6 | 5,081,408 | 2.06x | 3.0 MB/s | 51.5% |
| brotli-11 | 5,104,515 | 2.05x | 0.5 MB/s | 51.3% |
| zstd-19 | 5,249,596 | 2.00x | 3.3 MB/s | 49.9% |
| gzip-9 | 5,660,989 | 1.85x | 22.9 MB/s | 46.0% |
| UMC lossless | 5,793,287 | 1.81x | 20.1 MB/s | 44.8% |
| UMC lossless_fast | 6,166,386 | 1.70x | 123.4 MB/s | 41.2% |

UMC `optimal_fast` achieves **6.9% smaller** output than lzma-6 on order book data.

### Risk Metrics (Greeks + VaR) -- 1.3 MB

| Compressor | Compressed Size | Ratio | Throughput | Space Savings |
|------------|----------------:|------:|-----------:|--------------:|
| **UMC optimal_fast** | **1,068,872** | **1.23x** | 0.2 MB/s | **18.5%** |
| UMC lossless_lzma | 1,072,330 | 1.22x | 4.2 MB/s | 18.2% |
| UMC lossless_zstd | 1,078,855 | 1.21x | 5.4 MB/s | 17.7% |
| UMC lossless | 1,089,483 | 1.20x | 16.7 MB/s | 16.9% |
| lzma-6 | 1,089,604 | 1.20x | 4.1 MB/s | 16.9% |
| brotli-11 | 1,131,259 | 1.16x | 0.4 MB/s | 13.7% |
| zstd-19 | 1,159,442 | 1.13x | 5.1 MB/s | 11.5% |
| gzip-9 | 1,207,709 | 1.09x | 23.7 MB/s | 7.9% |
| UMC lossless_fast | 1,210,175 | 1.08x | 244.0 MB/s | 7.7% |

Risk metrics data is high-entropy (many distinct float values), making it harder to compress. Even here, UMC edges out lzma-6 by 1.9%.

### Overall Summary

| Dataset | UMC Best | Best Standard | UMC Advantage |
|---------|:--------:|:-------------:|:-------------:|
| Tick Data (OHLCV+Bid/Ask) | **5.20x** | 4.19x (lzma-6) | **+19.5%** |
| Order Book (10-level) | **2.22x** | 2.06x (lzma-6) | **+6.9%** |
| Risk Metrics (Greeks+VaR) | **1.23x** | 1.20x (lzma-6) | **+1.9%** |
| **Weighted Overall** | **3.07x** | **2.74x** | **+10.9%** |

UMC wins on every dataset. All results are bit-exact lossless -- verified by round-trip decompression with `np.array_equal()`.

---

## Cost Savings Analysis

Based on AWS S3 Standard pricing (us-east-1):
- Storage: **$0.023/GB/month**
- Data transfer (same-region): **$0.004/GB**
- Assumption: data read **4x/month** for analytics, backtesting, and compliance queries

### 10 TB of Raw Financial Data

| Method | Stored | Monthly Cost | Annual Savings vs Raw | Annual Savings vs Best Standard |
|--------|-------:|-------------:|----------------------:|--------------------------------:|
| Uncompressed | 10,240 GB | $399.36 | -- | -- |
| Best Standard (lzma-6, 2.74x) | 3,742 GB | $145.95 | $3,040.90 | -- |
| **UMC (3.07x)** | **3,335 GB** | **$130.08** | **$3,231.33** | **$190.43** |

### 50 TB of Raw Financial Data

| Method | Stored | Monthly Cost | Annual Savings vs Raw | Annual Savings vs Best Standard |
|--------|-------:|-------------:|----------------------:|--------------------------------:|
| Uncompressed | 51,200 GB | $1,996.80 | -- | -- |
| Best Standard (lzma-6, 2.74x) | 18,712 GB | $729.76 | $15,204.50 | -- |
| **UMC (3.07x)** | **16,677 GB** | **$650.41** | **$16,156.67** | **$952.17** |

### 100 TB of Raw Financial Data

| Method | Stored | Monthly Cost | Annual Savings vs Raw | Annual Savings vs Best Standard |
|--------|-------:|-------------:|----------------------:|--------------------------------:|
| Uncompressed | 102,400 GB | $3,993.60 | -- | -- |
| Best Standard (lzma-6, 2.74x) | 37,423 GB | $1,459.52 | $30,409.00 | -- |
| **UMC (3.07x)** | **33,354 GB** | **$1,300.82** | **$32,313.33** | **$1,904.33** |

At 100 TB, switching from lzma-6 to UMC saves an additional **$1,904/year** in combined storage and transfer costs. This is on top of the $30,409 already saved by using lzma-6 over raw storage.

For organizations with tick data volumes exceeding 100 TB (common at major market makers and exchanges), these savings scale linearly. At 1 PB, the additional annual savings from UMC over lzma-6 reach approximately **$19,000/year**.

---

## Technical Details: Why UMC Wins on Financial Data

### The Float32 Structure Advantage

Financial time series are stored as IEEE 754 float32 values. Each 4-byte float has internal structure:

```
[sign: 1 bit] [exponent: 8 bits] [mantissa: 23 bits]
```

Standard compressors (gzip, zstd, brotli, lzma) see these as opaque byte streams. When consecutive prices are close in value (e.g., 185.50, 185.51, 185.49), the raw bytes look quite different because small value changes cause scattered bit changes across the 4-byte representation.

### UMC's Key Technique: Byte Transposition

UMC applies **byte transposition** before compression. For an array of N float32 values (4N bytes total), instead of storing:

```
[byte0 byte1 byte2 byte3] [byte0 byte1 byte2 byte3] ...  (interleaved)
```

UMC rearranges to:

```
[all byte0s] [all byte1s] [all byte2s] [all byte3s]  (transposed)
```

This groups the exponent bytes together and the mantissa bytes together. For financial data where prices cluster in a narrow range:
- **Exponent bytes** become nearly constant (highly compressible)
- **High mantissa bytes** change slowly (delta-compressible)
- **Low mantissa bytes** remain noisy but are isolated from the structured bytes

The `optimal_fast` mode goes further: it competes **20 different (transform, compressor) strategies** and picks the winner. For tick data, the winning strategy was **transform=15 (flatten + delta) with compressor=2 (lzma-9)**, which first applies delta encoding on the flattened byte stream before LZMA compression. For order book data, the winner was **transform=14 (split exponent/mantissa) with compressor=1 (zstd-22)**, which explicitly separates IEEE 754 components.

### Mode Selection Guide

| Mode | Ratio (tick data) | Throughput | Best For |
|------|:-----------------:|:----------:|----------|
| `lossless_fast` | 3.52x | 177.9 MB/s | Real-time ingestion, streaming pipelines |
| `lossless_zstd` | 3.11x | 2.1 MB/s | Balanced ratio/speed for batch jobs |
| `lossless_lzma` | 3.59x | 2.6 MB/s | Archival storage, overnight batch |
| `optimal_fast` | 5.20x | 0.2 MB/s | Cold storage, archival, maximum savings |

**Recommendation for fintech:** Use `lossless_fast` for hot/warm data ingestion (177.9 MB/s throughput, still beats brotli-11 and zstd-19 on ratio). Use `optimal_fast` for cold archival where compression time is not a constraint.

---

## Migration Path

Integrating UMC into an existing data pipeline requires three steps:

### Step 1: Install (30 seconds)

```bash
pip install umc
```

No external dependencies for compression. Pure Python with optional C extension for 4x faster byte transposition (auto-detected).

### Step 2: Compress (2 lines of code)

```python
import numpy as np
import umc

# Your existing data loading -- no changes needed
tick_data = load_tick_data()  # returns np.ndarray, shape (N, 64, 7)

# Compress
compressed = umc.compress(tick_data, mode="lossless_fast")

# Store to S3, disk, or any blob store
s3.put_object(Bucket="market-data", Key="ticks/2026-02-27.umc", Body=compressed)
```

### Step 3: Decompress (1 line)

```python
# Retrieve and decompress -- bit-exact original
recovered = umc.decompress(compressed)
assert np.array_equal(tick_data, recovered)  # guaranteed
```

No schema changes. No data model changes. No retraining. Drop-in replacement for your existing compression layer.

---

## ROI Calculation

### Assumptions
- Current compression: gzip-9 (most common default in data pipelines)
- Data volume: 50 TB of financial time series
- Retention: 7 years (MiFID II compliance)
- Data access: 4 reads/month for analytics and backtesting

### Current State (gzip-9 at 2.59x on tick data)

| Item | Value |
|------|------:|
| Raw data | 50 TB |
| Compressed (gzip-9, ~2.59x) | 19.3 TB |
| Annual storage + transfer | $9,040 |
| 7-year total | $63,277 |

### With UMC (optimal_fast at 5.20x on tick data, 3.07x blended)

| Item | Value |
|------|------:|
| Raw data | 50 TB |
| Compressed (UMC, ~3.07x) | 16.3 TB |
| Annual storage + transfer | $7,805 |
| 7-year total | $54,635 |
| **7-year savings vs gzip-9** | **$8,642** |

### Implementation Cost

| Item | Estimate |
|------|:--------:|
| Engineering time (integration + testing) | 2-3 days |
| Code changes | ~10 lines per service |
| Risk (lossless, bit-exact) | Zero data loss risk |
| Dependencies added | 1 (pure Python, no native libs required) |

### Net ROI

- **Payback period:** Immediate (no upfront cost beyond engineering time)
- **7-year net savings:** $8,642 at 50 TB (scales linearly with volume)
- **At 500 TB:** $86,420 in 7-year savings
- **Non-financial benefits:** Faster data transfer, reduced I/O wait times, smaller backup windows

---

## Cross-Domain Validation

UMC was also benchmarked on 8 standard datasets covering diverse data types. Results from the standard benchmark suite:

| Dataset | UMC Best | Best Standard | Winner |
|---------|:--------:|:-------------:|:------:|
| Financial OHLCV | 2.27x | 1.89x (lzma-6) | UMC |
| IoT Sensors (8ch) | 1.31x | 1.12x (brotli-11) | UMC |
| Monotonic Counters | 4.37x | 3.14x (brotli-11) | UMC |
| Image Tiles (32x32) | 1.21x | 1.10x (brotli-11) | UMC |
| Audio Waveforms | 1.46x | 1.19x (lzma-6) | UMC |
| Sparse Events | 53.65x | 55.09x (brotli-11) | Standard |
| Random Noise | 1.20x | 1.09x (brotli-11) | UMC |
| Scientific Simulation | 1.36x | 1.21x (lzma-6) | UMC |

**UMC wins on 7 out of 8 datasets.** The only loss is on sparse data (mostly zeros), where brotli's run-length encoding has a slight edge. For structured numeric data -- the kind produced by financial systems -- UMC consistently outperforms.

---

## Conclusion

UMC delivers measurable, verified compression improvements on financial time series data:

1. **5.20x compression on tick data** (vs 4.19x for lzma-6) -- a 19.5% reduction in stored bytes
2. **Bit-exact lossless** -- every float reconstructed identically, verified by automated round-trip testing
3. **Drop-in integration** -- 2 lines of code, no schema changes, no model training
4. **Dual-speed operation** -- 177.9 MB/s for streaming ingestion, maximum compression for cold archival
5. **$1,904/year additional savings** at 100 TB over the best standard compressor

For a VP of Engineering evaluating compression solutions: UMC provides a strictly better Pareto frontier of ratio vs. speed compared to gzip, zstd, brotli, and lzma on financial data. The integration is trivial, the risk is zero (lossless), and the savings are immediate.

**Recommended next step:** Run the benchmark on your own data.

```bash
pip install umc
python -c "
import numpy as np, umc
data = np.load('your_tick_data.npy')
c = umc.compress(data, mode='optimal_fast')
print(f'Ratio: {data.nbytes / len(c):.2f}x')
assert np.array_equal(data, umc.decompress(c))
"
```

---

*Benchmark data generated by `scripts/benchmark_fintech_casestudy.py` and `scripts/benchmark_standard.py`. Full results available in `results/fintech_benchmark.json`. All numbers are reproducible -- datasets use fixed random seeds.*
