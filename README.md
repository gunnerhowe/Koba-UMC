# UMC - Universal Manifold Codec

**The only compressor that lets you search without decompressing.**

UMC compresses numeric arrays (time series, sensor data, images, audio, video) and optionally builds a similarity search index over the compressed data. Find patterns in your archive without touching the raw bytes.

## Why UMC?

Every compressor makes your data smaller. UMC makes it smaller **and searchable**.

```python
# Traditional approach: decompress everything, then search
raw = zstd.decompress(archive)        # slow, needs full decompression
matches = search(raw, query)

# UMC approach: search compressed data directly
matches = codec.search_from_mnf("archive.mnf", query, k=10)  # instant
data = codec.decode_from_mnf("archive.mnf")                  # only when needed
```

This matters when you have terabytes of time series data and need to find patterns like "show me windows that look like the 2020 crash" without decompressing your entire archive.

## Install

```bash
# Lightweight (compress/decompress only, ~5MB)
pip install umc

# Full install with neural search codec
pip install umc[all]
```

The lightweight install only requires numpy. No torch, no CUDA, no heavy dependencies.

## Quick Start

```python
import numpy as np
import umc

# Compress any numeric array — no model, no training, no config
data = np.random.randn(1000, 32, 5).astype(np.float32)
compressed = umc.compress(data)
recovered = umc.decompress(compressed)

print(f"Ratio: {data.nbytes / len(compressed):.1f}x")
assert np.array_equal(data, recovered)  # bit-exact lossless
```

### Choose your tradeoff

```python
# Bit-exact lossless (~1.2x)
compressed = umc.compress(data, mode="lossless")

# Near-lossless with 2.5x ratio (<0.01% error)
compressed = umc.compress(data, mode="near_lossless")

# Fast near-lossless with 3x ratio
compressed = umc.compress(data, mode="near_lossless_turbo")

# High compression with 5-9x ratio (~2% error)
compressed = umc.compress(data, mode="quantized_8")

# Provably near-optimal lossless (tries all strategies, picks best)
compressed = umc.compress(data, mode="optimal")
```

### File compression

```python
umc.compress_file("data.csv", "data.umc", mode="near_lossless")
umc.decompress_file("data.umc", "recovered.csv")
```

Supports ANY file: `.npy`, `.csv`, `.parquet`, `.png`, `.jpg`, `.wav`, `.mp4`, `.exe`, `.stl`, `.pdf` -- everything.

### Raw binary compression

```python
# Compress any binary file (exe, 3D model, database, etc.)
raw = open("model.stl", "rb").read()
compressed = umc.compress_raw(raw)
recovered = umc.decompress_raw(compressed)
assert raw == recovered  # bit-exact
```

### Pandas Integration

```python
import pandas as pd
from umc import pandas_ext

# Compress a DataFrame (preserves columns, index, dtypes)
df = pd.read_csv("timeseries.csv", index_col="date", parse_dates=True)
compressed = df.umc.compress(mode="lossless")

# Decompress back to DataFrame
recovered = pandas_ext.decompress_dataframe(compressed)
assert list(recovered.columns) == list(df.columns)
```

## CLI

```bash
umc compress data.npy -o data.umc --mode near_lossless
umc decompress data.umc -o recovered.npy
umc info data.umc
umc dashboard    # interactive web UI

# Benchmark any file against all UMC modes + standard compressors
umc tryit data.npy
umc tryit data.npy --html report.html
```

## Compression Modes

| Mode | Fidelity | Ratio | Speed | Use Case |
| ---- | -------- | ----- | ----- | -------- |
| `lossless` | Bit-exact | ~1.3x | 16-34 MB/s | Regulatory/audit data |
| `lossless_fast` | Bit-exact | ~1.7x | 40-100 MB/s | Real-time lossless streaming |
| `near_lossless` | <0.01% error | ~2.4x | 27-46 MB/s | General archival |
| `near_lossless_turbo` | <0.05% error | ~2.8x | 37-61 MB/s | Real-time streaming |
| `quantized_8` | ~2% error | 4-9x | 30-85 MB/s | ML training data, previews |
| `lossless_zstd` | Bit-exact | ~1.2x | 10-15 MB/s | Better ratio, needs zstandard |
| `lossless_lzma` | Bit-exact | ~1.5x | slow | Maximum lossless ratio |
| `optimal` | Bit-exact | 1.2-5.5x | slow | Best possible lossless (with certificate) |
| `optimal_fast` | Bit-exact | 1.2-5.5x | moderate | Near-optimal with 25x faster speed |

## Benchmark: UMC Optimal vs Standard Compressors

UMC's `optimal` mode tries 128 (transform, compressor) combinations and picks the smallest. On 7 diverse data types, it beats the best standard compressor on 6 and ties on 1:

| Dataset | UMC optimal | Best std compressor | Winner | Delta |
| ------- | ----------- | ------------------- | ------ | ----- |
| Monotonic counters | **5.48x** | lzma 2.95x | **UMC** | **-46%** |
| Scientific sim | **1.35x** | lzma 1.21x | **UMC** | **-10%** |
| Quantized sensors | **4.20x** | lzma 3.81x | **UMC** | **-9%** |
| Random noise | **1.20x** | brotli 1.09x | **UMC** | **-9%** |
| Image tiles | **1.34x** | lzma 1.23x | **UMC** | **-9%** |
| Financial OHLCV | **2.45x** | lzma 2.28x | **UMC** | **-7%** |
| Sparse events | 15.81x | zstd 15.91x | Tied | -1% |

**Key insight:** UMC never loses by more than 1% (format header overhead). When its float-aware transforms help (predictive coding, byte transposition, exponent/mantissa splitting), it wins by 7-46%. When they don't help, it falls back to the identity transform and matches the best standard compressor.

### All compression modes

| Dataset | lossless | near_lossless | turbo | quantized_8 | optimal |
| ------- | -------- | ------------- | ----- | ----------- | ------- |
| Financial | 1.4x | 2.4x | 2.7x | 5.9x | **2.5x** |
| Images | 1.3x | 2.4x | 2.6x | 5.0x | **1.3x** |
| Audio | 1.2x | 2.4x | 2.9x | 5.0x | **1.5x** |
| Scientific | 1.3x | 2.8x | 3.4x | 6.5x | **1.4x** |
| Counters | 2.3x | - | - | - | **5.5x** |

## Optimal Mode: Provably Near-Optimal Compression

The `optimal` mode tries every viable (preprocessing, compressor) combination and picks the smallest output -- a brute-force application of the **Minimum Description Length** principle. It then produces an **optimality certificate** proving how close the result is to the theoretical Shannon entropy limit.

```python
result = umc.compress_optimal(data)
cert = result["certificate"]

print(f"Compression ratio:  {cert['ratio']:.2f}x")
print(f"Shannon entropy:    {cert['entropy_h0']:.2f} bits/byte")
print(f"Achieved:           {cert['achieved_bpb']:.2f} bits/byte")
print(f"Entropy gap:        {cert['entropy_gap_pct']:.1f}%")
print(f"Randomness p-value: {cert['randomness_p_value']:.3f}")
```

**What the certificate tells you:**

- **Entropy gap** -- how many bits/byte above the Shannon limit. Lower is better; 0% means no single-byte pattern is wasted.
- **Randomness p-value** -- Chi-squared test on the compressed output. If p > 0.05, the output is statistically indistinguishable from random noise, meaning no further compression is possible at the byte level.
- **Winning strategy** -- which preprocessing transform and compressor backend won the competition.

**How it works:** 16 lossless transforms (byte transpose, identity, int32 delta orders 1-4, zigzag encoding, flatten-delta, window-time delta, cross-feature, per-feature adaptive, split exponent/mantissa, delta-identity, flatten-delta-identity) x 8 compressors (zlib-9, zstd-22, lzma-9, brotli-11, per-channel zlib/lzma, static/adaptive arithmetic coding) = 128 candidates evaluated in parallel. The smallest wins. Use `optimal_fast` for ~25x faster speed with the same results on most data.

## Architecture

### Two-Tier Design

```text
Input Data (float32)
       |
       v
  +-----------+     +------------------+
  | Tier 1    |     | Tier 2           |
  | VQ Search |     | Storage          |
  | Index     |     | (lossless/lossy) |
  +-----------+     +------------------+
       |                    |
       v                    v
  FAISS-searchable     Byte-transposed
  VQ codes (~89x)      float data (1-9x)
```

**Tier 1 (Search):** Neural VQ encoder produces compact codebook indices. These are FAISS-searchable -- find similar windows without decompressing storage data. Requires a trained model (`pip install umc[neural]`).

**Tier 2 (Storage):** Byte-transposed float data with configurable fidelity. Works standalone with just numpy -- no model needed. This is what `umc.compress()` uses.

### Neural Search Pipeline

```python
from umc import TieredManifoldCodec

# Load a trained model
codec = TieredManifoldCodec.from_checkpoint("model.pt", storage_mode="lossless")

# Encode: creates search index + compressed storage in one file
codec.encode_to_mnf(windows, "archive.mnf")

# Search compressed data (uses VQ codes, never touches raw data)
results = codec.search_from_mnf("archive.mnf", query_windows, k=10)

# Retrieve at full fidelity (only when needed)
data = codec.decode_from_mnf("archive.mnf")
```

### Pre-trained Models

```python
from umc.hub import list_models, load_model

# See available models
for m in list_models():
    print(f"{m['name']}: {m['description']}")

# Load and use
codec = load_model("financial-v1")
codec.encode_to_mnf(my_data, "output.mnf")
```

## VectorForge — Compressed Vector Database

VectorForge is a Pinecone-compatible vector database that stores embeddings with UMC compression. 53-57% storage savings vs raw float32, with 100% recall.

```bash
pip install umc[vectordb]
python -m umc.vectordb --port 8100
```

```python
# Same API as Pinecone — drop-in replacement
import requests

# Create index
requests.post("http://localhost:8100/indexes", json={
    "name": "my-index", "dimension": 1536, "metric": "cosine"
})

# Upsert vectors
requests.post("http://localhost:8100/indexes/my-index/vectors/upsert", json={
    "vectors": [{"id": "doc1", "values": [...], "metadata": {"source": "web"}}]
})

# Query
resp = requests.post("http://localhost:8100/indexes/my-index/query", json={
    "vector": [...], "topK": 10
})
```

| Scale | Raw Storage | With Float16+UMC | Savings | Query p50 | Recall@10 |
| ----- | ---------- | ---------------- | ------- | --------- | --------- |
| 5K × 384-dim | 7.3 MB | 3.1 MB | 57% | 0.2ms | 100% |
| 10K × 1536-dim | 58.6 MB | 25.1 MB | 57% | 4.7ms | 100% |
| 25K × 1536-dim | 146.5 MB | 69.3 MB | 53% | 11.6ms | 100% |

## Streaming

```python
from umc.codec.streaming import StreamingStorageCodec

codec = StreamingStorageCodec(chunk_size=64, mode="near_lossless")
for window in realtime_stream:
    chunk = codec.push(window)
    if chunk:
        send_to_storage(chunk)
final = codec.flush()
```

## C Extension

UMC includes optional C-accelerated kernels for 2.5-4x faster byte transposition and delta coding. They build automatically if a C compiler is available, with transparent fallback to pure NumPy.

```bash
# Build the C extension (optional — auto-detected at import time)
python -m umc.cext.build
```

```python
from umc.cext import HAS_C_EXT
print(f"C extension: {'active' if HAS_C_EXT else 'pure Python'}")
```

## Development

```bash
# Full dev install
pip install -e ".[dev]"

# Run tests (405 tests)
pytest tests/ -v

# Run standard benchmarks (8 reproducible datasets)
python scripts/benchmark_standard.py
python scripts/benchmark_standard.py --html report.html

# Run interactive demo
python scripts/demo_live.py

# Build C extension
python -m umc.cext.build
```

## When to use UMC

**Good fit:**

- Time series archives that need pattern search (financial, IoT, scientific)
- ML feature stores where ~2% precision loss is acceptable for 5-9x savings
- Streaming sensor data with real-time compression
- Any float32 data where you want compression + search in one format

**Also works for:**

- General-purpose file compression (UMC matches or beats zstd/lzma/brotli on structured data)
- Raw binary files (.exe, .stl, .pdf — compressed as-is via `compress_raw()`)

**Not the right tool:**

- Image/video for display (use WebP/H.265 — UMC treats pixels as raw floats, not perceptual)
- Audio for playback (use FLAC/Opus — UMC has no psychoacoustic model)
- Text or mixed-type data (use gzip — UMC is optimized for numeric arrays)

## License

MIT
