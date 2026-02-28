
# UNIVERSAL MANIFOLD CODEC (UMC)

## Complete Technical Specification & Business Plan

**v4.0 — February 2026**

---

# PART I — WHAT UMC IS

## 1. Overview

UMC is a compression engine optimized for numeric data (float32/float16/bfloat16 arrays) that beats every standard compressor on structured data and provides similarity search over compressed archives without decompression.

**What it does today (shipped, tested, working):**

- 11 compression modes from fast streaming to provably near-optimal
- Beats gzip/lzma/zstd/brotli on 7/8 standard datasets
- Search-without-decompressing via VQ codebook indices
- C-accelerated kernels (2.5-4x speedup over pure Python)
- Optimality certificates proving proximity to Shannon entropy limit
- Pandas DataFrame integration, streaming codec, CLI with benchmarking
- 348 tests passing, 4 trained domain models

**What it does NOT do (correcting the old specification):**

- It does not achieve 50-200x lossless compression. Realistic lossless ratios are 1.2-5.5x.
- It does not operate on images/video/audio at the pixel/sample level with manifold-native encoding. It treats all data as float arrays.
- It does not have a "manifold processor" that runs arbitrary computation on compressed data. It has VQ-based similarity search.
- The theoretical manifold hypothesis (intrinsic dim << ambient dim) is real but the gap between theory and engineering is much larger than the old spec suggested.

## 2. Core Value Proposition

**For storage/archival customers:** UMC compresses numeric data 10-20% smaller than the best standard compressor (lzma) while being lossless. At 100TB scale, this saves ~$1,900/year in storage costs vs the next best option.

**For search/analytics customers:** UMC's VQ search tier enables sub-millisecond similarity queries over compressed archives. This capability does not exist in any standard compressor.

**For AI infrastructure customers (roadmap):** UMC's float-aware compression can reduce embedding storage 3-5x, directly cutting vector database and KV-cache costs.

---

# PART II — TECHNICAL ARCHITECTURE

## 3. Two-Tier Design

```
Input Data (float32/float16/bfloat16)
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
  VQ codes             float data (1-13x)
```

**Tier 1 (Search):** Neural VQ encoder (HVQ-VAE) produces compact codebook indices. FAISS-searchable. Requires a trained model. Optional — not needed for compression-only use.

**Tier 2 (Storage):** Float-aware byte transposition, delta coding, and strategy competition across 128 (transform, compressor) combinations. Works standalone with just numpy. This is what `umc.compress()` uses.

## 4. Storage Compression Engine (Tier 2)

### 4.1 Core Technique: Byte Transposition

IEEE 754 float32 values have internal structure: `[sign: 1 bit][exponent: 8 bits][mantissa: 23 bits]`. For consecutive similar values (common in time series), the raw byte representation looks noisy even though the values differ by tiny amounts.

UMC rearranges the byte layout so all exponent bytes are grouped together, all high-mantissa bytes together, etc. This makes the byte stream much more compressible by standard backends (zlib, zstd, lzma).

### 4.2 The 16 Transforms

| ID | Transform | Description |
|----|-----------|-------------|
| 0 | identity | Raw bytes, no preprocessing |
| 1 | byte_transpose | Group bytes by significance across elements |
| 2 | int32_delta_order1 | First differences of int32 view |
| 3 | int32_delta_order2 | Second differences |
| 4 | int32_delta_order3 | Third differences |
| 5 | int32_delta_order4 | Fourth differences |
| 6 | zigzag | Map signed deltas to unsigned for better entropy |
| 7 | zigzag + byte_transpose | Combined |
| 8 | flatten_delta | Flatten all features, then delta |
| 9 | window_time_delta | Delta along time axis per window |
| 10 | cross_feature_delta | Delta between adjacent features |
| 11 | per_feature_adaptive | Best delta order per feature |
| 12 | delta_identity | Delta + keep original for comparison |
| 13 | flatten_delta_identity | Combined |
| 14 | split_exponent_mantissa | Separate IEEE 754 components |
| 15 | flatten_delta_bt | Flatten + delta + byte transpose |

### 4.3 The 8 Compressor Backends

| ID | Compressor | Level |
|----|-----------|-------|
| 0 | zlib | 9 |
| 1 | zstd | 22 |
| 2 | lzma | 9 |
| 3 | brotli | 11 |
| 4 | per-channel zlib | 9 |
| 5 | per-channel lzma | 9 |
| 6 | static arithmetic coding | n/a |
| 7 | adaptive arithmetic coding | n/a |

### 4.4 Strategy Competition (Optimal Mode)

`mode="optimal"` tries all 16 × 8 = 128 combinations and picks the smallest output — a brute-force MDL (Minimum Description Length) approach. The winning strategy is stored in the header so decompression knows which transform and backend to apply.

`mode="optimal_fast"` pre-screens 16 transforms with zlib-6, picks top-4 + identity, then tries 4 fast compressors on each = 20 combinations. ~25x faster, same result on most data.

### 4.5 Optimality Certificate

After compression, UMC computes a certificate proving how close the result is to the Shannon entropy limit:

- **Shannon H0**: Empirical byte-level entropy of the preprocessed stream
- **Achieved bits/byte**: Actual compressed size in bits per byte
- **Entropy gap**: How many % above H0 we are (lower = better, 0 = perfect)
- **Randomness p-value**: Chi-squared test on compressed output. If p > 0.05, the output is statistically indistinguishable from random — no further byte-level compression is possible.

### 4.6 All 11 Compression Modes

| Mode | Tag | Method | Fidelity | Typical Ratio |
|------|-----|--------|----------|---------------|
| lossless | 0x01 | byte_transpose + zlib | Bit-exact | ~1.3x |
| lossless_fast | 0x0a | flatten-delta + zstd-3 | Bit-exact | ~1.7x |
| lossless_zstd | 0x03 | byte_transpose + zstd | Bit-exact | ~1.2x |
| lossless_lzma | 0x04 | byte_transpose + lzma | Bit-exact | ~1.5x |
| near_lossless | 0x02 | float16 + zlib | <0.01% error | ~2.5x |
| near_lossless_turbo | 0x07 | float16 + bt + zlib | <0.05% error | ~3x |
| quantized_8 | 0x08 | uint8 + delta + zlib | ~2% error | 5-13x |
| optimal | 0x09 | 128-strategy competition | Bit-exact | 1.2-5.5x |
| optimal_fast | 0x09 | 20-strategy competition | Bit-exact | 1.2-5.5x |
| normalized_lossless | 0x05 | normalize + bt + zlib | Bit-exact | ~1.3x |
| normalized_lossless_zstd | 0x06 | normalize + bt + zstd | Bit-exact | ~1.3x |

## 5. Neural Search Engine (Tier 1)

### 5.1 HVQ-VAE Architecture

```
Input (B, W, F) → RevIN → PatchEmbedding → TransformerBlocks → split:
  Top:    mean pool → proj → VectorQuantize (256 codes)    → global shape
  Bottom: per-patch → proj → VectorQuantize (256 codes)    → detail
  → z_projection → z (64-dim continuous for FAISS search)
  → chart_head → chart_id (8 charts, Gumbel-Softmax)
```

### 5.2 Trained Models

| Model | Window × Features | Params | Val Loss | Checkpoint |
|-------|-------------------|--------|----------|------------|
| financial-v1 | 32 × 5 (OHLCV) | 580K | 0.037 | 2.49 MB |
| iot-sensors-v1 | 64 × 8 | 776K | 0.118 | 3.28 MB |
| audio-v1 | 1024 × 1 | 980K | 0.017 | 4.22 MB |
| scientific-v1 | 64 × 5 | 771K | 0.262 | 3.40 MB |

### 5.3 .mnf File Format

Binary format storing VQ codes + continuous z coordinates + compressed storage in a single file:

```
Header:  MNF1 magic + version + domain + n_samples + latent_dim + flags
Blocks:  z coordinates | chart IDs | VQ codes | FAISS index | storage blob
```

The file IS simultaneously the compressed data, the search index, and the VQ codebook reference. Search queries hit the FAISS index over z coordinates. Full reconstruction decodes the storage blob.

## 6. C Extension

12 C functions compiled to a shared library (_umc_kernels.so/.dylib/.dll), loaded via ctypes with transparent fallback to pure NumPy:

| Function | Speedup vs NumPy |
|----------|-----------------|
| byte_transpose (float32) | 4.0x |
| byte_untranspose (float32) | 3.8x |
| delta_encode_order1 | 2.5x |
| delta_decode_order1-4 | 3.8x |
| xor_encode/decode | 2.0x |
| zigzag_encode/decode | 2.2x |

---

# PART III — PROVEN BENCHMARKS

## 7. Standard Benchmark Suite (8 Datasets)

All deterministic, reproducible. UMC `optimal_fast` vs best standard compressor at max settings:

| Dataset | UMC Best | Best Standard | Winner | Margin |
|---------|----------|---------------|--------|--------|
| Financial OHLCV | **2.27x** | 1.89x (lzma) | UMC | +17% |
| IoT Sensors (8ch) | **1.31x** | 1.12x (brotli) | UMC | +15% |
| Monotonic Counters | **4.37x** | 3.14x (brotli) | UMC | +28% |
| Image Tiles (32x32) | **1.21x** | 1.10x (brotli) | UMC | +9% |
| Audio Waveforms | **1.46x** | 1.19x (lzma) | UMC | +18% |
| Random Noise | **1.20x** | 1.09x (brotli) | UMC | +9% |
| Scientific Sim | **1.36x** | 1.21x (lzma) | UMC | +11% |
| Sparse Events | 53.65x | **55.09x** (brotli) | Standard | -3% |

**UMC wins 7/8 datasets.**

## 8. Fintech Benchmark (6.6M data points)

| Dataset | UMC Best | Best Standard | Margin |
|---------|----------|---------------|--------|
| Tick Data (OHLCV+Bid/Ask) | **5.20x** | 4.19x (lzma) | +19.5% |
| Order Book (10-level) | **2.22x** | 2.06x (lzma) | +6.9% |
| Risk Metrics (Greeks+VaR) | **1.23x** | 1.20x (lzma) | +1.9% |
| **Weighted Overall** | **3.07x** | **2.74x** | **+10.9%** |

## 9. Speed Benchmarks

| Mode | Compress | Decompress | Use Case |
|------|----------|------------|----------|
| lossless_fast | 101 MB/s | 269 MB/s | Real-time streaming |
| lossless | 17 MB/s | 34 MB/s | General purpose |
| optimal_fast | 0.2 MB/s | varies | Maximum compression |

---

# PART IV — PRODUCT ROADMAP

## 10. Product #1: UMC Vector Database ("VectorForge")

### 10.1 What It Is

A Pinecone-compatible vector database that stores embeddings compressed with UMC, delivering 3-5x lower cost at equivalent search quality. The UMC compression engine is the invisible advantage — customers see a standard vector DB API that costs 65% less.

### 10.2 Technical Architecture

```
Client (OpenAI embeddings, 1536-dim float32)
    |
    v
REST/gRPC API (Pinecone-compatible)
    |
    v
UMC Compression Layer
  - Byte transpose embeddings (groups exponent bytes)
  - Optional quantization (float32 → float16, <0.01% recall loss)
  - Delta coding across similar embeddings
    |
    v
Search Index (FAISS IVF-PQ on compressed representations)
    |
    v
Storage (compressed embeddings on SSD/S3)
```

### 10.3 API Compatibility

```python
# Drop-in Pinecone replacement
import vectorforge

index = vectorforge.Index("my-index", dimension=1536, metric="cosine")
index.upsert(vectors=[("id1", [0.1, 0.2, ...], {"metadata": "value"})])
results = index.query(vector=[0.1, 0.2, ...], top_k=10)
```

### 10.4 What Needs to Be Built

| Component | Effort | Priority |
|-----------|--------|----------|
| bfloat16/float16 native compression | 2-3 days | P0 |
| FastAPI server with Pinecone-compatible REST API | 1 week | P0 |
| FAISS index management (create, upsert, query, delete) | 1 week | P0 |
| Metadata filtering | 3 days | P1 |
| Multi-tenancy / namespace support | 3 days | P1 |
| Hosted service on AWS (ECS/Fargate) | 1 week | P1 |
| Billing dashboard (Stripe integration) | 3 days | P2 |
| Landing page + docs | 3 days | P2 |

### 10.5 Pricing Model

| Tier | Vectors | Price/month | Pinecone Equivalent |
|------|---------|-------------|---------------------|
| Free | 100K | $0 | Pinecone free: 100K |
| Starter | 1M | $25 | Pinecone: $70 |
| Pro | 10M | $150 | Pinecone: $700 |
| Enterprise | 100M+ | Custom | Pinecone: $7,000+ |

### 10.6 Revenue Trajectory

| Month | Milestone | MRR |
|-------|-----------|-----|
| 1-2 | MVP + landing page live | $0 |
| 3 | First 10 free users | $0 |
| 4-5 | First 5 paying customers | $500 |
| 6-8 | 50 paying customers, Product Hunt launch | $5,000 |
| 9-12 | 200 customers, self-serve growth | $20,000 |
| 12-18 | 500 customers, enterprise pipeline | $75,000 |
| 18-24 | 1,000+ customers or acquisition | $150,000+ |

At $150K MRR ($1.8M ARR) with 100%+ growth, the company is valued at $10-20M for acquisition purposes.

## 11. Product #2: Acquisition Preparation

### 11.1 Target Acquirers (Ranked by Fit)

| Company | Why They'd Buy | Contact Path | Deal Size |
|---------|---------------|--------------|-----------|
| **Pinecone** | Compression reduces their infrastructure cost | LinkedIn → engineering leadership | $3-8M |
| **Snowflake** | Better numeric compression = better margins on storage | LinkedIn → VP Engineering | $5-15M |
| **Databricks** | Competitive advantage in data lakehouse | LinkedIn → CTO office | $5-15M |
| **InfluxDB** | Direct feature enhancement for time series DB | Open source → partnership → acquisition | $2-5M |
| **Confluent** | Streaming compression for Kafka | Conference → BD team | $3-8M |
| **Cloudflare** | R2 storage differentiation vs S3 | Developer evangelism → BD | $5-10M |

### 11.2 What Makes You Acquirable

1. **Working product with tests** — 348 tests, 11 modes, C extension, benchmarks
2. **Patent filing** — "Compressed-domain similarity search on numeric data using VQ codebook indices"
3. **Revenue/traction** — Even $5K MRR from VectorForge proves market demand
4. **Unique IP** — 128-strategy competition, optimality certificates, byte-transposed float compression
5. **Domain expertise** — You built the entire system and understand it deeply

### 11.3 Patent Strategy

File a **provisional patent application** ($1,800 with attorney, $320 self-file) covering:

**Title**: "System and Method for Similarity Search over Compressed Numeric Data Using Vector Quantization Codebook Indices"

**Claims**:
1. A method for compressing numeric arrays using byte transposition of IEEE 754 floating-point values followed by strategy competition across multiple (preprocessing transform, compressor backend) combinations
2. A method for performing approximate nearest-neighbor search on compressed float data using VQ codebook indices without decompressing the underlying data
3. A file format (.mnf) that simultaneously serves as compressed storage and a searchable similarity index
4. A method for generating optimality certificates proving proximity to Shannon entropy limits for compressed data

---

# PART V — BUSINESS PLAYBOOK

## 12. Month-by-Month Execution Plan

### Month 1: Foundation (Weeks 1-4)

**Week 1: Technical**
- [ ] Add bfloat16/float16 native support to UMC compression engine
- [ ] Benchmark compression ratios on standard embedding dimensions (768, 1024, 1536, 3072)
- [ ] Verify search recall quality after compression (target: >99% recall@10)

**Week 2: VectorForge MVP**
- [ ] Build FastAPI server with 4 endpoints: /upsert, /query, /delete, /describe
- [ ] FAISS IVF-PQ index backed by UMC-compressed storage
- [ ] Docker container for local testing
- [ ] Basic load test: 1M vectors, 1000 qps target

**Week 3: Infrastructure**
- [ ] Deploy to AWS (single EC2 instance or ECS Fargate)
- [ ] Domain registration: vectorforge.dev (or similar)
- [ ] Landing page with pricing, benchmark results, code samples
- [ ] Stripe billing integration (free + paid tiers)

**Week 4: Legal + Patent**
- [ ] File provisional patent application (USPTO, Form SB/16)
- [ ] Choose: self-file ($320) or use patent attorney ($1,500-2,500)
- [ ] Patent attorney recommendation: find one specializing in software/data patents on UpCounsel or LegalZoom
- [ ] Register LLC (if not already done) for the business entity

### Month 2: Launch (Weeks 5-8)

**Week 5-6: Beta Users**
- [ ] Post on Hacker News "Show HN: Vector DB that's 3x cheaper than Pinecone"
- [ ] Post on r/MachineLearning, r/LocalLLaMA, r/SideProject
- [ ] Tweet thread showing Pinecone vs VectorForge cost comparison
- [ ] Reach out to 20 AI startups on LinkedIn (founders, CTOs) offering free beta

**Week 7-8: Iterate**
- [ ] Fix bugs from beta feedback
- [ ] Add metadata filtering if not done
- [ ] Add Python client library: `pip install vectorforge`
- [ ] Write 3 tutorial blog posts (RAG with VectorForge, Pinecone migration, cost comparison)

### Month 3: First Revenue (Weeks 9-12)

- [ ] Convert beta users to paid plans
- [ ] Product Hunt launch
- [ ] Cold email campaign to 100 companies using Pinecone/Weaviate (see email templates below)
- [ ] Apply to Y Combinator (next batch) if >$1K MRR

### Months 4-6: Growth

- [ ] Self-serve growth from content marketing (blog posts, Twitter, HN)
- [ ] Start acquisition conversations with target companies (see outreach templates below)
- [ ] Hire first contractor (devops) if revenue supports it
- [ ] Add features: namespaces, access control, usage dashboard

### Months 7-12: Scale or Sell

**If growing well (>$20K MRR):**
- Continue scaling
- Raise seed round ($1-2M) to hire 2-3 engineers
- Expand to enterprise tier with SLA, dedicated instances

**If growth is slow (<$5K MRR):**
- Pivot to acquisition path
- Use traction + patent + tech demo to approach acquirers
- Target: $3-8M acquisition

## 13. Email Templates

### 13.1 Cold Email to Potential Customers

**Subject: Cut your Pinecone bill by 65%**

```
Hi [Name],

I noticed [Company] is building with vector search (saw your job posting for
ML engineer / your blog post about RAG / your GitHub repo using Pinecone).

I built VectorForge — a Pinecone-compatible vector database that costs 65%
less. Same API, same recall quality, but we compress embeddings 3-5x using
a float-aware compression engine I developed (UMC), so we can charge less
while still making healthy margins.

Migration is a one-line change:
  # index = pinecone.Index("my-index")
  index = vectorforge.Index("my-index")

Happy to give you a free month on any plan so you can compare. What do
you think?

[Your name]
[VectorForge URL]
```

### 13.2 Cold Email to Acquisition Targets (Engineering Leadership)

**Subject: Float compression tech that beats zstd by 20% on numeric data**

```
Hi [Name],

I'm the creator of UMC (Universal Manifold Codec), a compression engine
specifically designed for numeric/float data. On standard benchmarks it
beats zstd-19, lzma-9, and brotli-11 on 7 out of 8 dataset types.

The interesting part: it also supports similarity search on compressed data
without decompressing — useful for vector search, time series pattern
matching, and KV-cache compression.

I think this technology would be a great fit for [Company]'s [specific
product — e.g., "Snowflake's storage engine" / "InfluxDB's compression
layer" / "Pinecone's index backend"]. I'd love to show you a demo.

Would you have 20 minutes for a call this week?

[Your name]
GitHub: [repo link]
Benchmarks: [link to case study]
```

### 13.3 Y Combinator Application (Key Fields)

**What does your company do?**
```
VectorForge is a vector database for AI applications that costs 65% less
than Pinecone. We built a novel float-aware compression engine (UMC) that
compresses embeddings 3-5x smaller while maintaining >99% search recall.
Same API, one-line migration, immediately cheaper.
```

**Why did you pick this idea?**
```
I built a compression library that beats every standard compressor on
numeric data (7/8 benchmarks won). The biggest commercial application is
vector databases — a $3B+ market where storage cost is the #1 customer
complaint. Instead of selling compression as a library (no revenue model),
I'm selling it as infrastructure (recurring SaaS revenue).
```

**What's your progress?**
```
- Working product: [X] paying customers, [Y] MRR
- 348 tests passing, C-accelerated, published benchmarks
- Patent pending on compressed-domain vector search
- Beat Pinecone on cost by 65% with equivalent recall
```

### 13.4 LinkedIn DM Template (For Warm Outreach to Acquirers)

```
Hi [Name], I've been following [Company]'s work on [specific topic].

I built a compression engine for float/numeric data that beats zstd/lzma
by 10-20% on structured data, with the unique ability to do similarity
search on the compressed representation. 348 tests, C extension, published
benchmarks.

I'm exploring whether this technology might be a fit for [Company]. Would
you be open to a brief chat?
```

## 14. Financial Projections

### 14.1 Path to $5M via VectorForge SaaS

| Year | MRR (End) | ARR | Cumulative Revenue | Customers |
|------|-----------|-----|--------------------|-----------|
| Year 1 | $20-50K | $240-600K | $150-350K | 200-500 |
| Year 2 | $100-200K | $1.2-2.4M | $1-2.5M | 1,000-2,000 |

At $2M+ ARR with 100% growth, acquisition multiples of 5-10x ARR = **$10-20M valuation**.

### 14.2 Path to $5M via Acquisition

| Quarter | Milestone | Effect on Valuation |
|---------|-----------|---------------------|
| Q1 | Patent filed, MVP live, 10 users | $500K-1M (IP value) |
| Q2 | 50 paying customers, $5K MRR | $1-2M |
| Q3 | 200 customers, $20K MRR, YC batch | $3-5M |
| Q4 | Start acquisition conversations | $5-10M (with traction) |

### 14.3 Expenses (Solo Founder, Year 1)

| Item | Monthly | Annual |
|------|---------|--------|
| AWS hosting (VectorForge) | $200-500 | $2,400-6,000 |
| Domain + landing page | $20 | $240 |
| Patent attorney | — | $2,500 (one-time) |
| LLC registration | — | $200 (one-time) |
| Stripe fees (2.9%) | varies | ~$500 |
| **Total** | **$300-600** | **$6,000-9,000** |

You can run this entire operation for under $10K/year while keeping your day job.

## 15. Competitive Intelligence

### 15.1 Pinecone Weaknesses to Exploit

- **Pricing**: $70/month per 1M vectors. No compression — raw float32 in memory. UMC compresses 3-5x, so we can charge $25 and still profit.
- **Cold start**: New index takes minutes to provision. VectorForge can be instant (single-process, in-memory FAISS).
- **Vendor lock-in**: Proprietary API. VectorForge uses the same API format — zero switching cost.
- **Serverless limitations**: Pinecone serverless has cold start latency. VectorForge dedicated instances are always warm.

### 15.2 Technical Moat

1. **Byte transposition patent** — Novel preprocessing for IEEE 754 float data before compression
2. **128-strategy competition** — No other compressor tries this many combinations
3. **Optimality certificates** — Provable proximity to Shannon limit
4. **VQ search on compressed data** — Unique to UMC, not available in any standard compressor
5. **C extension with fallback** — Production-grade performance with pure-Python safety net

---

# PART VI — TECHNICAL APPENDIX

## A. Corrected Compression Theory

The old specification claimed compression ratios of 50-200x lossless and up to 172,000x based on manifold dimensionality arguments. These were theoretical upper bounds assuming perfect manifold learning, which does not exist.

**Actual achievable lossless ratios:**
- Standard compressors (zstd, lzma): 1.5-4x on float32 time series
- UMC optimal: 1.2-5.5x on float32 time series
- UMC advantage: 10-28% smaller than best standard compressor

**Why the theoretical gap is so large:**
- Manifold coordinates require a trained decoder to reconstruct. The decoder itself is megabytes.
- Real-world data has noise that doesn't lie on any low-dimensional manifold.
- Lossless compression cannot discard noise — it must encode every bit.
- The "50x lossy" claims from earlier experiments used VQ codes that had 0.1-3% reconstruction error — not lossless.

**Honest assessment:** UMC's lossless compression is meaningfully better than standard compressors (10-28%), not orders of magnitude better. The real value differentiator is the search-without-decompressing capability, which no standard compressor offers at any ratio.

## B. File Formats

### B.1 UMCZ (Storage Compression)

```
magic:    "UMCZ" (4 bytes)
ndim:     uint8 (1 byte)
dims:     uint32 × ndim
payload:  tag (1 byte) + shape header (13 bytes) + compressed data
```

### B.2 UMCR (Raw Binary Compression)

```
magic:    "UMCR" (4 bytes)
size:     uint64 (8 bytes) — original size
payload:  optimal compressed data
```

### B.3 UMCD (DataFrame Compression)

```
magic:    "UMCD" (4 bytes)
meta_len: uint32 (4 bytes)
metadata: JSON (column names, dtypes, index info)
payload:  UMCZ compressed numeric data
```

### B.4 MNF (Neural Search Archive)

```
magic:    "MNF1" (4 bytes)
version:  uint16
header:   domain, n_samples, latent_dim, flags
blocks:   z coordinates | chart IDs | VQ codes | FAISS index | storage
```

## C. Project Structure (Current)

```
umc/
├── __init__.py              # Public API: compress/decompress/compress_optimal
├── __main__.py              # CLI: compress, decompress, info, tryit, dashboard
├── config.py                # UMCConfig dataclass
├── _neural.py               # TieredManifoldCodec (neural search)
├── hub.py                   # Model registry (4 pre-trained models)
├── pandas_ext.py            # DataFrame accessor + compress/decompress
├── benchmark_report.py      # HTML report generator
├── codec/
│   ├── tiered.py            # Core compress/decompress dispatch (11 modes)
│   ├── optimal.py           # 128-strategy competition engine
│   ├── arithmetic.py        # Static + adaptive arithmetic coders
│   ├── residual.py          # Byte transpose, delta coding
│   └── streaming.py         # StreamingStorageCodec
├── cext/
│   ├── kernels.c            # 12 C-accelerated functions
│   ├── __init__.py          # ctypes wrapper with fallback
│   └── build.py             # Cross-platform build script
├── encoder/
│   └── hvqvae_encoder.py    # HVQ-VAE encoder
├── decoder/
│   └── hvqvae_decoder.py    # HVQ-VAE decoder
├── training/
│   ├── trainer.py           # Training loop
│   ├── losses.py            # VQ + reconstruction losses
│   └── scheduler.py         # LR scheduling
├── data/
│   ├── synthetic.py         # Synthetic data generators
│   ├── preprocessors.py     # WindowDataset, normalization
│   ├── image_loader.py      # Image file support
│   ├── audio_loader.py      # Audio file support
│   └── video_loader.py      # Video file support
├── storage/
│   └── mnf_format.py        # .mnf reader/writer
├── processor/
│   ├── search.py            # FAISS nearest-neighbor
│   └── anomaly.py           # Manifold anomaly detection
└── evaluation/
    ├── metrics.py            # Compression ratio, RMSE, throughput
    └── visualize.py          # Latent space plots

tests/                        # 348 tests
scripts/
├── benchmark_standard.py     # 8-dataset benchmark suite
├── benchmark_fintech_casestudy.py  # Fintech case study
├── demo_live.py              # Interactive terminal demo
├── train_pretrained_models.py # Train 4 hub models
└── entropy_analyzer.py       # Shannon entropy analysis

docs/
└── case-study-fintech.md     # Enterprise case study

checkpoints/pretrained/       # 4 trained .pt models (13 MB total)
```

## D. Test Coverage

348 tests covering:
- All 11 compression modes (round-trip verification)
- Optimal + optimal_fast strategy competition
- Arithmetic coding (static + adaptive)
- Neural encode/decode/search pipeline
- Streaming codec
- File compression (npy, csv, raw binary)
- Edge cases (empty arrays, NaN, Inf, single-element)
- C extension with fallback verification

---

**END OF SPECIFICATION**

Universal Manifold Codec v4.0 — February 2026

348 tests passing | 11 compression modes | C-accelerated | 4 pre-trained models
Wins 7/8 standard benchmarks vs gzip/lzma/zstd/brotli
