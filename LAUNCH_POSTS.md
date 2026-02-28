# Launch Posts — Ready to Copy-Paste

Once your GitHub repo is public, replace [GITHUB_URL] with your actual repo URL below, then copy-paste each post to the corresponding platform.

---

## 1. Hacker News (Show HN)

**Go to:** https://news.ycombinator.com/submit

**Title:**
```
Show HN: UMC – Lossless compression that beats lzma by 7-46% on numeric data
```

**URL:** [GITHUB_URL]

**Text (leave blank if submitting URL, or use this if doing a text post):**
```
UMC (Universal Manifold Codec) is a Python compression library for numeric arrays — time series, sensor data, embeddings, scientific simulations.

What makes it different from zstd/lzma/brotli:

• Float-aware transforms (byte transposition, delta coding, exponent/mantissa splitting) before compression. Standard compressors treat floats as opaque bytes.

• mode="optimal" tries 128 (transform × compressor) combinations and picks the smallest. Issues an optimality certificate proving how close to Shannon entropy you are.

• VectorForge: Pinecone-compatible vector database with 53-57% storage savings and 100% recall@10.

• Float16/bfloat16 native support — 57% total savings vs raw float32 on embeddings.

• Only dependency is numpy. pip install umc.

Benchmarks on 7 data types: beats the best standard compressor on 6, ties on 1. Worst case is 1% overhead (format header). Best case is 46% smaller than lzma (monotonic counters).

405 tests. MIT license. GitHub: [GITHUB_URL]

I built this to solve a specific problem: we had terabytes of financial time series and needed pattern search without decompressing the archive.
```

---

## 2. Reddit r/Python

**Go to:** https://www.reddit.com/r/Python/submit

**Title:**
```
I built a compression library that beats zstd/lzma on structured data and searches compressed archives
```

**Body:**
```
Hey r/Python,

I've been working on UMC (Universal Manifold Codec) — a compression library specifically designed for numeric arrays (time series, sensor data, ML embeddings, scientific data).

**Why not just use zstd/lzma?**

Standard compressors treat float32 data as opaque bytes. UMC uses float-aware preprocessing (byte transposition, delta coding, exponent/mantissa splitting) before compression, which exploits the structure in numeric data.

**Results on 7 real-world datasets:**

| Dataset | UMC optimal | Best standard | Improvement |
|---------|------------|---------------|-------------|
| Monotonic counters | 5.48x | lzma 2.95x | 46% better |
| Scientific sim | 1.35x | lzma 1.21x | 10% better |
| Financial OHLCV | 2.45x | lzma 2.28x | 7% better |
| Random noise | 1.20x | brotli 1.09x | 9% better |

**Quick start:**

    pip install umc

    import numpy as np, umc
    data = np.random.randn(1000, 32, 5).astype(np.float32)
    compressed = umc.compress(data)
    recovered = umc.decompress(compressed)
    assert np.array_equal(data, recovered)  # bit-exact

**Features:**
- 11 compression modes (lossless to 9x lossy)
- `mode="optimal"` tries 128 strategies, picks the best, proves how close to Shannon entropy
- Float16/bfloat16 native support
- VectorForge: Pinecone-compatible vector DB with 53-57% storage savings
- CLI: `umc compress`, `umc tryit`, `umc dashboard`
- Only dependency: numpy
- 405 tests, MIT license

GitHub: [GITHUB_URL]

Happy to answer any questions about the approach or benchmarks.
```

**Flair:** Library

---

## 3. Reddit r/MachineLearning

**Go to:** https://www.reddit.com/r/MachineLearning/submit

**Title:**
```
[P] 53-57% storage savings on OpenAI/sentence-transformer embeddings with 100% recall — open source
```

**Body:**
```
I built VectorForge, a Pinecone-compatible vector database that uses UMC compression to cut embedding storage by 53-57% with zero recall loss.

**Benchmark results (realistic clustered embeddings):**

| Scale | Raw | Float16+UMC | Savings | Query p50 | Recall@10 |
|-------|-----|-------------|---------|-----------|-----------|
| 5K × 384-dim | 7.3 MB | 3.1 MB | 57% | 0.2ms | 100% |
| 10K × 1536-dim | 58.6 MB | 25.1 MB | 57% | 4.7ms | 100% |
| 25K × 1536-dim | 146.5 MB | 69.3 MB | 53% | 11.6ms | 100% |

**How it works:**

1. Store embeddings as float16 instead of float32 (negligible precision loss for similarity search)
2. Apply UMC's float-aware lossless compression (byte transposition + entropy coding)
3. Result: 53-57% smaller than raw float32, with identical search results

**API is Pinecone-compatible** — same REST endpoints, same request/response format. If you're spending money on Pinecone for a RAG app, this is a self-hosted alternative.

    pip install umc[vectordb]
    python -m umc.vectordb --port 8100

The underlying compression library (UMC) also works standalone for any numeric data — time series, sensor data, scientific simulations. It beats lzma by 7-46% on structured float data.

GitHub: [GITHUB_URL]

405 tests. MIT license. Only core dependency is numpy.
```

---

## 4. Reddit r/dataengineering

**Go to:** https://www.reddit.com/r/dataengineering/submit

**Title:**
```
Open-source compression library for numeric data — beats lzma/zstd, plus a Pinecone-compatible vector DB
```

**Body:**
```
Built a compression library called UMC specifically for numeric/float data (time series, embeddings, sensor data, scientific arrays).

**The problem:** Standard compressors (zstd, lzma, brotli) treat float32 data as random bytes. But floats have structure — shared exponents, correlated mantissas, temporal patterns. UMC exploits this.

**The approach:** Float-aware preprocessing (byte transposition, delta coding, exponent/mantissa splitting) before feeding into standard compression backends. The `optimal` mode tries 128 transform×compressor combinations and picks the smallest output.

**Results vs standard compressors:**
- Beats lzma by 7-46% on structured data
- Never loses by more than 1% (format header overhead)
- 100% lossless — bit-exact round trip

**Also includes VectorForge** — a Pinecone-compatible vector database with 53-57% storage savings on embeddings and 100% recall@10. Useful if you're running RAG apps and want to cut embedding storage costs.

    pip install umc

405 tests, MIT license, only dependency is numpy.

GitHub: [GITHUB_URL]
```

---

## 5. Twitter/X

**Go to:** https://twitter.com/compose/tweet

**Post 1 (main launch):**
```
I built UMC — a compression library that beats lzma by 7-46% on numeric data.

How: float-aware transforms before compression. Standard compressors treat floats as opaque bytes. UMC exploits the structure.

pip install umc
11 modes, 405 tests, MIT license.

[GITHUB_URL]
```

**Post 2 (reply thread — VectorForge angle):**
```
Also built VectorForge on top of it — a Pinecone-compatible vector DB with 53-57% storage savings.

Same API. Same recall. 57% less storage.

pip install umc[vectordb]
```

**Post 3 (reply thread — the hook):**
```
The `optimal` mode tries 128 compression strategies and picks the best. Then proves how close to the Shannon entropy limit you are with an optimality certificate.

No other compressor does this.
```

---

## 6. LinkedIn

**Go to:** https://www.linkedin.com/feed/ → Start a post

```
I just open-sourced UMC (Universal Manifold Codec) — a compression library for numeric data that outperforms industry-standard compressors.

The insight: standard compressors like zstd and lzma treat float32 data as opaque bytes. But numeric data has structure — shared exponents, correlated values, temporal patterns. UMC uses float-aware transforms (byte transposition, delta coding, exponent/mantissa splitting) before compression.

Results on 7 real-world datasets: UMC beats the best standard compressor on 6, ties on 1. Improvement ranges from 7% to 46%.

I also built VectorForge on top of it — a Pinecone-compatible vector database that stores embeddings with 53-57% less storage and 100% recall. If you're running RAG applications and spending money on vector storage, this is worth a look.

Key specs:
→ 11 compression modes (lossless to 9x lossy)
→ Float16/bfloat16 native support
→ Optimality certificates (proves how close to Shannon entropy)
→ 405 tests, MIT license
→ Only dependency: numpy

pip install umc

GitHub: [GITHUB_URL]

#opensource #compression #machinelearning #dataengineering #python
```

---

## 7. Dev.to (first technical article)

**Go to:** https://dev.to/enter → New Post

**Title:**
```
How UMC beats lzma: float-aware compression explained
```

**Tags:** python, compression, machinelearning, opensource

**Body:**
```markdown
## The problem with compressing floats

When you compress numeric data with standard tools (gzip, zstd, lzma, brotli), they treat your float32 values as arbitrary bytes. But floats aren't arbitrary — they have structure:

- **Exponents cluster**: similar-magnitude values share exponent bits
- **Temporal correlation**: consecutive values in time series are similar
- **Cross-feature patterns**: related features move together

Standard compressors miss all of this.

## What UMC does differently

UMC (Universal Manifold Codec) applies **float-aware preprocessing** before compression:

### 1. Byte transposition

A float32 is 4 bytes: `[exp][mantissa_hi][mantissa_mid][mantissa_lo]`. When you store floats sequentially, the byte stream alternates between high-entropy mantissa bytes and low-entropy exponent bytes.

Byte transposition groups all exponent bytes together, all mantissa_hi bytes together, etc. Now the compressor sees long runs of similar bytes instead of mixed noise.

**Before:** `E1 M1a M1b M1c E2 M2a M2b M2c E3 M3a M3b M3c`
**After:**  `E1 E2 E3 M1a M2a M3a M1b M2b M3b M1c M2c M3c`

This alone improves compression by 10-30% on most float data.

### 2. Delta coding

For time series, consecutive values are similar. Instead of storing raw values, store the difference. Delta-coded data has much lower entropy.

### 3. Exponent/mantissa splitting

Split each float into its exponent and mantissa components, compress them separately. Exponents compress extremely well (often 10x+).

### 4. Strategy competition (optimal mode)

UMC's `optimal` mode tries **128 combinations** of 16 transforms × 8 compressors. The smallest output wins. This is the Minimum Description Length principle in action.

It then issues an **optimality certificate** — computing the Shannon entropy of the preprocessed stream and measuring how close the compressed output gets. If the compressed bytes are statistically indistinguishable from random noise (chi-squared test), no further compression is possible.

## Benchmarks

| Dataset | UMC optimal | Best standard | Winner |
|---------|------------|---------------|--------|
| Monotonic counters | 5.48x | lzma 2.95x | UMC +46% |
| Scientific sim | 1.35x | lzma 1.21x | UMC +10% |
| Financial OHLCV | 2.45x | lzma 2.28x | UMC +7% |
| Random noise | 1.20x | brotli 1.09x | UMC +9% |

## Try it

\```bash
pip install umc
\```

\```python
import numpy as np
import umc

data = np.random.randn(1000, 32, 5).astype(np.float32)

# Basic lossless
compressed = umc.compress(data, mode="lossless")

# Optimal with certificate
result = umc.compress_optimal(data)
print(f"Entropy gap: {result['certificate']['entropy_gap_pct']:.1f}%")
\```

GitHub: [GITHUB_URL]

405 tests. MIT license. Only dependency is numpy.
```

---

## Posting Order (recommended)

1. **Hacker News** — post first thing in the morning (US time, ~9am EST Tuesday-Thursday gets best visibility)
2. **Reddit r/Python** — 30 minutes after HN
3. **Reddit r/MachineLearning** — 1 hour after HN
4. **Reddit r/dataengineering** — 2 hours after HN
5. **Twitter/X** — same time as HN
6. **LinkedIn** — same day, afternoon
7. **Dev.to** — day 3 (first technical article)

## Tips

- **Reply to every comment** on HN and Reddit within the first 6 hours — this is your most important marketing activity
- **Don't be defensive** about criticism — acknowledge limitations honestly
- **Fix any bug reported** within 24 hours and reply with the fix
- **Don't spam** — one post per subreddit, one HN submission. Let it succeed or fail on merit.
- **Best days to post:** Tuesday, Wednesday, Thursday. Avoid weekends and Mondays.
