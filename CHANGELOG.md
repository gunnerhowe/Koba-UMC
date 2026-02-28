# Changelog

## v0.2.0 (2026-02-27)

### New Features
- **Optimal compression mode** — tries 128 (transform, compressor) combinations and picks the smallest output. Beats the best standard compressor on 6/7 test datasets (7-46% improvement). Includes optimality certificate proving proximity to Shannon entropy limit.
- **Optimal fast mode** — same results as optimal on most data, ~25x faster.
- **Float16 / bfloat16 native support** — compress half-precision data directly without float32 conversion. UMC2 binary format preserves original dtype through round-trip.
- **VectorForge** — Pinecone-compatible vector database with UMC compression. 53-57% storage savings on embeddings with 100% recall@10. REST API server included.
- **Pandas integration** — `df.umc.compress()` / `umc.pandas_ext.decompress_dataframe()` preserves columns, index, and dtypes.
- **C extension** — optional C-accelerated byte transposition and delta coding (2.5-4x faster). Auto-builds with fallback to pure Python.
- **CLI dashboard** — `umc dashboard` launches interactive web UI for exploring compression results.
- **HTML benchmark reports** — `umc tryit data.npy --html report.html` generates standalone comparison reports.
- **Pre-trained model hub** — `umc.hub.list_models()` / `umc.hub.load_model()` for domain-specific neural codecs.

### Compression Modes (11 total)
- `lossless` — bit-exact, ~1.3x ratio
- `lossless_fast` — bit-exact, ~1.7x ratio, 2-3x faster
- `lossless_zstd` — bit-exact, ~1.2x ratio (requires zstandard)
- `lossless_lzma` — bit-exact, ~1.5x ratio, maximum lossless
- `near_lossless` — <0.01% error, ~2.4x ratio
- `near_lossless_turbo` — <0.05% error, ~2.8x ratio
- `normalized_lossless` — bit-exact, 1.0-1.5x, best for normalized data
- `normalized_lossless_zstd` — bit-exact, normalized + zstd backend
- `quantized_8` — ~2% error, 4-9x ratio
- `optimal` — bit-exact, 1.2-5.5x, tries all strategies
- `optimal_fast` — bit-exact, 1.2-5.5x, ~25x faster than optimal

### Testing
- 405 tests across 24 test modules
- CI: Python 3.10, 3.11, 3.12 on Linux, macOS, Windows
- Cross-platform wheel builds (x86_64, aarch64, arm64)

## v0.1.0 (2024)

- Initial release with lossless and near-lossless compression
- Two-tier architecture (VQ search + storage)
- MNF file format
- Basic CLI
