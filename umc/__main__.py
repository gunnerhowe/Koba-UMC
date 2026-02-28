"""CLI entry point: python -m umc <command>"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="umc",
        description="Universal Manifold Codec CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train a new codec")
    train_parser.add_argument("--data", type=str, help="Path to CSV data or directory")
    train_parser.add_argument("--symbols", type=str, default="SPY,AAPL,MSFT",
                              help="Comma-separated ticker symbols (for Yahoo Finance)")
    train_parser.add_argument("--period", type=str, default="5y")
    train_parser.add_argument("--interval", type=str, default="1d")
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--latent-dim", type=int, default=128)
    train_parser.add_argument("--output", type=str, default="checkpoints/financial_v1")
    train_parser.add_argument("--config", type=str, default=None)

    # --- encode ---
    encode_parser = subparsers.add_parser("encode", help="Encode data to .mnf")
    encode_parser.add_argument("--input", type=str, required=True,
                                help="Input: CSV file or .npy windows array")
    encode_parser.add_argument("--output", type=str, required=True, help="Output .mnf file")
    encode_parser.add_argument("--codec", type=str, required=True,
                                help="Path to checkpoint (.pt) or pretrained directory")
    encode_parser.add_argument("--storage-mode", type=str, default="lossless",
                                choices=["lossless", "near_lossless", "lossless_zstd", "lossless_lzma",
                                         "normalized_lossless", "normalized_lossless_zstd",
                                         "near_lossless_turbo", "quantized_8",
                                         "optimal", "optimal_fast"],
                                help="Storage compression mode (default: lossless)")
    encode_parser.add_argument("--batch-size", type=int, default=32)
    encode_parser.add_argument("--device", type=str, default="cpu")

    # --- decode ---
    decode_parser = subparsers.add_parser("decode", help="Decode .mnf back to data")
    decode_parser.add_argument("--input", type=str, required=True, help="Input .mnf file")
    decode_parser.add_argument("--output", type=str, required=True,
                                help="Output .npy or .csv file")
    decode_parser.add_argument("--codec", type=str, required=True)
    decode_parser.add_argument("--mode", type=str, default="storage",
                                choices=["storage", "search"],
                                help="Decode mode: storage (full fidelity) or search (VQ lossy)")
    decode_parser.add_argument("--batch-size", type=int, default=32)
    decode_parser.add_argument("--device", type=str, default="cpu")

    # --- search ---
    search_parser = subparsers.add_parser("search", help="Search for similar patterns")
    search_parser.add_argument("--index", type=str, required=True, help="Path to .mnf file")
    search_parser.add_argument("--query", type=str, required=True,
                                help="Query: .npy windows or CSV file")
    search_parser.add_argument("--k", type=int, default=10)
    search_parser.add_argument("--codec", type=str, required=True)
    search_parser.add_argument("--batch-size", type=int, default=32)
    search_parser.add_argument("--device", type=str, default="cpu")

    # --- stats ---
    stats_parser = subparsers.add_parser("stats", help="Show compression statistics")
    stats_parser.add_argument("--input", type=str, required=True, help="Path to .mnf file")
    stats_parser.add_argument("--codec", type=str, required=True)
    stats_parser.add_argument("--device", type=str, default="cpu")

    # --- benchmark ---
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    bench_parser.add_argument("--data-dir", type=str, default="./data/test/")
    bench_parser.add_argument("--codec-path", type=str, required=True)
    bench_parser.add_argument("--output", type=str, default="./results/benchmark.json")

    # --- tryit (model-free benchmark) ---
    tryit_parser = subparsers.add_parser(
        "tryit", help="Benchmark UMC on your data — compare all modes + competitors",
    )
    tryit_parser.add_argument("input", type=str, help="Any file (.npy, .csv, .parquet, .png, .wav, .exe, ...)")
    tryit_parser.add_argument("--html", type=str, default=None,
                               help="Save HTML report to this path")

    # --- compress (model-free) ---
    compress_parser = subparsers.add_parser("compress", help="Compress a file (no model needed)")
    compress_parser.add_argument("input", type=str, help="Input file (.npy, .csv, .parquet, .png, .wav)")
    compress_parser.add_argument("-o", "--output", type=str, required=True, help="Output file path")
    compress_parser.add_argument("-m", "--mode", type=str, default="lossless",
                                  choices=["lossless", "near_lossless", "lossless_zstd", "lossless_lzma",
                                           "normalized_lossless", "normalized_lossless_zstd",
                                           "near_lossless_turbo", "quantized_8",
                                           "optimal", "optimal_fast", "lossless_fast"],
                                  help="Compression mode (default: lossless)")

    # --- decompress (model-free) ---
    decompress_parser = subparsers.add_parser("decompress", help="Decompress a UMC file")
    decompress_parser.add_argument("input", type=str, help="Compressed UMC file")
    decompress_parser.add_argument("-o", "--output", type=str, required=True,
                                    help="Output file (.npy, .csv, .png, .wav)")

    # --- info ---
    info_parser = subparsers.add_parser("info", help="Show compression info for a UMC file")
    info_parser.add_argument("input", type=str, help="Compressed UMC file")

    # --- dashboard ---
    subparsers.add_parser("dashboard", help="Launch the Streamlit web dashboard")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "encode":
        _cmd_encode(args)
    elif args.command == "decode":
        _cmd_decode(args)
    elif args.command == "search":
        _cmd_search(args)
    elif args.command == "stats":
        _cmd_stats(args)
    elif args.command == "benchmark":
        _cmd_benchmark(args)
    elif args.command == "tryit":
        _cmd_tryit(args)
    elif args.command == "compress":
        _cmd_compress(args)
    elif args.command == "decompress":
        _cmd_decompress_file(args)
    elif args.command == "info":
        _cmd_info(args)
    elif args.command == "dashboard":
        _cmd_dashboard(args)


def _cmd_train(args):
    from .config import UMCConfig
    from . import ManifoldCodec
    from .data.loaders import load_yahoo_finance, combine_datasets

    config = UMCConfig(
        epochs=args.epochs,
        max_latent_dim=args.latent_dim,
    )

    symbols = [s.strip() for s in args.symbols.split(",")]
    print(f"Loading data for: {symbols}")
    datasets = load_yahoo_finance(symbols, period=args.period, interval=args.interval)
    df = combine_datasets(datasets)
    print(f"Total samples: {len(df)}")

    codec = ManifoldCodec(config)
    history = codec.fit(df, verbose=True)
    codec.save(args.output)
    print(f"Codec saved to {args.output}")
    print(f"Final val reconstruction loss: {history[-1]['val']['reconstruction']:.6f}")


def _load_tiered_codec(codec_path, device="cpu", storage_mode="lossless"):
    """Load a TieredManifoldCodec from a checkpoint or pretrained directory."""
    from . import TieredManifoldCodec
    from pathlib import Path

    path = Path(codec_path)
    if path.is_dir():
        return TieredManifoldCodec.from_pretrained(str(path), device=device, storage_mode=storage_mode)
    else:
        return TieredManifoldCodec.from_checkpoint(str(path), device=device, storage_mode=storage_mode)


def _load_windows(input_path):
    """Load windows from .npy or CSV file."""
    import numpy as np
    from pathlib import Path

    path = Path(input_path)
    if path.suffix == ".npy":
        return np.load(str(path))
    elif path.suffix in (".csv", ".txt"):
        import pandas as pd
        df = pd.read_csv(str(path), index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        from .data.preprocessors import create_windows
        return create_windows(df.values.astype(np.float32), window_size=32)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}. Use .npy or .csv")


def _cmd_encode(args):
    import numpy as np
    from .cli_formatting import console, print_encode_results

    codec = _load_tiered_codec(args.codec, device=args.device, storage_mode=args.storage_mode)
    windows = _load_windows(args.input)

    console.print(f"[bold]Encoding {len(windows)} windows[/bold] ({windows.shape})...")
    bytes_written = codec.encode_to_mnf(windows, args.output, batch_size=args.batch_size)

    print_encode_results(
        n_windows=len(windows),
        shape=windows.shape,
        raw_bytes=windows.nbytes,
        mnf_bytes=bytes_written,
        storage_mode=args.storage_mode,
        output=args.output,
    )


def _cmd_decode(args):
    import numpy as np
    from pathlib import Path
    from .cli_formatting import console, print_decode_results

    codec = _load_tiered_codec(args.codec, device=args.device)

    console.print(f"[bold]Decoding[/bold] {args.input} (mode={args.mode})...")
    decoded = codec.decode_from_mnf(args.input, mode=args.mode, batch_size=args.batch_size)

    output_path = Path(args.output)
    if output_path.suffix == ".npy":
        np.save(str(output_path), decoded)
    else:
        flat = decoded.reshape(-1, decoded.shape[-1])
        np.savetxt(str(output_path), flat, delimiter=",", comments="")

    print_decode_results(
        n_windows=decoded.shape[0],
        shape=decoded.shape,
        dtype=str(decoded.dtype),
        output=args.output,
    )


def _cmd_search(args):
    from .cli_formatting import console, print_search_results

    codec = _load_tiered_codec(args.codec, device=args.device)
    query_windows = _load_windows(args.query)

    console.print(f"[bold]Searching[/bold] {args.index} for {args.k} nearest neighbors...")
    result = codec.search_from_mnf(
        args.index, query_windows, k=args.k, batch_size=args.batch_size,
    )
    print_search_results(result, k=args.k)


def _cmd_stats(args):
    from .cli_formatting import print_stats

    codec = _load_tiered_codec(args.codec, device=args.device)
    stats = codec.stats_from_mnf(args.input)
    print_stats(stats)


def _cmd_benchmark(args):
    import json
    from pathlib import Path
    import numpy as np
    from torch.utils.data import DataLoader
    from . import ManifoldCodec
    from .evaluation.benchmarks import run_all_baselines
    from .data.preprocessors import WindowDataset

    codec = ManifoldCodec.from_pretrained(args.codec_path)

    # Load test data
    data_dir = Path(args.data_dir)
    if data_dir.is_file():
        import pandas as pd
        df = pd.read_csv(data_dir, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
    else:
        from .data.loaders import load_yahoo_finance, combine_datasets
        datasets = load_yahoo_finance(["SPY", "AAPL", "MSFT"], period="2y", interval="1d")
        df = combine_datasets(datasets)
        print("No data dir found -- using default symbols (SPY, AAPL, MSFT)")

    # Prepare windows
    from .data.preprocessors import OHLCVPreprocessor, create_windows
    preprocessor = OHLCVPreprocessor(codec.config)
    normalized = preprocessor.fit_transform(df)
    windows = create_windows(normalized, codec.config.window_size)
    print(f"Test data: {len(windows)} windows of size {codec.config.window_size}")

    # Evaluate codec
    test_loader = DataLoader(WindowDataset(windows), batch_size=codec.config.batch_size)
    eval_metrics = codec.trainer.evaluate(test_loader)

    # Run baselines
    baselines = run_all_baselines(windows)

    results = {
        "codec": eval_metrics,
        "baselines": baselines,
    }

    # Print summary
    print("\n=== UMC Benchmark Results ===")
    print(f"  RMSE:            {eval_metrics['rmse']:.6f}")
    print(f"  RMSE % of range: {eval_metrics['rmse_pct_of_range']:.4f}%")
    print(f"  Active dims:     {eval_metrics['active_dims']} / {eval_metrics['max_latent_dim']}")
    print(f"  Compression:     {eval_metrics['compression_ratio']:.1f}x")
    print(f"  Encode:          {eval_metrics['encode_throughput_candles_per_sec']:,.0f} candles/sec")
    print(f"  Decode:          {eval_metrics['decode_throughput_candles_per_sec']:,.0f} candles/sec")
    print("\n  Baselines:")
    for b in baselines:
        if "ratio" in b:
            print(f"    {b['method']}: {b['ratio']:.1f}x")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


def _cmd_tryit(args):
    """Benchmark UMC on user's data — compare all modes + competitors."""
    import time
    import numpy as np
    from pathlib import Path

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {args.input}")
        return

    # Load data
    raw_file_bytes = input_path.stat().st_size
    suffix = input_path.suffix.lower()
    is_raw_binary = False

    try:
        if suffix == ".npy":
            data = np.load(str(input_path))
        elif suffix in (".csv", ".tsv", ".txt"):
            import pandas as pd
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(str(input_path), sep=sep)
            data = df.select_dtypes(include=[np.number]).values.astype(np.float32)
        elif suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(str(input_path))
            data = df.select_dtypes(include=[np.number]).values.astype(np.float32)
        else:
            is_raw_binary = True
            raw_bytes = input_path.read_bytes()
    except Exception as e:
        print(f"Could not load as numeric, treating as raw binary: {e}")
        is_raw_binary = True
        raw_bytes = input_path.read_bytes()

    if not is_raw_binary:
        arr = np.asarray(data, dtype=np.float32)
        raw_size = arr.nbytes
        print(f"\nFile: {input_path.name}")
        print(f"Shape: {arr.shape}, dtype: float32, raw size: {raw_size:,} bytes")
    else:
        raw_size = len(raw_bytes)
        print(f"\nFile: {input_path.name}")
        print(f"Binary file, raw size: {raw_size:,} bytes")

    print(f"File size on disk: {raw_file_bytes:,} bytes\n")

    # ---- UMC modes ----
    umc_modes = [
        "lossless", "lossless_fast", "lossless_zstd", "lossless_lzma",
        "near_lossless", "near_lossless_turbo", "quantized_8",
        "optimal_fast",
    ]
    results = []

    if is_raw_binary:
        # For raw binary, only optimal compression makes sense
        from . import compress_raw
        print("=" * 70)
        print(f"{'Compressor':<25} {'Size':>10} {'Ratio':>8} {'Time':>8}")
        print("-" * 70)

        t0 = time.perf_counter()
        compressed = compress_raw(raw_bytes)
        elapsed = time.perf_counter() - t0
        ratio = raw_size / max(len(compressed), 1)
        results.append(("UMC optimal", len(compressed), ratio, elapsed, True))
        print(f"{'UMC optimal':<25} {len(compressed):>10,} {ratio:>7.2f}x {elapsed:>7.2f}s")
    else:
        from . import compress
        print("=" * 70)
        print(f"{'Compressor':<25} {'Size':>10} {'Ratio':>8} {'Speed':>10} {'Lossless':>9}")
        print("-" * 70)

        for mode in umc_modes:
            try:
                t0 = time.perf_counter()
                c = compress(arr, mode=mode)
                elapsed = time.perf_counter() - t0
                ratio = raw_size / max(len(c), 1)
                speed = raw_size / max(elapsed, 1e-9) / 1e6
                lossless = mode in ("lossless", "lossless_fast", "lossless_zstd",
                                    "lossless_lzma", "optimal_fast")
                tag = "yes" if lossless else "no"
                results.append((f"UMC {mode}", len(c), ratio, elapsed, lossless))
                print(f"{'UMC ' + mode:<25} {len(c):>10,} {ratio:>7.2f}x {speed:>8.1f} MB/s {tag:>9}")
            except Exception as e:
                print(f"{'UMC ' + mode:<25} {'SKIP':>10} — {e}")

    # ---- Standard competitors ----
    if is_raw_binary:
        test_bytes = raw_bytes
    else:
        test_bytes = arr.tobytes()

    competitors = []

    # gzip
    import zlib
    t0 = time.perf_counter()
    gz = zlib.compress(test_bytes, 9)
    elapsed = time.perf_counter() - t0
    ratio = raw_size / max(len(gz), 1)
    speed = raw_size / max(elapsed, 1e-9) / 1e6
    competitors.append(("gzip-9", len(gz), ratio, elapsed))
    print(f"{'gzip-9':<25} {len(gz):>10,} {ratio:>7.2f}x {speed:>8.1f} MB/s {'yes':>9}")

    # lzma
    import lzma
    t0 = time.perf_counter()
    lz = lzma.compress(test_bytes, preset=6)
    elapsed = time.perf_counter() - t0
    ratio = raw_size / max(len(lz), 1)
    speed = raw_size / max(elapsed, 1e-9) / 1e6
    competitors.append(("lzma-6", len(lz), ratio, elapsed))
    print(f"{'lzma-6':<25} {len(lz):>10,} {ratio:>7.2f}x {speed:>8.1f} MB/s {'yes':>9}")

    # zstd
    try:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=19)
        t0 = time.perf_counter()
        zs = cctx.compress(test_bytes)
        elapsed = time.perf_counter() - t0
        ratio = raw_size / max(len(zs), 1)
        speed = raw_size / max(elapsed, 1e-9) / 1e6
        competitors.append(("zstd-19", len(zs), ratio, elapsed))
        print(f"{'zstd-19':<25} {len(zs):>10,} {ratio:>7.2f}x {speed:>8.1f} MB/s {'yes':>9}")
    except ImportError:
        pass

    # brotli
    try:
        import brotli
        t0 = time.perf_counter()
        br = brotli.compress(test_bytes, quality=11)
        elapsed = time.perf_counter() - t0
        ratio = raw_size / max(len(br), 1)
        speed = raw_size / max(elapsed, 1e-9) / 1e6
        competitors.append(("brotli-11", len(br), ratio, elapsed))
        print(f"{'brotli-11':<25} {len(br):>10,} {ratio:>7.2f}x {speed:>8.1f} MB/s {'yes':>9}")
    except ImportError:
        pass

    print("=" * 70)

    # Find winner
    all_results = results + [(n, s, r, t, True) for n, s, r, t in competitors]
    if all_results:
        best = max(all_results, key=lambda x: x[2])
        print(f"\nWinner: {best[0]} ({best[2]:.2f}x)")

    # HTML report
    if args.html:
        from .benchmark_report import generate_report
        report_data = {
            "file": str(input_path),
            "file_size": raw_file_bytes,
            "raw_size": raw_size,
            "is_raw_binary": is_raw_binary,
            "results": [
                {"name": r[0], "size": r[1], "ratio": r[2], "time": r[3],
                 "lossless": r[4] if len(r) > 4 else True}
                for r in results
            ],
            "competitors": [
                {"name": c[0], "size": c[1], "ratio": c[2], "time": c[3]}
                for c in competitors
            ],
        }
        generate_report(report_data, args.html)
        print(f"\nHTML report saved to: {args.html}")


def _cmd_compress(args):
    from . import compress_file
    try:
        from .cli_formatting import console
        console.print(f"[bold]Compressing[/bold] {args.input} (mode={args.mode})...")
    except Exception:
        print(f"Compressing {args.input} (mode={args.mode})...")

    stats = compress_file(args.input, args.output, mode=args.mode)

    try:
        from .cli_formatting import console
        from rich.table import Table
        table = Table(title="Compression Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Input", stats["input_path"])
        table.add_row("Output", stats["output_path"])
        table.add_row("Shape", str(stats["original_shape"]))
        table.add_row("Raw Size", f"{stats['raw_bytes']:,} bytes")
        table.add_row("File Size", f"{stats['input_file_bytes']:,} bytes")
        table.add_row("Compressed", f"{stats['compressed_bytes']:,} bytes")
        table.add_row("Ratio (raw)", f"{stats['ratio']:.1f}x")
        table.add_row("Ratio (file)", f"{stats['file_ratio']:.1f}x")
        table.add_row("Mode", stats["mode"])
        table.add_row("Time", f"{stats['elapsed_sec']*1000:.0f}ms")
        console.print(table)
    except Exception:
        print(f"  Raw: {stats['raw_bytes']:,} -> {stats['compressed_bytes']:,} bytes "
              f"({stats['ratio']:.1f}x)")
        print(f"  Saved to {stats['output_path']}")


def _cmd_decompress_file(args):
    from . import decompress_file

    try:
        from .cli_formatting import console
        console.print(f"[bold]Decompressing[/bold] {args.input}...")
    except Exception:
        print(f"Decompressing {args.input}...")

    stats = decompress_file(args.input, args.output)

    try:
        from .cli_formatting import console
        console.print(f"  Shape: {stats['shape']}, dtype: {stats['dtype']}")
        console.print(f"  Saved to {stats['output_path']} ({stats['elapsed_sec']*1000:.0f}ms)")
    except Exception:
        print(f"  Shape: {stats['shape']}, dtype: {stats['dtype']}")
        print(f"  Saved to {stats['output_path']}")


def _cmd_info(args):
    from pathlib import Path
    import struct

    path = Path(args.input)
    if not path.exists():
        print(f"Error: file not found: {args.input}")
        return

    data = path.read_bytes()
    file_size = len(data)

    if data[:4] == b"UMCZ":
        # UMC compressed stream
        offset = 4
        ndim = struct.unpack("<B", data[offset:offset + 1])[0]
        offset += 1
        shape = []
        for _ in range(ndim):
            dim = struct.unpack("<I", data[offset:offset + 4])[0]
            shape.append(dim)
            offset += 4

        # Read storage tag
        storage_tag = data[offset:offset + 1]
        tag_map = {
            b'\x01': 'lossless', b'\x02': 'near_lossless',
            b'\x03': 'lossless_zstd', b'\x04': 'lossless_lzma',
            b'\x05': 'normalized_lossless', b'\x06': 'normalized_lossless_zstd',
            b'\x07': 'near_lossless_turbo', b'\x08': 'quantized_8',
            b'\x09': 'optimal', b'\x0a': 'lossless_fast',
        }
        mode = tag_map.get(storage_tag, 'unknown')

        import numpy as np
        raw_bytes = int(np.prod(shape)) * 4

        try:
            from .cli_formatting import console
            from rich.table import Table
            table = Table(title=f"UMC File Info: {path.name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Format", "UMC Compressed Stream (UMCZ)")
            table.add_row("Original Shape", str(tuple(shape)))
            table.add_row("Storage Mode", mode)
            table.add_row("File Size", f"{file_size:,} bytes")
            table.add_row("Raw Size (est)", f"{raw_bytes:,} bytes")
            table.add_row("Ratio", f"{raw_bytes / max(file_size, 1):.1f}x")
            console.print(table)
        except Exception:
            print(f"Format: UMC Compressed Stream (UMCZ)")
            print(f"Shape:  {tuple(shape)}")
            print(f"Mode:   {mode}")
            print(f"Size:   {file_size:,} bytes (raw: {raw_bytes:,}, ratio: {raw_bytes / max(file_size, 1):.1f}x)")
    else:
        print(f"Unknown file format (magic: {data[:4]!r})")


def _cmd_dashboard(args):
    import subprocess
    import sys
    print("Launching UMC Dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    "umc/dashboard.py", "--theme.base=dark"])


if __name__ == "__main__":
    main()
