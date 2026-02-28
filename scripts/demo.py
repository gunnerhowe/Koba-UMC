#!/usr/bin/env python3
"""Interactive demo of the Universal Manifold Codec."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="UMC Interactive Demo")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--codec", type=str, default=None, help="Path to pretrained codec")
    parser.add_argument("--quick", action="store_true", help="Quick demo with synthetic data")
    args = parser.parse_args()

    if args.quick:
        run_synthetic_demo()
    else:
        run_financial_demo(args.codec)


def run_synthetic_demo():
    """Quick demo using synthetic data (no downloads needed)."""
    from umc.config import UMCConfig
    from umc.data.synthetic import SyntheticManifoldGenerator
    from umc.encoder.vae import VAEEncoder
    from umc.decoder.mlp_decoder import MLPDecoder
    from umc.training.trainer import UMCTrainer
    from umc.processor.cluster import ManifoldCluster
    from umc.storage.mnf_format import MNFWriter, MNFReader

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    print("=== UMC Synthetic Demo ===\n")

    # Generate synthetic data
    config = UMCConfig(
        window_size=64,
        max_latent_dim=32,
        epochs=30,
        batch_size=128,
        encoder_hidden=(128, 64),
        decoder_hidden=(64, 128),
    )

    print("Generating synthetic financial data (intrinsic dim = 10)...")
    gen = SyntheticManifoldGenerator(seed=42)
    data, true_dim = gen.generate_financial_synthetic(
        n_samples=5000,
        intrinsic_dim=10,
        window_size=config.window_size,
        n_features=config.n_features,
    )
    print(f"  Shape: {data.shape}, True intrinsic dim: {true_dim}\n")

    # Train
    print("Training encoder-decoder...")
    encoder = VAEEncoder(config)
    decoder = MLPDecoder(config)

    tensor_data = torch.from_numpy(data).float()
    n_train = int(0.8 * len(tensor_data))
    train_loader = DataLoader(
        TensorDataset(tensor_data[:n_train]),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(tensor_data[n_train:]),
        batch_size=config.batch_size,
    )

    # Wrap datasets to return just the tensor (not a tuple)
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, tensors):
            self.tensors = tensors
        def __len__(self):
            return len(self.tensors)
        def __getitem__(self, idx):
            return self.tensors[idx]

    train_loader = DataLoader(SimpleDataset(tensor_data[:n_train]),
                              batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(SimpleDataset(tensor_data[n_train:]),
                            batch_size=config.batch_size)

    trainer = UMCTrainer(encoder, decoder, config)
    history = trainer.train(train_loader, val_loader, verbose=True)

    # Encode
    print("\nEncoding all data...")
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        x = tensor_data.to(trainer.device)
        enc_result = encoder.encode(x)
        z = enc_result.z.cpu().numpy()
        x_hat = decoder.decode(enc_result.z, enc_result.chart_id).cpu().numpy()

    # Metrics
    rmse = np.sqrt(np.mean((data - x_hat) ** 2))
    data_range = data.max() - data.min()
    rmse_pct = (rmse / (data_range + 1e-8)) * 100

    var_per_dim = np.var(z, axis=0)
    sorted_var = np.sort(var_per_dim)[::-1]
    # Active dims: use relative threshold (1% of max variance)
    var_threshold = sorted_var[0] * 0.01 if sorted_var[0] > 0 else 1e-6
    active_dims = int(np.sum(var_per_dim > var_threshold))
    if active_dims == 0:
        active_dims = config.max_latent_dim  # Fallback: all dims active

    raw_bytes = data.nbytes
    # Compression: coords stored as float16
    coord_bytes_active = z.shape[0] * active_dims * 2
    coord_bytes_full = z.shape[0] * config.max_latent_dim * 2
    ratio_active = raw_bytes / coord_bytes_active
    ratio_full = raw_bytes / coord_bytes_full

    print(f"\n=== Results ===")
    print(f"Reconstruction RMSE: {rmse:.6f} ({rmse_pct:.4f}% of data range)")
    print(f"Active dimensions: {active_dims} / {config.max_latent_dim}")
    print(f"Top 5 dim variances: {sorted_var[:5]}")
    print(f"Raw data: {raw_bytes:,} bytes ({raw_bytes/1024/1024:.1f} MB)")
    print(f"Manifold coords (active dims): {coord_bytes_active:,} bytes")
    print(f"Manifold coords (all dims): {coord_bytes_full:,} bytes")
    print(f"Compression ratio (active): {ratio_active:.1f}x")
    print(f"Compression ratio (full latent): {ratio_full:.1f}x")

    # Clustering
    print(f"\nClustering in manifold space...")
    clusterer = ManifoldCluster()
    clusters = clusterer.cluster(z, n_clusters=5)
    print(f"  {clusters.n_clusters} clusters found, inertia={clusters.inertia:.2f}")

    # Save .mnf
    mnf_path = "demo_output.mnf"
    writer = MNFWriter()
    from umc.storage.manifest import DecoderManifest
    decoder_hash = DecoderManifest.compute_hash_bytes(decoder.state_dict())
    bytes_written = writer.write(
        mnf_path,
        coordinates=z,
        chart_ids=enc_result.chart_id.cpu().numpy().astype(np.uint8),
        decoder_hash=decoder_hash,
        confidences=enc_result.confidence.cpu().numpy(),
    )
    print(f"\n.mnf file saved: {mnf_path} ({bytes_written:,} bytes)")

    # Read it back
    reader = MNFReader()
    mnf = reader.read(mnf_path)
    print(f"Read back: {mnf.header.n_samples} samples, dim={mnf.header.latent_dim}")
    print(f"Coordinates match: {np.allclose(z.astype(np.float16), mnf.coordinates, atol=1e-3)}")

    print("\n=== Demo Complete ===")


def run_financial_demo(codec_path=None):
    """Demo with real financial data."""
    from umc.config import UMCConfig
    from umc import ManifoldCodec
    from umc.data.loaders import load_yahoo_finance, combine_datasets

    print("=== UMC Financial Demo ===\n")

    if codec_path and Path(codec_path).exists():
        print(f"Loading pretrained codec from {codec_path}...")
        codec = ManifoldCodec.from_pretrained(codec_path)
    else:
        print("Training a new codec (this may take a few minutes)...")
        config = UMCConfig(epochs=50, max_latent_dim=64)
        codec = ManifoldCodec(config)

        symbols = ["SPY", "AAPL", "MSFT"]
        datasets = load_yahoo_finance(symbols, period="2y")
        df = combine_datasets(datasets)
        print(f"  Loaded {len(df)} rows from {len(datasets)} symbols\n")

        codec.fit(df, verbose=True)
        codec.save("checkpoints/demo_codec")
        print(f"\n  Codec saved to checkpoints/demo_codec\n")

        # Encode
        mnf = codec.encode(df)
        print(f"Encoded: {mnf.n_samples} windows, latent_dim={mnf.latent_dim}")

        # Save
        bytes_written = mnf.save("demo_financial.mnf")
        raw_bytes = len(df.to_csv().encode())
        print(f".mnf size: {bytes_written:,} bytes")
        print(f"Raw CSV: {raw_bytes:,} bytes")
        print(f"Compression ratio: {raw_bytes / bytes_written:.1f}x")

        # Cluster
        clusters = mnf.cluster(n_clusters=5)
        print(f"\nMarket regimes: {clusters.n_clusters} clusters detected")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
