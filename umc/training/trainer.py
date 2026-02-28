"""Training loop for UMC encoder-decoder pairs."""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import UMCConfig
from ..encoder.base import BaseEncoder
from ..decoder.base import BaseDecoder
from .losses import total_loss, vq_total_loss, kl_divergence_per_dim
from .scheduler import CosineWarmupScheduler


class UMCTrainer:
    """Trains an encoder-decoder pair for manifold-native representation."""

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        config: UMCConfig,
        device: Optional[str] = None,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # Combine parameters for joint optimization
        self.optimizer = Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=config.learning_rate,
        )
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=5,
            max_epochs=config.epochs,
        )

        self.history: list[dict] = []
        self.best_val_loss = float("inf")

        # Detect VQ-VAE mode
        self._is_vqvae = config.encoder_type == "hvqvae"

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> list[dict]:
        """Full training loop with early stopping.

        Args:
            train_loader: Training data loader (batches of windows).
            val_loader: Validation data loader.
            verbose: Print progress.

        Returns:
            List of per-epoch metrics dicts.
        """
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Anneal Gumbel-Softmax temperature (1.0 -> 0.1 over training)
            if hasattr(self.encoder, 'gumbel_temperature'):
                progress = epoch / max(1, self.config.epochs - 1)
                self.encoder.gumbel_temperature = max(0.1, 1.0 - 0.9 * progress)

            # --- Train ---
            self.encoder.train()
            self.decoder.train()
            train_metrics = self._run_epoch(train_loader, epoch, train=True)

            # VQ-VAE: periodic dead code reset
            if self._is_vqvae and epoch > 0 and epoch % 50 == 0:
                if hasattr(self.encoder, 'reset_dead_codes'):
                    n_top, n_bottom = self.encoder.reset_dead_codes()
                    if verbose and (n_top > 0 or n_bottom > 0):
                        print(f"  Dead code reset: {n_top} top, {n_bottom} bottom codes reinitialized")

            # --- Validate ---
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                val_metrics = self._run_epoch(val_loader, epoch, train=False)

            self.scheduler.step()

            # Record
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": self.scheduler.get_last_lr()[0],
            }
            self.history.append(record)

            # Early stopping on val total loss (accounts for beta changes)
            val_total = val_metrics["total"]
            val_recon = val_metrics["reconstruction"]
            if val_total < self.best_val_loss:
                self.best_val_loss = val_total
                patience_counter = 0
                self._save_checkpoint("best")
            else:
                patience_counter += 1

            if verbose and (epoch % 10 == 0 or epoch == self.config.epochs - 1):
                if self._is_vqvae:
                    vq_commit = train_metrics.get('vq_commitment', 0.0)
                    top_ppl = train_metrics.get('top_perplexity', 0.0)
                    bot_ppl = train_metrics.get('bottom_perplexity', 0.0)
                    print(
                        f"Epoch {epoch:4d} | "
                        f"Train Loss: {train_metrics['total']:.6f} | "
                        f"Val Recon: {val_recon:.6f} | "
                        f"Val Total: {val_total:.6f} | "
                        f"VQ: {vq_commit:.4f} | "
                        f"Perplexity: {top_ppl:.0f}/{bot_ppl:.0f} | "
                        f"Active Dims: {train_metrics['active_dims']:.0f}"
                    )
                else:
                    print(
                        f"Epoch {epoch:4d} | "
                        f"Train Loss: {train_metrics['total']:.6f} | "
                        f"Val Recon: {val_recon:.6f} | "
                        f"Val Total: {val_total:.6f} | "
                        f"Active Dims: {train_metrics['active_dims']:.0f} | "
                        f"Beta: {train_metrics['beta']:.4f}"
                    )

            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        return self.history

    def _run_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        train: bool,
    ) -> dict:
        """Run one epoch of training or validation."""
        if self._is_vqvae:
            return self._run_epoch_vqvae(loader, epoch, train)
        return self._run_epoch_vae(loader, epoch, train)

    def _run_epoch_vae(
        self,
        loader: DataLoader,
        epoch: int,
        train: bool,
    ) -> dict:
        """Standard VAE epoch."""
        total_metrics = {
            "reconstruction": 0.0,
            "kl": 0.0,
            "sparsity": 0.0,
            "smoothness": 0.0,
            "multiscale": 0.0,
            "spectral": 0.0,
            "total": 0.0,
            "beta": 0.0,
            "active_dims": 0,
        }
        n_batches = 0

        for batch in loader:
            # Handle WindowDataset with per-window scale factors (returns tuple)
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)

            # Forward pass
            enc_result = self.encoder.encode(x)
            x_hat = self.decoder.decode(enc_result.z, enc_result.chart_id)

            # Compute loss
            loss, components = total_loss(
                x, x_hat, enc_result.mu, enc_result.log_var,
                enc_result.z, self.config, epoch,
            )

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    max_norm=1.0,
                )
                self.optimizer.step()

            # Accumulate metrics
            for k, v in components.items():
                total_metrics[k] += v
            total_metrics["active_dims"] += enc_result.active_dims
            n_batches += 1

        # Average
        if n_batches > 0:
            for k in total_metrics:
                total_metrics[k] /= n_batches

        return total_metrics

    def _run_epoch_vqvae(
        self,
        loader: DataLoader,
        epoch: int,
        train: bool,
    ) -> dict:
        """VQ-VAE epoch with direct code decode path."""
        total_metrics = {
            "reconstruction": 0.0,
            "vq_commitment": 0.0,
            "sparsity": 0.0,
            "smoothness": 0.0,
            "multiscale": 0.0,
            "spectral": 0.0,
            "total": 0.0,
            "active_dims": 0,
            "top_perplexity": 0.0,
            "bottom_perplexity": 0.0,
        }
        n_batches = 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)

            # Forward: encoder produces VQ codes + continuous z
            enc_result = self.encoder.encode(x)

            # Decode via direct VQ code path (higher fidelity during training)
            if hasattr(self.decoder, 'decode_from_codes') and hasattr(self.encoder, '_last_top_quantized'):
                x_hat_raw = self.decoder.decode_from_codes(
                    self.encoder._last_top_quantized,
                    self.encoder._last_bottom_quantized,
                )
                # Apply RevIN inverse
                x_hat = self.encoder.revin.inverse(x_hat_raw)
            else:
                # Fallback to standard decode path
                x_hat = self.decoder.decode(enc_result.z, enc_result.chart_id)

            # VQ loss
            loss, components = vq_total_loss(
                x, x_hat, enc_result.z,
                self.encoder.vq_loss, self.config,
            )

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    max_norm=1.0,
                )
                self.optimizer.step()

            # Accumulate metrics
            for k, v in components.items():
                total_metrics[k] += v
            total_metrics["active_dims"] += enc_result.active_dims
            if hasattr(self.encoder, 'top_perplexity'):
                total_metrics["top_perplexity"] += self.encoder.top_perplexity
                total_metrics["bottom_perplexity"] += self.encoder.bottom_perplexity
            n_batches += 1

        # Average
        if n_batches > 0:
            for k in total_metrics:
                total_metrics[k] /= n_batches

        return total_metrics

    def evaluate(self, test_loader: DataLoader) -> dict:
        """Full evaluation on test set.

        Returns:
            Dict with reconstruction RMSE, compression metrics, throughput, etc.
        """
        self.encoder.eval()
        self.decoder.eval()

        all_originals = []
        all_reconstructed = []
        all_latents = []
        total_encode_time = 0.0
        total_decode_time = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)

                # Encode timing
                t0 = time.perf_counter()
                enc_result = self.encoder.encode(x)
                t1 = time.perf_counter()
                total_encode_time += t1 - t0

                # Decode timing
                t0 = time.perf_counter()
                x_hat = self.decoder.decode(enc_result.z, enc_result.chart_id)
                t1 = time.perf_counter()
                total_decode_time += t1 - t0

                all_originals.append(x.cpu().numpy())
                all_reconstructed.append(x_hat.cpu().numpy())
                all_latents.append(enc_result.z.cpu().numpy())
                total_samples += x.shape[0]

        originals = np.concatenate(all_originals, axis=0)
        reconstructed = np.concatenate(all_reconstructed, axis=0)
        latents = np.concatenate(all_latents, axis=0)

        # Reconstruction RMSE
        rmse = np.sqrt(np.mean((originals - reconstructed) ** 2))
        price_range = originals.max() - originals.min()
        rmse_pct = (rmse / (price_range + 1e-8)) * 100

        # Effective dimensionality (dims with variance above threshold)
        latent_var = np.var(latents, axis=0)
        active_dims = int(np.sum(latent_var > 0.01))

        # Throughput
        candles_per_window = self.config.window_size
        encode_throughput = (total_samples * candles_per_window) / (total_encode_time + 1e-8)
        decode_throughput = (total_samples * candles_per_window) / (total_decode_time + 1e-8)

        # Compression ratio (approximate)
        raw_bytes_per_window = self.config.window_size * self.config.n_features * 4  # float32
        coord_bytes_per_window = active_dims * 2  # float16
        compression_ratio = raw_bytes_per_window / (coord_bytes_per_window + 1e-8)

        return {
            "rmse": float(rmse),
            "rmse_pct_of_range": float(rmse_pct),
            "active_dims": active_dims,
            "max_latent_dim": self.config.max_latent_dim,
            "compression_ratio": float(compression_ratio),
            "encode_throughput_candles_per_sec": float(encode_throughput),
            "decode_throughput_candles_per_sec": float(decode_throughput),
            "total_test_samples": total_samples,
        }

    def _save_checkpoint(self, tag: str) -> None:
        """Save encoder and decoder weights."""
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "config": self.config,
                "best_val_loss": self.best_val_loss,
            },
            ckpt_dir / f"umc_{tag}.pt",
        )

    def load_checkpoint(self, tag: str) -> None:
        """Load encoder and decoder weights."""
        ckpt_path = Path(self.config.checkpoint_dir) / f"umc_{tag}.pt"
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
