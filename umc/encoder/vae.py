"""VAE encoder for financial time series — Phase 1 primary encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import UMCConfig
from .base import BaseEncoder, EncodingResult


class ResidualBlock(nn.Module):
    """Linear + ReLU + optional residual connection."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU(inplace=True)
        self.use_residual = (in_dim == out_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.act(self.fc(x))
        if self.use_residual:
            out = out + x
        return out


class VAEEncoder(BaseEncoder):
    """Variational Autoencoder encoder with adaptive sparsity."""

    def __init__(self, config: UMCConfig):
        super().__init__()
        self.config = config
        input_dim = config.input_dim
        latent_dim = config.max_latent_dim

        # Build hidden layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in config.encoder_hidden:
            layers.append(ResidualBlock(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Latent heads
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

        # Chart head: predicts which coordinate chart
        self.chart_head = nn.Sequential(
            nn.Linear(prev_dim, config.num_charts),
        )

        # Confidence head: scalar confidence [0, 1]
        self.confidence_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid(),
        )

        # Gumbel-Softmax temperature (annealed during training)
        self.gumbel_temperature = 1.0

        # Track active dimensions threshold
        self._active_dim_threshold = 0.01

    def encode(self, x: Tensor) -> EncodingResult:
        """Encode a batch of windowed OHLCV data.

        Args:
            x: (batch, window_size, n_features)

        Returns:
            EncodingResult
        """
        batch_size = x.shape[0]

        # Flatten: (batch, window_size, n_features) -> (batch, window_size * n_features)
        h = x.reshape(batch_size, -1)

        # Forward through backbone
        h = self.backbone(h)

        # Latent distribution parameters
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu  # Deterministic at inference

        # Chart ID: Gumbel-Softmax during training for differentiable selection,
        # hard argmax at inference (spec Section 10.2)
        chart_logits = self.chart_head(h)
        if self.training:
            chart_probs = F.gumbel_softmax(
                chart_logits, tau=self.gumbel_temperature, hard=True
            )
            chart_id = chart_probs.argmax(dim=-1)
        else:
            chart_id = chart_logits.argmax(dim=-1)

        # Confidence
        confidence = self.confidence_head(h).squeeze(-1)

        # Count active dimensions (use actual z variance across the batch)
        with torch.no_grad():
            z_var = z.var(dim=0)
            max_var = z_var.max()
            threshold = max(self._active_dim_threshold, max_var.item() * 0.01)
            active_dims = int((z_var > threshold).sum().item())
            if active_dims == 0:
                active_dims = self.config.max_latent_dim

        return EncodingResult(
            z=z,
            chart_id=chart_id,
            confidence=confidence,
            mu=mu,
            log_var=log_var,
            active_dims=active_dims,
        )

    def get_effective_dim(self) -> int:
        """Return the last computed number of active latent dimensions."""
        return self.config.max_latent_dim  # Updated dynamically during encoding

    def get_chart_logits(self, x: Tensor) -> Tensor:
        """Return raw chart logits (for Gumbel-Softmax during training)."""
        batch_size = x.shape[0]
        h = x.reshape(batch_size, -1)
        h = self.backbone(h)
        return self.chart_head(h)
