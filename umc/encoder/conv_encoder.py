"""1D-CNN encoder for financial time series — captures temporal patterns."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import UMCConfig
from .base import BaseEncoder, EncodingResult


class ConvResBlock(nn.Module):
    """1D convolutional residual block with batch normalization and dropout."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class ConvEncoder(BaseEncoder):
    """1D-CNN VAE encoder that preserves temporal structure.

    Supports two bottleneck modes:
      - FC bottleneck (default): AdaptivePool -> Flatten -> FC -> Latent
      - Conv bottleneck (conv_bottleneck=True): Conv1x1 channel reduction ->
        Flatten -> small FC -> Latent. Much fewer params, preserves all
        temporal positions.

    Adapts depth based on window_size:
      - window_size <= 64:  2 downsampling blocks (channels: 32->64->128)
      - window_size > 64:   3 downsampling blocks (channels: 32->64->128->256)
    """

    def __init__(self, config: UMCConfig):
        super().__init__()
        self.config = config
        n_features = config.n_features
        latent_dim = config.max_latent_dim

        # Adaptive depth based on window size
        if config.window_size > 64:
            channels = [32, 64, 128, 256]
        else:
            channels = [32, 64, 128]

        # Initial projection from n_features channels
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_features, channels[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
        )

        # Downsampling blocks: each halves temporal dimension
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_blocks.append(nn.Sequential(
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(channels[i + 1]),
                nn.GELU(),
                ConvResBlock(channels[i + 1], dropout=0.15),
            ))

        n_downsample = len(self.down_blocks)
        compressed_len = config.window_size // (2 ** n_downsample)
        self._use_conv_bottleneck = config.conv_bottleneck

        if config.conv_bottleneck:
            # Conv-bottleneck: 1x1 conv reduces channels, preserves ALL temporal positions
            c_red = config.conv_bottleneck_channels
            self.channel_reduce = nn.Sequential(
                nn.Conv1d(channels[-1], c_red, kernel_size=1),
                nn.BatchNorm1d(c_red),
                nn.GELU(),
            )
            flat_dim = c_red * compressed_len
            # Small FC to latent (much smaller than FC-bottleneck)
            self.fc_mu = nn.Linear(flat_dim, latent_dim)
            self.fc_log_var = nn.Linear(flat_dim, latent_dim)
            # Chart/confidence heads from same flat features
            head_dim = flat_dim
        else:
            # FC-bottleneck: adaptive pool -> flatten -> FC
            if config.pool_size_cap > 0:
                pool_size = min(compressed_len, config.pool_size_cap)
            else:
                pool_size = compressed_len
            self.adaptive_pool = nn.AdaptiveAvgPool1d(pool_size)
            flat_dim = channels[-1] * pool_size

            bottleneck_dim = config.bottleneck_dim
            self.bottleneck = nn.Sequential(
                nn.Linear(flat_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(config.bottleneck_dropout),
            )
            self.fc_mu = nn.Linear(bottleneck_dim, latent_dim)
            self.fc_log_var = nn.Linear(bottleneck_dim, latent_dim)
            head_dim = bottleneck_dim

        # Chart head
        self.chart_head = nn.Linear(head_dim, config.num_charts)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(head_dim, 1),
            nn.Sigmoid(),
        )

        # Gumbel-Softmax temperature (annealed during training)
        self.gumbel_temperature = 1.0
        self._active_dim_threshold = 0.01

    def _encode_features(self, x: Tensor) -> Tensor:
        """Shared forward pass through conv blocks to bottleneck features."""
        h = x.transpose(1, 2)
        h = self.input_proj(h)
        for block in self.down_blocks:
            h = block(h)

        if self._use_conv_bottleneck:
            h = self.channel_reduce(h)
            h = h.reshape(h.shape[0], -1)
        else:
            h = self.adaptive_pool(h)
            h = h.reshape(h.shape[0], -1)
            h = self.bottleneck(h)
        return h

    def encode(self, x: Tensor) -> EncodingResult:
        """Encode windowed OHLCV data using temporal convolutions.

        Args:
            x: (batch, window_size, n_features)

        Returns:
            EncodingResult
        """
        h = self._encode_features(x)

        # Latent distribution
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)

        # Reparameterization
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        # Chart ID with Gumbel-Softmax
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

        # Active dimensions
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
        return self.config.max_latent_dim

    def get_chart_logits(self, x: Tensor) -> Tensor:
        """Return raw chart logits."""
        h = self._encode_features(x)
        return self.chart_head(h)
