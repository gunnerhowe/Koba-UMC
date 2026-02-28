"""1D-CNN decoder that reconstructs temporal structure from manifold coordinates."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import UMCConfig
from .base import BaseDecoder


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


class ConvDecoder(BaseDecoder):
    """1D-CNN decoder that mirrors the ConvEncoder architecture.

    Supports two bottleneck modes to match the encoder:
      - FC bottleneck (default): Latent -> FC -> Reshape -> Upsample
      - Conv bottleneck: Latent -> small FC -> Reshape -> Conv1x1 expand -> Upsample
    """

    def __init__(self, config: UMCConfig):
        super().__init__()
        self.config = config
        n_features = config.n_features
        latent_dim = config.max_latent_dim

        # Mirror encoder's channel progression (reversed)
        if config.window_size > 64:
            channels = [256, 128, 64, 32]
        else:
            channels = [128, 64, 32]

        n_upsample = len(channels) - 1
        self.init_channels = channels[0]
        n_upsample_blocks = len(channels) - 1
        compressed_len = config.window_size // (2 ** n_upsample_blocks)

        # Chart embedding
        self.chart_embedding = nn.Embedding(
            config.num_charts, config.chart_embedding_dim
        )

        in_dim = latent_dim + config.chart_embedding_dim
        self._use_conv_bottleneck = config.conv_bottleneck

        if config.conv_bottleneck:
            # Conv-bottleneck: small FC -> reshape -> Conv1x1 channel expansion
            c_red = config.conv_bottleneck_channels
            self.init_len = compressed_len  # No pooling — use full compressed length
            self._init_c_red = c_red
            flat_dim = c_red * self.init_len

            self.fc_project = nn.Sequential(
                nn.Linear(in_dim, flat_dim),
                nn.GELU(),
            )
            # Expand channels from reduced to full
            self.channel_expand = nn.Sequential(
                nn.Conv1d(c_red, channels[0], kernel_size=1),
                nn.BatchNorm1d(channels[0]),
                nn.GELU(),
            )
        else:
            # FC-bottleneck: standard approach
            if config.pool_size_cap > 0:
                self.init_len = min(compressed_len, config.pool_size_cap)
            else:
                self.init_len = compressed_len

            flat_dim = channels[0] * self.init_len
            bottleneck_dim = config.bottleneck_dim
            self.fc_project = nn.Sequential(
                nn.Linear(in_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(config.bottleneck_dropout),
                nn.Linear(bottleneck_dim, flat_dim),
                nn.GELU(),
            )

        # Upsampling blocks: each doubles temporal dimension
        self.up_blocks = nn.ModuleList()
        for i in range(n_upsample):
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose1d(
                    channels[i], channels[i + 1],
                    kernel_size=4, stride=2, padding=1,
                ),
                nn.BatchNorm1d(channels[i + 1]),
                nn.GELU(),
                ConvResBlock(channels[i + 1], dropout=0.15),
            ))

        self.upsampled_len = self.init_len * (2 ** n_upsample)

        # Final projection to n_features channels + refinement
        self.output_proj = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1], kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(channels[-1], n_features, kernel_size=1),
        )

    def decode(self, z: Tensor, chart_id: Tensor) -> Tensor:
        """Decode manifold coordinates to OHLCV windows.

        Args:
            z: (batch, latent_dim)
            chart_id: (batch,) int tensor

        Returns:
            x_hat: (batch, window_size, n_features)
        """
        # Concatenate latent coords with chart embedding
        chart_emb = self.chart_embedding(chart_id)
        decoder_input = torch.cat([z, chart_emb], dim=-1)

        # Project to initial feature map
        h = self.fc_project(decoder_input)

        if self._use_conv_bottleneck:
            h = h.reshape(-1, self._init_c_red, self.init_len)
            h = self.channel_expand(h)
        else:
            h = h.reshape(-1, self.init_channels, self.init_len)

        # Progressive upsampling via transposed convolutions
        for block in self.up_blocks:
            h = block(h)

        # Interpolate to exact window_size if needed
        if self.upsampled_len != self.config.window_size:
            h = F.interpolate(h, size=self.config.window_size, mode='linear', align_corners=False)

        # Final projection to n_features
        h = self.output_proj(h)

        # Transpose back to (batch, window_size, n_features)
        x_hat = h.transpose(1, 2)

        return x_hat
