"""MLP-based decoder for Phase 1 (fast, simple)."""

import torch
import torch.nn as nn
from torch import Tensor

from ..config import UMCConfig
from .base import BaseDecoder


class MLPDecoder(BaseDecoder):
    """MLP decoder that maps manifold coordinates back to OHLCV windows."""

    def __init__(self, config: UMCConfig):
        super().__init__()
        self.config = config
        output_dim = config.input_dim  # window_size * n_features

        # Chart embedding
        self.chart_embedding = nn.Embedding(
            config.num_charts, config.chart_embedding_dim
        )

        # Input dimension: latent coords + chart embedding
        in_dim = config.max_latent_dim + config.chart_embedding_dim

        # Build hidden layers (mirror of encoder)
        layers = []
        prev_dim = in_dim
        for hidden_dim in config.decoder_hidden:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def decode(self, z: Tensor, chart_id: Tensor) -> Tensor:
        """Decode manifold coordinates to OHLCV windows.

        Args:
            z: (batch, latent_dim)
            chart_id: (batch,) int tensor

        Returns:
            x_hat: (batch, window_size, n_features)
        """
        # Get chart embeddings
        chart_emb = self.chart_embedding(chart_id)  # (batch, chart_embedding_dim)

        # Concatenate coordinates with chart embedding
        decoder_input = torch.cat([z, chart_emb], dim=-1)

        # Forward through network
        flat_output = self.network(decoder_input)

        # Reshape to (batch, window_size, n_features)
        x_hat = flat_output.reshape(
            -1, self.config.window_size, self.config.n_features
        )
        return x_hat
