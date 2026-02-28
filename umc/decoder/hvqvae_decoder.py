"""Hierarchical VQ-VAE Decoder with Transformer backbone.

Architecture:
    Top VQ code (broadcast) + Bottom VQ codes → TransformerDecoder → Unpatchify → RevIN.inverse
    Also supports decode from continuous z (for .mnf inference).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from ..config import UMCConfig
from .base import BaseDecoder


class TransformerTSBlock(nn.Module):
    """Pre-norm transformer block (same architecture as encoder)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class HVQVAEDecoder(BaseDecoder):
    """Hierarchical VQ-VAE decoder conforming to BaseDecoder interface.

    Two decode paths:
        1. decode_from_codes(): Direct VQ path (training + high-fidelity inference)
           Input: top_quantized + bottom_quantized + RevIN stats
        2. decode(): From continuous z (BaseDecoder interface, .mnf inference)
           Projects z back to approximate VQ codes, then decodes
    """

    def __init__(self, config: UMCConfig):
        super().__init__()
        self.config = config

        n_features = config.n_features
        d_model = config.d_model
        n_patches = config.window_size // config.patch_size
        vq_dim = config.vq_dim
        latent_dim = config.max_latent_dim

        self.n_patches = n_patches
        self._use_grad_checkpoint = False

        # Project VQ codes back to d_model for transformer
        # Input per patch: top_code (broadcast) + bottom_code = 2 * vq_dim
        self.code_proj = nn.Sequential(
            nn.Linear(2 * vq_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Learnable positional encoding for decoder
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        # Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerTSBlock(d_model, config.n_heads, config.d_ff, config.transformer_dropout)
            for _ in range(config.n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Unpatchify: project patches back to temporal signal
        self.unpatchify = nn.ConvTranspose1d(
            d_model, n_features,
            kernel_size=config.patch_size, stride=config.patch_size,
        )

        # Chart embedding (for decode() interface compatibility)
        self.chart_embedding = nn.Embedding(config.num_charts, config.chart_embedding_dim)

        # z → approximate VQ codes path (for .mnf decode)
        self.z_to_codes = nn.Sequential(
            nn.Linear(latent_dim + config.chart_embedding_dim, 2 * vq_dim * 2),
            nn.GELU(),
            nn.Linear(2 * vq_dim * 2, (1 + n_patches) * vq_dim),
        )

    def decode_from_codes(
        self,
        top_quantized: Tensor,
        bottom_quantized: Tensor,
    ) -> Tensor:
        """Decode from VQ codes directly (used during training).

        Args:
            top_quantized: (batch, vq_dim) — top level code
            bottom_quantized: (batch, n_patches, vq_dim) — bottom level codes

        Returns:
            x_hat: (batch, window_size, n_features) — before RevIN inverse
        """
        B = top_quantized.shape[0]

        # Broadcast top code to all patches and concatenate with bottom
        top_broadcast = top_quantized.unsqueeze(1).expand(-1, self.n_patches, -1)
        decoder_input = torch.cat([top_broadcast, bottom_quantized], dim=-1)  # (B, P, 2*vq_dim)

        # Project to d_model
        h = self.code_proj(decoder_input) + self.pos_embed  # (B, P, d_model)

        # Transformer decoder
        for block in self.decoder_blocks:
            if self._use_grad_checkpoint and self.training:
                h = grad_checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        h = self.decoder_norm(h)

        # Unpatchify: (B, P, d_model) → (B, d_model, P) → ConvTranspose1d → (B, n_features, window_size)
        h = h.transpose(1, 2)
        h = self.unpatchify(h)

        # (B, n_features, window_size) → (B, window_size, n_features)
        x_hat = h.transpose(1, 2)
        return x_hat

    def decode(self, z: Tensor, chart_id: Tensor) -> Tensor:
        """Decode from continuous z (BaseDecoder interface, .mnf inference).

        Args:
            z: (batch, latent_dim) — continuous manifold coordinates
            chart_id: (batch,) — chart identifier

        Returns:
            x_hat: (batch, window_size, n_features)
        """
        B = z.shape[0]
        vq_dim = self.config.vq_dim

        # Concatenate z with chart embedding
        chart_emb = self.chart_embedding(chart_id)
        z_input = torch.cat([z, chart_emb], dim=-1)

        # Project to approximate VQ codes
        codes_flat = self.z_to_codes(z_input)  # (B, (1 + n_patches) * vq_dim)
        codes_flat = codes_flat.reshape(B, 1 + self.n_patches, vq_dim)

        top_approx = codes_flat[:, 0, :]  # (B, vq_dim)
        bottom_approx = codes_flat[:, 1:, :]  # (B, n_patches, vq_dim)

        return self.decode_from_codes(top_approx, bottom_approx)
