"""Hierarchical VQ-VAE Encoder with Transformer backbone.

Architecture:
    Input → RevIN → PatchEmbedding → TransformerEncoder → Two-level VQ
    - Top VQ: global shape (mean-pooled, single codebook)
    - Bottom VQ: local detail (per-patch, Residual VQ with N levels)
    - z_projection: continuous output for FAISS search / .mnf storage

Uses vector-quantize-pytorch library for battle-tested VQ implementation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from vector_quantize_pytorch import VectorQuantize, ResidualVQ as LibResidualVQ
from vector_quantize_pytorch import FSQ, ResidualFSQ

from ..config import UMCConfig
from .base import BaseEncoder, EncodingResult
from .vq_layers import RevIN, PatchEmbedding


class TransformerTSBlock(nn.Module):
    """Pre-norm transformer block for time series."""

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
        """Pre-norm transformer block.

        Args:
            x: (batch, n_patches, d_model)
        """
        # Self-attention with pre-norm
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        return x


class HVQVAEEncoder(BaseEncoder):
    """Hierarchical VQ-VAE encoder conforming to BaseEncoder interface.

    Flow:
        x → RevIN → PatchEmbed → N×TransformerBlock → split into:
            Top: mean pool → proj → VQ(512) → top_code (64-dim)
            Bottom: per-patch → proj → VQ(1024) → bottom_codes (n_patches, 64-dim)
        → z_projection(cat(top, mean(bottom))) → z (max_latent_dim)
        → chart_head, confidence_head on encoder features
    """

    def __init__(self, config: UMCConfig):
        super().__init__()
        self.config = config

        n_features = config.n_features
        d_model = config.d_model
        n_patches = config.window_size // config.patch_size
        latent_dim = config.max_latent_dim
        vq_dim = config.vq_dim

        # RevIN for instance normalization
        self.revin = RevIN(n_features)

        # Patch tokenization
        self.patch_embed = PatchEmbedding(n_features, d_model, config.patch_size, n_patches)

        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerTSBlock(d_model, config.n_heads, config.d_ff, config.transformer_dropout)
            for _ in range(config.n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        self._vq_type = getattr(config, 'vq_type', 'vq')
        self._bottom_n_levels = getattr(config, 'vq_bottom_n_levels', 1)

        # Top and bottom projections (shared by both VQ and FSQ)
        self.top_proj = nn.Linear(d_model, vq_dim)
        self.bottom_proj = nn.Linear(d_model, vq_dim)

        if self._vq_type == 'fsq':
            # --- FSQ mode: no learned codebooks, no collapse possible ---
            fsq_top_levels = list(getattr(config, 'fsq_top_levels', (8, 8)))
            fsq_bottom_levels = list(getattr(config, 'fsq_bottom_levels', (8, 8, 4)))

            self.vq_top = FSQ(levels=fsq_top_levels, dim=vq_dim)
            self.vq_bottom = ResidualFSQ(
                levels=fsq_bottom_levels,
                num_quantizers=self._bottom_n_levels,
                dim=vq_dim,
            )
            # Store implicit codebook sizes for perplexity/storage
            self._top_n_codes = int(math.prod(fsq_top_levels))
            self._bottom_n_codes = int(math.prod(fsq_bottom_levels))
        else:
            # --- Standard VQ mode (with optional DAC-style kwargs) ---
            vq_kwargs = {}
            cb_dim = getattr(config, 'vq_codebook_dim', 0)
            if cb_dim > 0:
                vq_kwargs['codebook_dim'] = cb_dim
            if getattr(config, 'vq_use_cosine_sim', False):
                vq_kwargs['use_cosine_sim'] = True
            if getattr(config, 'vq_rotation_trick', False):
                vq_kwargs['rotation_trick'] = True

            self.vq_top = VectorQuantize(
                dim=vq_dim,
                codebook_size=config.vq_top_n_codes,
                decay=config.vq_ema_decay,
                commitment_weight=config.vq_commitment_weight,
                kmeans_init=True,
                threshold_ema_dead_code=config.vq_dead_code_threshold,
                **vq_kwargs,
            )
            self.vq_bottom = LibResidualVQ(
                dim=vq_dim,
                num_quantizers=self._bottom_n_levels,
                codebook_size=config.vq_bottom_n_codes,
                quantize_dropout=self._bottom_n_levels > 1,
                kmeans_init=True,
                threshold_ema_dead_code=config.vq_dead_code_threshold,
                commitment_weight=config.vq_commitment_weight,
                decay=config.vq_ema_decay,
                **vq_kwargs,
            )
            self._top_n_codes = config.vq_top_n_codes
            self._bottom_n_codes = config.vq_bottom_n_codes

        # z_projection: continuous coords for FAISS / .mnf
        # Input: cat(top_code, mean(bottom_codes)) = 2 * vq_dim
        self.z_projection = nn.Sequential(
            nn.Linear(2 * vq_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Chart and confidence heads (on encoder features)
        head_input_dim = d_model
        self.chart_head = nn.Linear(head_input_dim, config.num_charts)
        self.confidence_head = nn.Sequential(
            nn.Linear(head_input_dim, 1),
            nn.Sigmoid(),
        )

        # Gumbel-Softmax temperature (annealed during training)
        self.gumbel_temperature = 1.0
        self._active_dim_threshold = 0.01
        self._use_grad_checkpoint = False

        # VQ loss accumulators (set during forward)
        self._vq_loss = torch.tensor(0.0)
        self._top_perplexity: float = 0.0
        self._bottom_perplexity: list[float] = []

        # Store VQ outputs for decoder access during training
        self._last_top_quantized: Tensor | None = None
        self._last_bottom_quantized: Tensor | None = None
        self._last_top_indices: Tensor | None = None
        self._last_bottom_indices: Tensor | None = None

    def _encode_transformer(self, x: Tensor) -> Tensor:
        """Shared forward pass: RevIN → PatchEmbed → Transformer.

        Args:
            x: (batch, window_size, n_features)

        Returns:
            encoder output: (batch, n_patches, d_model)
        """
        h = self.revin(x)
        h = self.patch_embed(h)
        for block in self.encoder_blocks:
            if self._use_grad_checkpoint and self.training:
                h = grad_checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        return self.encoder_norm(h)

    def encode(self, x: Tensor) -> EncodingResult:
        """Encode windowed data through hierarchical VQ-VAE.

        Args:
            x: (batch, window_size, n_features)

        Returns:
            EncodingResult with continuous z (for FAISS), chart_id, confidence.
        """
        B = x.shape[0]
        enc_out = self._encode_transformer(x)  # (B, n_patches, d_model)

        # --- Top VQ: global shape ---
        top_z_e = self.top_proj(enc_out.mean(dim=1))  # (B, vq_dim)

        if self._vq_type == 'fsq':
            # FSQ returns (quantized, indices), no commit_loss
            top_z_q, top_indices = self.vq_top(top_z_e.unsqueeze(1))
            top_z_q = top_z_q.squeeze(1)        # (B, vq_dim)
            top_indices = top_indices.squeeze(1)  # (B,)
            top_commit_loss = torch.tensor(0.0, device=top_z_q.device)
        else:
            # VQ returns (quantized, indices, commit_loss)
            top_z_q, top_indices, top_commit_loss = self.vq_top(top_z_e.unsqueeze(1))
            top_z_q = top_z_q.squeeze(1)        # (B, vq_dim)
            top_indices = top_indices.squeeze(1)  # (B,)

        # --- Bottom VQ: per-patch detail (Residual VQ / ResidualFSQ) ---
        bottom_z_e = self.bottom_proj(enc_out)  # (B, n_patches, vq_dim)

        if self._vq_type == 'fsq':
            # ResidualFSQ returns (quantized, indices), no commit_loss
            bottom_z_q, bottom_indices_packed = self.vq_bottom(bottom_z_e)
            bottom_commit_loss = torch.tensor(0.0, device=bottom_z_q.device)
        else:
            # RVQ returns (quantized, indices, commit_loss)
            bottom_z_q, bottom_indices_packed, bottom_commit_loss = self.vq_bottom(bottom_z_e)

        # Convert packed indices (B, P, n_quantizers) to list of (B, P) for storage
        bottom_indices = [
            bottom_indices_packed[:, :, i]
            for i in range(bottom_indices_packed.shape[-1])
        ]

        # Store for decoder access
        self._last_top_quantized = top_z_q
        self._last_bottom_quantized = bottom_z_q
        self._last_top_indices = top_indices
        self._last_bottom_indices = bottom_indices

        # Total VQ loss (FSQ has no commitment loss, but keep interface consistent)
        self._vq_loss = top_commit_loss.sum() + bottom_commit_loss.sum()

        # Compute perplexity from indices
        with torch.no_grad():
            top_flat = top_indices.reshape(-1).long()
            top_probs = torch.bincount(top_flat, minlength=self._top_n_codes).float()
            top_probs = top_probs / top_probs.sum().clamp(min=1)
            self._top_perplexity = torch.exp(-torch.sum(top_probs * torch.log(top_probs + 1e-10))).item()

            self._bottom_perplexity = []
            for level_indices in bottom_indices:
                flat = level_indices.reshape(-1).long()
                valid = flat[flat >= 0]
                if valid.numel() > 0:
                    probs = torch.bincount(valid, minlength=self._bottom_n_codes).float()
                    probs = probs / probs.sum().clamp(min=1)
                    perp = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10))).item()
                else:
                    perp = 0.0
                self._bottom_perplexity.append(perp)

        # --- Continuous z for FAISS / .mnf ---
        bottom_mean = bottom_z_q.mean(dim=1)  # (B, vq_dim)
        z_cat = torch.cat([top_z_q, bottom_mean], dim=-1)  # (B, 2*vq_dim)
        z = self.z_projection(z_cat)  # (B, max_latent_dim)

        # --- Chart ID (Gumbel-Softmax) ---
        enc_global = enc_out.mean(dim=1)  # (B, d_model)
        chart_logits = self.chart_head(enc_global)
        if self.training:
            chart_probs = F.gumbel_softmax(
                chart_logits, tau=self.gumbel_temperature, hard=True
            )
            chart_id = chart_probs.argmax(dim=-1)
        else:
            chart_id = chart_logits.argmax(dim=-1)

        # --- Confidence ---
        confidence = self.confidence_head(enc_global).squeeze(-1)

        # --- Active dims ---
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
            mu=z,  # No VAE — z is deterministic
            log_var=torch.zeros_like(z),  # Placeholder for interface
            active_dims=active_dims,
        )

    def get_effective_dim(self) -> int:
        return self.config.max_latent_dim

    def get_chart_logits(self, x: Tensor) -> Tensor:
        """Return raw chart logits."""
        enc_out = self._encode_transformer(x)
        return self.chart_head(enc_out.mean(dim=1))

    def reset_dead_codes(self, threshold: int | None = None) -> tuple[int, int]:
        """No-op: library handles dead code reset automatically via threshold_ema_dead_code."""
        return 0, 0

    @property
    def vq_loss(self) -> Tensor:
        """Combined VQ commitment loss from both levels."""
        return self._vq_loss

    @property
    def top_perplexity(self) -> float:
        return self._top_perplexity

    @property
    def bottom_perplexity(self) -> float:
        """Average perplexity across bottom RVQ levels."""
        if isinstance(self._bottom_perplexity, list) and self._bottom_perplexity:
            return sum(self._bottom_perplexity) / len(self._bottom_perplexity)
        return 0.0

    @property
    def per_level_perplexity(self) -> list[float]:
        """Per-level perplexity for bottom RVQ."""
        if isinstance(self._bottom_perplexity, list):
            return self._bottom_perplexity
        return []

    @property
    def entropy_loss(self) -> torch.Tensor:
        """Entropy loss placeholder (library doesn't expose this separately)."""
        return torch.tensor(0.0)

    def clear_cached(self) -> None:
        """Clear cached tensors to free GPU memory between epochs."""
        self._last_top_quantized = None
        self._last_bottom_quantized = None
        self._last_top_indices = None
        self._last_bottom_indices = None

    def _lookup_and_project(self, vq_layer, indices: Tensor) -> Tensor:
        """Look up codebook vectors and project back to vq_dim if using codebook_dim.

        Args:
            vq_layer: A VectorQuantize layer (has _codebook.embed, may have project_out).
            indices: (...) int indices into the codebook.

        Returns:
            Quantized vectors in vq_dim space: (..., vq_dim).
        """
        cb = vq_layer._codebook.embed  # (1, n_codes, codebook_dim) or (n_codes, codebook_dim)
        if cb.dim() == 3:
            cb = cb.squeeze(0)  # (n_codes, codebook_dim)
        vectors = F.embedding(indices.long(), cb)  # (..., codebook_dim)

        # If codebook_dim != vq_dim, project back through the learned project_out
        if hasattr(vq_layer, 'project_out') and vq_layer.project_out is not None:
            proj = vq_layer.project_out
            if hasattr(proj, 'weight'):  # It's a real linear, not identity
                orig_shape = vectors.shape
                vectors = proj(vectors.reshape(-1, vectors.shape[-1]))
                vectors = vectors.reshape(*orig_shape[:-1], -1)

        return vectors

    def indices_to_quantized(
        self,
        top_indices: Tensor,
        bottom_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Look up quantized vectors from VQ/FSQ indices (no encoder forward pass).

        This is the decode-from-storage path: given only the integer indices,
        reconstruct the quantized vectors that the decoder needs.

        Works for both VQ mode (learned codebooks) and FSQ mode (fixed grid).

        Args:
            top_indices: (batch,) int — top VQ/FSQ codebook indices.
            bottom_indices: (batch, n_patches, n_levels) int — bottom RVQ/ResidualFSQ indices.

        Returns:
            top_quantized: (batch, vq_dim) — top quantized vectors.
            bottom_quantized: (batch, n_patches, vq_dim) — reconstructed bottom vectors.
        """
        if self._vq_type == 'fsq':
            return self._indices_to_quantized_fsq(top_indices, bottom_indices)
        else:
            return self._indices_to_quantized_vq(top_indices, bottom_indices)

    def _indices_to_quantized_fsq(
        self,
        top_indices: Tensor,
        bottom_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """FSQ path: use FSQ.indices_to_codes and ResidualFSQ.get_output_from_indices."""
        # Top FSQ: indices_to_codes handles projection automatically
        top_quantized = self.vq_top.indices_to_codes(top_indices.unsqueeze(1))  # (B, 1, vq_dim)
        top_quantized = top_quantized.squeeze(1)  # (B, vq_dim)

        # Bottom ResidualFSQ: get_output_from_indices reconstructs from all levels
        bottom_quantized = self.vq_bottom.get_output_from_indices(bottom_indices)  # (B, P, vq_dim)

        return top_quantized, bottom_quantized

    def _indices_to_quantized_vq(
        self,
        top_indices: Tensor,
        bottom_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """VQ path: manual codebook lookup with codebook_dim support."""
        # Top VQ: single codebook lookup + project
        top_quantized = self._lookup_and_project(self.vq_top, top_indices)  # (B, vq_dim)

        # Bottom RVQ: sum per-level lookups in codebook_dim, then project once
        B, P, L = bottom_indices.shape
        codebook_dim = self.vq_bottom.layers[0]._codebook.embed.shape[-1]
        bottom_sum = torch.zeros(
            B, P, codebook_dim, device=top_indices.device,
            dtype=self.vq_top._codebook.embed.dtype,
        )
        for i, layer in enumerate(self.vq_bottom.layers):
            cb = layer._codebook.embed
            if cb.dim() == 3:
                cb = cb.squeeze(0)
            level_idx = bottom_indices[:, :, i].long()  # (B, P)
            valid_mask = level_idx >= 0
            safe_idx = level_idx.clamp(min=0)
            level_vectors = F.embedding(safe_idx, cb)  # (B, P, codebook_dim)
            level_vectors = level_vectors * valid_mask.unsqueeze(-1).float()
            bottom_sum = bottom_sum + level_vectors

        # Project from codebook_dim back to vq_dim (if using factorized codebooks)
        rvq_proj = self.vq_bottom.project_out
        if hasattr(rvq_proj, 'weight'):  # Real linear, not Identity
            orig_shape = bottom_sum.shape
            bottom_quantized = rvq_proj(bottom_sum.reshape(-1, codebook_dim))
            bottom_quantized = bottom_quantized.reshape(B, P, -1)
        else:
            bottom_quantized = bottom_sum

        return top_quantized, bottom_quantized

    def set_ema_decay(self, decay: float) -> None:
        """No-op: library manages EMA decay internally."""
        pass
