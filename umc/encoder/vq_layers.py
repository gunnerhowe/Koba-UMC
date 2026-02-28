"""Shared building blocks for Hierarchical VQ-VAE.

Components:
    RevIN: Reversible Instance Normalization (differentiable nn.Module)
    PatchEmbedding: Non-overlapping patch tokenization with positional encoding
    VectorQuantizerEMA: EMA-updated vector quantization with dead code reset
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RevIN(nn.Module):
    """Reversible Instance Normalization.

    Normalizes each sample independently per feature, with learnable affine
    parameters. Stores normalization stats so the decoder can reverse them.
    """

    def __init__(self, n_features: int, eps: float = 1e-5):
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(n_features))
        self.affine_bias = nn.Parameter(torch.zeros(n_features))
        # Stored during forward for inverse
        self._mean: Tensor | None = None
        self._std: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Normalize input per-instance per-feature.

        Args:
            x: (batch, seq_len, n_features)

        Returns:
            Normalized tensor, same shape.
        """
        self._mean = x.mean(dim=1, keepdim=True)  # (B, 1, F)
        self._std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        x_norm = (x - self._mean) / self._std
        return x_norm * self.affine_weight + self.affine_bias

    def inverse(self, x: Tensor) -> Tensor:
        """Reverse normalization using stored stats.

        Args:
            x: (batch, seq_len, n_features) — decoder output

        Returns:
            Denormalized tensor.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("RevIN.inverse() called before forward()")
        x_unaffine = (x - self.affine_bias) / (self.affine_weight + 1e-8)
        return x_unaffine * self._std + self._mean

    def set_stats(self, mean: Tensor, std: Tensor) -> None:
        """Manually set stats for inference without encoder forward pass."""
        self._mean = mean
        self._std = std


class PatchEmbedding(nn.Module):
    """Non-overlapping patch tokenization with learnable positional encoding.

    Splits temporal input into patches and projects each to d_model dimensions.
    """

    def __init__(self, n_features: int, d_model: int, patch_size: int, n_patches: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.proj = nn.Conv1d(
            n_features, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Tokenize input into patch embeddings.

        Args:
            x: (batch, seq_len, n_features)

        Returns:
            (batch, n_patches, d_model)
        """
        # Conv1d expects (B, C, L)
        h = self.proj(x.transpose(1, 2))  # (B, d_model, n_patches)
        h = h.transpose(1, 2)  # (B, n_patches, d_model)
        return h + self.pos_embed


class VectorQuantizerEMA(nn.Module):
    """Vector Quantizer with Exponential Moving Average codebook updates.

    More stable than gradient-based VQ: codebook entries track the running
    mean of assigned encoder outputs. Includes dead code reset and
    perplexity monitoring.
    """

    def __init__(
        self,
        n_codes: int,
        code_dim: int,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps

        # Codebook embeddings — initialized from data on first forward
        self.register_buffer("codebook", torch.randn(n_codes, code_dim))

        # EMA tracking
        self.register_buffer("ema_count", torch.ones(n_codes))
        self.register_buffer("ema_weight", self.codebook.clone())

        # Usage tracking for dead code detection
        self.register_buffer("usage_count", torch.zeros(n_codes))

        # Data-driven initialization flag
        self._needs_init = True

        # Metrics
        self._perplexity: float = 0.0
        self._entropy_loss: Tensor = torch.tensor(0.0)

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize encoder outputs to nearest codebook entries.

        Args:
            z_e: (batch, code_dim) or (batch, n_patches, code_dim)

        Returns:
            z_q: Quantized output (same shape as z_e), with straight-through grad
            commit_loss: Commitment loss scalar
            indices: (batch,) or (batch, n_patches) codebook indices
        """
        # Ensure fp32 for VQ operations (AMP autocast may pass fp16)
        z_e_float = z_e.float()
        flat_input = z_e_float.reshape(-1, self.code_dim)  # (N, D)

        # Data-driven init: initialize codebook from first batch of encoder outputs
        if self._needs_init and self.training:
            self._init_from_data(flat_input)
            self._needs_init = False

        # Compute distances: ||z - e||^2 = ||z||^2 - 2*z*e + ||e||^2
        distances = (
            flat_input.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_input @ self.codebook.T
            + self.codebook.pow(2).sum(dim=1, keepdim=True).T
        )

        # Find nearest codes
        indices = distances.argmin(dim=1)  # (N,)
        z_q_flat = self.codebook[indices]  # (N, D)

        # EMA update (training only)
        if self.training:
            self._ema_update(flat_input, indices)

        # Commitment loss: encourage encoder to commit to codebook entries
        commit_loss = self.commitment_weight * F.mse_loss(z_e_float, z_q_flat.reshape_as(z_e_float).detach())

        # Straight-through estimator: gradients flow through z_q to z_e
        # Use original z_e (may be fp16 in AMP) for gradient flow
        z_q = z_e + (z_q_flat.reshape_as(z_e).to(z_e.dtype) - z_e).detach()

        # Reshape indices to match input batch dims
        indices = indices.reshape(z_e_float.shape[:-1])

        # Compute perplexity and entropy loss for codebook utilization
        with torch.no_grad():
            encodings = F.one_hot(indices.reshape(-1), self.n_codes).float()
            avg_probs = encodings.mean(dim=0)
            self._perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()

        # Entropy loss: monitor codebook utilization (detached — no gradient retention)
        # Previously this retained large (N, K) tensors in the autograd graph,
        # causing OOM with many RVQ levels. Now fully detached.
        with torch.no_grad():
            cb_det = self.codebook.detach().float()
            fi_det = flat_input.detach().float()
            ent_distances = (
                fi_det.pow(2).sum(dim=1, keepdim=True)
                - 2 * fi_det @ cb_det.T
                + cb_det.pow(2).sum(dim=1, keepdim=True).T
            )
            soft_assign = F.softmax(-ent_distances / max(self.code_dim ** 0.5, 1.0), dim=1)
            avg_soft = soft_assign.mean(dim=0).clamp(min=1e-10)
            entropy = -torch.sum(avg_soft * torch.log(avg_soft))
            max_entropy = math.log(self.n_codes)
            ent_val = (max_entropy - entropy) / max_entropy
            self._entropy_loss = ent_val if torch.isfinite(ent_val) else torch.tensor(0.0, device=z_e.device)

        return z_q, commit_loss, indices

    def _init_from_data(self, flat_input: Tensor) -> None:
        """Initialize codebook from encoder outputs for better starting point."""
        n = flat_input.shape[0]
        if n >= self.n_codes:
            # Sample n_codes random inputs
            perm = torch.randperm(n, device=flat_input.device)[:self.n_codes]
            self.codebook.copy_(flat_input[perm].detach())
        else:
            # Not enough inputs — tile and add noise
            repeats = (self.n_codes + n - 1) // n
            tiled = flat_input.detach().repeat(repeats, 1)[:self.n_codes]
            tiled = tiled + torch.randn_like(tiled) * 0.01
            self.codebook.copy_(tiled)
        self.ema_weight.copy_(self.codebook)
        self.ema_count.fill_(1.0)

    def _ema_update(self, flat_input: Tensor, indices: Tensor) -> None:
        """Update codebook via EMA on encoder outputs."""
        encodings = F.one_hot(indices, self.n_codes).float()  # (N, K)

        # Update counts with Laplace smoothing
        batch_count = encodings.sum(dim=0)  # (K,)
        self.ema_count.mul_(self.decay).add_(batch_count, alpha=1 - self.decay)

        # Laplace smoothing
        n = self.ema_count.sum()
        smoothed_count = (
            (self.ema_count + self.eps) / (n + self.n_codes * self.eps) * n
        )

        # Update sum of assigned vectors
        batch_sum = encodings.T @ flat_input  # (K, D)
        self.ema_weight.mul_(self.decay).add_(batch_sum, alpha=1 - self.decay)

        # Update codebook
        self.codebook.copy_(self.ema_weight / smoothed_count.unsqueeze(1))

        # Track usage
        self.usage_count.add_(batch_count)

    def reset_dead_codes(self, z_e: Tensor, threshold: int = 2) -> int:
        """Reinitialize codebook entries that are rarely used.

        Args:
            z_e: Recent encoder outputs (N, code_dim) to sample replacements from.
            threshold: Codes used fewer than this many times are considered dead.

        Returns:
            Number of codes reset.
        """
        dead_mask = self.usage_count < threshold
        n_dead = dead_mask.sum().item()

        if n_dead > 0 and z_e.shape[0] > 0:
            # Sample from encoder outputs (with noise)
            replace_idx = torch.randint(0, z_e.shape[0], (n_dead,), device=z_e.device)
            new_codes = z_e[replace_idx].detach() + torch.randn(n_dead, self.code_dim, device=z_e.device) * 0.01
            self.codebook[dead_mask] = new_codes
            self.ema_weight[dead_mask] = new_codes
            self.ema_count[dead_mask] = 1.0

        # Reset usage counter after each reset cycle
        self.usage_count.zero_()
        return n_dead

    @property
    def perplexity(self) -> float:
        """Codebook utilization metric. Higher = more codes used."""
        return self._perplexity

    @property
    def entropy_loss(self) -> Tensor:
        """Entropy loss for encouraging uniform codebook usage. Lower = more uniform."""
        return self._entropy_loss

    def get_codes(self, indices: Tensor) -> Tensor:
        """Look up codebook entries by index.

        Args:
            indices: Integer indices into codebook.

        Returns:
            Codebook vectors at those indices.
        """
        return self.codebook[indices]


class ResidualVQ(nn.Module):
    """Residual Vector Quantization (RVQ).

    Iteratively quantizes the residual error through multiple VQ codebooks.
    Each level captures increasingly fine detail. Used in modern neural codecs
    (SoundStream, EnCodec) for high-fidelity discrete representation.

    Level 1: quantize input         → z_q_1, residual_1 = input - z_q_1
    Level 2: quantize residual_1    → z_q_2, residual_2 = residual_1 - z_q_2
    ...
    Final:   z_q = z_q_1 + z_q_2 + ... + z_q_K
    """

    def __init__(
        self,
        n_levels: int,
        n_codes: int,
        code_dim: int,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_codes = n_codes
        self.code_dim = code_dim

        self.vq_layers = nn.ModuleList([
            VectorQuantizerEMA(n_codes, code_dim, commitment_weight, decay)
            for _ in range(n_levels)
        ])

        # Progressive training: only use first _active_levels layers
        self._active_levels = n_levels

    def set_active_levels(self, n: int) -> None:
        """Set the number of active RVQ levels for progressive training.

        During progressive training, we start with few levels and gradually
        activate more. This stabilizes training since later levels see
        meaningful residuals only after earlier levels converge.
        """
        self._active_levels = max(1, min(n, self.n_levels))

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Quantize through active residual VQ levels.

        Args:
            z_e: (batch, code_dim) or (batch, n_patches, code_dim)

        Returns:
            z_q: Sum of all quantized outputs (same shape as z_e)
            total_commit_loss: Sum of commitment losses across active levels
            all_indices: List of index tensors, one per active level
        """
        residual = z_e
        z_q_sum = torch.zeros_like(z_e)
        total_commit_loss = torch.tensor(0.0, device=z_e.device)
        all_indices = []

        for i, vq in enumerate(self.vq_layers[:self._active_levels]):
            z_q_level, commit_loss, indices = vq(residual)
            z_q_sum = z_q_sum + z_q_level
            total_commit_loss = total_commit_loss + commit_loss
            all_indices.append(indices)
            # Detach residual to break gradient chain between RVQ levels.
            # Each level's commitment loss provides encoder gradients independently.
            # The STE on z_q_sum still gives the decoder gradient to the encoder.
            # This prevents O(levels) memory growth in the autograd graph.
            residual = (residual - z_q_level).detach()

        return z_q_sum, total_commit_loss, all_indices

    def decode_from_indices(self, all_indices: list[Tensor]) -> Tensor:
        """Reconstruct quantized vector from stored indices.

        Args:
            all_indices: List of index tensors, one per RVQ level.
                Can be shorter than n_levels (for progressive training).

        Returns:
            z_q: Sum of codebook lookups across provided levels.
        """
        z_q = None
        for vq, indices in zip(self.vq_layers, all_indices):
            codes = vq.get_codes(indices)
            z_q = codes if z_q is None else z_q + codes
        return z_q

    @property
    def active_levels(self) -> int:
        """Number of currently active RVQ levels."""
        return self._active_levels

    @property
    def perplexity(self) -> float:
        """Average perplexity across active levels."""
        n = self._active_levels
        return sum(vq.perplexity for vq in self.vq_layers[:n]) / max(n, 1)

    @property
    def entropy_loss(self) -> Tensor:
        """Sum of entropy losses across active levels."""
        return sum(vq.entropy_loss for vq in self.vq_layers[:self._active_levels])

    @property
    def per_level_perplexity(self) -> list[float]:
        """Perplexity for each active RVQ level."""
        return [vq.perplexity for vq in self.vq_layers[:self._active_levels]]

    def reset_dead_codes(self, z_e: Tensor, threshold: int = 2) -> int:
        """Reset dead codes across active levels."""
        total_reset = 0
        residual = z_e.detach()
        for vq in self.vq_layers[:self._active_levels]:
            n_reset = vq.reset_dead_codes(residual, threshold)
            total_reset += n_reset
            # Compute residual for next level's reset
            with torch.no_grad():
                z_q, _, _ = vq(residual)
                residual = residual - z_q
        return total_reset

    def set_decay(self, decay: float) -> None:
        """Set EMA decay on all VQ levels."""
        for vq in self.vq_layers:
            vq.decay = decay
