"""Adaptive dimensionality discovery encoder.

Wraps a VAE encoder with automatic intrinsic dimension detection.
Uses structured sparsity in the latent space to discover the
minimal number of dimensions needed to represent the data.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..config import UMCConfig
from .base import BaseEncoder, EncodingResult
from .vae import VAEEncoder


class AdaptiveEncoder(BaseEncoder):
    """VAE encoder with adaptive dimensionality discovery.

    During training, dimensions with low variance are automatically
    pruned. The encoder discovers the intrinsic dimensionality of
    the data manifold.
    """

    def __init__(self, config: UMCConfig):
        super().__init__()
        self.config = config
        self.vae = VAEEncoder(config)

        # Per-dimension gate: learnable parameter that scales each latent dim
        # Initialized to 1.0 (all dimensions active)
        self.dim_gates = nn.Parameter(torch.ones(config.max_latent_dim))

        # Tracking
        self._dim_variance_ema: torch.Tensor | None = None
        self._ema_decay = 0.99

    def encode(self, x: Tensor) -> EncodingResult:
        """Encode with adaptive dimensionality gating.

        Dimensions with small gate values are effectively pruned.
        """
        result = self.vae.encode(x)

        # Apply soft gates to latent coordinates
        gates = torch.sigmoid(self.dim_gates * 10)  # Sharp sigmoid
        z_gated = result.z * gates.unsqueeze(0)
        mu_gated = result.mu * gates.unsqueeze(0)
        log_var_gated = result.log_var  # Don't gate variance (for KL)

        # Update EMA of dimension variance
        with torch.no_grad():
            batch_var = z_gated.var(dim=0)
            if self._dim_variance_ema is None:
                self._dim_variance_ema = batch_var.clone()
            else:
                self._dim_variance_ema = (
                    self._ema_decay * self._dim_variance_ema
                    + (1 - self._ema_decay) * batch_var
                )

        # Count active dimensions
        active_dims = int((gates > 0.5).sum().item())

        return EncodingResult(
            z=z_gated,
            chart_id=result.chart_id,
            confidence=result.confidence,
            mu=mu_gated,
            log_var=log_var_gated,
            active_dims=active_dims,
        )

    def get_effective_dim(self) -> int:
        """Return the number of active (non-pruned) dimensions."""
        with torch.no_grad():
            gates = torch.sigmoid(self.dim_gates * 10)
            return int((gates > 0.5).sum().item())

    def get_gate_values(self) -> np.ndarray:
        """Return the gate values for all dimensions."""
        with torch.no_grad():
            return torch.sigmoid(self.dim_gates * 10).cpu().numpy()

    def gate_sparsity_loss(self) -> Tensor:
        """L1 penalty on gate values to encourage dimension pruning."""
        gates = torch.sigmoid(self.dim_gates * 10)
        return gates.sum()

    def get_active_mask(self, threshold: float = 0.5) -> np.ndarray:
        """Boolean mask of active dimensions."""
        gates = self.get_gate_values()
        return gates > threshold

    def prune(self, threshold: float = 0.5) -> int:
        """Hard-prune dimensions below threshold.

        After pruning, gate values below threshold are set to -10
        (effectively zero after sigmoid).

        Returns:
            Number of pruned dimensions.
        """
        with torch.no_grad():
            gates = torch.sigmoid(self.dim_gates * 10)
            mask = gates < threshold
            self.dim_gates.data[mask] = -10.0
            return int(mask.sum().item())
