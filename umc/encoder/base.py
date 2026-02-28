"""Abstract encoder interface for UMC."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class EncodingResult:
    """Output of an encoder's encode() method."""
    z: Tensor               # (batch, latent_dim) — manifold coordinates
    chart_id: Tensor         # (batch,) — int, chart identifier
    confidence: Tensor       # (batch,) — float [0, 1]
    mu: Tensor               # (batch, latent_dim) — mean
    log_var: Tensor          # (batch, latent_dim) — log variance
    active_dims: int         # number of non-collapsed dimensions


class BaseEncoder(nn.Module):
    """Abstract base for all UMC encoders."""

    @abstractmethod
    def encode(self, x: Tensor) -> EncodingResult:
        """Encode ambient-space data to manifold coordinates.

        Args:
            x: Tensor of shape (batch, window_size, n_features).

        Returns:
            EncodingResult with coordinates, chart IDs, confidence, etc.
        """
        ...

    @abstractmethod
    def get_effective_dim(self) -> int:
        """Return the discovered intrinsic dimensionality."""
        ...

    def forward(self, x: Tensor) -> EncodingResult:
        return self.encode(x)
