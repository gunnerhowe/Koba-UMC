"""Abstract decoder interface for UMC."""

from abc import abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseDecoder(nn.Module):
    """Abstract base for all UMC decoders."""

    @abstractmethod
    def decode(self, z: Tensor, chart_id: Tensor) -> Tensor:
        """Decode manifold coordinates back to ambient space.

        Args:
            z: Tensor (batch, latent_dim) — manifold coordinates.
            chart_id: Tensor (batch,) — chart identifier (int).

        Returns:
            x_hat: Tensor (batch, window_size, n_features).
        """
        ...

    def forward(self, z: Tensor, chart_id: Tensor) -> Tensor:
        return self.decode(z, chart_id)
