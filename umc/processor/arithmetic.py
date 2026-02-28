"""Semantic arithmetic on manifold coordinates.

Enables operations like: z(AAPL_bull) - z(AAPL_bear) + z(MSFT_bear) ≈ z(MSFT_bull)
"""

import numpy as np
from typing import Optional


def manifold_add(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Element-wise addition in manifold space."""
    return z1 + z2


def manifold_subtract(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Element-wise subtraction in manifold space."""
    return z1 - z2


def manifold_mean(coords: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute the mean point in manifold space (centroid)."""
    return np.mean(coords, axis=axis)


def analogy(
    z_a: np.ndarray,
    z_b: np.ndarray,
    z_c: np.ndarray,
) -> np.ndarray:
    """Compute the analogy: a is to b as c is to ?

    z_result = z_c + (z_b - z_a)

    Example: bull_AAPL is to bear_AAPL as bull_MSFT is to ?
        result = analogy(z_bull_aapl, z_bear_aapl, z_bull_msft)

    Args:
        z_a, z_b, z_c: Manifold coordinates.

    Returns:
        Resulting manifold coordinate.
    """
    return z_c + (z_b - z_a)


def manifold_distance(z1: np.ndarray, z2: np.ndarray) -> float:
    """Euclidean distance between two manifold points."""
    return float(np.linalg.norm(z1 - z2))


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise distance matrix in manifold space.

    Args:
        coords: (n_samples, latent_dim) array.

    Returns:
        (n_samples, n_samples) distance matrix.
    """
    from scipy.spatial.distance import cdist
    return cdist(coords, coords, metric="euclidean")


def direction_vector(
    coords: np.ndarray,
    labels: np.ndarray,
    label_a: int,
    label_b: int,
) -> np.ndarray:
    """Compute the direction vector between two cluster centroids.

    Useful for finding "trend directions" in manifold space.

    Args:
        coords: (n_samples, latent_dim) coordinates.
        labels: (n_samples,) cluster/class labels.
        label_a: Source cluster label.
        label_b: Target cluster label.

    Returns:
        Direction vector (latent_dim,).
    """
    centroid_a = coords[labels == label_a].mean(axis=0)
    centroid_b = coords[labels == label_b].mean(axis=0)
    return centroid_b - centroid_a
