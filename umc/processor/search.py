"""Nearest-neighbor search in manifold coordinate space."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SearchResult:
    """Result of a manifold-space nearest neighbor query."""
    indices: np.ndarray       # (n_queries, k) int
    distances: np.ndarray     # (n_queries, k) float
    coordinates: Optional[np.ndarray] = None  # (n_queries, k, latent_dim)


class ManifoldSearch:
    """FAISS-based nearest neighbor search in manifold space."""

    def __init__(
        self,
        coordinates: np.ndarray,
        use_ivf: bool = False,
        n_lists: int = 100,
    ):
        """Build a search index over manifold coordinates.

        Args:
            coordinates: (n_samples, latent_dim) float array.
            use_ivf: Use IVF index for large datasets.
            n_lists: Number of IVF partitions.
        """
        import faiss

        self.coordinates = np.ascontiguousarray(coordinates.astype(np.float32))
        self.n_samples, self.dim = self.coordinates.shape

        if use_ivf and self.n_samples > n_lists * 40:
            quantizer = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, n_lists)
            self.index.train(self.coordinates)
            self.index.nprobe = min(10, n_lists)
        else:
            self.index = faiss.IndexFlatL2(self.dim)

        self.index.add(self.coordinates)

    def query(self, q: np.ndarray, k: int = 10) -> SearchResult:
        """Find k nearest neighbors in manifold space.

        Args:
            q: Query coordinates, shape (latent_dim,) or (n_queries, latent_dim).
            k: Number of neighbors to return.

        Returns:
            SearchResult with indices, distances, and optionally coordinates.
        """
        q = np.ascontiguousarray(q.astype(np.float32))
        if q.ndim == 1:
            q = q.reshape(1, -1)

        k = min(k, self.n_samples)
        distances, indices = self.index.search(q, k)

        # Gather neighbor coordinates
        neighbor_coords = self.coordinates[indices]

        return SearchResult(
            indices=indices,
            distances=distances,
            coordinates=neighbor_coords,
        )

    def range_query(self, q: np.ndarray, radius: float) -> SearchResult:
        """Find all points within a given radius in manifold space.

        Args:
            q: Query coordinate, shape (latent_dim,).
            radius: Search radius (L2 distance).

        Returns:
            SearchResult with variable-length results.
        """
        import faiss

        q = np.ascontiguousarray(q.astype(np.float32).reshape(1, -1))

        # FAISS range_search returns (lims, D, I)
        lims, distances, indices = self.index.range_search(q, radius ** 2)

        n_results = int(lims[1] - lims[0])
        indices = indices[:n_results].reshape(1, -1)
        distances = np.sqrt(distances[:n_results]).reshape(1, -1)

        neighbor_coords = self.coordinates[indices.flatten()].reshape(1, n_results, -1)

        return SearchResult(
            indices=indices,
            distances=distances,
            coordinates=neighbor_coords,
        )

    def batch_query(self, queries: np.ndarray, k: int = 10) -> SearchResult:
        """Batch nearest neighbor search.

        Args:
            queries: (n_queries, latent_dim) array.
            k: Number of neighbors.

        Returns:
            SearchResult.
        """
        return self.query(queries, k=k)
