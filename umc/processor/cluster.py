"""Clustering in manifold coordinate space."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans, DBSCAN


@dataclass
class ClusterResult:
    """Result of manifold-space clustering."""
    labels: np.ndarray          # (n_samples,) int cluster labels
    centroids: np.ndarray       # (n_clusters, latent_dim) centroid coordinates
    n_clusters: int
    inertia: float              # Sum of squared distances to closest centroid


@dataclass
class Regime:
    """A detected market regime."""
    label: int
    centroid: np.ndarray        # (latent_dim,) centroid coordinates
    start_idx: int
    end_idx: int
    size: int


class ManifoldCluster:
    """Clustering operations in manifold coordinate space."""

    def cluster(
        self,
        z: np.ndarray,
        n_clusters: int,
        random_state: int = 42,
    ) -> ClusterResult:
        """K-means clustering in manifold space.

        Args:
            z: (n_samples, latent_dim) manifold coordinates.
            n_clusters: Number of clusters.
            random_state: Random seed for reproducibility.

        Returns:
            ClusterResult with labels, centroids, and inertia.
        """
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(z)

        return ClusterResult(
            labels=labels,
            centroids=kmeans.cluster_centers_,
            n_clusters=n_clusters,
            inertia=float(kmeans.inertia_),
        )

    def cluster_dbscan(
        self,
        z: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> ClusterResult:
        """DBSCAN clustering (density-based, no n_clusters needed).

        Args:
            z: (n_samples, latent_dim) manifold coordinates.
            eps: Maximum distance between neighbors.
            min_samples: Minimum cluster size.

        Returns:
            ClusterResult.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(z)

        unique_labels = set(labels) - {-1}
        n_clusters = len(unique_labels)

        # Compute centroids for each cluster
        centroids = np.zeros((n_clusters, z.shape[1]))
        for i, label in enumerate(sorted(unique_labels)):
            mask = labels == label
            centroids[i] = z[mask].mean(axis=0)

        inertia = 0.0
        for i, label in enumerate(sorted(unique_labels)):
            mask = labels == label
            inertia += np.sum((z[mask] - centroids[i]) ** 2)

        return ClusterResult(
            labels=labels,
            centroids=centroids,
            n_clusters=n_clusters,
            inertia=float(inertia),
        )

    def find_regimes(
        self,
        z: np.ndarray,
        n_clusters: int = 5,
        min_regime_length: int = 5,
    ) -> list[Regime]:
        """Identify market regimes via manifold clustering.

        Assumes z is ordered by time. Finds contiguous segments
        where the cluster label is stable.

        Args:
            z: (n_samples, latent_dim) time-ordered coordinates.
            n_clusters: Number of regime types.
            min_regime_length: Minimum regime duration.

        Returns:
            List of Regime objects.
        """
        result = self.cluster(z, n_clusters)
        labels = result.labels
        centroids = result.centroids

        regimes = []
        current_label = labels[0]
        start_idx = 0

        for i in range(1, len(labels)):
            if labels[i] != current_label:
                length = i - start_idx
                if length >= min_regime_length:
                    regimes.append(Regime(
                        label=int(current_label),
                        centroid=centroids[current_label],
                        start_idx=start_idx,
                        end_idx=i,
                        size=length,
                    ))
                current_label = labels[i]
                start_idx = i

        # Last regime
        length = len(labels) - start_idx
        if length >= min_regime_length:
            regimes.append(Regime(
                label=int(current_label),
                centroid=centroids[current_label],
                start_idx=start_idx,
                end_idx=len(labels),
                size=length,
            ))

        return regimes

    def find_optimal_k(
        self,
        z: np.ndarray,
        k_range: range = range(2, 15),
    ) -> dict:
        """Find optimal number of clusters using the elbow method.

        Returns:
            Dict with inertias per k and suggested optimal k.
        """
        inertias = []
        for k in k_range:
            result = self.cluster(z, n_clusters=k)
            inertias.append(result.inertia)

        # Simple elbow detection: largest second derivative
        inertias_arr = np.array(inertias)
        if len(inertias_arr) >= 3:
            second_deriv = np.diff(np.diff(inertias_arr))
            optimal_idx = int(np.argmax(second_deriv)) + 2  # offset for diff
            optimal_k = list(k_range)[min(optimal_idx, len(list(k_range)) - 1)]
        else:
            optimal_k = list(k_range)[0]

        return {
            "k_values": list(k_range),
            "inertias": inertias,
            "optimal_k": optimal_k,
        }
