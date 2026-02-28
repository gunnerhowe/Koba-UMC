"""Anomaly detection via manifold distance and reconstruction confidence."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class ManifoldAnomalyDetector:
    """Anomaly detection operating entirely in manifold coordinate space."""

    def __init__(
        self,
        coordinates: np.ndarray,
        method: str = "combined",
        k_neighbors: int = 20,
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        """Fit anomaly detection model on manifold coordinates.

        Args:
            coordinates: (n_samples, latent_dim) training coordinates.
            method: 'knn', 'isolation_forest', 'reconstruction_confidence',
                    or 'combined'.
            k_neighbors: Number of neighbors for density-based methods.
            contamination: Expected fraction of anomalies.
            random_state: Random seed.
        """
        self.coordinates = coordinates.astype(np.float32)
        self.method = method
        self.k_neighbors = k_neighbors

        self._models = {}
        self._recon_errors = None
        self._recon_threshold = None

        if method in ("knn", "combined"):
            # k-distance: average distance to k nearest neighbors
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k_neighbors)
            nn.fit(self.coordinates)
            distances, _ = nn.kneighbors(self.coordinates)
            self._knn_mean_dist = distances.mean(axis=1)
            self._knn_threshold = np.percentile(
                self._knn_mean_dist, (1 - contamination) * 100
            )
            self._models["knn"] = nn

        if method in ("isolation_forest", "combined"):
            iso = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=100,
            )
            iso.fit(self.coordinates)
            self._models["isolation_forest"] = iso

    def fit_reconstruction_confidence(
        self,
        originals: np.ndarray,
        reconstructed: np.ndarray,
        contamination: float = 0.1,
    ) -> None:
        """Fit reconstruction confidence anomaly detector.

        Points with high reconstruction error are anomalous — the decoder
        can't faithfully reconstruct them, indicating they lie far from the
        learned manifold (spec Section 14.3).

        Args:
            originals: (n_samples, ...) original data.
            reconstructed: (n_samples, ...) decoder output.
            contamination: Expected anomaly fraction for threshold.
        """
        originals = originals.reshape(len(originals), -1).astype(np.float32)
        reconstructed = reconstructed.reshape(len(reconstructed), -1).astype(np.float32)
        self._recon_errors = np.sqrt(np.mean((originals - reconstructed) ** 2, axis=1))
        self._recon_threshold = np.percentile(
            self._recon_errors, (1 - contamination) * 100
        )
        self._models["reconstruction_confidence"] = True

    def score(self, z: np.ndarray, reconstruction_errors: np.ndarray | None = None) -> np.ndarray:
        """Return anomaly score [0, 1] for each coordinate vector.

        Higher score = more anomalous.

        Args:
            z: (n_samples, latent_dim) coordinates to score.
            reconstruction_errors: Optional per-sample RMSE from decoder.
                Required if 'reconstruction_confidence' is in the active methods.

        Returns:
            (n_samples,) array of anomaly scores in [0, 1].
        """
        z = z.astype(np.float32)
        scores = []

        if "knn" in self._models:
            nn = self._models["knn"]
            distances, _ = nn.kneighbors(z)
            mean_dist = distances.mean(axis=1)
            # Normalize to [0, 1] using training distribution
            knn_score = mean_dist / (self._knn_threshold * 2 + 1e-8)
            knn_score = np.clip(knn_score, 0, 1)
            scores.append(knn_score)

        if "isolation_forest" in self._models:
            iso = self._models["isolation_forest"]
            # score_samples returns negative anomaly scores (lower = more anomalous)
            raw_scores = -iso.score_samples(z)
            # Normalize to [0, 1]
            iso_score = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
            scores.append(iso_score)

        if "reconstruction_confidence" in self._models:
            if reconstruction_errors is not None:
                recon_errs = reconstruction_errors
            elif self._recon_errors is not None and len(self._recon_errors) == len(z):
                recon_errs = self._recon_errors
            else:
                raise ValueError(
                    "reconstruction_confidence method requires reconstruction_errors "
                    "parameter or fit_reconstruction_confidence() to have been called "
                    "with matching sample count."
                )
            # Normalize using training threshold
            threshold = self._recon_threshold if self._recon_threshold else (recon_errs.max() + 1e-8)
            recon_score = recon_errs / (threshold * 2 + 1e-8)
            recon_score = np.clip(recon_score, 0, 1)
            scores.append(recon_score)

        if not scores:
            raise ValueError(f"No valid method configured: {self.method}")

        # Combine scores (average for 'combined')
        combined = np.mean(scores, axis=0)
        return combined

    def predict(self, z: np.ndarray, threshold: float = 0.5, **kwargs) -> np.ndarray:
        """Binary anomaly prediction.

        Args:
            z: (n_samples, latent_dim) coordinates.
            threshold: Score threshold for anomaly flag.
            **kwargs: Passed to score() (e.g., reconstruction_errors).

        Returns:
            (n_samples,) binary array (1 = anomaly, 0 = normal).
        """
        scores = self.score(z, **kwargs)
        return (scores > threshold).astype(np.int32)
