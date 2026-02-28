"""Tests for UMC manifold processor modules."""

import pytest
import numpy as np

from umc.processor.cluster import ManifoldCluster
from umc.processor.interpolate import linear_interpolate, spherical_interpolate
from umc.processor.arithmetic import analogy, manifold_distance, manifold_mean


@pytest.fixture
def sample_coordinates():
    """Generate sample manifold coordinates."""
    rng = np.random.RandomState(42)
    # 3 clusters in 10-dimensional space
    coords = np.vstack([
        rng.randn(100, 10) + np.array([5] * 10),
        rng.randn(100, 10) + np.array([-5] * 10),
        rng.randn(100, 10) + np.array([0] * 10),
    ]).astype(np.float32)
    return coords


try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


@pytest.mark.skipif(not _HAS_FAISS, reason="faiss not installed")
class TestManifoldSearch:
    def test_query_returns_correct_k(self, sample_coordinates):
        """Search returns exactly k neighbors."""
        from umc.processor.search import ManifoldSearch
        search = ManifoldSearch(sample_coordinates)
        result = search.query(sample_coordinates[0], k=5)

        assert result.indices.shape == (1, 5)
        assert result.distances.shape == (1, 5)

    def test_self_is_nearest(self, sample_coordinates):
        """A point should be its own nearest neighbor."""
        from umc.processor.search import ManifoldSearch
        search = ManifoldSearch(sample_coordinates)
        result = search.query(sample_coordinates[0], k=1)

        assert result.indices[0, 0] == 0
        assert result.distances[0, 0] < 1e-5

    def test_batch_query(self, sample_coordinates):
        """Batch query works correctly."""
        from umc.processor.search import ManifoldSearch
        search = ManifoldSearch(sample_coordinates)
        queries = sample_coordinates[:5]
        result = search.batch_query(queries, k=3)

        assert result.indices.shape == (5, 3)

    def test_distances_sorted(self, sample_coordinates):
        """Returned distances are sorted ascending."""
        from umc.processor.search import ManifoldSearch
        search = ManifoldSearch(sample_coordinates)
        result = search.query(sample_coordinates[0], k=10)

        dists = result.distances[0]
        assert np.all(dists[:-1] <= dists[1:])


class TestManifoldCluster:
    def test_cluster_labels(self, sample_coordinates):
        """Clustering produces valid labels."""
        clusterer = ManifoldCluster()
        result = clusterer.cluster(sample_coordinates, n_clusters=3)

        assert result.labels.shape == (300,)
        assert set(result.labels).issubset({0, 1, 2})
        assert result.centroids.shape == (3, 10)

    def test_cluster_stability(self, sample_coordinates):
        """Clustering is deterministic with same seed."""
        clusterer = ManifoldCluster()
        r1 = clusterer.cluster(sample_coordinates, n_clusters=3, random_state=42)
        r2 = clusterer.cluster(sample_coordinates, n_clusters=3, random_state=42)

        assert np.array_equal(r1.labels, r2.labels)

    def test_find_regimes(self, sample_coordinates):
        """Regime detection returns valid regimes."""
        clusterer = ManifoldCluster()
        regimes = clusterer.find_regimes(sample_coordinates, n_clusters=3)

        assert len(regimes) > 0
        for r in regimes:
            assert r.size >= 5  # min_regime_length default
            assert r.start_idx < r.end_idx

    def test_find_optimal_k(self, sample_coordinates):
        """Optimal k search returns reasonable results."""
        clusterer = ManifoldCluster()
        result = clusterer.find_optimal_k(sample_coordinates, k_range=range(2, 8))

        assert "optimal_k" in result
        assert 2 <= result["optimal_k"] <= 7
        assert len(result["inertias"]) == 6


class TestInterpolation:
    def test_linear_endpoints(self):
        """Linear interpolation starts and ends at correct points."""
        z1 = np.array([0.0, 0.0, 0.0])
        z2 = np.array([1.0, 1.0, 1.0])
        result = linear_interpolate(z1, z2, n_steps=5)

        assert result.shape == (5, 3)
        assert np.allclose(result[0], z1)
        assert np.allclose(result[-1], z2)

    def test_spherical_endpoints(self):
        """Spherical interpolation starts and ends at correct points."""
        z1 = np.array([1.0, 0.0, 0.0])
        z2 = np.array([0.0, 1.0, 0.0])
        result = spherical_interpolate(z1, z2, n_steps=5)

        assert result.shape == (5, 3)
        assert np.allclose(result[0], z1, atol=1e-6)
        assert np.allclose(result[-1], z2, atol=1e-6)

    def test_linear_midpoint(self):
        """Linear midpoint is the average."""
        z1 = np.zeros(5)
        z2 = np.ones(5) * 2
        result = linear_interpolate(z1, z2, n_steps=3)

        assert np.allclose(result[1], np.ones(5))


class TestArithmetic:
    def test_analogy(self):
        """Analogy operation: a:b :: c:?"""
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 1.0])
        c = np.array([2.0, 0.0])
        result = analogy(a, b, c)

        # Expected: c + (b - a) = [2, 0] + [0, 1] = [2, 1]
        assert np.allclose(result, [2.0, 1.0])

    def test_distance(self):
        """Manifold distance is Euclidean."""
        z1 = np.array([0.0, 0.0])
        z2 = np.array([3.0, 4.0])

        assert abs(manifold_distance(z1, z2) - 5.0) < 1e-6

    def test_mean(self):
        """Manifold mean is centroid."""
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        mean = manifold_mean(coords)

        assert np.allclose(mean, [2 / 3, 2 / 3])
