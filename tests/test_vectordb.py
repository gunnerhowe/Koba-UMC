"""Tests for VectorForge — compressed vector database engine."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from umc.vectordb.engine import VectorEngine, VectorNamespace, VectorRecord


class TestVectorNamespace:
    """Tests for the core VectorNamespace storage engine."""

    def test_upsert_and_count(self):
        ns = VectorNamespace(dimension=4)
        records = [
            VectorRecord(id="a", values=np.array([1, 0, 0, 0], dtype=np.float32)),
            VectorRecord(id="b", values=np.array([0, 1, 0, 0], dtype=np.float32)),
        ]
        count = ns.upsert(records)
        assert count == 2
        assert ns.count == 2

    def test_upsert_update(self):
        ns = VectorNamespace(dimension=3)
        ns.upsert([VectorRecord(id="x", values=np.ones(3, dtype=np.float32))])
        ns.upsert([VectorRecord(id="x", values=np.zeros(3, dtype=np.float32))])
        assert ns.count == 1
        fetched = ns.fetch(["x"])
        assert np.allclose(fetched["x"].values, 0.0)

    def test_upsert_wrong_dimension(self):
        ns = VectorNamespace(dimension=4)
        with pytest.raises(ValueError, match="Expected dimension 4"):
            ns.upsert([VectorRecord(id="x", values=np.ones(5, dtype=np.float32))])

    def test_query_cosine(self):
        ns = VectorNamespace(dimension=3, metric="cosine")
        ns.upsert([
            VectorRecord(id="a", values=np.array([1, 0, 0], dtype=np.float32)),
            VectorRecord(id="b", values=np.array([0, 1, 0], dtype=np.float32)),
            VectorRecord(id="c", values=np.array([1, 1, 0], dtype=np.float32)),
        ])
        # Query close to [1,0,0] — should find "a" first
        results = ns.query(np.array([1, 0.1, 0], dtype=np.float32), top_k=3)
        assert len(results) == 3
        assert results[0].id == "a"

    def test_query_euclidean(self):
        ns = VectorNamespace(dimension=2, metric="euclidean")
        ns.upsert([
            VectorRecord(id="origin", values=np.array([0, 0], dtype=np.float32)),
            VectorRecord(id="far", values=np.array([10, 10], dtype=np.float32)),
        ])
        results = ns.query(np.array([0.1, 0.1], dtype=np.float32), top_k=2)
        assert results[0].id == "origin"

    def test_query_dotproduct(self):
        ns = VectorNamespace(dimension=3, metric="dotproduct")
        ns.upsert([
            VectorRecord(id="a", values=np.array([1, 0, 0], dtype=np.float32)),
            VectorRecord(id="b", values=np.array([0, 1, 0], dtype=np.float32)),
        ])
        results = ns.query(np.array([1, 0, 0], dtype=np.float32), top_k=2)
        assert results[0].id == "a"

    def test_query_include_values(self):
        ns = VectorNamespace(dimension=3)
        ns.upsert([
            VectorRecord(id="a", values=np.array([1, 2, 3], dtype=np.float32)),
        ])
        results = ns.query(np.array([1, 2, 3], dtype=np.float32),
                          top_k=1, include_values=True)
        assert results[0].values is not None
        assert len(results[0].values) == 3

    def test_query_metadata_filter(self):
        ns = VectorNamespace(dimension=3, metric="cosine")
        ns.upsert([
            VectorRecord(id="a", values=np.array([1, 0, 0], dtype=np.float32),
                        metadata={"color": "red"}),
            VectorRecord(id="b", values=np.array([1, 0.1, 0], dtype=np.float32),
                        metadata={"color": "blue"}),
        ])
        # Filter for blue only
        results = ns.query(
            np.array([1, 0, 0], dtype=np.float32),
            top_k=2,
            filter={"color": "blue"},
        )
        assert len(results) == 1
        assert results[0].id == "b"

    def test_query_empty_namespace(self):
        ns = VectorNamespace(dimension=3)
        results = ns.query(np.array([1, 0, 0], dtype=np.float32), top_k=5)
        assert results == []

    def test_fetch(self):
        ns = VectorNamespace(dimension=3)
        ns.upsert([
            VectorRecord(id="a", values=np.ones(3, dtype=np.float32),
                        metadata={"key": "value"}),
        ])
        fetched = ns.fetch(["a", "nonexistent"])
        assert "a" in fetched
        assert "nonexistent" not in fetched
        assert fetched["a"].metadata == {"key": "value"}

    def test_delete(self):
        ns = VectorNamespace(dimension=3)
        ns.upsert([
            VectorRecord(id="a", values=np.ones(3, dtype=np.float32)),
            VectorRecord(id="b", values=np.zeros(3, dtype=np.float32)),
        ])
        assert ns.count == 2
        deleted = ns.delete(["a"])
        assert deleted == 1
        assert ns.count == 1
        assert ns.fetch(["a"]) == {}
        assert "b" in ns.fetch(["b"])

    def test_delete_nonexistent(self):
        ns = VectorNamespace(dimension=3)
        deleted = ns.delete(["doesnt_exist"])
        assert deleted == 0

    def test_stats(self):
        ns = VectorNamespace(dimension=128, compression_mode="lossless")
        rng = np.random.default_rng(42)
        records = [
            VectorRecord(id=f"v{i}", values=rng.standard_normal(128).astype(np.float32))
            for i in range(100)
        ]
        ns.upsert(records)
        stats = ns.stats()
        assert stats["vector_count"] == 100
        assert stats["dimension"] == 128
        assert stats["raw_bytes"] == 100 * 128 * 4
        assert stats["compressed_bytes"] > 0
        assert stats["compression_ratio"] > 1.0  # Should achieve some compression

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = VectorNamespace(dimension=4, metric="cosine")
            ns.upsert([
                VectorRecord(id="a", values=np.array([1, 2, 3, 4], dtype=np.float32),
                            metadata={"label": "test"}),
            ])
            ns.save(Path(tmpdir))

            loaded = VectorNamespace.load(Path(tmpdir))
            assert loaded.count == 1
            assert loaded.dimension == 4
            fetched = loaded.fetch(["a"])
            assert "a" in fetched
            assert np.allclose(fetched["a"].values, [1, 2, 3, 4])
            assert fetched["a"].metadata == {"label": "test"}


class TestVectorEngine:
    """Tests for the multi-index VectorEngine."""

    def test_create_index(self):
        engine = VectorEngine()
        result = engine.create_index("test", dimension=128)
        assert result["name"] == "test"
        assert result["dimension"] == 128

    def test_create_duplicate_index(self):
        engine = VectorEngine()
        engine.create_index("test", dimension=128)
        with pytest.raises(ValueError, match="already exists"):
            engine.create_index("test", dimension=128)

    def test_list_indexes(self):
        engine = VectorEngine()
        engine.create_index("idx1", dimension=64)
        engine.create_index("idx2", dimension=128)
        indexes = engine.list_indexes()
        assert len(indexes) == 2

    def test_delete_index(self):
        engine = VectorEngine()
        engine.create_index("test", dimension=128)
        engine.delete_index("test")
        assert len(engine.list_indexes()) == 0

    def test_delete_nonexistent_index(self):
        engine = VectorEngine()
        with pytest.raises(KeyError):
            engine.delete_index("nope")

    def test_upsert_and_query(self):
        engine = VectorEngine()
        engine.create_index("embeddings", dimension=4, metric="cosine")

        engine.upsert("embeddings", [
            {"id": "v1", "values": [1, 0, 0, 0], "metadata": {"text": "hello"}},
            {"id": "v2", "values": [0, 1, 0, 0], "metadata": {"text": "world"}},
            {"id": "v3", "values": [1, 1, 0, 0], "metadata": {"text": "mixed"}},
        ])

        result = engine.query("embeddings", vector=[1, 0.1, 0, 0], top_k=2)
        assert len(result["matches"]) == 2
        assert result["matches"][0]["id"] == "v1"

    def test_upsert_and_fetch(self):
        engine = VectorEngine()
        engine.create_index("test", dimension=3)
        engine.upsert("test", [
            {"id": "a", "values": [1, 2, 3]},
        ])
        result = engine.fetch("test", ["a"])
        assert "a" in result["vectors"]
        assert result["vectors"]["a"]["values"] == [1.0, 2.0, 3.0]

    def test_namespaces(self):
        engine = VectorEngine()
        engine.create_index("test", dimension=3)

        engine.upsert("test", [
            {"id": "a", "values": [1, 0, 0]},
        ], namespace="ns1")

        engine.upsert("test", [
            {"id": "b", "values": [0, 1, 0]},
        ], namespace="ns2")

        stats = engine.describe_index("test")
        assert stats["total_vector_count"] == 2
        assert "ns1" in stats["namespaces"]
        assert "ns2" in stats["namespaces"]

    def test_delete_vectors(self):
        engine = VectorEngine()
        engine.create_index("test", dimension=3)
        engine.upsert("test", [
            {"id": "a", "values": [1, 0, 0]},
            {"id": "b", "values": [0, 1, 0]},
        ])
        result = engine.delete("test", ["a"])
        assert result["deleted_count"] == 1

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = VectorEngine(data_dir=tmpdir)
            engine.create_index("test", dimension=4, metric="cosine")
            engine.upsert("test", [
                {"id": "v1", "values": [1, 0, 0, 0], "metadata": {"label": "x"}},
                {"id": "v2", "values": [0, 1, 0, 0], "metadata": {"label": "y"}},
            ])
            engine.save()

            # Load into new engine
            loaded = VectorEngine.load(tmpdir)
            assert len(loaded.list_indexes()) == 1

            result = loaded.query("test", vector=[1, 0, 0, 0], top_k=1)
            assert result["matches"][0]["id"] == "v1"

    def test_compression_stats(self):
        engine = VectorEngine()
        engine.create_index("test", dimension=128, compression_mode="lossless")
        rng = np.random.default_rng(42)
        vectors = [
            {"id": f"v{i}", "values": rng.standard_normal(128).tolist()}
            for i in range(100)
        ]
        engine.upsert("test", vectors)
        stats = engine.describe_index("test")
        ns_stats = stats["namespaces"][""]
        assert ns_stats["compression_ratio"] > 1.0
        print(f"Compression ratio: {ns_stats['compression_ratio']:.2f}x")


class TestMetadataFilter:
    """Tests for Pinecone-style metadata filtering."""

    def _make_engine(self):
        engine = VectorEngine()
        engine.create_index("test", dimension=3, metric="cosine")
        engine.upsert("test", [
            {"id": "v1", "values": [1, 0, 0], "metadata": {"color": "red", "count": 10}},
            {"id": "v2", "values": [0, 1, 0], "metadata": {"color": "blue", "count": 20}},
            {"id": "v3", "values": [0, 0, 1], "metadata": {"color": "red", "count": 30}},
        ])
        return engine

    def test_eq_filter(self):
        engine = self._make_engine()
        result = engine.query("test", vector=[1, 1, 1], top_k=10,
                             filter={"color": {"$eq": "red"}})
        ids = {m["id"] for m in result["matches"]}
        assert ids == {"v1", "v3"}

    def test_ne_filter(self):
        engine = self._make_engine()
        result = engine.query("test", vector=[1, 1, 1], top_k=10,
                             filter={"color": {"$ne": "red"}})
        assert len(result["matches"]) == 1
        assert result["matches"][0]["id"] == "v2"

    def test_gt_filter(self):
        engine = self._make_engine()
        result = engine.query("test", vector=[1, 1, 1], top_k=10,
                             filter={"count": {"$gt": 15}})
        ids = {m["id"] for m in result["matches"]}
        assert ids == {"v2", "v3"}

    def test_in_filter(self):
        engine = self._make_engine()
        result = engine.query("test", vector=[1, 1, 1], top_k=10,
                             filter={"color": {"$in": ["red", "green"]}})
        ids = {m["id"] for m in result["matches"]}
        assert ids == {"v1", "v3"}

    def test_simple_equality_filter(self):
        engine = self._make_engine()
        result = engine.query("test", vector=[1, 1, 1], top_k=10,
                             filter={"color": "blue"})
        assert len(result["matches"]) == 1
        assert result["matches"][0]["id"] == "v2"


class TestFastAPIServer:
    """Tests for the FastAPI REST API endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from umc.vectordb.server import create_app
        app = create_app()
        return TestClient(app)

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "VectorForge"

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_create_and_list_indexes(self, client):
        response = client.post("/indexes", json={
            "name": "test-idx",
            "dimension": 128,
            "metric": "cosine",
        })
        assert response.status_code == 200

        response = client.get("/indexes")
        assert response.status_code == 200
        indexes = response.json()["indexes"]
        assert len(indexes) >= 1

    def test_upsert_query_flow(self, client):
        # Create index
        client.post("/indexes", json={
            "name": "search-test",
            "dimension": 4,
        })

        # Upsert vectors
        response = client.post("/indexes/search-test/vectors/upsert", json={
            "vectors": [
                {"id": "v1", "values": [1, 0, 0, 0], "metadata": {"text": "hello"}},
                {"id": "v2", "values": [0, 1, 0, 0], "metadata": {"text": "world"}},
            ],
            "namespace": "",
        })
        assert response.status_code == 200
        assert response.json()["upserted_count"] == 2

        # Query
        response = client.post("/indexes/search-test/query", json={
            "vector": [1, 0.1, 0, 0],
            "topK": 2,
            "includeMetadata": True,
        })
        assert response.status_code == 200
        matches = response.json()["matches"]
        assert len(matches) == 2
        assert matches[0]["id"] == "v1"
        assert matches[0]["metadata"]["text"] == "hello"

    def test_fetch_vectors(self, client):
        client.post("/indexes", json={"name": "fetch-test", "dimension": 3})
        client.post("/indexes/fetch-test/vectors/upsert", json={
            "vectors": [{"id": "a", "values": [1, 2, 3]}],
        })

        response = client.post("/indexes/fetch-test/vectors/fetch", json={
            "ids": ["a"],
        })
        assert response.status_code == 200
        assert "a" in response.json()["vectors"]

    def test_delete_vectors(self, client):
        client.post("/indexes", json={"name": "del-test", "dimension": 3})
        client.post("/indexes/del-test/vectors/upsert", json={
            "vectors": [
                {"id": "a", "values": [1, 0, 0]},
                {"id": "b", "values": [0, 1, 0]},
            ],
        })

        response = client.post("/indexes/del-test/vectors/delete", json={
            "ids": ["a"],
        })
        assert response.status_code == 200
        assert response.json()["deleted_count"] == 1

    def test_index_stats(self, client):
        client.post("/indexes", json={"name": "stats-test", "dimension": 4})
        client.post("/indexes/stats-test/vectors/upsert", json={
            "vectors": [{"id": "v1", "values": [1, 2, 3, 4]}],
        })

        response = client.get("/indexes/stats-test/stats")
        assert response.status_code == 200
        stats = response.json()
        assert stats["total_vector_count"] == 1

    def test_404_on_missing_index(self, client):
        response = client.post("/indexes/nonexistent/query", json={
            "vector": [1, 0, 0],
            "topK": 1,
        })
        assert response.status_code == 404

    def test_duplicate_index_409(self, client):
        client.post("/indexes", json={"name": "dup", "dimension": 3})
        response = client.post("/indexes", json={"name": "dup", "dimension": 3})
        assert response.status_code == 409
