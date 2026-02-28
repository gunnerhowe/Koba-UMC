"""VectorForge storage engine — compressed vector storage with FAISS search.

Core data structure: each namespace holds vectors compressed with UMC
and indexed with FAISS for nearest-neighbor search. Metadata is stored
as a side-car dict per vector ID.

Storage layout on disk:
    {data_dir}/{index_name}/{namespace}/
        vectors.umc          # UMC-compressed float32/float16 vectors
        metadata.json         # ID -> metadata mapping
        faiss.index           # FAISS IVF index (optional, for large collections)
        config.json           # Namespace config (dimension, metric, etc.)
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class VectorRecord:
    """A single vector with ID and optional metadata."""
    id: str
    values: np.ndarray       # (dim,) float32 or float16
    metadata: dict[str, Any] = field(default_factory=dict)
    sparse_values: Optional[dict] = None  # Pinecone compat (unused in MVP)


@dataclass
class QueryResult:
    """Result of a nearest-neighbor query."""
    id: str
    score: float
    values: Optional[np.ndarray] = None
    metadata: Optional[dict[str, Any]] = None


class VectorNamespace:
    """A namespace within an index — holds vectors, metadata, and search index.

    Thread-safe for concurrent reads. Writes acquire a lock.
    """

    def __init__(self, dimension: int, metric: str = "cosine",
                 compression_mode: str = "lossless"):
        self.dimension = dimension
        self.metric = metric
        self.compression_mode = compression_mode

        # In-memory storage
        self._ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._vectors: Optional[np.ndarray] = None  # (N, dim) float32
        self._metadata: dict[str, dict[str, Any]] = {}
        self._compressed: Optional[bytes] = None
        self._dirty = False  # vectors changed since last compression

        # FAISS index (lazy-built)
        self._faiss_index = None

        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        return len(self._ids)

    def upsert(self, records: list[VectorRecord]) -> int:
        """Insert or update vectors. Returns number of upserted vectors."""
        with self._lock:
            for rec in records:
                if rec.values.shape != (self.dimension,):
                    raise ValueError(
                        f"Expected dimension {self.dimension}, "
                        f"got {rec.values.shape}"
                    )
                if rec.id in self._id_to_idx:
                    # Update existing
                    idx = self._id_to_idx[rec.id]
                    self._vectors[idx] = rec.values.astype(np.float32)
                else:
                    # Insert new
                    idx = len(self._ids)
                    self._ids.append(rec.id)
                    self._id_to_idx[rec.id] = idx
                    if self._vectors is None:
                        self._vectors = rec.values.astype(np.float32).reshape(1, -1)
                    else:
                        self._vectors = np.vstack([
                            self._vectors,
                            rec.values.astype(np.float32).reshape(1, -1)
                        ])
                self._metadata[rec.id] = rec.metadata
                self._dirty = True

            # Rebuild FAISS index
            self._rebuild_faiss()
            # Recompress
            self._recompress()

            return len(records)

    def query(self, vector: np.ndarray, top_k: int = 10,
              include_values: bool = False,
              include_metadata: bool = True,
              filter: Optional[dict] = None) -> list[QueryResult]:
        """Find nearest neighbors to the query vector."""
        if self._vectors is None or len(self._ids) == 0:
            return []

        q = vector.astype(np.float32).reshape(1, -1)

        if self.metric == "cosine":
            # Normalize for cosine similarity
            q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)
        else:
            q_norm = q

        k = min(top_k, len(self._ids))

        if self._faiss_index is not None:
            try:
                import faiss
                distances, indices = self._faiss_index.search(q_norm, k)
                distances = distances[0]
                indices = indices[0]
            except ImportError:
                distances, indices = self._brute_force_search(q_norm, k)
        else:
            distances, indices = self._brute_force_search(q_norm, k)

        # Apply metadata filter
        results = []
        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self._ids):
                continue
            vec_id = self._ids[idx]
            meta = self._metadata.get(vec_id, {})

            if filter and not self._matches_filter(meta, filter):
                continue

            score = float(dist)
            if self.metric == "cosine":
                # FAISS returns L2 on normalized vectors; convert to cosine sim
                score = 1.0 - score / 2.0

            results.append(QueryResult(
                id=vec_id,
                score=score,
                values=self._vectors[idx].copy() if include_values else None,
                metadata=meta if include_metadata else None,
            ))

        return results[:top_k]

    def fetch(self, ids: list[str]) -> dict[str, VectorRecord]:
        """Fetch vectors by ID."""
        result = {}
        for vec_id in ids:
            if vec_id in self._id_to_idx:
                idx = self._id_to_idx[vec_id]
                result[vec_id] = VectorRecord(
                    id=vec_id,
                    values=self._vectors[idx].copy(),
                    metadata=self._metadata.get(vec_id, {}),
                )
        return result

    def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID. Returns count deleted."""
        with self._lock:
            deleted = 0
            keep_mask = np.ones(len(self._ids), dtype=bool)

            for vec_id in ids:
                if vec_id in self._id_to_idx:
                    idx = self._id_to_idx[vec_id]
                    keep_mask[idx] = False
                    del self._id_to_idx[vec_id]
                    del self._metadata[vec_id]
                    deleted += 1

            if deleted > 0:
                self._ids = [self._ids[i] for i in range(len(keep_mask)) if keep_mask[i]]
                if self._vectors is not None:
                    self._vectors = self._vectors[keep_mask]
                # Rebuild index mapping
                self._id_to_idx = {vid: i for i, vid in enumerate(self._ids)}
                self._dirty = True
                self._rebuild_faiss()
                self._recompress()

            return deleted

    def stats(self) -> dict:
        """Return namespace statistics."""
        raw_bytes = 0
        if self._vectors is not None:
            raw_bytes = self._vectors.nbytes

        compressed_bytes = len(self._compressed) if self._compressed else 0

        return {
            "vector_count": len(self._ids),
            "dimension": self.dimension,
            "metric": self.metric,
            "raw_bytes": raw_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": raw_bytes / max(compressed_bytes, 1) if compressed_bytes > 0 else 0,
            "compression_mode": self.compression_mode,
        }

    def _brute_force_search(self, q_norm: np.ndarray, k: int):
        """Fallback brute-force search when FAISS is unavailable."""
        vecs = self._vectors
        if self.metric == "cosine":
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
            vecs_norm = vecs / norms
            # L2 distance on normalized vectors = 2 - 2*cos_sim
            diffs = vecs_norm - q_norm
            distances = np.sum(diffs ** 2, axis=1)
        elif self.metric == "euclidean":
            diffs = vecs - q_norm
            distances = np.sum(diffs ** 2, axis=1)
        elif self.metric == "dotproduct":
            distances = -np.dot(vecs, q_norm.T).ravel()
        else:
            diffs = vecs - q_norm
            distances = np.sum(diffs ** 2, axis=1)

        top_k_idx = np.argsort(distances)[:k]
        return distances[top_k_idx], top_k_idx

    def _rebuild_faiss(self):
        """Rebuild FAISS index from current vectors."""
        if self._vectors is None or len(self._ids) < 10:
            self._faiss_index = None
            return

        try:
            import faiss
            dim = self.dimension
            if self.metric == "cosine":
                # Normalize vectors, then use L2
                norms = np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-10
                normed = (self._vectors / norms).astype(np.float32)
                index = faiss.IndexFlatL2(dim)
                index.add(normed)
            elif self.metric == "dotproduct":
                index = faiss.IndexFlatIP(dim)
                index.add(self._vectors.astype(np.float32))
            else:
                index = faiss.IndexFlatL2(dim)
                index.add(self._vectors.astype(np.float32))
            self._faiss_index = index
        except ImportError:
            self._faiss_index = None

    def _recompress(self):
        """Compress vectors with UMC."""
        if self._vectors is None or not self._dirty:
            return
        try:
            import umc
            # Reshape for UMC: (1, n_vectors, dim) — one "window" of all vectors
            data = self._vectors.reshape(1, -1, self.dimension)
            self._compressed = umc.compress(data, mode=self.compression_mode)
            self._dirty = False
        except Exception:
            self._compressed = None

    @staticmethod
    def _matches_filter(metadata: dict, filter_dict: dict) -> bool:
        """Check if metadata matches Pinecone-style filter."""
        for key, condition in filter_dict.items():
            if key not in metadata:
                return False
            if isinstance(condition, dict):
                # Operator-based: {"$eq": val}, {"$in": [...]}, etc.
                for op, val in condition.items():
                    actual = metadata[key]
                    if op == "$eq" and actual != val:
                        return False
                    elif op == "$ne" and actual == val:
                        return False
                    elif op == "$in" and actual not in val:
                        return False
                    elif op == "$nin" and actual in val:
                        return False
                    elif op == "$gt" and not (actual > val):
                        return False
                    elif op == "$gte" and not (actual >= val):
                        return False
                    elif op == "$lt" and not (actual < val):
                        return False
                    elif op == "$lte" and not (actual <= val):
                        return False
            else:
                # Simple equality
                if metadata[key] != condition:
                    return False
        return True

    def save(self, path: Path):
        """Save namespace to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "dimension": self.dimension,
            "metric": self.metric,
            "compression_mode": self.compression_mode,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))

        # Save compressed vectors
        if self._compressed:
            (path / "vectors.umc").write_bytes(self._compressed)

        # Save raw vectors (for quick startup without decompression)
        if self._vectors is not None:
            np.save(str(path / "vectors.npy"), self._vectors)

        # Save metadata and IDs
        meta_data = {
            "ids": self._ids,
            "metadata": self._metadata,
        }
        (path / "metadata.json").write_text(json.dumps(meta_data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "VectorNamespace":
        """Load namespace from disk."""
        config = json.loads((path / "config.json").read_text())
        ns = cls(
            dimension=config["dimension"],
            metric=config.get("metric", "cosine"),
            compression_mode=config.get("compression_mode", "lossless"),
        )

        # Load vectors
        vectors_path = path / "vectors.npy"
        if vectors_path.exists():
            ns._vectors = np.load(str(vectors_path))

        # Load metadata and IDs
        meta_path = path / "metadata.json"
        if meta_path.exists():
            meta_data = json.loads(meta_path.read_text())
            ns._ids = meta_data["ids"]
            ns._metadata = meta_data["metadata"]
            ns._id_to_idx = {vid: i for i, vid in enumerate(ns._ids)}

        # Load compressed data
        umc_path = path / "vectors.umc"
        if umc_path.exists():
            ns._compressed = umc_path.read_bytes()

        # Rebuild FAISS index
        ns._rebuild_faiss()

        return ns


class VectorEngine:
    """Multi-index vector database engine.

    Manages multiple indexes, each with multiple namespaces.
    Pinecone-compatible API surface.
    """

    def __init__(self, data_dir: str = "./vectorforge_data"):
        self.data_dir = Path(data_dir)
        self._indexes: dict[str, dict[str, VectorNamespace]] = {}
        self._index_configs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create_index(self, name: str, dimension: int,
                     metric: str = "cosine",
                     compression_mode: str = "lossless") -> dict:
        """Create a new vector index."""
        with self._lock:
            if name in self._indexes:
                raise ValueError(f"Index '{name}' already exists")
            self._indexes[name] = {}
            self._index_configs[name] = {
                "name": name,
                "dimension": dimension,
                "metric": metric,
                "compression_mode": compression_mode,
                "created_at": time.time(),
            }
            return self._index_configs[name]

    def delete_index(self, name: str):
        """Delete an index and all its namespaces."""
        with self._lock:
            if name not in self._indexes:
                raise KeyError(f"Index '{name}' not found")
            del self._indexes[name]
            del self._index_configs[name]

    def list_indexes(self) -> list[dict]:
        """List all indexes."""
        return list(self._index_configs.values())

    def describe_index(self, name: str) -> dict:
        """Get index configuration and stats."""
        if name not in self._indexes:
            raise KeyError(f"Index '{name}' not found")
        config = self._index_configs[name].copy()
        namespaces = {}
        total_vectors = 0
        for ns_name, ns in self._indexes[name].items():
            ns_stats = ns.stats()
            namespaces[ns_name] = ns_stats
            total_vectors += ns_stats["vector_count"]
        config["namespaces"] = namespaces
        config["total_vector_count"] = total_vectors
        return config

    def _get_namespace(self, index_name: str,
                       namespace: str = "") -> VectorNamespace:
        """Get or create a namespace within an index."""
        if index_name not in self._indexes:
            raise KeyError(f"Index '{index_name}' not found")

        config = self._index_configs[index_name]
        if namespace not in self._indexes[index_name]:
            self._indexes[index_name][namespace] = VectorNamespace(
                dimension=config["dimension"],
                metric=config.get("metric", "cosine"),
                compression_mode=config.get("compression_mode", "lossless"),
            )
        return self._indexes[index_name][namespace]

    def upsert(self, index_name: str, vectors: list[dict],
               namespace: str = "") -> dict:
        """Upsert vectors into an index namespace.

        Args:
            index_name: Target index.
            vectors: List of {"id": str, "values": list[float], "metadata": dict}.
            namespace: Namespace within the index (default: "").

        Returns:
            {"upserted_count": int}
        """
        ns = self._get_namespace(index_name, namespace)
        records = []
        for v in vectors:
            records.append(VectorRecord(
                id=v["id"],
                values=np.array(v["values"], dtype=np.float32),
                metadata=v.get("metadata", {}),
            ))
        count = ns.upsert(records)
        return {"upserted_count": count}

    def query(self, index_name: str, vector: list[float],
              top_k: int = 10, namespace: str = "",
              include_values: bool = False,
              include_metadata: bool = True,
              filter: Optional[dict] = None) -> dict:
        """Query nearest neighbors.

        Returns:
            {"matches": [{"id": str, "score": float, "values": list, "metadata": dict}]}
        """
        ns = self._get_namespace(index_name, namespace)
        q = np.array(vector, dtype=np.float32)
        results = ns.query(
            q, top_k=top_k,
            include_values=include_values,
            include_metadata=include_metadata,
            filter=filter,
        )
        matches = []
        for r in results:
            match = {"id": r.id, "score": r.score}
            if include_values and r.values is not None:
                match["values"] = r.values.tolist()
            if include_metadata and r.metadata is not None:
                match["metadata"] = r.metadata
            matches.append(match)
        return {"matches": matches, "namespace": namespace}

    def fetch(self, index_name: str, ids: list[str],
              namespace: str = "") -> dict:
        """Fetch vectors by ID."""
        ns = self._get_namespace(index_name, namespace)
        records = ns.fetch(ids)
        vectors = {}
        for vec_id, rec in records.items():
            vectors[vec_id] = {
                "id": rec.id,
                "values": rec.values.tolist(),
                "metadata": rec.metadata,
            }
        return {"vectors": vectors, "namespace": namespace}

    def delete(self, index_name: str, ids: list[str],
               namespace: str = "") -> dict:
        """Delete vectors by ID."""
        ns = self._get_namespace(index_name, namespace)
        count = ns.delete(ids)
        return {"deleted_count": count}

    def describe_index_stats(self, index_name: str) -> dict:
        """Get detailed statistics for an index."""
        return self.describe_index(index_name)

    def save(self):
        """Persist all indexes to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Save engine config
        engine_config = {
            "indexes": self._index_configs,
        }
        (self.data_dir / "engine.json").write_text(
            json.dumps(engine_config, indent=2, default=str)
        )

        # Save each namespace
        for idx_name, namespaces in self._indexes.items():
            for ns_name, ns in namespaces.items():
                ns_dir = self.data_dir / idx_name / (ns_name or "_default")
                ns.save(ns_dir)

    @classmethod
    def load(cls, data_dir: str = "./vectorforge_data") -> "VectorEngine":
        """Load engine state from disk."""
        engine = cls(data_dir=data_dir)
        config_path = engine.data_dir / "engine.json"

        if not config_path.exists():
            return engine

        engine_config = json.loads(config_path.read_text())
        engine._index_configs = engine_config.get("indexes", {})

        for idx_name, idx_config in engine._index_configs.items():
            engine._indexes[idx_name] = {}
            idx_dir = engine.data_dir / idx_name
            if idx_dir.exists():
                for ns_dir in idx_dir.iterdir():
                    if ns_dir.is_dir() and (ns_dir / "config.json").exists():
                        ns_name = ns_dir.name
                        if ns_name == "_default":
                            ns_name = ""
                        ns = VectorNamespace.load(ns_dir)
                        engine._indexes[idx_name][ns_name] = ns

        return engine
