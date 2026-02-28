"""VectorForge REST API — Pinecone-compatible vector database server.

Usage:
    python -m umc.vectordb.server                    # Start on port 8100
    python -m umc.vectordb.server --port 9000        # Custom port
    python -m umc.vectordb.server --data-dir ./data  # Custom data directory

API compatibility: Mirrors Pinecone's REST API for easy migration.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .engine import VectorEngine

# ---------------------------------------------------------------------------
# Pydantic models (Pinecone-compatible request/response shapes)
# ---------------------------------------------------------------------------


class CreateIndexRequest(BaseModel):
    name: str
    dimension: int
    metric: str = "cosine"
    compression_mode: str = "lossless"


class UpsertVector(BaseModel):
    id: str
    values: list[float]
    metadata: Optional[dict[str, Any]] = None


class UpsertRequest(BaseModel):
    vectors: list[UpsertVector]
    namespace: str = ""


class QueryRequest(BaseModel):
    vector: list[float]
    top_k: int = Field(default=10, alias="topK")
    namespace: str = ""
    include_values: bool = Field(default=False, alias="includeValues")
    include_metadata: bool = Field(default=True, alias="includeMetadata")
    filter: Optional[dict[str, Any]] = None

    model_config = {"populate_by_name": True}


class FetchRequest(BaseModel):
    ids: list[str]
    namespace: str = ""


class DeleteRequest(BaseModel):
    ids: list[str]
    namespace: str = ""


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def create_app(data_dir: str = "./vectorforge_data") -> FastAPI:
    """Create the VectorForge FastAPI application."""

    app = FastAPI(
        title="VectorForge",
        description=(
            "Compressed vector database powered by UMC. "
            "Pinecone-compatible REST API with 3-5x storage compression."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    engine = VectorEngine(data_dir=data_dir)

    # Try to load existing state
    try:
        engine = VectorEngine.load(data_dir)
    except Exception:
        pass

    _start_time = time.time()

    # ---- Health & Info ----

    @app.get("/")
    async def root():
        return {
            "name": "VectorForge",
            "version": "0.1.0",
            "engine": "UMC (Universal Manifold Codec)",
            "uptime_seconds": time.time() - _start_time,
        }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ---- Index Management ----

    @app.post("/indexes")
    async def create_index(req: CreateIndexRequest):
        try:
            result = engine.create_index(
                name=req.name,
                dimension=req.dimension,
                metric=req.metric,
                compression_mode=req.compression_mode,
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.get("/indexes")
    async def list_indexes():
        return {"indexes": engine.list_indexes()}

    @app.get("/indexes/{index_name}")
    async def describe_index(index_name: str):
        try:
            return engine.describe_index(index_name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")

    @app.delete("/indexes/{index_name}")
    async def delete_index(index_name: str):
        try:
            engine.delete_index(index_name)
            return {"status": "ok"}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")

    # ---- Vector Operations ----

    @app.post("/indexes/{index_name}/vectors/upsert")
    async def upsert_vectors(index_name: str, req: UpsertRequest):
        try:
            vectors = [
                {"id": v.id, "values": v.values, "metadata": v.metadata or {}}
                for v in req.vectors
            ]
            result = engine.upsert(index_name, vectors, namespace=req.namespace)
            return result
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/indexes/{index_name}/query")
    async def query_vectors(index_name: str, req: QueryRequest):
        try:
            result = engine.query(
                index_name,
                vector=req.vector,
                top_k=req.top_k,
                namespace=req.namespace,
                include_values=req.include_values,
                include_metadata=req.include_metadata,
                filter=req.filter,
            )
            return result
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")

    @app.post("/indexes/{index_name}/vectors/fetch")
    async def fetch_vectors(index_name: str, req: FetchRequest):
        try:
            result = engine.fetch(index_name, req.ids, namespace=req.namespace)
            return result
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")

    @app.post("/indexes/{index_name}/vectors/delete")
    async def delete_vectors(index_name: str, req: DeleteRequest):
        try:
            result = engine.delete(index_name, req.ids, namespace=req.namespace)
            return result
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")

    @app.get("/indexes/{index_name}/stats")
    async def index_stats(index_name: str):
        try:
            return engine.describe_index_stats(index_name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")

    # ---- Persistence ----

    @app.post("/admin/save")
    async def save_state():
        engine.save()
        return {"status": "saved"}

    @app.get("/admin/compression-stats")
    async def compression_stats():
        """Show UMC compression statistics across all indexes."""
        stats = {}
        for idx_config in engine.list_indexes():
            idx_name = idx_config["name"]
            try:
                idx_stats = engine.describe_index(idx_name)
                stats[idx_name] = idx_stats
            except Exception:
                pass
        return stats

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="VectorForge — Compressed vector database server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8100, help="Bind port")
    parser.add_argument("--data-dir", default="./vectorforge_data", help="Data directory")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    import uvicorn

    print(f"Starting VectorForge on http://{args.host}:{args.port}")
    print(f"Data directory: {args.data_dir}")
    print(f"API docs: http://localhost:{args.port}/docs")

    # Create app with data_dir
    app = create_app(data_dir=args.data_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
