"""VectorForge — Compressed vector database powered by UMC.

A Pinecone-compatible REST API for storing, searching, and managing
vector embeddings with UMC compression (3-5x storage reduction).
"""

from .engine import VectorEngine, VectorNamespace

__all__ = ["VectorEngine", "VectorNamespace"]
