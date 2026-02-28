"""Decoder registry and versioning for manifold storage."""

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch


@dataclass
class DecoderRecord:
    """Registry entry for a decoder model."""
    decoder_hash: str          # SHA-256 hex digest
    domain_id: int
    latent_dim: int
    model_path: str
    description: str = ""
    created_at: str = ""


class DecoderManifest:
    """Registry mapping decoder hashes to weight files."""

    def __init__(self, manifest_path: str = "manifests/decoders.json"):
        self.manifest_path = Path(manifest_path)
        self._records: dict[str, DecoderRecord] = {}
        if self.manifest_path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.manifest_path) as f:
            data = json.load(f)
        for key, val in data.items():
            self._records[key] = DecoderRecord(**val)

    def _save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in self._records.items()}
        with open(self.manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def register(self, record: DecoderRecord) -> None:
        """Register a decoder in the manifest."""
        self._records[record.decoder_hash] = record
        self._save()

    def lookup(self, decoder_hash: str) -> Optional[DecoderRecord]:
        """Look up a decoder by its hash."""
        return self._records.get(decoder_hash)

    def list_all(self) -> list[DecoderRecord]:
        """List all registered decoders."""
        return list(self._records.values())

    @staticmethod
    def compute_hash(state_dict: dict) -> str:
        """Compute SHA-256 hash of a model's state dict."""
        hasher = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            tensor_bytes = state_dict[key].cpu().numpy().tobytes()
            hasher.update(key.encode())
            hasher.update(tensor_bytes)
        return hasher.hexdigest()

    @staticmethod
    def compute_hash_bytes(state_dict: dict) -> bytes:
        """Compute SHA-256 hash as raw bytes (for .mnf header)."""
        hex_hash = DecoderManifest.compute_hash(state_dict)
        return bytes.fromhex(hex_hash)[:32]
