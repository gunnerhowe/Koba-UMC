"""Neural codec classes (torch-dependent).

These classes require torch, faiss, and other heavy dependencies.
They are lazy-imported from umc.__init__ so that `import umc` and
`umc.compress()` work without torch installed.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .config import UMCConfig
from .encoder.base import EncodingResult
from .encoder.vae import VAEEncoder
from .encoder.conv_encoder import ConvEncoder
from .encoder.adaptive import AdaptiveEncoder
from .encoder.hvqvae_encoder import HVQVAEEncoder
from .decoder.mlp_decoder import MLPDecoder
from .decoder.conv_decoder import ConvDecoder
from .decoder.hvqvae_decoder import HVQVAEDecoder
from .data.preprocessors import OHLCVPreprocessor, WindowNormalizer, create_windows, prepare_dataloaders
from .storage.mnf_format import MNFWriter, MNFReader, MNFFile, MNFHeader
from .storage.manifest import DecoderManifest
from .training.trainer import UMCTrainer
from .processor.search import ManifoldSearch, SearchResult
from .processor.cluster import ManifoldCluster, ClusterResult
from .processor.anomaly import ManifoldAnomalyDetector
from .codec.tiered import TieredCodec, TieredEncoding, serialize_tiered, deserialize_tiered


class ManifoldCodecResult:
    """Wrapper around encoded manifold data with processor operations."""

    def __init__(
        self,
        coordinates: np.ndarray,
        chart_ids: np.ndarray,
        confidences: np.ndarray,
        decoder_hash: bytes,
        config: UMCConfig,
        scale_means: Optional[np.ndarray] = None,
        scale_stds: Optional[np.ndarray] = None,
    ):
        self.coordinates = coordinates
        self.chart_ids = chart_ids
        self.confidences = confidences
        self.decoder_hash = decoder_hash
        self.config = config
        self.scale_means = scale_means
        self.scale_stds = scale_stds
        self._search_index: Optional[ManifoldSearch] = None

    @property
    def n_samples(self) -> int:
        return self.coordinates.shape[0]

    @property
    def latent_dim(self) -> int:
        return self.coordinates.shape[1]

    def save(self, path: str) -> int:
        """Save to .mnf file format."""
        writer = MNFWriter()
        return writer.write(
            path=path,
            coordinates=self.coordinates,
            chart_ids=self.chart_ids,
            decoder_hash=self.decoder_hash,
            confidences=self.confidences,
            scale_means=self.scale_means,
            scale_stds=self.scale_stds,
            domain_id=0,
            coord_dtype=self.config.coordinate_dtype,
        )

    @classmethod
    def load(cls, path: str, config: Optional[UMCConfig] = None) -> "ManifoldCodecResult":
        """Load from .mnf file."""
        reader = MNFReader()
        mnf = reader.read(path)
        if config is None:
            config = UMCConfig()
        return cls(
            coordinates=mnf.coordinates.astype(np.float32),
            chart_ids=mnf.chart_ids,
            confidences=mnf.confidences if mnf.confidences is not None else np.ones(mnf.header.n_samples, dtype=np.float32),
            decoder_hash=mnf.header.decoder_hash,
            config=config,
            scale_means=mnf.scale_means.astype(np.float32) if mnf.scale_means is not None else None,
            scale_stds=mnf.scale_stds.astype(np.float32) if mnf.scale_stds is not None else None,
        )

    def _get_search_index(self) -> ManifoldSearch:
        if self._search_index is None:
            self._search_index = ManifoldSearch(self.coordinates)
        return self._search_index

    def search(self, query: np.ndarray, k: int = 10) -> SearchResult:
        """Find k nearest neighbors in manifold space (never decodes)."""
        return self._get_search_index().query(query, k=k)

    def cluster(self, n_clusters: int = 5) -> ClusterResult:
        """K-means clustering in manifold space (never decodes)."""
        return ManifoldCluster().cluster(self.coordinates, n_clusters=n_clusters)

    def anomaly_scores(self, **kwargs) -> np.ndarray:
        """Anomaly detection in manifold space (never decodes)."""
        detector = ManifoldAnomalyDetector(self.coordinates, **kwargs)
        return detector.score(self.coordinates)

    def manifold_distance(self, other: "ManifoldCodecResult") -> float:
        """Distance between two manifold datasets (centroid distance)."""
        c1 = self.coordinates.mean(axis=0)
        c2 = other.coordinates.mean(axis=0)
        return float(np.linalg.norm(c1 - c2))


class ManifoldCodec:
    """High-level codec API for encoding/decoding data via manifold representation.

    Requires: torch, faiss-cpu, pandas, scipy, scikit-learn
    """

    def __init__(self, config: Optional[UMCConfig] = None):
        self.config = config or UMCConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build encoder/decoder based on config type
        if self.config.encoder_type == "hvqvae":
            self.encoder = HVQVAEEncoder(self.config)
        elif self.config.encoder_type == "conv":
            self.encoder = ConvEncoder(self.config)
        elif self.config.encoder_type == "adaptive":
            self.encoder = AdaptiveEncoder(self.config)
        else:
            self.encoder = VAEEncoder(self.config)

        if self.config.encoder_type == "hvqvae":
            self.decoder = HVQVAEDecoder(self.config)
        elif self.config.decoder_type == "conv":
            self.decoder = ConvDecoder(self.config)
        else:
            self.decoder = MLPDecoder(self.config)

        self.preprocessor = OHLCVPreprocessor(self.config)
        self.window_normalizer = WindowNormalizer() if self.config.per_window_normalize else None
        self.trainer: Optional[UMCTrainer] = None
        self._fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        val_split: float = 0.1,
        test_split: float = 0.1,
        verbose: bool = True,
    ) -> list[dict]:
        """Train the codec on OHLCV data."""
        train_loader, val_loader, test_loader = prepare_dataloaders(
            df, self.config, val_split=val_split, test_split=test_split,
        )

        self.trainer = UMCTrainer(
            self.encoder, self.decoder, self.config, device=str(self.device),
        )
        history = self.trainer.train(train_loader, val_loader, verbose=verbose)
        self._fitted = True
        return history

    def encode(self, df: pd.DataFrame) -> ManifoldCodecResult:
        """Encode raw data to manifold coordinates."""
        if not self._fitted:
            raise RuntimeError("Codec not fitted. Call .fit() first.")

        normalized = self.preprocessor.transform(df)
        windows = create_windows(normalized, self.config.window_size)

        scale_means = None
        scale_stds = None
        if self.window_normalizer is not None:
            windows, scale_means, scale_stds = self.window_normalizer.normalize(windows)

        self.encoder.eval()
        with torch.no_grad():
            x = torch.from_numpy(windows).float().to(self.device)
            result = self.encoder.encode(x)

        decoder_hash = DecoderManifest.compute_hash_bytes(self.decoder.state_dict())

        return ManifoldCodecResult(
            coordinates=result.z.cpu().numpy(),
            chart_ids=result.chart_id.cpu().numpy().astype(np.uint8),
            confidences=result.confidence.cpu().numpy(),
            decoder_hash=decoder_hash,
            config=self.config,
            scale_means=scale_means,
            scale_stds=scale_stds,
        )

    def decode(self, mnf_result: ManifoldCodecResult) -> np.ndarray:
        """Decode manifold coordinates back to time series windows."""
        if not self._fitted:
            raise RuntimeError("Codec not fitted. Call .fit() first.")

        self.decoder.eval()
        with torch.no_grad():
            z = torch.from_numpy(mnf_result.coordinates).float().to(self.device)
            chart_ids = torch.from_numpy(mnf_result.chart_ids.astype(np.int64)).to(self.device)
            x_hat = self.decoder.decode(z, chart_ids)

        result = x_hat.cpu().numpy()

        if mnf_result.scale_means is not None and mnf_result.scale_stds is not None:
            normalizer = WindowNormalizer()
            result = normalizer.denormalize(result, mnf_result.scale_means, mnf_result.scale_stds)

        return result

    def save(self, directory: str) -> None:
        """Save the full codec (encoder + decoder + config + preprocessor)."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "config": self.config,
            "preprocessor_params": self.preprocessor._fit_params,
        }, path / "codec.pt")

    @classmethod
    def from_pretrained(cls, directory: str) -> "ManifoldCodec":
        """Load a pretrained codec."""
        path = Path(directory)
        checkpoint = torch.load(path / "codec.pt", map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        codec = cls(config)
        codec.encoder.load_state_dict(checkpoint["encoder"])
        codec.decoder.load_state_dict(checkpoint["decoder"])
        codec.preprocessor._fit_params = checkpoint["preprocessor_params"]
        codec._fitted = True
        return codec


class TieredManifoldCodec:
    """Production-grade tiered compression codec.

    Two independent tiers:
        Tier 1 (Search):  VQ codebook indices for FAISS similarity search
        Tier 2 (Storage): Byte-transposed compressed data for retrieval

    Requires: torch, faiss-cpu

    Usage:
        codec = TieredManifoldCodec.from_checkpoint("results/v15b_best_state.pt")
        codec.encode_to_mnf(windows, "data.mnf")
        data = codec.decode_from_mnf("data.mnf")
        approx = codec.decode_from_mnf("data.mnf", mode="search")
        stats = codec.stats_from_mnf("data.mnf")
    """

    def __init__(
        self,
        encoder: HVQVAEEncoder,
        decoder: HVQVAEDecoder,
        config: UMCConfig,
        device: str = "cpu",
        storage_mode: str = "lossless",
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.device = device
        self.storage_mode = storage_mode
        self._codec = TieredCodec(
            encoder, decoder, device=device, storage_mode=storage_mode,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        storage_mode: str = "lossless",
    ) -> "TieredManifoldCodec":
        """Load from an experiment checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "config" in checkpoint and isinstance(checkpoint["config"], UMCConfig):
            config = checkpoint["config"]
        elif "config" in checkpoint and isinstance(checkpoint["config"], dict):
            config = UMCConfig(**checkpoint["config"])
        else:
            raise ValueError("Checkpoint missing 'config' key")

        encoder = HVQVAEEncoder(config)
        decoder = HVQVAEDecoder(config)

        if "encoder" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder"])
        elif "encoder_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder_state_dict"])

        if "decoder" in checkpoint:
            decoder.load_state_dict(checkpoint["decoder"])
        elif "decoder_state_dict" in checkpoint:
            decoder.load_state_dict(checkpoint["decoder_state_dict"])

        return cls(encoder, decoder, config, device=device, storage_mode=storage_mode)

    @classmethod
    def from_pretrained(
        cls,
        directory: str,
        device: str = "cpu",
        storage_mode: str = "lossless",
    ) -> "TieredManifoldCodec":
        """Load from a ManifoldCodec pretrained directory."""
        path = Path(directory) / "codec.pt"
        return cls.from_checkpoint(str(path), device=device, storage_mode=storage_mode)

    def encode(
        self, windows: np.ndarray, batch_size: int = 32
    ) -> TieredEncoding:
        """Encode windows into tiered compressed form."""
        return self._codec.encode(windows, batch_size=batch_size)

    def encode_to_mnf(
        self,
        windows: np.ndarray,
        path: str,
        batch_size: int = 32,
    ) -> int:
        """Encode windows and save as .mnf file."""
        encoding = self._codec.encode(windows, batch_size=batch_size)
        tiered_bytes = serialize_tiered(encoding)

        decoder_hash = DecoderManifest.compute_hash_bytes(self.decoder.state_dict())

        n_windows = encoding.n_windows
        coords = np.zeros((n_windows, 1), dtype=np.float16)
        chart_ids = np.zeros(n_windows, dtype=np.uint8)

        writer = MNFWriter()
        return writer.write(
            path=path,
            coordinates=coords,
            chart_ids=chart_ids,
            decoder_hash=decoder_hash,
            coord_dtype="float16",
            tiered_data=tiered_bytes,
        )

    def decode_from_mnf(
        self,
        path: str,
        mode: str = "storage",
        batch_size: int = 32,
    ) -> np.ndarray:
        """Decode a .mnf file back to original data."""
        reader = MNFReader()
        mnf = reader.read(path)

        if mnf.tiered_data is None:
            raise ValueError("MNF file does not contain tiered encoding data")

        encoding = deserialize_tiered(mnf.tiered_data)

        if mode == "storage":
            return self._codec.decode_storage(encoding)
        elif mode == "search":
            return self._codec.decode_search(encoding, batch_size=batch_size)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'storage' or 'search'.")

    def stats_from_mnf(self, path: str) -> dict:
        """Get compression statistics from a .mnf file."""
        file_path = Path(path)
        mnf_size = file_path.stat().st_size

        reader = MNFReader()
        mnf = reader.read(path)

        if mnf.tiered_data is None:
            raise ValueError("MNF file does not contain tiered encoding data")

        encoding = deserialize_tiered(mnf.tiered_data)
        stats = self._codec.compression_stats(encoding)
        stats["mnf_file_bytes"] = mnf_size
        stats["mnf_compression"] = encoding.raw_bytes / max(mnf_size, 1)
        return stats

    def search_from_mnf(
        self,
        index_path: str,
        query_windows: np.ndarray,
        k: int = 10,
        batch_size: int = 32,
    ) -> "SearchResult":
        """Search an encoded .mnf file for similar windows."""
        reader = MNFReader()
        mnf = reader.read(index_path)

        if mnf.tiered_data is None:
            raise ValueError("MNF file does not contain tiered encoding data")

        index_encoding = deserialize_tiered(mnf.tiered_data)

        query_encoding = self._codec.encode(query_windows, batch_size=batch_size)

        index_vq = index_encoding.vq_indices
        n_idx = len(index_vq.top_indices)
        idx_flat = np.concatenate([
            index_vq.top_indices.reshape(n_idx, 1).astype(np.float32),
            index_vq.bottom_indices.reshape(n_idx, -1).astype(np.float32),
        ], axis=1)

        query_vq = query_encoding.vq_indices
        n_q = len(query_vq.top_indices)
        q_flat = np.concatenate([
            query_vq.top_indices.reshape(n_q, 1).astype(np.float32),
            query_vq.bottom_indices.reshape(n_q, -1).astype(np.float32),
        ], axis=1)

        search_index = ManifoldSearch(idx_flat)
        return search_index.query(q_flat, k=k)
