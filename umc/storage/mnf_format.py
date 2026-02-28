"""Binary .mnf file format: manifold-native file storage.

HEADER (64 bytes fixed):
    magic: bytes[4] = b'MNF1'
    version: uint16
    domain_id: uint16          # 0=financial, 1=image, 2=audio...
    n_samples: uint32
    latent_dim: uint16
    n_charts: uint16
    coord_dtype: uint8         # 0=float16, 1=float32
    decoder_hash: bytes[32]    # SHA-256 of decoder weights
    flags: uint8               # bit 0: has_index, bit 1: has_confidence, bit 2: has_scale_factors, bit 3: has_vq_codes, bit 4: has_residual, bit 5: has_tiered
    reserved: bytes[14]

COORDINATE BLOCK:
    n_samples * latent_dim * sizeof(coord_dtype) bytes

CHART ID BLOCK:
    n_samples * 1 byte (uint8)

CONFIDENCE BLOCK (if flags.has_confidence):
    n_samples * 2 bytes (float16)

SCALE FACTORS BLOCK (if flags.has_scale_factors):
    n_features: uint16
    means: n_samples * n_features * 2 bytes (float16)
    stds: n_samples * n_features * 2 bytes (float16)

VQ CODES BLOCK (if flags.has_vq_codes):
    Length: uint64
    Data: compressed VQ indices from entropy.py

RESIDUAL BLOCK (if flags.has_residual):
    Length: uint64
    Data: serialized LosslessEncoding bytes from codec.lossless

TIERED BLOCK (if flags.has_tiered):
    Length: uint64
    Data: serialized TieredEncoding bytes from codec.tiered
    Contains VQ search index + compressed storage (lossless or near-lossless)

INDEX BLOCK (if flags.has_index):
    FAISS index serialized bytes (prefixed with uint64 length)
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

MAGIC = b"MNF1"
HEADER_SIZE = 64
HEADER_FORMAT = "<4sHHIHHB32sB14s"  # total = 4+2+2+4+2+2+1+32+1+14 = 64


@dataclass
class MNFHeader:
    """Parsed .mnf file header."""
    magic: bytes
    version: int
    domain_id: int
    n_samples: int
    latent_dim: int
    n_charts: int
    coord_dtype: int           # 0=float16, 1=float32
    decoder_hash: bytes
    flags: int
    reserved: bytes = b"\x00" * 14

    @property
    def has_index(self) -> bool:
        return bool(self.flags & 0x01)

    @property
    def has_confidence(self) -> bool:
        return bool(self.flags & 0x02)

    @property
    def has_scale_factors(self) -> bool:
        return bool(self.flags & 0x04)

    @property
    def has_vq_codes(self) -> bool:
        return bool(self.flags & 0x08)

    @property
    def has_residual(self) -> bool:
        return bool(self.flags & 0x10)

    @property
    def has_tiered(self) -> bool:
        return bool(self.flags & 0x20)

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.float16 if self.coord_dtype == 0 else np.float32

    @property
    def bytes_per_coord(self) -> int:
        return 2 if self.coord_dtype == 0 else 4

    def to_bytes(self) -> bytes:
        return struct.pack(
            HEADER_FORMAT,
            self.magic,
            self.version,
            self.domain_id,
            self.n_samples,
            self.latent_dim,
            self.n_charts,
            self.coord_dtype,
            self.decoder_hash,
            self.flags,
            self.reserved,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "MNFHeader":
        fields = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
        return cls(
            magic=fields[0],
            version=fields[1],
            domain_id=fields[2],
            n_samples=fields[3],
            latent_dim=fields[4],
            n_charts=fields[5],
            coord_dtype=fields[6],
            decoder_hash=fields[7],
            flags=fields[8],
            reserved=fields[9],
        )


@dataclass
class MNFFile:
    """In-memory representation of a .mnf file."""
    header: MNFHeader
    coordinates: np.ndarray       # (n_samples, latent_dim)
    chart_ids: np.ndarray         # (n_samples,) uint8
    confidences: Optional[np.ndarray] = None  # (n_samples,) float16
    scale_means: Optional[np.ndarray] = None  # (n_samples, n_features) float16
    scale_stds: Optional[np.ndarray] = None   # (n_samples, n_features) float16
    index_data: Optional[bytes] = None        # Serialized FAISS index
    vq_code_data: Optional[bytes] = None      # Compressed VQ indices
    residual_data: Optional[bytes] = None     # Lossless residual (serialized LosslessEncoding)
    tiered_data: Optional[bytes] = None       # Serialized TieredEncoding (VQ search + storage)


class MNFWriter:
    """Write .mnf binary files."""

    def write(
        self,
        path: str,
        coordinates: np.ndarray,
        chart_ids: np.ndarray,
        decoder_hash: bytes,
        confidences: Optional[np.ndarray] = None,
        scale_means: Optional[np.ndarray] = None,
        scale_stds: Optional[np.ndarray] = None,
        domain_id: int = 0,
        version: int = 1,
        index_data: Optional[bytes] = None,
        coord_dtype: str = "float16",
        vq_code_data: Optional[bytes] = None,
        residual_data: Optional[bytes] = None,
        tiered_data: Optional[bytes] = None,
    ) -> int:
        """Write manifold data to a .mnf file.

        Args:
            path: Output file path.
            coordinates: (n_samples, latent_dim) array.
            chart_ids: (n_samples,) uint8 array.
            decoder_hash: SHA-256 hash of decoder weights (32 bytes).
            confidences: Optional (n_samples,) confidence scores.
            domain_id: Domain identifier (0=financial).
            version: Format version.
            index_data: Optional serialized FAISS index bytes.
            coord_dtype: 'float16' or 'float32'.

        Returns:
            Total bytes written.
        """
        n_samples, latent_dim = coordinates.shape
        dtype_code = 0 if coord_dtype == "float16" else 1
        np_dtype = np.float16 if coord_dtype == "float16" else np.float32

        flags = 0
        if index_data is not None:
            flags |= 0x01
        if confidences is not None:
            flags |= 0x02
        if scale_means is not None and scale_stds is not None:
            flags |= 0x04
        if vq_code_data is not None:
            flags |= 0x08
        if residual_data is not None:
            flags |= 0x10
        if tiered_data is not None:
            flags |= 0x20

        # Ensure decoder_hash is exactly 32 bytes
        if len(decoder_hash) < 32:
            decoder_hash = decoder_hash + b"\x00" * (32 - len(decoder_hash))
        decoder_hash = decoder_hash[:32]

        n_charts = int(chart_ids.max()) + 1 if len(chart_ids) > 0 else 0

        header = MNFHeader(
            magic=MAGIC,
            version=version,
            domain_id=domain_id,
            n_samples=n_samples,
            latent_dim=latent_dim,
            n_charts=n_charts,
            coord_dtype=dtype_code,
            decoder_hash=decoder_hash,
            flags=flags,
        )

        path = Path(path)
        total_written = 0

        with open(path, "wb") as f:
            # Header
            header_bytes = header.to_bytes()
            f.write(header_bytes)
            total_written += len(header_bytes)

            # Coordinate block
            coord_bytes = coordinates.astype(np_dtype).tobytes()
            f.write(coord_bytes)
            total_written += len(coord_bytes)

            # Chart ID block
            chart_bytes = chart_ids.astype(np.uint8).tobytes()
            f.write(chart_bytes)
            total_written += len(chart_bytes)

            # Confidence block
            if confidences is not None:
                conf_bytes = confidences.astype(np.float16).tobytes()
                f.write(conf_bytes)
                total_written += len(conf_bytes)

            # Scale factors block
            if scale_means is not None and scale_stds is not None:
                n_features = scale_means.shape[1]
                f.write(struct.pack("<H", n_features))
                means_bytes = scale_means.astype(np.float16).tobytes()
                stds_bytes = scale_stds.astype(np.float16).tobytes()
                f.write(means_bytes)
                f.write(stds_bytes)
                total_written += 2 + len(means_bytes) + len(stds_bytes)

            # VQ code block
            if vq_code_data is not None:
                f.write(struct.pack("<Q", len(vq_code_data)))
                f.write(vq_code_data)
                total_written += 8 + len(vq_code_data)

            # Residual block (lossless correction)
            if residual_data is not None:
                f.write(struct.pack("<Q", len(residual_data)))
                f.write(residual_data)
                total_written += 8 + len(residual_data)

            # Tiered block (VQ search + compressed storage)
            if tiered_data is not None:
                f.write(struct.pack("<Q", len(tiered_data)))
                f.write(tiered_data)
                total_written += 8 + len(tiered_data)

            # Index block
            if index_data is not None:
                # Prefix with length
                f.write(struct.pack("<Q", len(index_data)))
                f.write(index_data)
                total_written += 8 + len(index_data)

        return total_written


class MNFReader:
    """Read .mnf binary files."""

    def read(self, path: str) -> MNFFile:
        """Read a complete .mnf file into memory.

        Args:
            path: Path to .mnf file.

        Returns:
            MNFFile with all data loaded.
        """
        path = Path(path)
        with open(path, "rb") as f:
            # Header
            header_bytes = f.read(HEADER_SIZE)
            header = MNFHeader.from_bytes(header_bytes)

            if header.magic != MAGIC:
                raise ValueError(f"Invalid .mnf file: bad magic {header.magic!r}")

            np_dtype = header.numpy_dtype

            # Coordinate block
            coord_size = header.n_samples * header.latent_dim * header.bytes_per_coord
            coord_bytes = f.read(coord_size)
            coordinates = np.frombuffer(coord_bytes, dtype=np_dtype).reshape(
                header.n_samples, header.latent_dim
            ).copy()

            # Chart ID block
            chart_bytes = f.read(header.n_samples)
            chart_ids = np.frombuffer(chart_bytes, dtype=np.uint8).copy()

            # Confidence block
            confidences = None
            if header.has_confidence:
                conf_size = header.n_samples * 2  # float16
                conf_bytes = f.read(conf_size)
                confidences = np.frombuffer(conf_bytes, dtype=np.float16).copy()

            # Scale factors block
            scale_means = None
            scale_stds = None
            if header.has_scale_factors:
                n_features = struct.unpack("<H", f.read(2))[0]
                means_size = header.n_samples * n_features * 2  # float16
                stds_size = header.n_samples * n_features * 2
                means_bytes = f.read(means_size)
                stds_bytes = f.read(stds_size)
                scale_means = np.frombuffer(means_bytes, dtype=np.float16).reshape(
                    header.n_samples, n_features
                ).copy()
                scale_stds = np.frombuffer(stds_bytes, dtype=np.float16).reshape(
                    header.n_samples, n_features
                ).copy()

            # VQ code block
            vq_code_data = None
            if header.has_vq_codes:
                vq_len = struct.unpack("<Q", f.read(8))[0]
                vq_code_data = f.read(vq_len)

            # Residual block
            residual_data = None
            if header.has_residual:
                res_len = struct.unpack("<Q", f.read(8))[0]
                residual_data = f.read(res_len)

            # Tiered block
            tiered_data = None
            if header.has_tiered:
                tiered_len = struct.unpack("<Q", f.read(8))[0]
                tiered_data = f.read(tiered_len)

            # Index block
            index_data = None
            if header.has_index:
                index_len = struct.unpack("<Q", f.read(8))[0]
                index_data = f.read(index_len)

        return MNFFile(
            header=header,
            coordinates=coordinates,
            chart_ids=chart_ids,
            confidences=confidences,
            scale_means=scale_means,
            scale_stds=scale_stds,
            index_data=index_data,
            vq_code_data=vq_code_data,
            residual_data=residual_data,
            tiered_data=tiered_data,
        )

    def read_header(self, path: str) -> MNFHeader:
        """Read only the header (fast metadata check)."""
        with open(path, "rb") as f:
            return MNFHeader.from_bytes(f.read(HEADER_SIZE))

    def read_coordinates(
        self,
        path: str,
        indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Read coordinate block, optionally selecting specific samples."""
        with open(path, "rb") as f:
            header_bytes = f.read(HEADER_SIZE)
            header = MNFHeader.from_bytes(header_bytes)
            np_dtype = header.numpy_dtype

            coord_size = header.n_samples * header.latent_dim * header.bytes_per_coord
            coord_bytes = f.read(coord_size)
            coordinates = np.frombuffer(coord_bytes, dtype=np_dtype).reshape(
                header.n_samples, header.latent_dim
            ).copy()

        if indices is not None:
            return coordinates[indices]
        return coordinates
