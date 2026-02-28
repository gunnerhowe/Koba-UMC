"""Central configuration for the Universal Manifold Codec."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class UMCConfig:
    """All UMC hyperparameters in one place."""

    # --- Data ---
    window_size: int = 64
    features: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
    normalize: str = "returns"  # 'returns', 'minmax', 'zscore'
    per_window_normalize: bool = False  # Per-window zero-mean/unit-std normalization

    # --- Encoder ---
    max_latent_dim: int = 128
    min_latent_dim: int = 8
    encoder_hidden: Tuple[int, ...] = (256, 128, 64)
    encoder_type: str = "conv"  # 'conv', 'vae', 'vqvae', 'adaptive'
    num_charts: int = 16

    # --- Decoder ---
    decoder_hidden: Tuple[int, ...] = (64, 128, 256)
    decoder_type: str = "conv"  # 'conv', 'mlp', 'diffusion'
    chart_embedding_dim: int = 8

    # --- Architecture ---
    bottleneck_dim: int = 512  # FC bottleneck width in encoder/decoder
    pool_size_cap: int = 16  # Max adaptive pool size (0 = no cap, use full compressed length)
    bottleneck_dropout: float = 0.2  # Dropout in bottleneck layers
    conv_bottleneck: bool = False  # Use conv channel reduction instead of FC bottleneck
    conv_bottleneck_channels: int = 8  # Channels after reduction (flat_dim = this * compressed_len)

    # --- Hierarchical VQ-VAE ---
    d_model: int = 256  # Transformer model dimension
    n_heads: int = 8  # Number of attention heads
    n_encoder_layers: int = 6  # Transformer encoder depth
    n_decoder_layers: int = 6  # Transformer decoder depth
    d_ff: int = 1024  # Feed-forward hidden dimension
    patch_size: int = 8  # Non-overlapping patch size for tokenization
    vq_dim: int = 64  # VQ codebook vector dimension
    vq_top_n_codes: int = 512  # Top-level codebook size (global shape)
    vq_bottom_n_codes: int = 1024  # Bottom-level codebook size (local detail)
    vq_bottom_n_levels: int = 1  # Number of RVQ levels for bottom codes (1 = standard VQ)
    vq_commitment_weight: float = 0.25  # Commitment loss weight
    vq_ema_decay: float = 0.99  # EMA decay for codebook updates
    vq_dead_code_threshold: int = 2  # Min usage before code is considered dead
    vq_codebook_dim: int = 0  # Factorized codebook dimension (0 = same as vq_dim, no projection)
    vq_use_cosine_sim: bool = False  # Use cosine similarity for codebook lookup (L2-normalized)
    vq_rotation_trick: bool = False  # Use rotation trick for straight-through estimator
    vq_type: str = "vq"  # 'vq' (learned codebook) or 'fsq' (Finite Scalar Quantization, no collapse)
    fsq_top_levels: tuple = (8, 8)  # FSQ quantization levels for top VQ (product = implicit codebook size)
    fsq_bottom_levels: tuple = (8, 8, 4)  # FSQ levels per bottom residual quantizer (product = codes per level)
    transformer_dropout: float = 0.1  # Dropout in transformer blocks

    # --- Training ---
    batch_size: int = 256
    learning_rate: float = 1e-3
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_warmup_epochs: int = 20
    sparsity_weight: float = 0.01
    smoothness_weight: float = 0.001
    multiscale_weight: float = 0.0  # Multi-scale temporal loss (0 = disabled)
    spectral_weight: float = 0.0  # FFT spectral loss (0 = disabled)
    multiscale_scales: Tuple[int, ...] = (1, 4, 16)
    epochs: int = 200
    early_stopping_patience: int = 20
    checkpoint_dir: str = "checkpoints"

    # --- Storage ---
    coordinate_dtype: str = "float16"  # 'float16' or 'float32'
    mnf_version: int = 1

    # --- Evaluation ---
    close_weight: float = 2.0  # Extra weight on close price reconstruction
    volume_weight: float = 1.0  # Extra weight on volume reconstruction

    @property
    def input_dim(self) -> int:
        return self.window_size * len(self.features)

    @property
    def n_features(self) -> int:
        return len(self.features)
