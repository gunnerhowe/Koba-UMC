"""Neural byte prediction model for arithmetic coding.

A small causal model that predicts P(next_byte | context) for XOR
residual byte streams. Used by NeuralByteCompressor for Phase 3
compression approaching the Shannon entropy limit.

Architecture: Causal dilated 1D CNN over byte embeddings.
- Byte embedding: 256 → embed_dim
- Channel embedding: n_channels → embed_dim (byte position in float32)
- Stack of causal Conv1d layers with dilated receptive field
- Output: softmax over 256 byte values

The model is lightweight (~50-200K params) and runs fast on CPU.
Training is done on byte-transposed XOR residual streams from Phase 2.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D convolution with causal (left-only) padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class BytePredictor(nn.Module):
    """Causal CNN for byte-level probability prediction.

    Predicts P(byte[t] | byte[0..t-1], channel_id) for each position t
    in a byte stream. Channel_id identifies which byte position within
    a float32 element (0=MSB exponent, 3=LSB mantissa).

    Args:
        embed_dim: Dimension of byte/channel embeddings.
        hidden_dim: Width of causal conv layers.
        n_layers: Number of causal conv layers (receptive field grows exponentially).
        kernel_size: Kernel size for causal convolutions.
        n_channels: Number of byte channels (4 for float32).
        dropout: Dropout rate between layers.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        n_layers: int = 6,
        kernel_size: int = 3,
        n_channels: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_channels = n_channels

        # Byte embedding (256 possible byte values)
        self.byte_embed = nn.Embedding(256, embed_dim)

        # Channel embedding (which byte position in the float32)
        self.channel_embed = nn.Embedding(n_channels, embed_dim)

        # Input projection
        self.input_proj = nn.Conv1d(embed_dim, hidden_dim, 1)

        # Causal conv stack with exponentially increasing dilation
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.layers.append(nn.Sequential(
                CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation),
                nn.GELU(),
                nn.Dropout(dropout),
                CausalConv1d(hidden_dim, hidden_dim, 1),
                nn.GELU(),
            ))

        # Output head
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 256, 1),  # logits over 256 byte values
        )

        # Learnable start token
        self.start_token = nn.Parameter(torch.randn(embed_dim))

    def forward(
        self, byte_sequence: torch.Tensor, channel_id: int = 0
    ) -> torch.Tensor:
        """Predict byte probabilities for all positions.

        Args:
            byte_sequence: (B, L) uint8/long tensor of byte values.
            channel_id: Which byte channel (0-3 for float32).

        Returns:
            (B, L, 256) float32 tensor of log-probabilities (log-softmax).
            Position t contains log P(byte[t] | byte[0..t-1]).
        """
        B, L = byte_sequence.shape

        # Embed bytes: shift right by 1 (causal: predict t from 0..t-1)
        # Prepend start token, drop last
        byte_emb = self.byte_embed(byte_sequence)  # (B, L, E)
        start = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        shifted = torch.cat([start, byte_emb[:, :-1]], dim=1)  # (B, L, E)

        # Add channel embedding
        ch_emb = self.channel_embed(
            torch.tensor(channel_id, device=byte_sequence.device)
        )  # (E,)
        x = shifted + ch_emb.unsqueeze(0).unsqueeze(0)  # (B, L, E)

        # Conv expects (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)

        # Residual causal conv blocks
        for layer in self.layers:
            x = x + layer(x)

        # Output logits
        logits = self.output_proj(x)  # (B, 256, L)
        logits = logits.permute(0, 2, 1)  # (B, L, 256)

        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_all(
        self,
        byte_array: np.ndarray,
        channel_id: int = 0,
        device: str = "cpu",
    ) -> np.ndarray:
        """Predict probabilities for all positions in a byte array.

        Non-autoregressive: uses teacher forcing (ground truth context).
        Suitable for encoding (all symbols known).

        Args:
            byte_array: (N,) uint8 numpy array.
            channel_id: Which byte channel (0-3).
            device: Compute device.

        Returns:
            (N, 256) float32 numpy array of probabilities (not log-probs).
        """
        self.eval()
        x = torch.from_numpy(byte_array.astype(np.int64)).unsqueeze(0).to(device)
        log_probs = self.forward(x, channel_id=channel_id)  # (1, N, 256)
        probs = log_probs.exp().squeeze(0).cpu().numpy()  # (N, 256)

        # Ensure valid probability distribution
        probs = np.maximum(probs, 1e-10)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)

    @torch.no_grad()
    def predict_next(
        self,
        context: np.ndarray,
        channel_id: int = 0,
        device: str = "cpu",
    ) -> np.ndarray:
        """Predict probability of next byte given context.

        Autoregressive: used during decoding (one symbol at a time).

        Args:
            context: (M,) uint8 numpy array of previous bytes (may be empty).
            channel_id: Which byte channel (0-3).
            device: Compute device.

        Returns:
            (256,) float32 probability distribution over next byte.
        """
        self.eval()
        if len(context) == 0:
            # No context: use start token only
            x = torch.zeros(1, 1, dtype=torch.long, device=device)
            # We need at least 1 position; use dummy and take position 0
            log_probs = self.forward(x, channel_id=channel_id)  # (1, 1, 256)
            probs = log_probs.exp().squeeze(0).squeeze(0).cpu().numpy()
        else:
            # Feed context + dummy next token
            full = np.concatenate([context, np.zeros(1, dtype=np.uint8)])
            x = torch.from_numpy(full.astype(np.int64)).unsqueeze(0).to(device)
            log_probs = self.forward(x, channel_id=channel_id)
            probs = log_probs.exp()[0, -1].cpu().numpy()  # last position

        probs = np.maximum(probs, 1e-10)
        probs = probs / probs.sum()
        return probs.astype(np.float32)


def train_byte_predictor(
    residual_streams: list[np.ndarray],
    n_channels: int = 4,
    embed_dim: int = 32,
    hidden_dim: int = 64,
    n_layers: int = 6,
    batch_size: int = 32,
    seq_len: int = 1024,
    n_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> BytePredictor:
    """Train a BytePredictor on XOR residual byte streams.

    Args:
        residual_streams: List of byte-transposed XOR residual byte arrays.
            Each array should already be byte-transposed (channels concatenated).
        n_channels: Number of byte channels per element (4 for float32).
        embed_dim: Byte embedding dimension.
        hidden_dim: Hidden layer width.
        n_layers: Number of causal conv layers.
        batch_size: Training batch size.
        seq_len: Sequence length for training chunks.
        n_epochs: Number of training epochs.
        lr: Learning rate.
        device: Training device.

    Returns:
        Trained BytePredictor model.
    """
    model = BytePredictor(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_channels=n_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Prepare training data: split each stream into channels, then chunk
    all_chunks = {ch: [] for ch in range(n_channels)}
    for stream in residual_streams:
        symbols_per_channel = len(stream) // n_channels
        for ch in range(n_channels):
            start = ch * symbols_per_channel
            end = start + symbols_per_channel
            ch_data = np.frombuffer(stream[start:end], dtype=np.uint8)
            # Split into fixed-length chunks
            for i in range(0, len(ch_data) - seq_len, seq_len // 2):
                chunk = ch_data[i:i + seq_len].copy()
                if len(chunk) == seq_len:
                    all_chunks[ch].append(chunk)

    total_chunks = sum(len(v) for v in all_chunks.values())
    if total_chunks == 0:
        print("Warning: no training data. Returning untrained model.")
        return model

    print(f"Training BytePredictor: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Data: {total_chunks} chunks × {seq_len} bytes across {n_channels} channels")

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0

        for ch in range(n_channels):
            chunks = all_chunks[ch]
            if not chunks:
                continue
            indices = np.random.permutation(len(chunks))

            for b_start in range(0, len(indices), batch_size):
                b_end = min(b_start + batch_size, len(indices))
                batch_idx = indices[b_start:b_end]
                batch = np.stack([chunks[i] for i in batch_idx])

                x = torch.from_numpy(batch.astype(np.int64)).to(device)
                log_probs = model(x, channel_id=ch)  # (B, L, 256)

                # Cross-entropy loss
                loss = F.nll_loss(
                    log_probs.reshape(-1, 256),
                    x.reshape(-1),
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        bits_per_byte = avg_loss / np.log(2)
        print(
            f"  Epoch {epoch:3d}/{n_epochs} | Loss: {avg_loss:.4f} | "
            f"Bits/byte: {bits_per_byte:.3f} | LR: {scheduler.get_last_lr()[0]:.1e}"
        )

    model.eval()
    return model
