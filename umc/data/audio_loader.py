"""Load and save audio as float32 arrays for UMC compression."""

import numpy as np
from pathlib import Path


def load_audio(
    path: str,
    window_size: int = 1024,
) -> np.ndarray:
    """Load audio file as windowed float32 array.

    Args:
        path: Path to WAV file.
        window_size: Samples per window.

    Returns:
        (n_windows, window_size, channels) float32 array normalized to [-1, 1].
    """
    import scipy.io.wavfile as wavfile

    sample_rate, data = wavfile.read(path)

    # Convert to float32 in [-1, 1]
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        audio = data
    elif data.dtype == np.float64:
        audio = data.astype(np.float32)
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)

    # Ensure 2D (samples, channels)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    # Window the audio
    n_samples, n_channels = audio.shape
    n_windows = n_samples // window_size
    if n_windows == 0:
        # Pad short audio
        padded = np.zeros((window_size, n_channels), dtype=np.float32)
        padded[:n_samples] = audio
        return padded.reshape(1, window_size, n_channels)

    trimmed = audio[:n_windows * window_size]
    return trimmed.reshape(n_windows, window_size, n_channels)


def save_audio(
    data: np.ndarray,
    path: str,
    sample_rate: int = 44100,
) -> None:
    """Save float32 array back to WAV file.

    Args:
        data: Array from load_audio or decompress.
              Shape (n_windows, window_size, channels) or (samples, channels).
        path: Output WAV file path.
        sample_rate: Sample rate in Hz.
    """
    import scipy.io.wavfile as wavfile

    if data.ndim == 3:
        n_windows, window_size, channels = data.shape
        audio = data.reshape(-1, channels)
    elif data.ndim == 2:
        audio = data
    elif data.ndim == 1:
        audio = data[:, np.newaxis]
    else:
        raise ValueError(f"Unexpected audio shape: {data.shape}")

    # Clip to valid range
    audio = np.clip(audio, -1.0, 1.0)

    # Convert to int16 for WAV
    audio_int16 = (audio * 32767).astype(np.int16)

    if audio_int16.shape[1] == 1:
        audio_int16 = audio_int16.squeeze(-1)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(p), sample_rate, audio_int16)
