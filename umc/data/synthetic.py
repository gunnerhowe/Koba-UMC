"""Synthetic data generators with known intrinsic dimensionality.

Includes:
  - Manifold generators for testing dimensionality detection
  - Multi-domain generators for benchmarking compression across data types
"""

import numpy as np
from typing import Dict, Tuple


class SyntheticManifoldGenerator:
    """Generate synthetic data on manifolds with known dimension."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_sine_manifold(
        self,
        n_samples: int = 10000,
        intrinsic_dim: int = 5,
        ambient_dim: int = 320,
        noise_std: float = 0.01,
    ) -> Tuple[np.ndarray, int]:
        """Generate data on a sine-wave manifold.

        Returns:
            (data of shape (n_samples, ambient_dim), true intrinsic dim)
        """
        # Random latent coordinates
        z = self.rng.randn(n_samples, intrinsic_dim).astype(np.float32)

        # Random projection matrix from intrinsic -> ambient
        W = self.rng.randn(intrinsic_dim, ambient_dim).astype(np.float32)
        W /= np.sqrt(intrinsic_dim)

        # Nonlinear embedding: sin + linear
        data = np.sin(z @ W) + 0.5 * (z @ W)

        # Add noise
        data += self.rng.randn(*data.shape).astype(np.float32) * noise_std

        return data, intrinsic_dim

    def generate_swiss_roll(
        self,
        n_samples: int = 10000,
        noise_std: float = 0.01,
    ) -> Tuple[np.ndarray, int]:
        """Generate a Swiss roll (intrinsic dim = 2)."""
        t = 1.5 * np.pi * (1 + 2 * self.rng.rand(n_samples))
        height = 10 * self.rng.rand(n_samples)

        x = t * np.cos(t)
        y = height
        z_coord = t * np.sin(t)

        data = np.column_stack([x, y, z_coord]).astype(np.float32)
        data += self.rng.randn(*data.shape).astype(np.float32) * noise_std

        return data, 2

    def generate_financial_synthetic(
        self,
        n_samples: int = 10000,
        intrinsic_dim: int = 10,
        window_size: int = 64,
        n_features: int = 5,
        noise_std: float = 0.001,
    ) -> Tuple[np.ndarray, int]:
        """Generate synthetic financial-like windowed data.

        Returns:
            (data of shape (n_samples, window_size, n_features), true dim)
        """
        ambient_dim = window_size * n_features
        z = self.rng.randn(n_samples, intrinsic_dim).astype(np.float32)

        # Two-layer nonlinear projection
        W1 = self.rng.randn(intrinsic_dim, 64).astype(np.float32) / np.sqrt(intrinsic_dim)
        W2 = self.rng.randn(64, ambient_dim).astype(np.float32) / np.sqrt(64)

        hidden = np.tanh(z @ W1)
        data = hidden @ W2
        data += self.rng.randn(*data.shape).astype(np.float32) * noise_std

        data = data.reshape(n_samples, window_size, n_features)
        return data, intrinsic_dim

    def generate_with_anomalies(
        self,
        n_normal: int = 9000,
        n_anomalies: int = 1000,
        intrinsic_dim: int = 10,
        ambient_dim: int = 320,
        anomaly_scale: float = 5.0,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Generate normal + anomalous data for testing anomaly detection.

        Returns:
            (data, labels (0=normal, 1=anomaly), true intrinsic dim)
        """
        normal_data, _ = self.generate_sine_manifold(
            n_samples=n_normal,
            intrinsic_dim=intrinsic_dim,
            ambient_dim=ambient_dim,
        )

        # Anomalies: random points off the manifold
        anomalies = self.rng.randn(n_anomalies, ambient_dim).astype(np.float32) * anomaly_scale

        data = np.vstack([normal_data, anomalies])
        labels = np.concatenate([
            np.zeros(n_normal, dtype=np.int32),
            np.ones(n_anomalies, dtype=np.int32),
        ])

        # Shuffle
        perm = self.rng.permutation(len(data))
        return data[perm], labels[perm], intrinsic_dim


# ---- Multi-domain generators for compression benchmarking ----

def generate_financial(n_windows: int = 1000, window_size: int = 32,
                       n_features: int = 5, seed: int = 42) -> np.ndarray:
    """Generate synthetic financial OHLCV-like data.

    Features: Open, High, Low, Close, Volume with realistic structure
    (trends, volatility clustering, volume spikes).

    Returns:
        (n_windows, window_size, n_features) float32
    """
    rng = np.random.RandomState(seed)
    total = n_windows * window_size

    # Price series with GBM + volatility clustering
    returns = rng.randn(total) * 0.02
    vol = np.ones(total)
    for i in range(1, total):
        vol[i] = 0.9 * vol[i - 1] + 0.1 * abs(returns[i - 1]) / 0.02
    returns *= vol

    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.randn(total)) * 0.005)
    low = close * (1 - np.abs(rng.randn(total)) * 0.005)
    opn = low + rng.rand(total) * (high - low)
    volume = np.exp(10 + rng.randn(total) * 0.5 + np.abs(returns) * 20)

    data = np.column_stack([opn, high, low, close, volume]).astype(np.float32)
    return data[:n_windows * window_size].reshape(n_windows, window_size, n_features)


def generate_sine_waves(n_windows: int = 1000, window_size: int = 32,
                        n_features: int = 4, seed: int = 42) -> np.ndarray:
    """Generate multi-frequency sinusoidal data with noise.

    Returns:
        (n_windows, window_size, n_features) float32
    """
    rng = np.random.RandomState(seed)
    data = np.zeros((n_windows, window_size, n_features), dtype=np.float32)
    t = np.linspace(0, 2 * np.pi, window_size)

    for i in range(n_windows):
        for f in range(n_features):
            freq = rng.uniform(0.5, 5.0)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.5, 2.0)
            data[i, :, f] = amp * np.sin(freq * t + phase)
            data[i, :, f] += rng.randn(window_size) * 0.05

    return data


def generate_iot_sensors(n_windows: int = 1000, window_size: int = 32,
                         n_features: int = 3, seed: int = 42) -> np.ndarray:
    """Generate synthetic IoT sensor data (temperature, humidity, pressure).

    Includes drift, daily cycles, and occasional spikes.

    Returns:
        (n_windows, window_size, n_features) float32
    """
    rng = np.random.RandomState(seed)
    total = n_windows * window_size

    # Temperature: 20C baseline + daily cycle + drift + noise
    t = np.arange(total, dtype=np.float32)
    temp = 20.0 + 5.0 * np.sin(2 * np.pi * t / (24 * 60)) + t * 0.0001
    temp += rng.randn(total) * 0.3
    # Occasional spikes
    spikes = rng.rand(total) < 0.001
    temp[spikes] += rng.randn(spikes.sum()) * 10

    # Humidity: anti-correlated with temperature
    humidity = 60.0 - 0.5 * (temp - 20.0) + rng.randn(total) * 2.0
    humidity = np.clip(humidity, 0, 100)

    # Pressure: slow drift + noise
    pressure = 1013.25 + np.cumsum(rng.randn(total) * 0.01)

    data = np.column_stack([temp, humidity, pressure]).astype(np.float32)
    return data[:n_windows * window_size].reshape(n_windows, window_size, n_features)


def generate_weather(n_windows: int = 1000, window_size: int = 32,
                     n_features: int = 4, seed: int = 42) -> np.ndarray:
    """Generate synthetic weather data with seasonal patterns.

    Features: temperature, wind_speed, precipitation, cloud_cover.

    Returns:
        (n_windows, window_size, n_features) float32
    """
    rng = np.random.RandomState(seed)
    total = n_windows * window_size
    t = np.arange(total, dtype=np.float32)

    # Temperature: yearly + daily cycles
    temp = (15 + 10 * np.sin(2 * np.pi * t / 365)
            + 5 * np.sin(2 * np.pi * t / 1)
            + rng.randn(total) * 2)

    # Wind speed: positive, with gusts
    wind = np.abs(5 + rng.randn(total) * 3)
    gusts = rng.rand(total) < 0.02
    wind[gusts] *= rng.uniform(2, 4, size=gusts.sum())

    # Precipitation: mostly zero, occasional rain
    precip = np.zeros(total, dtype=np.float32)
    rain = rng.rand(total) < 0.15
    precip[rain] = np.abs(rng.randn(rain.sum()) * 5)

    # Cloud cover: 0-100, correlated with precipitation
    cloud = 30 + 40 * rain.astype(np.float32) + rng.randn(total) * 15
    cloud = np.clip(cloud, 0, 100)

    data = np.column_stack([temp, wind, precip, cloud]).astype(np.float32)
    return data[:n_windows * window_size].reshape(n_windows, window_size, n_features)


def generate_ecg(n_windows: int = 1000, window_size: int = 64,
                 n_features: int = 1, seed: int = 42) -> np.ndarray:
    """Generate synthetic ECG-like periodic waveforms.

    Returns:
        (n_windows, window_size, n_features) float32
    """
    rng = np.random.RandomState(seed)
    data = np.zeros((n_windows, window_size, n_features), dtype=np.float32)
    t = np.linspace(0, 1, window_size)

    for i in range(n_windows):
        # Heart rate variation
        hr = rng.uniform(0.8, 1.2)

        # P wave
        signal = 0.1 * np.exp(-((t * hr - 0.1) ** 2) / 0.001)
        # QRS complex
        signal += -0.1 * np.exp(-((t * hr - 0.2) ** 2) / 0.0005)
        signal += 1.0 * np.exp(-((t * hr - 0.25) ** 2) / 0.0003)
        signal += -0.2 * np.exp(-((t * hr - 0.3) ** 2) / 0.0005)
        # T wave
        signal += 0.3 * np.exp(-((t * hr - 0.5) ** 2) / 0.003)
        # Baseline noise
        signal += rng.randn(window_size) * 0.02

        data[i, :, 0] = signal.astype(np.float32)

    return data


def generate_audio_spectrogram(n_windows: int = 1000, window_size: int = 32,
                               n_features: int = 16, seed: int = 42) -> np.ndarray:
    """Generate synthetic mel-spectrogram-like frequency bins.

    Returns:
        (n_windows, window_size, n_features) float32
    """
    rng = np.random.RandomState(seed)
    data = np.zeros((n_windows, window_size, n_features), dtype=np.float32)

    for i in range(n_windows):
        # Base energy per frequency bin (lower bins = more energy)
        base_energy = np.exp(-np.arange(n_features) / 5.0)

        # Temporal envelope (attack-sustain-release)
        attack = int(rng.uniform(1, window_size // 4))
        release = int(rng.uniform(window_size // 2, window_size))
        envelope = np.ones(window_size, dtype=np.float32)
        envelope[:attack] = np.linspace(0, 1, attack)
        if release < window_size:
            envelope[release:] = np.linspace(1, 0, window_size - release)

        # Harmonic structure: a few dominant frequency bins
        n_harmonics = rng.randint(1, 4)
        fundamental = rng.randint(0, n_features // 2)
        for h in range(n_harmonics):
            bin_idx = min(fundamental * (h + 1), n_features - 1)
            strength = 1.0 / (h + 1)
            data[i, :, bin_idx] += envelope * strength

        # Background + noise
        data[i] += base_energy * 0.1
        data[i] += np.abs(rng.randn(window_size, n_features) * 0.02)

    return data.astype(np.float32)


def generate_all_types(n_windows: int = 500, seed: int = 42) -> Dict[str, np.ndarray]:
    """Generate all data types for benchmarking.

    Returns:
        Dict mapping type name to (n_windows, window_size, n_features) float32 array.
    """
    return {
        "financial": generate_financial(n_windows, seed=seed),
        "sine_waves": generate_sine_waves(n_windows, seed=seed),
        "iot_sensors": generate_iot_sensors(n_windows, seed=seed),
        "weather": generate_weather(n_windows, seed=seed),
        "ecg": generate_ecg(n_windows, seed=seed),
        "audio_spectrogram": generate_audio_spectrogram(n_windows, seed=seed),
    }
