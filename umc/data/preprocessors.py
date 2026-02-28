"""Data preprocessing: normalization, windowing, cleaning."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional

from ..config import UMCConfig


class OHLCVPreprocessor:
    """Normalize and window OHLCV data for encoder input."""

    def __init__(self, config: UMCConfig):
        self.config = config
        self.normalize_method = config.normalize
        self._fit_params: dict = {}

    def fit(self, df: pd.DataFrame) -> "OHLCVPreprocessor":
        """Compute normalization parameters from training data."""
        features = list(self.config.features)
        data = df[features].values.astype(np.float64)

        if self.normalize_method == "returns":
            # Store first values for reconstruction; returns are self-normalizing
            self._fit_params["method"] = "returns"
        elif self.normalize_method == "minmax":
            self._fit_params["method"] = "minmax"
            self._fit_params["min"] = data.min(axis=0)
            self._fit_params["max"] = data.max(axis=0)
        elif self.normalize_method == "zscore":
            self._fit_params["method"] = "zscore"
            self._fit_params["mean"] = data.mean(axis=0)
            self._fit_params["std"] = data.std(axis=0) + 1e-8
        else:
            raise ValueError(f"Unknown normalize method: {self.normalize_method}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Normalize raw OHLCV data.

        Returns:
            Array of shape (n_samples, n_features) with normalized values.
        """
        features = list(self.config.features)
        data = df[features].values.astype(np.float64)

        method = self._fit_params.get("method", self.normalize_method)
        if method == "returns":
            # Log returns: stable, stationary, bounded
            eps = 1e-10
            price_cols = [i for i, f in enumerate(features) if f != "volume"]
            vol_col = [i for i, f in enumerate(features) if f == "volume"]

            normalized = np.zeros_like(data)
            # Log returns for price columns
            for col in price_cols:
                normalized[1:, col] = np.log(data[1:, col] / (data[:-1, col] + eps) + eps)
            normalized[0, price_cols] = 0.0

            # Volume: log-normalize then robust scale to ~[-1, 1]
            if vol_col:
                vc = vol_col[0]
                vol = data[:, vc]
                rolling_mean = pd.Series(vol).rolling(20, min_periods=1).mean().values + eps
                log_vol = np.log(vol / rolling_mean + eps)
                # Robust scaling: clip to ±3 IQR, then scale
                q25, q75 = np.percentile(log_vol, [25, 75])
                iqr = q75 - q25 + eps
                log_vol = (log_vol - np.median(log_vol)) / (iqr + eps)
                log_vol = np.clip(log_vol, -5, 5)
                normalized[:, vc] = log_vol

            # Clip all values to prevent extreme outliers
            normalized = np.clip(normalized, -10, 10)

            return normalized.astype(np.float32)

        elif method == "minmax":
            mn = self._fit_params["min"]
            mx = self._fit_params["max"]
            return ((data - mn) / (mx - mn + 1e-8)).astype(np.float32)

        elif method == "zscore":
            mean = self._fit_params["mean"]
            std = self._fit_params["std"]
            return ((data - mean) / std).astype(np.float32)

        raise ValueError(f"Unknown method: {method}")

    def inverse_transform(self, normalized: np.ndarray, reference_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Reverse normalization to get original-scale data."""
        method = self._fit_params.get("method", self.normalize_method)

        if method == "returns":
            # Cannot fully invert without reference prices
            # Return the normalized (returns) data as-is if no reference
            if reference_df is None:
                return normalized
            features = list(self.config.features)
            price_cols = [i for i, f in enumerate(features) if f != "volume"]
            ref = reference_df[features].values.astype(np.float64)
            result = np.zeros_like(normalized, dtype=np.float64)
            for col in price_cols:
                result[0, col] = ref[0, col]
                for t in range(1, len(normalized)):
                    result[t, col] = result[t - 1, col] * np.exp(normalized[t, col])
            return result.astype(np.float32)

        elif method == "minmax":
            mn = self._fit_params["min"]
            mx = self._fit_params["max"]
            return (normalized * (mx - mn + 1e-8) + mn).astype(np.float32)

        elif method == "zscore":
            mean = self._fit_params["mean"]
            std = self._fit_params["std"]
            return (normalized * std + mean).astype(np.float32)

        raise ValueError(f"Unknown method: {method}")

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


def create_windows(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """Slice normalized data into overlapping windows.

    Args:
        data: Array of shape (n_samples, n_features).
        window_size: Number of timesteps per window.
        stride: Step size between windows.

    Returns:
        Array of shape (n_windows, window_size, n_features).
    """
    n_samples, n_features = data.shape
    n_windows = (n_samples - window_size) // stride + 1
    if n_windows <= 0:
        raise ValueError(
            f"Data length {n_samples} too short for window_size {window_size}"
        )

    windows = np.zeros((n_windows, window_size, n_features), dtype=data.dtype)
    for i in range(n_windows):
        start = i * stride
        windows[i] = data[start : start + window_size]
    return windows


class WindowNormalizer:
    """Per-window zero-mean, unit-std normalization with stored scale factors.

    After global preprocessing and windowing, each window is independently
    standardized per feature. The mean and std per window per feature are
    stored as metadata for reconstruction.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def normalize(
        self, windows: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize each window to zero-mean, unit-std per feature.

        Args:
            windows: (n_windows, window_size, n_features)

        Returns:
            normalized: (n_windows, window_size, n_features)
            means: (n_windows, n_features)
            stds: (n_windows, n_features)
        """
        means = windows.mean(axis=1)  # (n_windows, n_features)
        stds = windows.std(axis=1) + self.eps  # (n_windows, n_features)
        normalized = (windows - means[:, np.newaxis, :]) / stds[:, np.newaxis, :]
        normalized = np.clip(normalized, -10, 10)
        return normalized.astype(np.float32), means.astype(np.float32), stds.astype(np.float32)

    def denormalize(
        self, normalized: np.ndarray, means: np.ndarray, stds: np.ndarray
    ) -> np.ndarray:
        """Reverse per-window normalization.

        Args:
            normalized: (n_windows, window_size, n_features)
            means: (n_windows, n_features)
            stds: (n_windows, n_features)

        Returns:
            windows: (n_windows, window_size, n_features) in original scale
        """
        return (normalized * stds[:, np.newaxis, :] + means[:, np.newaxis, :]).astype(np.float32)


class UniversalPreprocessor:
    """Domain-agnostic preprocessor for any numeric 2D data.

    Applies z-score normalization per feature with optional clipping.
    Makes the neural encoder pipeline work on non-financial data.

    Usage:
        prep = UniversalPreprocessor()
        prep.fit(data)          # (n_samples, n_features)
        normed = prep.transform(data)
        original = prep.inverse_transform(normed)
    """

    def __init__(self, clip_range: float = 5.0):
        self.clip_range = clip_range
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "UniversalPreprocessor":
        """Compute per-feature mean and std from data.

        Args:
            data: (n_samples, n_features) or (n_windows, window_size, n_features).
        """
        if data.ndim == 3:
            flat = data.reshape(-1, data.shape[-1])
        elif data.ndim == 2:
            flat = data
        elif data.ndim == 1:
            flat = data.reshape(-1, 1)
        else:
            flat = data.reshape(-1, data.shape[-1])

        self._mean = flat.mean(axis=0).astype(np.float64)
        self._std = flat.std(axis=0).astype(np.float64)
        self._std = np.maximum(self._std, 1e-8)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalize and clip.

        Args:
            data: Same shape convention as fit().

        Returns:
            Normalized array, same shape as input.
        """
        if self._mean is None:
            raise RuntimeError("Call fit() first")
        result = (data.astype(np.float64) - self._mean) / self._std
        if self.clip_range > 0:
            result = np.clip(result, -self.clip_range, self.clip_range)
        return result.astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the normalization.

        Args:
            data: Normalized array from transform().

        Returns:
            Array in original scale.
        """
        if self._mean is None:
            raise RuntimeError("Call fit() first")
        return (data.astype(np.float64) * self._std + self._mean).astype(np.float32)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(data).transform(data)

    def state_dict(self) -> dict:
        """Serialize parameters for saving."""
        return {"mean": self._mean, "std": self._std, "clip_range": self.clip_range}

    def load_state_dict(self, state: dict) -> "UniversalPreprocessor":
        """Load from serialized parameters."""
        self._mean = state["mean"]
        self._std = state["std"]
        self.clip_range = state.get("clip_range", 5.0)
        return self


class WindowDataset(Dataset):
    """PyTorch dataset wrapping windowed OHLCV data with optional scale factors."""

    def __init__(
        self,
        windows: np.ndarray,
        means: Optional[np.ndarray] = None,
        stds: Optional[np.ndarray] = None,
    ):
        self.windows = torch.from_numpy(windows).float()
        self.means = torch.from_numpy(means).float() if means is not None else None
        self.stds = torch.from_numpy(stds).float() if stds is not None else None

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        if self.means is not None and self.stds is not None:
            return self.windows[idx], self.means[idx], self.stds[idx]
        return self.windows[idx]


def prepare_dataloaders(
    df: pd.DataFrame,
    config: UMCConfig,
    val_split: float = 0.1,
    test_split: float = 0.1,
    stride: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Full pipeline: normalize -> window -> split -> DataLoaders."""
    preprocessor = OHLCVPreprocessor(config)
    normalized = preprocessor.fit_transform(df)
    windows = create_windows(normalized, config.window_size, stride=stride)

    n = len(windows)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    train_windows = windows[:n_train]
    val_windows = windows[n_train : n_train + n_val]
    test_windows = windows[n_train + n_val :]

    train_loader = DataLoader(
        WindowDataset(train_windows),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        WindowDataset(val_windows),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        WindowDataset(test_windows),
        batch_size=config.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader
