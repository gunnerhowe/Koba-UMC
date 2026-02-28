"""Geodesic interpolation in manifold coordinate space."""

import numpy as np


def linear_interpolate(
    z1: np.ndarray,
    z2: np.ndarray,
    n_steps: int = 10,
) -> np.ndarray:
    """Linear interpolation between two manifold points.

    This is an approximation of geodesic interpolation that works
    well when the manifold is locally flat.

    Args:
        z1: Starting coordinate (latent_dim,).
        z2: Ending coordinate (latent_dim,).
        n_steps: Number of interpolation steps (including endpoints).

    Returns:
        (n_steps, latent_dim) array of interpolated coordinates.
    """
    alphas = np.linspace(0, 1, n_steps)
    return np.array([(1 - a) * z1 + a * z2 for a in alphas])


def spherical_interpolate(
    z1: np.ndarray,
    z2: np.ndarray,
    n_steps: int = 10,
) -> np.ndarray:
    """Spherical linear interpolation (slerp) between two points.

    Better than linear interpolation for latent spaces with
    approximately uniform norm.

    Args:
        z1: Starting coordinate (latent_dim,).
        z2: Ending coordinate (latent_dim,).
        n_steps: Number of interpolation steps.

    Returns:
        (n_steps, latent_dim) array of interpolated coordinates.
    """
    z1_norm = z1 / (np.linalg.norm(z1) + 1e-8)
    z2_norm = z2 / (np.linalg.norm(z2) + 1e-8)

    dot = np.clip(np.dot(z1_norm, z2_norm), -1.0, 1.0)
    omega = np.arccos(dot)

    if np.abs(omega) < 1e-8:
        # Points are nearly identical, fall back to linear
        return linear_interpolate(z1, z2, n_steps)

    alphas = np.linspace(0, 1, n_steps)
    result = []
    for a in alphas:
        coeff1 = np.sin((1 - a) * omega) / np.sin(omega)
        coeff2 = np.sin(a * omega) / np.sin(omega)
        result.append(coeff1 * z1 + coeff2 * z2)

    return np.array(result)


def multi_point_interpolate(
    points: np.ndarray,
    n_steps_per_segment: int = 10,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate through a sequence of manifold points.

    Args:
        points: (n_points, latent_dim) waypoints.
        n_steps_per_segment: Steps between each pair.
        method: 'linear' or 'spherical'.

    Returns:
        (total_steps, latent_dim) interpolated path.
    """
    interp_fn = spherical_interpolate if method == "spherical" else linear_interpolate

    segments = []
    for i in range(len(points) - 1):
        segment = interp_fn(points[i], points[i + 1], n_steps_per_segment)
        if i < len(points) - 2:
            segment = segment[:-1]  # Avoid duplicating endpoints
        segments.append(segment)

    return np.concatenate(segments, axis=0)
