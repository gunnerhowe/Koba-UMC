"""Loss functions for UMC training."""

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import UMCConfig


def reconstruction_loss(
    x: Tensor,
    x_hat: Tensor,
    config: UMCConfig,
    loss_type: str = "mse",
) -> Tensor:
    """Reconstruction loss between original and decoded data.

    For financial data, weights close prices higher.

    Args:
        x: Original data (batch, window_size, n_features).
        x_hat: Reconstructed data, same shape.
        config: UMC config (for feature weighting).
        loss_type: 'mse' or 'huber'.

    Returns:
        Scalar loss tensor.
    """
    # Feature weighting: close price gets extra weight
    features = list(config.features)
    weights = torch.ones(len(features), device=x.device)
    if "close" in features:
        close_idx = features.index("close")
        weights[close_idx] = config.close_weight

    # Normalize weights
    weights = weights / weights.sum() * len(features)

    # Apply weights along feature dimension
    diff = x - x_hat
    weighted_diff = diff * weights.unsqueeze(0).unsqueeze(0)

    if loss_type == "mse":
        return (weighted_diff ** 2).mean()
    elif loss_type == "huber":
        return torch.nn.functional.smooth_l1_loss(
            weighted_diff, torch.zeros_like(weighted_diff)
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def kl_divergence(mu: Tensor, log_var: Tensor) -> Tensor:
    """KL divergence KL(q(z|x) || N(0, I)).

    Args:
        mu: Mean of approximate posterior (batch, latent_dim).
        log_var: Log variance of approximate posterior (batch, latent_dim).

    Returns:
        Scalar KL loss (averaged over batch).
    """
    # -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    return kl_per_dim.sum(dim=-1).mean()


def kl_divergence_per_dim(mu: Tensor, log_var: Tensor) -> Tensor:
    """Per-dimension KL divergence for monitoring active dims.

    Returns:
        Tensor of shape (latent_dim,) with KL for each dimension.
    """
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    return kl_per_dim.mean(dim=0)


def sparsity_loss(z: Tensor) -> Tensor:
    """L1 penalty on latent activations to encourage dimension collapse.

    Args:
        z: Latent coordinates (batch, latent_dim).

    Returns:
        Scalar L1 loss.
    """
    return z.abs().mean()


def smoothness_loss(z: Tensor) -> Tensor:
    """Penalize high-frequency variations in adjacent latent codes.

    Extension beyond base spec: this regularizer encourages temporal
    coherence in manifold coordinates when batch samples come from
    sequential windows. Controlled by config.smoothness_weight (default
    0.001, low enough to avoid interfering with reconstruction quality).

    Assumes batch samples are somewhat sequential (from windowed data).

    Args:
        z: Latent coordinates (batch, latent_dim).

    Returns:
        Scalar smoothness loss.
    """
    if z.shape[0] < 2:
        return torch.tensor(0.0, device=z.device)
    diff = z[1:] - z[:-1]
    return (diff ** 2).mean()


def multiscale_reconstruction_loss(
    x: Tensor,
    x_hat: Tensor,
    scales: tuple = (1, 4, 16),
) -> Tensor:
    """Reconstruction loss at multiple temporal resolutions.

    Downsamples both x and x_hat by averaging over time windows of different
    sizes, then computes MSE at each scale.

    Args:
        x, x_hat: (batch, window_size, n_features)
        scales: Tuple of averaging window sizes. 1 = full resolution.
    """
    total = torch.tensor(0.0, device=x.device)
    for s in scales:
        if s == 1:
            total = total + F.mse_loss(x, x_hat)
        else:
            # avg_pool1d expects (batch, channels, length)
            x_t = x.transpose(1, 2)
            x_hat_t = x_hat.transpose(1, 2)
            if x_t.shape[2] >= s:
                x_pooled = F.avg_pool1d(x_t, kernel_size=s, stride=s)
                x_hat_pooled = F.avg_pool1d(x_hat_t, kernel_size=s, stride=s)
                total = total + F.mse_loss(x_pooled, x_hat_pooled)
    return total / len(scales)


def spectral_loss(x: Tensor, x_hat: Tensor) -> Tensor:
    """Loss on frequency content via FFT magnitude spectrum.

    Args:
        x, x_hat: (batch, window_size, n_features)
    """
    x_fft = torch.fft.rfft(x, dim=1)
    x_hat_fft = torch.fft.rfft(x_hat, dim=1)
    return F.mse_loss(torch.abs(x_fft), torch.abs(x_hat_fft))


def total_loss(
    x: Tensor,
    x_hat: Tensor,
    mu: Tensor,
    log_var: Tensor,
    z: Tensor,
    config: UMCConfig,
    epoch: int,
) -> tuple[Tensor, dict]:
    """Combined loss with beta annealing.

    Args:
        x: Original data.
        x_hat: Reconstructed data.
        mu: Encoder mean.
        log_var: Encoder log variance.
        z: Sampled latent coordinates.
        config: UMC configuration.
        epoch: Current epoch (for beta annealing).

    Returns:
        (total_loss, dict of individual loss components)
    """
    recon = reconstruction_loss(x, x_hat, config)
    kl = kl_divergence(mu, log_var)
    sparse = sparsity_loss(z)
    smooth = smoothness_loss(z)

    # Beta annealing: linearly increase KL weight
    if config.beta_warmup_epochs > 0:
        beta = min(
            config.beta_end,
            config.beta_start + (config.beta_end - config.beta_start)
            * epoch / config.beta_warmup_epochs,
        )
    else:
        beta = config.beta_end

    loss = recon + beta * kl + config.sparsity_weight * sparse + config.smoothness_weight * smooth

    # Multi-scale temporal loss
    ms_val = torch.tensor(0.0, device=x.device)
    if config.multiscale_weight > 0:
        ms_val = multiscale_reconstruction_loss(x, x_hat, config.multiscale_scales)
        loss = loss + config.multiscale_weight * ms_val

    # Spectral (FFT) loss
    spec_val = torch.tensor(0.0, device=x.device)
    if config.spectral_weight > 0:
        spec_val = spectral_loss(x, x_hat)
        loss = loss + config.spectral_weight * spec_val

    components = {
        "reconstruction": recon.item(),
        "kl": kl.item(),
        "sparsity": sparse.item(),
        "smoothness": smooth.item(),
        "multiscale": ms_val.item(),
        "spectral": spec_val.item(),
        "beta": beta,
        "total": loss.item(),
    }
    return loss, components


def vq_total_loss(
    x: Tensor,
    x_hat: Tensor,
    z: Tensor,
    vq_loss: Tensor,
    config: UMCConfig,
) -> tuple[Tensor, dict]:
    """Combined loss for VQ-VAE training (no KL divergence).

    Args:
        x: Original data (batch, window_size, n_features).
        x_hat: Reconstructed data, same shape.
        z: Continuous z_projection output (for sparsity/smoothness on FAISS coords).
        vq_loss: Combined commitment loss from VQ layers.
        config: UMC configuration.

    Returns:
        (total_loss, dict of individual loss components)
    """
    recon = reconstruction_loss(x, x_hat, config)
    sparse = sparsity_loss(z)
    smooth = smoothness_loss(z)

    loss = recon + vq_loss + config.sparsity_weight * sparse + config.smoothness_weight * smooth

    # Multi-scale temporal loss
    ms_val = torch.tensor(0.0, device=x.device)
    if config.multiscale_weight > 0:
        ms_val = multiscale_reconstruction_loss(x, x_hat, config.multiscale_scales)
        loss = loss + config.multiscale_weight * ms_val

    # Spectral (FFT) loss
    spec_val = torch.tensor(0.0, device=x.device)
    if config.spectral_weight > 0:
        spec_val = spectral_loss(x, x_hat)
        loss = loss + config.spectral_weight * spec_val

    components = {
        "reconstruction": recon.item(),
        "vq_commitment": vq_loss.item(),
        "sparsity": sparse.item(),
        "smoothness": smooth.item(),
        "multiscale": ms_val.item(),
        "spectral": spec_val.item(),
        "total": loss.item(),
    }
    return loss, components
