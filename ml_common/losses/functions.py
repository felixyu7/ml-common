"""Loss functions for angular/directional reconstruction and energy regression."""

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Dict, Tuple

try:
    from spherical_harmonics_loss import SphericalHarmonicLoss, get_n_coeffs
except ImportError:  # Package is optional
    SphericalHarmonicLoss = None  # type: ignore[assignment]
    get_n_coeffs = None  # type: ignore[assignment]

_SH_LOSS_CACHE: Dict[Tuple[int, int, int, str, str, torch.dtype], torch.nn.Module] = {}


def angular_distance_loss(
    pred: Tensor,
    truth: Tensor,
    eps: float = 1e-7,
    reduction: str = "mean"
) -> Tensor:
    """Angular distance loss, normalized to [0, 1]."""
    # Normalize pred and truth to unit vectors
    pred = F.normalize(pred, p=2, dim=1)
    truth = F.normalize(truth, p=2, dim=1)

    # Clamp prevents invalid input to arccos
    cos_sim = F.cosine_similarity(pred, truth)
    angle = torch.acos(torch.clamp(cos_sim, min=-1.0 + eps, max=1.0 - eps))
    loss = angle / np.pi

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def gaussian_nll_loss(mu: Tensor, var: Tensor, target: Tensor) -> Tensor:
    """Gaussian NLL loss for regression with uncertainty (var passed through softplus)."""
    # Ensure var is positive and stable
    var = F.softplus(var) + 1e-6

    # NLL for each sample
    nll = 0.5 * ((target - mu)**2 / var + torch.log(var))

    # Return average over batch
    return torch.mean(nll)


def von_mises_fisher_loss(n_pred: Tensor, n_true: Tensor, eps: float = 1e-8) -> Tensor:
    """von Mises-Fisher loss for directional data (kappa from ||n_pred||)."""
    kappa = torch.norm(n_pred, dim=1)
    logC = -kappa + torch.log((kappa + eps) / (1 - torch.exp(-2 * kappa) + 2 * eps))
    return -((n_true * n_pred).sum(dim=1) + logC).mean()


def spherical_harmonic_loss(
    coeffs: Tensor,
    target_dirs: Tensor,
    *,
    l_max: int,
    n_theta: int = 64,
    n_lambda: int = 128,
    grid_type: str = "legendre-gauss",
    reduction: str = "mean",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Negative log-likelihood loss using spherical harmonic expansions.

    Wraps :class:`spherical_harmonics_loss.SphericalHarmonicLoss` with basic caching so
    the loss can be selected from :mod:`ml_common.losses` when the optional dependency
    is installed.
    """
    if SphericalHarmonicLoss is None:
        raise ImportError(
            "spherical_harmonics_loss is not installed. Install it to use "
            "`spherical_harmonic_loss`."
        )

    if coeffs.dim() != 2:
        raise ValueError(f"`coeffs` must be 2D with shape (batch, n_coeffs); got {coeffs.shape}")
    if target_dirs.dim() != 2 or target_dirs.size(-1) != 3:
        raise ValueError(
            "`target_dirs` must be 2D with shape (batch, 3) of direction vectors; "
            f"got {target_dirs.shape}"
        )

    if get_n_coeffs is not None:
        expected_coeffs = get_n_coeffs(l_max)
        if coeffs.size(-1) != expected_coeffs:
            raise ValueError(
                f"Expected {expected_coeffs} SH coefficients for l_max={l_max}, "
                f"got {coeffs.size(-1)}"
            )

    cache_key = (l_max, n_theta, n_lambda, grid_type, reduction, dtype)
    loss_module = _SH_LOSS_CACHE.get(cache_key)
    if loss_module is None:
        loss_module = SphericalHarmonicLoss(
            l_max=l_max,
            n_theta=n_theta,
            n_lambda=n_lambda,
            grid_type=grid_type,
            reduction=reduction,
            dtype=dtype,
        )
        _SH_LOSS_CACHE[cache_key] = loss_module

    loss_module = loss_module.to(coeffs.device)
    return loss_module(coeffs, target_dirs)
