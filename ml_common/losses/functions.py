"""Loss functions for angular/directional reconstruction and energy regression."""

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Dict, Tuple

# Constants for numerical stability
_SQRT_2 = np.sqrt(2.0)
_LOG_2PI = np.log(2.0 * np.pi)

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


def von_mises_fisher_loss(
    n_pred: Tensor, n_true: Tensor, kappa_reg: float = 0.01, eps: float = 1e-8
) -> Tensor:
    """
    von Mises-Fisher loss with decoupled direction and κ.

    Expects n_pred [B,4]: direction = n_pred[:,:3], κ = softplus(n_pred[:,3]).
    The κ head is regularized to prevent explosion while preserving vMF gradient scaling.
    """
    direction = F.normalize(n_pred[:, :3], p=2, dim=1)
    kappa = F.softplus(n_pred[:, 3]) + 0.1
    cos_sim = (direction * n_true).sum(dim=1)
    log_C = -kappa + torch.log((kappa + eps) / (1 - torch.exp(-2 * kappa) + 2 * eps))
    return (-(kappa * cos_sim + log_C) + kappa_reg * kappa).mean()


def _log_M2(alpha: Tensor) -> Tensor:
    """
    Compute log(M_2(alpha)) numerically stably.

    M_2(alpha) = (1 + alpha^2) * Phi(alpha) + alpha * phi(alpha)

    where Phi is the standard normal CDF and phi is the standard normal PDF.
    This function arises in the angular Gaussian distribution density.

    Reference: Paine et al. (2018), "An elliptically symmetric angular Gaussian
    distribution", Stat Comput 28:689-697, Equation (4).
    """
    # Standard normal PDF: phi(alpha) = exp(-alpha^2/2) / sqrt(2*pi)
    log_phi = -0.5 * alpha ** 2 - 0.5 * _LOG_2PI
    phi = torch.exp(log_phi)

    # Standard normal CDF: Phi(alpha) = 0.5 * (1 + erf(alpha / sqrt(2)))
    Phi = 0.5 * (1.0 + torch.erf(alpha / _SQRT_2))

    # M_2(alpha) = (1 + alpha^2) * Phi(alpha) + alpha * phi(alpha)
    M2 = (1.0 + alpha ** 2) * Phi + alpha * phi

    # Clamp to avoid log(0) for very negative alpha where M2 underflows
    M2 = torch.clamp(M2, min=1e-40)

    return torch.log(M2)


def _construct_orthonormal_basis(mu: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Construct two orthonormal vectors perpendicular to mu using Gram-Schmidt.

    This avoids the singularity in the paper's formula (Eq. 14) when mu is
    aligned with the x-axis.

    Args:
        mu: [B, 3] mean direction vectors (not necessarily normalized)

    Returns:
        xi1, xi2: [B, 3] orthonormal vectors perpendicular to mu
    """
    B = mu.shape[0]
    device, dtype = mu.device, mu.dtype
    mu_norm = F.normalize(mu, p=2, dim=1)  # [B, 3]

    # Reference vectors: use (1,0,0) unless mu is close to x-axis
    ref1 = torch.zeros(B, 3, device=device, dtype=dtype)
    ref1[:, 0] = 1.0  # (1, 0, 0)

    ref2 = torch.zeros(B, 3, device=device, dtype=dtype)
    ref2[:, 1] = 1.0  # (0, 1, 0)

    # Use ref2 when mu is close to x-axis (|mu · (1,0,0)| > 0.9)
    dot1 = torch.abs((mu_norm * ref1).sum(dim=1))
    use_ref2 = dot1 > 0.9
    ref = torch.where(use_ref2.unsqueeze(1), ref2, ref1)

    # Gram-Schmidt: xi1 = ref - (ref · mu_norm) * mu_norm, then normalize
    xi1 = ref - (ref * mu_norm).sum(dim=1, keepdim=True) * mu_norm
    xi1 = F.normalize(xi1, p=2, dim=1)

    # xi2 = mu_norm × xi1 (cross product gives third orthonormal vector)
    xi2 = torch.cross(mu_norm, xi1, dim=1)

    return xi1, xi2


def iag_nll_loss(pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Isotropic Angular Gaussian (IAG) negative log-likelihood loss.

    The IAG distribution is the angular Gaussian with V = I (identity covariance),
    making it rotationally symmetric about the mean direction. It is a special
    case of ESAG with gamma = (0, 0).

    The density is:
        f_IAG(y) = (1/2π) * exp[0.5 * ((y·μ)² - ||μ||²)] * M_2(y·μ)

    where M_2(α) = (1 + α²)Φ(α) + αφ(α), with φ and Φ being the standard
    normal PDF and CDF respectively.

    Reference: Paine et al. (2018), "An elliptically symmetric angular Gaussian
    distribution", Stat Comput 28:689-697.

    Args:
        pred: [B, 3] predicted mean vectors μ. The magnitude ||μ|| controls
              concentration (higher = more peaked), and μ/||μ|| is the mean direction.
        y_true: [B, 3] true unit direction vectors on S².

    Returns:
        Scalar mean NLL loss over the batch.
    """
    mu = pred  # [B, 3]

    # Normalize y_true to ensure unit vectors
    y = F.normalize(y_true, p=2, dim=1)  # [B, 3]

    # Compute terms
    mu_norm_sq = (mu ** 2).sum(dim=1)  # ||μ||² [B]
    y_dot_mu = (y * mu).sum(dim=1)     # y·μ [B]

    # log(M_2(y·μ))
    log_M2 = _log_M2(y_dot_mu)

    # NLL = log(2π) + 0.5*(||μ||² - (y·μ)²) - log(M_2(y·μ))
    nll = _LOG_2PI + 0.5 * (mu_norm_sq - y_dot_mu ** 2) - log_M2

    return nll.mean()


def esag_nll_loss(pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Elliptically Symmetric Angular Gaussian (ESAG) negative log-likelihood loss.

    The ESAG distribution has ellipse-like contours on the sphere, enabling
    modeling of anisotropic directional uncertainty. It generalizes IAG by
    adding shape parameters γ = (γ₁, γ₂) that control the ellipticity.

    The density is:
        f_ESAG(y) = C_3 / (y'V⁻¹y)^(3/2) * exp[0.5*((y·μ)²/(y'V⁻¹y) - ||μ||²)]
                    * M_2(y·μ / √(y'V⁻¹y))

    where V⁻¹ is constructed from μ and γ per Equation (18) of the paper.

    Reference: Paine et al. (2018), "An elliptically symmetric angular Gaussian
    distribution", Stat Comput 28:689-697.

    Args:
        pred: [B, 5] predictions where:
              - pred[:, :3] = μ (mean vector, magnitude controls concentration)
              - pred[:, 3:5] = γ = (γ₁, γ₂) (shape parameters for ellipticity)
              Setting γ = (0, 0) recovers the IAG distribution.
        y_true: [B, 3] true unit direction vectors on S².

    Returns:
        Scalar mean NLL loss over the batch.
    """
    mu = pred[:, :3]      # [B, 3]
    gamma1 = pred[:, 3]   # [B]
    gamma2 = pred[:, 4]   # [B]

    # Normalize y_true to ensure unit vectors
    y = F.normalize(y_true, p=2, dim=1)  # [B, 3]

    # Basic terms
    mu_norm_sq = (mu ** 2).sum(dim=1)  # ||μ||² [B]
    y_dot_mu = (y * mu).sum(dim=1)     # y·μ [B]

    # Construct orthonormal basis {ξ₁, ξ₂} perpendicular to μ
    xi1, xi2 = _construct_orthonormal_basis(mu)  # [B, 3] each

    # Projections of y onto the basis vectors
    a = (y * xi1).sum(dim=1)  # y·ξ₁ [B]
    b = (y * xi2).sum(dim=1)  # y·ξ₂ [B]

    # Compute y'V⁻¹y using Equation (18):
    # y'V⁻¹y = 1 + γ₁(a² - b²) + 2γ₂ab + (√(1 + γ₁² + γ₂²) - 1)(a² + b²)
    gamma_sq = gamma1 ** 2 + gamma2 ** 2
    sqrt_term = torch.sqrt(1.0 + gamma_sq)
    a_sq_plus_b_sq = a ** 2 + b ** 2

    y_Vinv_y = (1.0
                + gamma1 * (a ** 2 - b ** 2)
                + 2.0 * gamma2 * a * b
                + (sqrt_term - 1.0) * a_sq_plus_b_sq)

    # Clamp for numerical stability (V⁻¹ is positive definite, so this should be > 0)
    y_Vinv_y = torch.clamp(y_Vinv_y, min=1e-8)

    # Argument for M_2
    sqrt_y_Vinv_y = torch.sqrt(y_Vinv_y)
    alpha = y_dot_mu / sqrt_y_Vinv_y

    # log(M_2(alpha))
    log_M2 = _log_M2(alpha)

    # NLL = log(2π) + 1.5*log(y'V⁻¹y) + 0.5*(||μ||² - (y·μ)²/(y'V⁻¹y)) - log(M_2(α))
    nll = (_LOG_2PI
           + 1.5 * torch.log(y_Vinv_y)
           + 0.5 * (mu_norm_sq - y_dot_mu ** 2 / y_Vinv_y)
           - log_M2)

    return nll.mean()


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
