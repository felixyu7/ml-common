"""Loss functions for angular/directional reconstruction and energy regression."""

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor


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