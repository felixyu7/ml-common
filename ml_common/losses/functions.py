"""Loss functions for angular/directional reconstruction and energy regression."""

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from directional_distributions import (
    von_mises_fisher_loss,
    iag_nll_loss,
    esag_nll_loss,
    gag_nll_loss,
    sipc_nll_loss,
    sespc_nll_loss,
    gspc_nll_loss,
    ps_nll_loss,
    ipt_nll_loss,
    ept_nll_loss,
    gpt_nll_loss,
)


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
