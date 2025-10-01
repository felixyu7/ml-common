"""Loss functions for ML tasks."""

from .functions import (
    angular_distance_loss,
    gaussian_nll_loss,
    von_mises_fisher_loss,
)

__all__ = [
    'angular_distance_loss',
    'gaussian_nll_loss',
    'von_mises_fisher_loss',
]