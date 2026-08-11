"""Loss functions for ML tasks.

Directional NLL losses (vMF/IAG/ESAG/GAG) live in ``directional_distributions``;
import them from there directly.
"""

from .functions import angular_distance_loss, gaussian_nll_loss

__all__ = ['angular_distance_loss', 'gaussian_nll_loss']
