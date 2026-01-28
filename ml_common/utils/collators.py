"""Data collators for batching variable-length samples."""

import numpy as np
import torch
from typing import List, Tuple, Any, Union, Dict


def _to_tensor(x: Any) -> torch.Tensor:
    """Convert input to float32 tensor without redundant copies."""
    if isinstance(x, torch.Tensor):
        return x.float() if x.dtype != torch.float32 else x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32, copy=False))
    return torch.as_tensor(x, dtype=torch.float32)


class IrregularDataCollator:
    """
    Collates variable-length point cloud data without padding.
    Returns concatenated points with batch indices for segment-wise pooling.
    """

    def __call__(
        self, batch: List[Union[Tuple[Any, Any, Any], Dict[str, Any]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate batch of variable-length events into concatenated tensors.

        Supports tuple format (coords, features, labels) or dict format
        with keys 'coords', 'features', and 'labels'/'targets'/'target'.

        Returns:
            coords_b: [sum_i N_i, 1+D] with batch index in column 0
            features_b: [sum_i N_i, F]
            labels_b: [B, L]
        """
        batch_coords: List[torch.Tensor] = []
        batch_features: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []

        for batch_idx, event in enumerate(batch):
            if isinstance(event, dict):
                coords = event.get('coords')
                features = event.get('features')
                labels = event.get('labels') or event.get('targets') or event.get('target')
                if coords is None or features is None or labels is None:
                    raise KeyError(
                        "Dict sample must include 'coords', 'features', and 'labels'/'targets'/'target'"
                    )
            else:
                try:
                    coords, features, labels = event
                except Exception as e:
                    raise TypeError(
                        f"Sample must be tuple (coords, features, labels) or dict. Got: {type(event)}"
                    ) from e

            coords = _to_tensor(coords)
            features = _to_tensor(features)
            labels = _to_tensor(labels)

            # Add batch index as first column to coords
            batch_indices = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.float32)
            coords_with_batch = torch.cat([batch_indices, coords], dim=1)

            batch_coords.append(coords_with_batch)
            batch_features.append(features)
            batch_labels.append(labels)

        # Concatenate points, stack labels
        if not batch_coords:
            coords_b = torch.empty((0, 5), dtype=torch.float32)
            features_b = torch.empty((0, 0), dtype=torch.float32)
            labels_b = torch.empty((0,), dtype=torch.float32)
        else:
            coords_b = torch.cat(batch_coords, dim=0)
            features_b = torch.cat(batch_features, dim=0)
            labels_b = torch.stack(batch_labels, dim=0)

        return coords_b, features_b, labels_b