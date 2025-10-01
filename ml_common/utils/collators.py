"""Data collators for batching variable-length samples."""

import numpy as np
import torch
from typing import List, Tuple, Any, Union, Dict


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
        batch_coords: List[np.ndarray] = []
        batch_features: List[np.ndarray] = []
        batch_labels: List[np.ndarray] = []

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

            coords = np.asarray(coords, dtype=np.float32)
            features = np.asarray(features, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.float32)

            # Add batch index as first column to coords
            batch_indices = np.full((coords.shape[0], 1), batch_idx, dtype=np.float32)
            coords_with_batch = np.concatenate([batch_indices, coords], axis=1)

            batch_coords.append(coords_with_batch)
            batch_features.append(features)
            batch_labels.append(labels)

        # Concatenate points, stack labels
        if not batch_coords:
            coords_b = torch.empty((0, 5), dtype=torch.float32)
            features_b = torch.empty((0, 0), dtype=torch.float32)
            labels_b = torch.empty((0,), dtype=torch.float32)
        else:
            coords_b = torch.from_numpy(np.vstack(batch_coords)).float()
            features_b = torch.from_numpy(np.vstack(batch_features)).float()
            try:
                labels_np = np.stack(batch_labels, axis=0).astype(np.float32)
            except ValueError:
                labels_np = np.array(batch_labels, dtype=np.float32)
            labels_b = torch.from_numpy(labels_np).float()

        return coords_b, features_b, labels_b