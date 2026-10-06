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

    def __init__(self, dom_dropout: float = 0.0):
        self.dom_dropout = dom_dropout
        self.training = True

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
                # Explicit None checks: `or` would coerce array truthiness and
                # raise on any multi-element label tensor.
                labels = event.get('labels')
                if labels is None:
                    labels = event.get('targets')
                if labels is None:
                    labels = event.get('target')
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

            # DOM dropout: independent per-DOM Bernoulli during training,
            # always keeping at least one DOM. NOT fixed-count subsampling —
            # int(n*(1-p)) over-drops sparse events (a 2-DOM event loses 50%
            # at any p > 0) and never leaves an event intact. Points are grouped
            # into DOMs by position, so in pulse mode a dropped DOM takes all of
            # its pulses with it (per-pulse dropout would be a different,
            # much weaker augmentation).
            if self.dom_dropout > 0 and self.training:
                n = coords.shape[0]
                if n > 0:
                    # 1-D key from positions quantized to 1e-5 (1 cm in km; DOMs are
                    # >= 7 m apart): unique on int64 is ~40x faster than on rows.
                    qc = torch.round(coords[:, :3].double() * 1e5).long() + (1 << 20)
                    key = (qc[:, 0] << 42) | (qc[:, 1] << 21) | qc[:, 2]
                    dom_ids = torch.unique(key, return_inverse=True)[1]
                    n_dom = int(dom_ids.max()) + 1
                else:
                    n_dom = 0
                if n_dom == n:
                    # One point per DOM (summary stats): same draw as before.
                    keep_mask = torch.rand(n) >= self.dom_dropout
                    if n > 0 and not keep_mask.any():
                        keep_mask[torch.randint(n, (1,)).item()] = True
                else:
                    keep_dom = torch.rand(n_dom) >= self.dom_dropout
                    if not keep_dom.any():
                        keep_dom[torch.randint(n_dom, (1,)).item()] = True
                    keep_mask = keep_dom[dom_ids]
                coords = coords[keep_mask]
                features = features[keep_mask]

            # Add batch index as first column to coords
            batch_indices = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.float32)
            coords_with_batch = torch.cat([batch_indices, coords], dim=1)

            batch_coords.append(coords_with_batch)
            batch_features.append(features)
            batch_labels.append(labels)

        # Concatenate points, stack labels (DataLoader never yields an empty batch)
        coords_b = torch.cat(batch_coords, dim=0)
        features_b = torch.cat(batch_features, dim=0)
        labels_b = torch.stack(batch_labels, dim=0)

        return coords_b, features_b, labels_b
