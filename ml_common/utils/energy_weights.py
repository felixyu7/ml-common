"""Energy-dependent sample weighting for imbalanced MC spectra."""

import numpy as np
import pyarrow.parquet as pq
from typing import Tuple, Optional, Union, List


def extract_energies(dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Extract energies (GeV) and sub-dataset IDs from a dataset without calling __getitem__.

    Returns:
        energies: [N] array of energies in GeV
        dataset_ids: [N] array of integer sub-dataset indices
    """
    from ..dataloaders.mmap import MmapDataset, BinaryLabelDataset
    from ..dataloaders.parquet import ParquetDataset

    if isinstance(dataset, BinaryLabelDataset):
        e0, d0 = extract_energies(dataset.dataset_0)
        e1, d1 = extract_energies(dataset.dataset_1)
        # Offset dataset_ids for second dataset
        d1 = d1 + (d0.max() + 1 if len(d0) > 0 else 0)
        return np.concatenate([e0, e1]), np.concatenate([d0, d1])

    if isinstance(dataset, MmapDataset):
        all_energies = []
        all_ids = []
        for i, (events, _) in enumerate(dataset.datasets):
            all_energies.append(np.array(events['initial_energy'], dtype=np.float64))
            all_ids.append(np.full(len(events), i, dtype=np.int32))
        all_energies = np.concatenate(all_energies)
        all_ids = np.concatenate(all_ids)
        if dataset.indices is not None:
            all_energies = all_energies[dataset.indices]
            all_ids = all_ids[dataset.indices]
        return all_energies, all_ids

    if isinstance(dataset, ParquetDataset):
        all_energies = []
        all_ids = []
        for i, f in enumerate(dataset.files):
            table = pq.read_table(f, columns=['mc_truth'])
            mc = table['mc_truth']
            energies = mc.field('initial_state_energy').to_numpy(zero_copy_only=False)
            all_energies.append(energies.astype(np.float64))
            all_ids.append(np.full(len(energies), i, dtype=np.int32))
        all_energies = np.concatenate(all_energies)
        all_ids = np.concatenate(all_ids)
        if dataset.indices is not None:
            all_energies = all_energies[dataset.indices]
            all_ids = all_ids[dataset.indices]
        return all_energies, all_ids

    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def compute_energy_weights(
    energies_gev: np.ndarray,
    mode: str,
    spectral_index: Optional[Union[float, List[float]]] = None,
    dataset_ids: Optional[np.ndarray] = None,
    n_bins: int = 50,
) -> np.ndarray:
    """Compute per-event sampling weights based on energy distribution.

    Args:
        energies_gev: [N] energies in GeV
        mode: "si", "flat", or "si+flat"
        spectral_index: simulation spectral index (gamma in E^-gamma).
            Float for single dataset, list for per-dataset values. Required for si/si+flat.
        dataset_ids: [N] sub-dataset index per event (for per-dataset spectral_index)
        n_bins: number of log10(E) bins for flat weighting

    Returns:
        weights: [N] normalized so weights.sum() == len(weights)
    """
    log_e = np.log10(np.maximum(energies_gev, 1e-6))

    if mode == "si":
        weights = _si_weights(energies_gev, spectral_index, dataset_ids)
    elif mode == "flat":
        weights = _flat_weights(log_e, n_bins)
    elif mode == "si+flat":
        weights = _si_weights(energies_gev, spectral_index, dataset_ids)
        # Flatten residual non-uniformity after si correction
        weights *= _flat_weights(log_e, n_bins, sample_weights=weights)
    else:
        raise ValueError(f"Unknown energy_weighting mode: '{mode}'")

    weights *= len(weights) / weights.sum()
    return weights.astype(np.float64)


def _si_weights(
    energies_gev: np.ndarray,
    spectral_index: Optional[Union[float, List[float]]],
    dataset_ids: Optional[np.ndarray],
) -> np.ndarray:
    """Spectral index reweighting: E^(gamma - 1) undoes E^-gamma -> E^-1 (flat in log E)."""
    if spectral_index is None:
        raise ValueError("spectral_index required for 'si' and 'si+flat' modes")

    if isinstance(spectral_index, (list, np.ndarray)):
        gamma = np.array(spectral_index, dtype=np.float64)
        if dataset_ids is None:
            raise ValueError("dataset_ids required when spectral_index is a list")
        exponents = gamma[dataset_ids] - 1.0
    else:
        exponents = float(spectral_index) - 1.0

    return np.power(np.maximum(energies_gev, 1e-6), exponents)


def _flat_weights(
    log_e: np.ndarray,
    n_bins: int,
    sample_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Inverse-density weighting to flatten the log10(E) distribution."""
    bin_edges = np.linspace(log_e.min() - 1e-6, log_e.max() + 1e-6, n_bins + 1)
    bin_idx = np.digitize(log_e, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    if sample_weights is not None:
        counts = np.bincount(bin_idx, weights=sample_weights, minlength=n_bins)
    else:
        counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)

    # Inverse density, zero weight for empty bins
    safe_counts = np.where(counts > 0, counts, 1.0)
    bin_weights = np.where(counts > 0, 1.0 / safe_counts, 0.0)
    return bin_weights[bin_idx]
