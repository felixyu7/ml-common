"""Memory-mapped dataset for neutrino detector data (Prometheus, IceCube, etc.)."""

import torch
import numpy as np
from typing import Union, List, Tuple, Optional, Dict

try:
    import nt_summary_stats
    HAS_SUMMARY_STATS = True
except ImportError:
    nt_summary_stats = None
    HAS_SUMMARY_STATS = False

from ..utils.io import load_ntmmap

# Detector geometry center (meters), per dataset type. Vertex targets are
# expressed relative to this point and converted to km, so the network
# regresses an O(1) offset from the detector center.
VERTEX_CENTER_M = {
    'prometheus': (0.0, 0.0, -1942.0),  # icgeo instrumented-volume center
    'icecube': (0.0, 0.0, 0.0),         # IceCube detector frame is ~centered
}


def _charge_weighted_median(times: np.ndarray, charges: np.ndarray) -> float:
    """Charge-weighted median pulse time — a robust per-event time reference.

    Replaces the event-minimum reference. The minimum is an extreme-value
    statistic set by the single earliest (often scattered/noise) hit, which sits
    ~1 us differently in data vs. MC and shifts the absolute first-hit-time
    feature/coordinate the network consumes. The inter-DOM timing structure that
    actually encodes direction is invariant to the per-event reference, so
    centering on a robust (charge-weighted median) time makes the absolute time
    feature data-MC consistent without changing the directional signal.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.size == 0:
        return 0.0
    charges = np.asarray(charges, dtype=np.float64)
    order = np.argsort(times, kind='mergesort')
    ts = times[order]
    cw = np.cumsum(charges[order])
    total = cw[-1]
    if total <= 0:                      # degenerate weights -> plain median
        return float(ts[ts.size // 2])
    return float(ts[np.searchsorted(cw, 0.5 * total)])


def _normalize_summary_stats(sensor_stats: np.ndarray, extended: bool) -> np.ndarray:
    """Per-feature normalization for summary statistics.

    first_hit_time (col 3) is now signed — it is relative to the per-event
    charge-weighted-median reference (see _charge_weighted_median / __getitem__) —
    so it uses a sign-preserving log. Charges and Δt features are non-negative, so
    signed-log reduces to log1p for them.

    For the 25 extended features:
      - Absolute time percentiles (indices 4-7, 9-14) are converted to
        deltas relative to first_hit_time (reference-invariant, non-negative).
      - first_hit_time (3) uses sign-preserving log.
      - q_max_frac (22) and t_skewness (24) use identity (no transform).
      - All other features use log1p.
    """
    stats = sensor_stats.astype(np.float32)
    if not extended:
        # signed-log: log1p for non-negative cols (charges), sign-preserving for
        # the signed time cols (3-7).
        return np.sign(stats) * np.log1p(np.abs(stats))

    # Convert absolute time percentiles to Δt from first_hit_time (the per-event
    # reference cancels, so these deltas are unchanged and stay non-negative).
    first_t = stats[:, 3:4]  # [N, 1]
    dt_idx = [4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
    stats[:, dt_idx] = np.maximum(stats[:, dt_idx] - first_t, 0.0)

    # log1p for non-negative features; sign-preserving log for the signed
    # first_hit_time (3); identity for q_max_frac (22) and t_skewness (24).
    log1p_idx = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23]
    feats = np.empty_like(stats)
    feats[:, log1p_idx] = np.log1p(stats[:, log1p_idx])
    feats[:, 3] = np.sign(stats[:, 3]) * np.log1p(np.abs(stats[:, 3]))
    feats[:, 22] = stats[:, 22]
    feats[:, 24] = stats[:, 24]
    return feats


class MmapDataset(torch.utils.data.Dataset):
    """
    Memory-mapped dataset for neutrino detector data.

    Auto-detects dataset type from file headers. Supports multiple mmap files
    and train/val splitting.
    """

    def __init__(
        self,
        mmap_paths: Union[str, List[str]],
        use_summary_stats: bool = True,
        split: str = "full",
        val_split: float = 0.2,
        split_seed: int = 42,
        task: Optional[str] = None,
        extended_stats: bool = False,
        morphology_filter: Optional[Union[List[int], Dict[int, float]]] = None,
    ):
        """
        Initialize MmapDataset.

        Args:
            mmap_paths: Path(s) to mmap files (without .idx/.dat extension)
            use_summary_stats: Use nt-summary-stats processing if available
            split: "train", "val", or "full"
            val_split: Validation fraction (when split != "full")
            split_seed: Random seed for reproducible splitting
            task: Task identifier used for validation. Supported values are
                None/"event_reconstruction" (default) and "starting_classification".
                When set to "starting_classification" the dataset checks that the
                event record contains a 'starting' field.
            morphology_filter: If provided, restrict the dataset by morphology.
                List form (e.g. [1, 2, 3]) keeps all events of the listed classes.
                Dict form (e.g. {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.1}) keeps the given
                fraction of each class, subsampled deterministically using
                split_seed. IceCube codes: 0=cascade, 1=starting track,
                2=throughgoing track, 3=stopping track, 4=uncontained, 5=bundle.
                IceCube-only.
        """
        if use_summary_stats and not HAS_SUMMARY_STATS:
            raise ImportError("nt_summary_stats package is required for summary stats processing. Please do 'pip install nt-summary-stats'.")
        self.use_summary_stats = use_summary_stats and HAS_SUMMARY_STATS
        self.extended_stats = extended_stats

        if isinstance(mmap_paths, str):
            mmap_paths = [mmap_paths]

        self.datasets = []
        self.cumulative_lengths = []
        total_length = 0
        self.dataset_type = None

        # Load all mmap files
        for path in mmap_paths:
            events, photons_array, photon_dtype = load_ntmmap(path)

            # Auto-detect dataset type from first file
            if self.dataset_type is None:
                field_names = set(events.dtype.names)
                if 'bjorken_x' in field_names or 'column_depth' in field_names:
                    self.dataset_type = 'prometheus'
                else:
                    self.dataset_type = 'icecube'

            self.datasets.append((events, photons_array))
            total_length += len(events)
            self.cumulative_lengths.append(total_length)

        self.total_events = total_length
        self.photon_dtype = photon_dtype
        self.task = (task or 'event_reconstruction').lower()
        if self.task not in {'event_reconstruction', 'starting_classification'}:
            raise ValueError(f"Unsupported task '{task}'. Expected 'event_reconstruction' or 'starting_classification'.")

        # Optional morphology pre-filter (IceCube only)
        valid_pool = None
        if morphology_filter is not None:
            if self.dataset_type != 'icecube':
                raise ValueError("morphology_filter is only supported for IceCube datasets")
            if isinstance(morphology_filter, dict):
                keep_fracs = {int(k): float(v) for k, v in morphology_filter.items()}
            else:
                keep_fracs = {int(k): 1.0 for k in morphology_filter}
            morph = np.concatenate([
                np.asarray(events['morphology'], dtype=np.int64)
                for events, _ in self.datasets
            ])
            filt_rng = np.random.RandomState(split_seed)
            pools = []
            for cls, frac in keep_fracs.items():
                idx = np.where(morph == cls)[0]
                if frac < 1.0:
                    n = int(round(len(idx) * frac))
                    idx = filt_rng.choice(idx, size=n, replace=False)
                pools.append(idx)
                print(f"  morph {cls}: kept {len(idx):,} (frac={frac})")
            valid_pool = np.sort(np.concatenate(pools)) if pools else np.array([], dtype=np.int64)
            print(f"Morphology filter: kept {len(valid_pool):,} / {self.total_events:,} events")

        # Handle train/val splitting (over the filtered pool, if any)
        if split in ["train", "val"]:
            rng = np.random.RandomState(split_seed)
            pool = valid_pool if valid_pool is not None else np.arange(self.total_events)
            shuffled = rng.permutation(pool)

            val_size = int(len(shuffled) * val_split)
            if split == "val":
                self.indices = np.sort(shuffled[:val_size])  # sorted for disk order
            else:  # train
                self.indices = shuffled[val_size:]
        else:
            self.indices = valid_pool  # None => all events; array => filtered only

        split_length = len(self.indices) if self.indices is not None else self.total_events
        split_str = f" ({split} split from {self.total_events:,} total)" if split != "full" else ""
        print(f"Loaded {self.dataset_type} dataset: {split_length:,} events{split_str}")

    def __len__(self):
        return len(self.indices) if self.indices is not None else self.total_events

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get event at index.

        Returns:
            coords: [N, 4] (x, y, z, t)
            features: [N, F]
            labels: [log_energy, dir_x, dir_y, dir_z, pid, starting_flag, vertex_x, vertex_y, vertex_z]
                    vertex coordinates are detector-centered and in km
                    (see VERTEX_CENTER_M)
        """
        # Map split index to global index
        global_idx = self.indices[idx] if self.indices is not None else idx

        # Find which dataset contains this index
        dataset_idx = np.searchsorted(self.cumulative_lengths, global_idx + 1)
        local_idx = global_idx if dataset_idx == 0 else global_idx - self.cumulative_lengths[dataset_idx - 1]

        # Get event and photons
        events, photons_array = self.datasets[dataset_idx]
        event_record = events[local_idx]

        start_idx = int(event_record['photon_start_idx'])
        end_idx = int(event_record['photon_end_idx'])

        if start_idx >= end_idx:
            raise ValueError(f"Invalid photon indices: {start_idx} >= {end_idx}")

        photons = photons_array[start_idx:end_idx].copy()
        
        # center times on the charge-weighted median (robust per-event reference,
        # data-MC consistent) instead of the unstable event minimum. Times become
        # signed (early hits negative); inter-DOM structure is unchanged.
        photons['t'] -= _charge_weighted_median(photons['t'], photons['charge'])

        # Process photons
        if self.use_summary_stats and len(photons) > 0:
            photons_dict = {
                'sensor_pos_x': photons['x'],
                'sensor_pos_y': photons['y'],
                'sensor_pos_z': photons['z'],
                't': photons['t'],
                'charge': photons['charge'],
                'string_id': photons['string_id'],
                'sensor_id': photons['sensor_id']
            }

            if self.dataset_type == 'prometheus' and 'id_idx' in photons.dtype.names:
                photons_dict['id_idx'] = photons['id_idx']

            sensor_positions, sensor_stats = nt_summary_stats.process_event(
                photons_dict, extended=self.extended_stats
            )

            # 4D coords: x, y, z, first_hit_time (signed, relative to the
            # per-event charge-weighted-median reference)
            pos = np.column_stack([
                sensor_positions[:, 0],
                sensor_positions[:, 1],
                sensor_positions[:, 2],
                sensor_stats[:, 3]  # first hit time
            ]).astype(np.float32) / 1000.  # convert to km/microseconds

            feats = _normalize_summary_stats(sensor_stats, self.extended_stats)
        else:
            # Pulse-level: use time, charge, and DOM identifiers as features
            pos = np.column_stack([photons['x'], photons['y'], photons['z'], photons['t']]) / 1000.
            charge_feat = np.log(photons['charge'] + 1).reshape(-1, 1).astype(np.float32)
            time_feat = pos[:, 3:4].astype(np.float32)
            string_id = photons['string_id'].reshape(-1, 1).astype(np.float32)
            sensor_id = photons['sensor_id'].reshape(-1, 1).astype(np.float32)
            feats = np.concatenate([time_feat, charge_feat, string_id, sensor_id], axis=1)

        # Extract labels
        initial_zenith = event_record['initial_zenith']
        initial_azimuth = event_record['initial_azimuth']
        initial_energy = event_record['initial_energy']
        log_energy = np.log10(max(initial_energy, 1e-6))

        # Spherical to Cartesian direction
        dir_x = np.sin(initial_zenith) * np.cos(initial_azimuth)
        dir_y = np.sin(initial_zenith) * np.sin(initial_azimuth)
        dir_z = np.cos(initial_zenith)

        starting_available = 'starting' in event_record.dtype.names
        if not starting_available and self.task == 'starting_classification':
            raise KeyError("Event record does not contain 'starting'; cannot build starting_classification labels.")

        starting_flag = float(event_record['starting']) if starting_available else 0.0
        starting_flag = 1.0 if starting_flag >= 0.5 else 0.0

        if self.dataset_type == 'prometheus':
            # Use the interaction vertex (vertex_*), not the neutrino generation
            # point (initial_*), which can be hundreds of km from the detector.
            vertex_x = event_record['vertex_x']
            vertex_y = event_record['vertex_y']
            vertex_z = event_record['vertex_z']
            class_field = event_record['initial_type']  # pid
        else:
            # IceCube: vertex_x/y/z is the precomputed interaction point (starting)
            # or track entry point (throughgoing). NaN for uncontained/bundle.
            vertex_x = event_record['vertex_x']
            vertex_y = event_record['vertex_y']
            vertex_z = event_record['vertex_z']
            # 0: cascade, 1: starting track, 2: throughgoing track, 3: stopping track, 4: uncontained, 5: bundle
            class_field = event_record['morphology']

        # Vertex target: detector-centered and converted to km, matching the
        # km input coordinate frame used for `pos`.
        cx, cy, cz = VERTEX_CENTER_M[self.dataset_type]
        vertex_x = (vertex_x - cx) / 1000.0
        vertex_y = (vertex_y - cy) / 1000.0
        vertex_z = (vertex_z - cz) / 1000.0

        labels = np.array([log_energy, dir_x, dir_y, dir_z, class_field, starting_flag, vertex_x, vertex_y, vertex_z], dtype=np.float32)

        return pos, feats, labels


class BinaryLabelDataset(torch.utils.data.Dataset):
    """Wrapper that assigns binary labels to samples from two datasets."""

    def __init__(self, dataset_0: MmapDataset, dataset_1: MmapDataset):
        """
        Args:
            dataset_0: Dataset for class 0 (e.g., CORSIKA muons)
            dataset_1: Dataset for class 1 (e.g., NuGen neutrinos)
        """
        self.dataset_0 = dataset_0
        self.dataset_1 = dataset_1
        self.len_0 = len(dataset_0)
        self.len_1 = len(dataset_1)

    def __len__(self):
        return self.len_0 + self.len_1

    def __getitem__(self, idx):
        if idx < self.len_0:
            pos, feats, labels = self.dataset_0[idx]
            binary_label = 0.0
        else:
            pos, feats, labels = self.dataset_1[idx - self.len_0]
            binary_label = 1.0

        # Append binary label as last element (same convention as starting_classification)
        labels = np.append(labels, binary_label)
        return pos, feats, labels
