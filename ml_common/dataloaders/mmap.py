"""Memory-mapped dataset for neutrino detector data (Prometheus, IceCube, etc.)."""

import torch
import numpy as np
from typing import Union, List, Tuple, Optional

try:
    import nt_summary_stats
    HAS_SUMMARY_STATS = True
except ImportError:
    nt_summary_stats = None
    HAS_SUMMARY_STATS = False

from ..utils.io import load_ntmmap


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
        task: Optional[str] = None
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
        """
        if use_summary_stats and not HAS_SUMMARY_STATS:
            raise ImportError("nt_summary_stats package is required for summary stats processing. Please do 'pip install nt-summary-stats'.")
        self.use_summary_stats = use_summary_stats and HAS_SUMMARY_STATS

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

        # Handle train/val splitting
        if split in ["train", "val"]:
            rng = np.random.RandomState(split_seed)
            indices = rng.permutation(self.total_events)

            val_size = int(self.total_events * val_split)
            if split == "val":
                self.indices = np.sort(indices[:val_size])  # sorted for disk order
            else:  # train
                self.indices = indices[val_size:]
        else:
            self.indices = None  # full dataset

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
                    vertex coordinates are in meters
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
        
        # subtract minimum photon hit time so event starts at t=0
        photons['t'] -= photons['t'].min()

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

            sensor_positions, sensor_stats = nt_summary_stats.process_event(photons_dict)

            # 4D coords: x, y, z, first_hit_time
            pos = np.column_stack([
                sensor_positions[:, 0],
                sensor_positions[:, 1],
                sensor_positions[:, 2],
                sensor_stats[:, 3]  # first hit time
            ]).astype(np.float32) / 1000.  # convert to km/microseconds

            feats = np.log(sensor_stats.astype(np.float32) + 1)
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

        # True vertex position (meters)
        vertex_x = event_record['initial_x']
        vertex_y = event_record['initial_y']
        vertex_z = event_record['initial_z']

        starting_available = 'starting' in event_record.dtype.names
        if not starting_available and self.task == 'starting_classification':
            raise KeyError("Event record does not contain 'starting'; cannot build starting_classification labels.")

        starting_flag = float(event_record['starting']) if starting_available else 0.0
        starting_flag = 1.0 if starting_flag >= 0.5 else 0.0

        if self.dataset_type == 'prometheus':
            pid = event_record['initial_type']
            labels = np.array([log_energy, dir_x, dir_y, dir_z, pid, starting_flag, vertex_x, vertex_y, vertex_z], dtype=np.float32)
        else:
            # 0: cascade, 1: through-going track, 2: starting track, 3: stopping track, 4: passing track, 5: bundle/other
            morphology = event_record['morphology']
            labels = np.array([log_energy, dir_x, dir_y, dir_z, morphology, starting_flag, vertex_x, vertex_y, vertex_z], dtype=np.float32)

        return pos, feats, labels
