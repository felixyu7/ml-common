"""Parquet dataset for Prometheus MC with stochastic loss information.

Matches MmapDataset signature: returns (coords, features, labels) tuple.
Stochastic loss info is packed into extended labels array for compatibility with
existing collators and training infrastructure.
"""

import torch
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from typing import Union, List, Tuple, Optional
from collections import OrderedDict

try:
    import nt_summary_stats
    HAS_SUMMARY_STATS = True
except ImportError:
    nt_summary_stats = None
    HAS_SUMMARY_STATS = False


# Stochastic loss types from prometheus (exclude type 1 = delta ray/ionization)
# 0: bremsstrahlung, 2: pair production, 3: photonuclear, 4: muon pair production
STOCHASTIC_LOSS_TYPES = (0, 2, 3, 4)


class ParquetDataset(torch.utils.data.Dataset):
    """
    Parquet dataset for Prometheus MC with stochastic losses.

    Matches MmapDataset signature: returns (coords, features, labels) tuple.
    Stochastic loss info is packed into extended labels array.

    Extended labels format:
        [0]: log10(energy)
        [1:4]: direction (x, y, z)
        [4]: pid
        [5]: starting_flag
        [6:9]: vertex (x, y, z) in meters
        [9]: n_stochastic (actual count before padding)
        [10:10+max_stochastic]: stochastic_distances in meters
        [10+max_stochastic:10+2*max_stochastic]: stochastic_energies in GeV
    """

    def __init__(
        self,
        parquet_paths: Union[str, Path, List[str]],
        use_summary_stats: bool = True,
        split: str = "full",
        val_split: float = 0.2,
        split_seed: int = 42,
        task: Optional[str] = None,
        max_stochastic: int = 100,
        min_stochastic_energy: float = 0.5,
        cache_size: int = 20,
    ):
        """
        Initialize ParquetDataset.

        Args:
            parquet_paths: Path(s) to parquet files or directory containing them
            use_summary_stats: Use nt-summary-stats for DOM-level aggregation
            split: "train", "val", or "full"
            val_split: Validation fraction (when split != "full")
            split_seed: Random seed for reproducible splitting
            task: Task identifier (for API compatibility, not currently used)
            max_stochastic: Maximum number of stochastic losses to keep per event
            min_stochastic_energy: Minimum stochastic loss energy threshold in GeV
            cache_size: Number of parquet files to keep in LRU cache
        """
        if use_summary_stats and not HAS_SUMMARY_STATS:
            raise ImportError(
                "nt_summary_stats package is required for summary stats processing. "
                "Please do 'pip install nt-summary-stats'."
            )
        self.use_summary_stats = use_summary_stats and HAS_SUMMARY_STATS
        self.max_stochastic = max_stochastic
        self.min_stochastic_energy = min_stochastic_energy
        self.cache_size = cache_size
        self.task = (task or 'event_reconstruction').lower()

        # Find parquet files (handles single path, directory, or list of either)
        if isinstance(parquet_paths, (str, Path)):
            parquet_paths = [parquet_paths]

        self.files = []
        for p in parquet_paths:
            p = Path(p)
            if p.is_dir():
                self.files.extend(sorted(p.glob("*.parquet")))
            else:
                self.files.append(p)

        if not self.files:
            raise ValueError(f"No parquet files found at {parquet_paths}")

        # Get file lengths without loading data (only reads metadata)
        self.file_lengths = []
        self.cumulative_lengths = []
        total = 0
        for f in self.files:
            pf = pq.ParquetFile(f)
            n = pf.metadata.num_rows
            self.file_lengths.append(n)
            total += n
            self.cumulative_lengths.append(total)

        self.total_events = total
        self.dataset_type = 'prometheus'

        # Train/val split (matches MmapDataset logic)
        if split in ["train", "val"]:
            rng = np.random.RandomState(split_seed)
            indices = rng.permutation(self.total_events)
            val_size = int(self.total_events * val_split)
            if split == "val":
                self.indices = np.sort(indices[:val_size])  # sorted for disk locality
            else:  # train
                self.indices = indices[val_size:]
        else:
            self.indices = None  # full dataset

        # LRU cache for loaded dataframes
        self._cache: OrderedDict = OrderedDict()

        split_len = len(self.indices) if self.indices is not None else self.total_events
        split_str = f" ({split} split from {self.total_events:,} total)" if split != "full" else ""
        print(f"Loaded prometheus parquet: {split_len:,} events{split_str}")

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else self.total_events

    def _load_file(self, file_idx: int):
        """Load parquet file with LRU caching."""
        if file_idx in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(file_idx)
            return self._cache[file_idx]

        # Load file as arrow table (faster than pandas for row access)
        table = pq.read_table(self.files[file_idx])

        # Evict oldest if cache is full
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)

        self._cache[file_idx] = table
        return table

    def _get_row(self, file_idx: int, local_idx: int):
        """Get a single row efficiently."""
        table = self._load_file(file_idx)
        # Use take() for single-row access - much faster than pandas iloc
        row_table = table.take([local_idx])
        return row_table.to_pydict()

    def _get_file_and_local_idx(self, global_idx: int) -> Tuple[int, int]:
        """Map global index to (file_idx, local_idx within file)."""
        file_idx = np.searchsorted(self.cumulative_lengths, global_idx + 1)
        local_idx = global_idx if file_idx == 0 else global_idx - self.cumulative_lengths[file_idx - 1]
        return file_idx, local_idx

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get event at index.

        Returns:
            coords: [N, 4] (x, y, z, t) in km/microseconds (matches MmapDataset)
            features: [N, F] log-transformed sensor stats
            labels: [10 + 2*max_stochastic] extended labels with stochastic loss info
        """
        # Map split index to global index
        global_idx = self.indices[idx] if self.indices is not None else idx

        # Get file and local index
        file_idx, local_idx = self._get_file_and_local_idx(global_idx)

        # Load row data
        row = self._get_row(file_idx, local_idx)
        mc = row['mc_truth'][0]  # [0] because pydict returns lists
        photons = row['photons'][0]

        # === Process Photons (matches MmapDataset) ===
        hit_x = np.array(photons['sensor_pos_x'], dtype=np.float32)
        hit_y = np.array(photons['sensor_pos_y'], dtype=np.float32)
        hit_z = np.array(photons['sensor_pos_z'], dtype=np.float32)
        hit_t = np.array(photons['t'], dtype=np.float32)

        # Normalize time to start at 0
        if len(hit_t) > 0:
            hit_t = hit_t - hit_t.min()

        if self.use_summary_stats and len(hit_t) > 0:
            # Aggregate to DOM level using nt_summary_stats
            photons_dict = {
                'sensor_pos_x': hit_x,
                'sensor_pos_y': hit_y,
                'sensor_pos_z': hit_z,
                't': hit_t,
                'charge': np.ones_like(hit_t),  # count photons as charge
                'string_id': np.array(photons['string_id']),
                'sensor_id': np.array(photons['sensor_id']),
            }

            if 'id_idx' in photons:
                photons_dict['id_idx'] = np.array(photons['id_idx'])

            sensor_positions, sensor_stats = nt_summary_stats.process_event(photons_dict)

            # 4D coords: x, y, z, first_hit_time (in km/microseconds)
            pos = np.column_stack([
                sensor_positions[:, 0],
                sensor_positions[:, 1],
                sensor_positions[:, 2],
                sensor_stats[:, 3]  # first hit time
            ]).astype(np.float32) / 1000.  # convert m/ns to km/us

            feats = np.log(sensor_stats.astype(np.float32) + 1)
        else:
            # Pulse-level (no aggregation)
            # Features: [placeholder, log(charge+1), string_id, sensor_id] to match MmapDataset format
            pos = np.column_stack([hit_x, hit_y, hit_z, hit_t]).astype(np.float32) / 1000.
            string_ids = np.array(photons['string_id'], dtype=np.float32)
            sensor_ids = np.array(photons['sensor_id'], dtype=np.float32)
            log_charge = np.log(2.0) * np.ones(len(hit_t), dtype=np.float32)  # 1 photon = log(1+1)
            feats = np.column_stack([
                np.zeros(len(hit_t), dtype=np.float32),  # placeholder (column 0)
                log_charge,   # log(charge+1) (column 1)
                string_ids,   # string_id (column 2)
                sensor_ids,   # sensor_id (column 3)
            ])

        # === Extract Event Labels ===
        zenith = mc['initial_state_zenith']
        azimuth = mc['initial_state_azimuth']
        energy = mc['initial_state_energy']

        # Spherical to Cartesian direction
        dir_x = np.sin(zenith) * np.cos(azimuth)
        dir_y = np.sin(zenith) * np.sin(azimuth)
        dir_z = np.cos(zenith)

        # Vertex position (meters)
        vertex_x = mc['initial_state_x']
        vertex_y = mc['initial_state_y']
        vertex_z = mc['initial_state_z']

        pid = mc['initial_state_type']
        starting_flag = 1.0  # assume starting track for prometheus simulation

        # === Process Stochastic Losses ===
        loss_types = np.array(mc['loss_type'])
        loss_distances = np.array(mc['loss_distance'], dtype=np.float32)
        loss_energies = np.array(mc['loss_energy'], dtype=np.float32)

        # Filter: stochastic types only, above energy threshold
        mask = (
            np.isin(loss_types, STOCHASTIC_LOSS_TYPES) &
            (loss_energies >= self.min_stochastic_energy)
        )
        stochastic_distances = loss_distances[mask]
        stochastic_energies = loss_energies[mask]
        stochastic_types = loss_types[mask]

        # Map loss types to sequential indices: 0=Brems, 1=Epair, 2=Photonuclear, 3=MuPair
        # STOCHASTIC_LOSS_TYPES = (0, 2, 3, 4) -> map to (0, 1, 2, 3)
        type_map = {0: 0, 2: 1, 3: 2, 4: 3}
        stochastic_types = np.array([type_map.get(t, 4) for t in stochastic_types], dtype=np.float32)

        n_stochastic = len(stochastic_distances)

        # Truncate to max_stochastic (keep highest energy stochastic losses)
        if n_stochastic > self.max_stochastic:
            # Get indices of top-K by energy
            top_idx = np.argsort(stochastic_energies)[-self.max_stochastic:]
            # Sort by distance to maintain spatial ordering
            top_idx = np.sort(top_idx)
            stochastic_distances = stochastic_distances[top_idx]
            stochastic_energies = stochastic_energies[top_idx]
            stochastic_types = stochastic_types[top_idx]
            n_stochastic = self.max_stochastic

        # Pad to fixed size
        pad_distances = np.zeros(self.max_stochastic, dtype=np.float32)
        pad_energies = np.zeros(self.max_stochastic, dtype=np.float32)
        pad_types = np.zeros(self.max_stochastic, dtype=np.float32)
        pad_distances[:n_stochastic] = stochastic_distances
        pad_energies[:n_stochastic] = stochastic_energies
        pad_types[:n_stochastic] = stochastic_types

        # === Build Extended Labels ===
        # Format: [base_labels(9), n_stochastic(1), distances(max), energies(max), types(max)]
        labels = np.concatenate([
            np.array([
                np.log10(max(energy, 1e-6)),  # [0] log energy
                dir_x, dir_y, dir_z,           # [1:4] direction
                pid,                            # [4] particle type
                starting_flag,                  # [5] starting flag
                vertex_x, vertex_y, vertex_z,  # [6:9] vertex position
                float(n_stochastic),             # [9] stochastic loss count
            ], dtype=np.float32),
            pad_distances,  # [10:10+max_stochastic]
            pad_energies,   # [10+max_stochastic:10+2*max_stochastic]
            pad_types,      # [10+2*max_stochastic:10+3*max_stochastic]
        ])

        return pos, feats, labels


def unpack_stochastic_labels(
    labels: torch.Tensor,
    max_stochastic: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpack stochastic loss information from extended labels tensor.

    Args:
        labels: [B, 10 + 3*max_stochastic] extended labels from ParquetDataset
        max_stochastic: Number of stochastic loss slots (must match dataset config)

    Returns:
        base_labels: [B, 9] standard labels (energy, direction, pid, etc.)
        n_stochastic: [B] number of valid stochastic losses per event
        stochastic_distances: [B, max_stochastic] stochastic loss distances in meters
        stochastic_energies: [B, max_stochastic] stochastic loss energies in GeV
        stochastic_types: [B, max_stochastic] stochastic loss type indices (0=Brems, 1=Epair, 2=Photonuclear, 3=MuPair)
    """
    base_labels = labels[:, :9]
    n_stochastic = labels[:, 9].long()
    stochastic_distances = labels[:, 10:10 + max_stochastic]
    stochastic_energies = labels[:, 10 + max_stochastic:10 + 2 * max_stochastic]
    stochastic_types = labels[:, 10 + 2 * max_stochastic:10 + 3 * max_stochastic].long()

    return base_labels, n_stochastic, stochastic_distances, stochastic_energies, stochastic_types


def create_stochastic_mask(n_stochastic: torch.Tensor, max_stochastic: int = 100) -> torch.Tensor:
    """
    Create boolean mask for valid stochastic losses.

    Args:
        n_stochastic: [B] number of valid stochastic losses per event
        max_stochastic: Total stochastic loss slots

    Returns:
        mask: [B, max_stochastic] boolean mask (True = valid stochastic loss)
    """
    B = n_stochastic.shape[0]
    indices = torch.arange(max_stochastic, device=n_stochastic.device).expand(B, -1)
    return indices < n_stochastic.unsqueeze(1)


class FileGroupedSampler(torch.utils.data.Sampler):
    """Sampler that groups indices by file to minimize parquet file loading.

    Each epoch:
        1. Shuffles the order of files
        2. Shuffles event indices within each file
        3. Yields all events from one file before moving to the next

    This ensures each file is loaded once per epoch, dramatically reducing I/O.
    """

    def __init__(self, dataset: ParquetDataset, shuffle: bool = True, seed: int = 42):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._build_file_groups()

    def _build_file_groups(self):
        """Group dataset indices by their source file."""
        n_files = len(self.dataset.files)
        self.file_groups = [[] for _ in range(n_files)]

        for idx in range(len(self.dataset)):
            global_idx = self.dataset.indices[idx] if self.dataset.indices is not None else idx
            file_idx, _ = self.dataset._get_file_and_local_idx(global_idx)
            self.file_groups[file_idx].append(idx)

        # Filter out empty groups (files with no events in this split)
        self.file_groups = [g for g in self.file_groups if g]

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        # Shuffle file order
        file_order = list(range(len(self.file_groups)))
        if self.shuffle:
            rng.shuffle(file_order)

        # Yield indices file by file
        for file_idx in file_order:
            indices = self.file_groups[file_idx].copy()
            if self.shuffle:
                rng.shuffle(indices)
            yield from indices

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across epochs."""
        self.epoch = epoch
