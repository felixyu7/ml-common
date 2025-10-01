"""Legacy IceCube Kaggle dataset with separate batch/meta files."""

import os
import glob
import torch
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import random
from typing import List, Tuple
from pathlib import Path
from collections import OrderedDict
from bisect import bisect_right

try:
    import nt_summary_stats
    HAS_SUMMARY_STATS = True
except ImportError:
    nt_summary_stats = None
    HAS_SUMMARY_STATS = False


def load_sensor_geometry(geometry_path: str) -> np.ndarray:
    """Load IceCube sensor geometry from CSV."""
    geometry_df = pd.read_csv(geometry_path)
    return geometry_df[['x', 'y', 'z']].values.astype(np.float32)


def get_icecube_file_names(
    batch_dir: str,
    ranges: List[int],
    shuffle_files: bool = False
) -> List[str]:
    """Get batch file names from directory within specified ranges."""
    all_files = sorted(glob.glob(os.path.join(batch_dir, 'batch_*.parquet')))
    if shuffle_files:
        random.shuffle(all_files)

    if len(ranges) == 2:
        filtered_files = all_files[ranges[0]:ranges[1]]
    else:
        filtered_files = all_files

    return sorted(filtered_files)


class KaggleDataset(torch.utils.data.Dataset):
    """
    Dataset for legacy IceCube Kaggle format with separate batch and meta files.

    Uses OrderedDict caching for efficient file loading. Works with RandomChunkSampler
    for optimal I/O performance.
    """

    def __init__(
        self,
        meta_dir: str,
        batch_files: List[str],
        sensor_geometry: np.ndarray,
        cache_size: int = 2,
        use_summary_stats: bool = True
    ):
        """
        Initialize KaggleDataset.

        Args:
            meta_dir: Directory with metadata parquet files
            batch_files: List of batch file paths
            sensor_geometry: Sensor geometry array [n_sensors, 3]
            cache_size: Number of batch files to cache in memory
            use_summary_stats: Use nt-summary-stats if available
        """
        self.meta_dir = meta_dir
        self.batch_files = batch_files
        self.batch_file_names = [Path(f).name for f in batch_files]
        self.sensor_geometry = sensor_geometry
        self.cache_size = cache_size
        self.use_summary_stats = use_summary_stats and HAS_SUMMARY_STATS

        self.batch_cache = None
        self.meta_cache = None

        # Calculate dataset size
        self.chunks = []
        for batch_file in batch_files:
            batch_num = int(Path(batch_file).name.split('.')[0].split('_')[1])
            meta_batch_file = os.path.join(meta_dir, f"train_meta_{batch_num}.parquet")
            if os.path.exists(meta_batch_file):
                meta_table = pq.read_table(meta_batch_file, columns=['event_id'])
                self.chunks.append(len(meta_table))
            else:
                raise FileNotFoundError(f"Metadata file not found: {meta_batch_file}")

        self.chunk_cumsum = np.cumsum(self.chunks)
        self.total_events = self.chunk_cumsum[-1]

        print(f"Kaggle dataset: {self.total_events} events across {len(self.chunks)} batch files")

        self.batch_name_to_path = {Path(f).name: f for f in batch_files}

    def __len__(self):
        return self.total_events

    def _load_batch_data(self, batch_filename: str) -> pl.DataFrame:
        """Load batch data with OrderedDict caching."""
        if self.batch_cache is None:
            self.batch_cache = OrderedDict()

        if batch_filename not in self.batch_cache:
            batch_path = self.batch_name_to_path[batch_filename]
            batch_data = pl.read_parquet(batch_path)

            batch_data = batch_data.group_by("event_id").agg([
                pl.len().alias("count"),
                pl.col("sensor_id"),
                pl.col("time"),
                pl.col("charge"),
                pl.col("auxiliary"),
            ]).sort('event_id')

            self.batch_cache[batch_filename] = batch_data

            if len(self.batch_cache) > self.cache_size:
                oldest_key = next(iter(self.batch_cache))
                del self.batch_cache[oldest_key]

        return self.batch_cache[batch_filename]

    def _load_meta_data(self, batch_filename: str) -> pd.DataFrame:
        """Load metadata with OrderedDict caching."""
        if self.meta_cache is None:
            self.meta_cache = OrderedDict()

        if batch_filename not in self.meta_cache:
            batch_num = int(batch_filename.split('.')[0].split('_')[1])
            meta_batch_file = os.path.join(self.meta_dir, f"train_meta_{batch_num}.parquet")

            if not os.path.exists(meta_batch_file):
                raise FileNotFoundError(f"Metadata file not found: {meta_batch_file}")

            meta_table = pq.read_table(meta_batch_file)
            batch_meta = meta_table.to_pandas().sort_values('event_id').reset_index(drop=True)

            self.meta_cache[batch_filename] = batch_meta

            if len(self.meta_cache) > self.cache_size:
                oldest_key = next(iter(self.meta_cache))
                del self.meta_cache[oldest_key]

        return self.meta_cache[batch_filename]

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get event at index using chunk-aware indexing.

        Returns:
            coords: [N, 4] (x, y, z, t)
            features: [N, F]
            labels: [4] (log_energy, dir_x, dir_y, dir_z)
        """
        # Find which batch file this index belongs to
        batch_file_idx = bisect_right(self.chunk_cumsum, idx)
        batch_filename = self.batch_file_names[batch_file_idx]

        local_idx = idx - (self.chunk_cumsum[batch_file_idx - 1] if batch_file_idx > 0 else 0)

        batch_data = self._load_batch_data(batch_filename)
        batch_meta = self._load_meta_data(batch_filename)

        event_meta = batch_meta.iloc[local_idx]
        azimuth = event_meta['azimuth']
        zenith = event_meta['zenith']

        event_row = batch_data[int(local_idx)]
        sensor_ids = event_row['sensor_id'][0].to_numpy()
        times = event_row['time'][0].to_numpy()
        charges = event_row['charge'][0].to_numpy()

        if self.use_summary_stats:
            unique_sensors = np.unique(sensor_ids)
            sensor_positions_list = []
            sensor_stats_list = []

            for sensor_id in unique_sensors:
                sensor_mask = sensor_ids == sensor_id
                sensor_times = times[sensor_mask]
                sensor_charges = charges[sensor_mask]

                stats = nt_summary_stats.compute_summary_stats(sensor_times, sensor_charges)
                sensor_stats_list.append(stats)
                sensor_positions_list.append(self.sensor_geometry[sensor_id])

            sensor_positions = np.array(sensor_positions_list, dtype=np.float32)
            sensor_stats = np.array(sensor_stats_list, dtype=np.float32)

            pos = np.column_stack([
                sensor_positions[:, 0] / 500.0,
                sensor_positions[:, 1] / 500.0,
                sensor_positions[:, 2] / 500.0,
                (sensor_stats[:, 3] - 1e4) / 3e4  # first_pulse_time
            ]).astype(np.float32)

            feats = np.log(sensor_stats + 1).astype(np.float32)
        else:
            sensor_positions = self.sensor_geometry[sensor_ids]
            times_norm = (times - 1e4) / 3e4
            charges_norm = np.log10(charges) / 3.0

            pos = np.column_stack([
                sensor_positions[:, 0] / 1000.0,
                sensor_positions[:, 1] / 1000.0,
                sensor_positions[:, 2] / 1000.0,
                times / 1000.0,
            ]).astype(np.float32)

            feats = np.column_stack([times_norm, charges_norm]).astype(np.float32)

        # Sort by time
        if pos.shape[0] > 0:
            order = np.argsort(pos[:, 3])
            pos = pos[order]
            feats = feats[order]

        # Build labels
        log_energy = 0.0  # Dummy (Kaggle dataset doesn't provide energy)
        dir_x = np.sin(zenith) * np.cos(azimuth)
        dir_y = np.sin(zenith) * np.sin(azimuth)
        dir_z = np.cos(zenith)

        labels = np.array([log_energy, dir_x, dir_y, dir_z], dtype=np.float32)

        return pos, feats, labels