"""I/O utilities for loading memory-mapped datasets (ntmmap format)."""

import os
import pickle
import struct
import numpy as np
from typing import Tuple


def load_ntmmap(input_path: str) -> Tuple[np.memmap, np.memmap, np.dtype]:
    """Load ntmmap files (.idx and .dat) with automatic dtype detection."""
    idx_path = f"{input_path}.idx"
    dat_path = f"{input_path}.dat"

    if not os.path.exists(idx_path) or not os.path.exists(dat_path):
        raise FileNotFoundError(
            f"Memory-mapped files not found at: {input_path}\n"
            f"Expected: {idx_path} and {dat_path}"
        )

    # Load index file with header
    with open(idx_path, 'rb') as f:
        dtype_size = struct.unpack('<I', f.read(4))[0]
        event_dtype = pickle.loads(f.read(dtype_size))
        data_start = f.tell()

    index_mmap = np.memmap(idx_path, dtype=event_dtype, mode='r', offset=data_start)

    # Load data file with header
    with open(dat_path, 'rb') as f:
        dtype_size = struct.unpack('<I', f.read(4))[0]
        photon_dtype = pickle.loads(f.read(dtype_size))
        data_start = f.tell()

    # Memory map photons as structured array (not raw bytes)
    photons_array = np.memmap(dat_path, dtype=photon_dtype, mode='r', offset=data_start)

    return index_mmap, photons_array, photon_dtype


def batched_coordinates(coords: np.ndarray, batch_size: int) -> np.ndarray:
    """Add batch indices as first column to coordinates [N,D] -> [N,D+1]."""
    batch_indices = np.arange(len(coords)) // batch_size
    return np.column_stack([batch_indices, coords])