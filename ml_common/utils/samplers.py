"""Custom samplers for efficient data loading."""

import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from typing import Optional, Iterator, Sized


class RandomChunkSampler(Sampler[int]):
    """
    Random sampler that processes chunks completely before moving to next chunk.

    Ensures chunks (e.g., batch files) are processed in random order, with events
    within each chunk shuffled. Maximizes cache efficiency when loading chunked data.
    """

    def __init__(
        self,
        data_source: Sized,
        chunks: list,
        num_samples: Optional[int] = None,
        generator=None
    ) -> None:
        """
        Initialize RandomChunkSampler.

        Args:
            data_source: Dataset
            chunks: List of chunk sizes (number of events per chunk)
            num_samples: Number of samples (defaults to len(data_source))
            generator: Random number generator
        """
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator
        self.chunks = chunks

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive int, got {self.num_samples}")

    @property
    def num_samples(self) -> int:
        """Return number of samples to draw."""
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        """
        Generate sample indices in chunk-aware random order.

        Strategy:
        1. Randomize chunk order
        2. For each chunk, shuffle events within that chunk
        3. Yield all events from chunk before moving to next
        """
        cumsum = np.cumsum(self.chunks)

        # Setup generator
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # Randomize chunk order
        chunk_list = torch.randperm(len(self.chunks), generator=generator).tolist()

        # Process chunks in random order, events within chunks shuffled
        for chunk_idx in chunk_list:
            chunk_len = self.chunks[chunk_idx]
            offset = cumsum[chunk_idx - 1] if chunk_idx > 0 else 0

            # Generate shuffled indices for this chunk
            chunk_indices = offset + torch.randperm(chunk_len, generator=generator)

            # Yield all indices from this chunk
            yield from chunk_indices.tolist()

    def __len__(self) -> int:
        return self.num_samples


class LargeWeightedRandomSampler(Sampler[int]):
    """Weighted sampler with replacement that bypasses torch.multinomial's 2^24 limit.

    Uses numpy inverse-CDF sampling, so it works for datasets of any size.
    """

    def __init__(self, weights, num_samples: int, generator=None) -> None:
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError(f"weights must be 1D, got shape {weights.shape}")
        if (weights < 0).any():
            raise ValueError("weights must be non-negative")
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("weights must have finite, positive sum")

        self._cum_weights = np.cumsum(weights)
        self._total = float(self._cum_weights[-1])
        self._num_samples = int(num_samples)
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = int(torch.empty((), dtype=torch.int64).random_(generator=self.generator).item())
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.0, self._total, size=self._num_samples)
        indices = np.searchsorted(self._cum_weights, u, side='right')
        np.clip(indices, 0, self._cum_weights.shape[0] - 1, out=indices)
        yield from indices.tolist()

    def __len__(self) -> int:
        return self._num_samples