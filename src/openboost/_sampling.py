"""Sampling strategies for large-scale training.

Phase 17: GOSS (Gradient-based One-Side Sampling) and mini-batch support.

This module provides sampling strategies to speed up training while
maintaining model quality:
- GOSS: Keep high-gradient samples, subsample low-gradient samples
- Random: Standard random subsampling (baseline)
- MiniBatch: Chunked processing for datasets larger than memory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SamplingStrategy(str, Enum):
    """Available sampling strategies."""
    RANDOM = "random"
    GOSS = "goss"
    NONE = "none"


@dataclass
class GOSSConfig:
    """Configuration for Gradient-based One-Side Sampling (GOSS).
    
    From LightGBM paper: Keep all samples with large gradients,
    subsample those with small gradients.
    
    Effective sample ratio = top_rate + other_rate * (1 - top_rate)
    
    Args:
        top_rate: Fraction of samples with largest gradient magnitudes to keep.
                  These are the "important" samples that contribute most to learning.
        other_rate: Fraction of remaining samples to randomly sample.
                    These get upweighted to maintain unbiased gradients.
        seed: Random seed for reproducibility.
        
    Example:
        >>> # Keep top 20%, sample 10% of rest = 28% total samples
        >>> config = GOSSConfig(top_rate=0.2, other_rate=0.1)
        >>> 
        >>> # More aggressive (faster but less accurate)
        >>> config = GOSSConfig(top_rate=0.1, other_rate=0.05)  # 14.5% samples
        >>> 
        >>> # Conservative (slower but more accurate)
        >>> config = GOSSConfig(top_rate=0.3, other_rate=0.2)  # 44% samples
    """
    top_rate: float = 0.2
    other_rate: float = 0.1
    seed: int | None = None
    
    def __post_init__(self):
        if not 0 < self.top_rate < 1:
            raise ValueError(f"top_rate must be in (0, 1), got {self.top_rate}")
        if not 0 < self.other_rate <= 1:
            raise ValueError(f"other_rate must be in (0, 1], got {self.other_rate}")
    
    @property
    def effective_sample_rate(self) -> float:
        """Effective fraction of samples used."""
        return self.top_rate + self.other_rate * (1 - self.top_rate)


@dataclass
class MiniBatchConfig:
    """Configuration for mini-batch training.
    
    Enables training on datasets larger than memory by processing
    samples in chunks and accumulating histograms.
    
    Args:
        batch_size: Number of samples to process at a time.
                   Larger = faster (better GPU utilization) but more memory.
        shuffle: Whether to shuffle samples before each epoch.
        seed: Random seed for shuffling.
        
    Example:
        >>> # Process 100k samples at a time
        >>> config = MiniBatchConfig(batch_size=100_000)
        >>> 
        >>> # With shuffling for better convergence
        >>> config = MiniBatchConfig(batch_size=100_000, shuffle=True, seed=42)
    """
    batch_size: int = 100_000
    shuffle: bool = False
    seed: int | None = None
    
    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


@dataclass
class SamplingResult:
    """Result of a sampling operation.
    
    Attributes:
        indices: Indices of selected samples.
        weights: Sample weights (for upweighting subsampled samples).
        n_selected: Number of samples selected.
        n_original: Original number of samples.
    """
    indices: NDArray[np.int64]
    weights: NDArray[np.float32]
    n_selected: int
    n_original: int
    
    @property
    def sample_rate(self) -> float:
        """Fraction of samples selected."""
        return self.n_selected / self.n_original


def goss_sample(
    grad: NDArray,
    hess: NDArray | None = None,
    top_rate: float = 0.2,
    other_rate: float = 0.1,
    seed: int | None = None,
) -> SamplingResult:
    """Gradient-based One-Side Sampling (GOSS).
    
    GOSS keeps all samples with large gradient magnitudes (important for learning)
    and randomly samples from the rest. Small-gradient samples are upweighted
    to maintain unbiased gradient estimates.
    
    This gives ~3x speedup with minimal accuracy loss compared to random subsampling.
    
    Algorithm:
        1. Sort samples by |gradient|
        2. Keep top `top_rate` samples by gradient magnitude
        3. Randomly sample `other_rate` from the rest
        4. Upweight small-gradient samples by (1 - top_rate) / other_rate
    
    Args:
        grad: Gradient array, shape (n_samples,) or (n_samples, n_params).
              For multi-parameter distributions, uses sum of absolute gradients.
        hess: Hessian array (unused, for API compatibility).
        top_rate: Fraction of high-gradient samples to keep (default 0.2).
        other_rate: Fraction of low-gradient samples to sample (default 0.1).
        seed: Random seed for reproducibility.
        
    Returns:
        SamplingResult with selected indices and weights.
        
    Example:
        >>> grad = compute_gradients(pred, y)
        >>> result = goss_sample(grad, top_rate=0.2, other_rate=0.1)
        >>> 
        >>> # Use selected samples for histogram building
        >>> hist = build_histogram(X[result.indices], 
        ...                        grad[result.indices] * result.weights,
        ...                        hess[result.indices] * result.weights)
        
    References:
        - LightGBM paper: https://papers.nips.cc/paper/6907-lightgbm
    """
    n_samples = len(grad)
    
    # Handle multi-dimensional gradients (e.g., distributional GBDT)
    if grad.ndim > 1:
        abs_grad = np.sum(np.abs(grad), axis=1)
    else:
        abs_grad = np.abs(grad)
    
    # Number of samples to keep from each group
    n_top = int(n_samples * top_rate)
    n_other = int((n_samples - n_top) * other_rate)
    
    # Handle edge cases
    if n_top == 0:
        n_top = 1
    if n_other == 0:
        n_other = 1
    
    # Find top samples by gradient magnitude
    # Using argpartition is O(n) vs O(n log n) for full sort
    top_indices_unsorted = np.argpartition(abs_grad, -n_top)[-n_top:]
    
    # Get indices of "other" samples (not in top)
    all_indices = np.arange(n_samples)
    top_set = set(top_indices_unsorted)
    other_indices = np.array([i for i in all_indices if i not in top_set], dtype=np.int64)
    
    # Random sample from others
    rng = np.random.default_rng(seed)
    other_sample_indices = rng.choice(other_indices, size=min(n_other, len(other_indices)), replace=False)
    
    # Combine indices
    selected_indices = np.concatenate([top_indices_unsorted, other_sample_indices])
    
    # Compute weights
    # - Top samples: weight = 1.0
    # - Other samples: upweight to compensate for subsampling
    n_top_actual = len(top_indices_unsorted)
    n_other_actual = len(other_sample_indices)
    n_selected = n_top_actual + n_other_actual
    
    weights = np.ones(n_selected, dtype=np.float32)
    if n_other_actual > 0 and other_rate < 1.0:
        # Upweight factor: (1 - top_rate) / other_rate
        # This ensures gradient sum is unbiased
        upweight = (1 - top_rate) / other_rate
        weights[n_top_actual:] = upweight
    
    return SamplingResult(
        indices=selected_indices,
        weights=weights,
        n_selected=n_selected,
        n_original=n_samples,
    )


def random_sample(
    n_samples: int,
    sample_rate: float = 1.0,
    seed: int | None = None,
) -> SamplingResult:
    """Random subsampling (baseline).
    
    Simple random sampling without regard to gradient magnitudes.
    All selected samples have equal weight.
    
    Args:
        n_samples: Total number of samples.
        sample_rate: Fraction of samples to select (0 < rate <= 1).
        seed: Random seed for reproducibility.
        
    Returns:
        SamplingResult with selected indices and unit weights.
        
    Example:
        >>> result = random_sample(n_samples=1_000_000, sample_rate=0.3)
        >>> X_subset = X[result.indices]
    """
    if sample_rate >= 1.0:
        # No sampling needed
        return SamplingResult(
            indices=np.arange(n_samples, dtype=np.int64),
            weights=np.ones(n_samples, dtype=np.float32),
            n_selected=n_samples,
            n_original=n_samples,
        )
    
    n_selected = max(1, int(n_samples * sample_rate))
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_samples, size=n_selected, replace=False)
    weights = np.ones(n_selected, dtype=np.float32)
    
    return SamplingResult(
        indices=indices,
        weights=weights,
        n_selected=n_selected,
        n_original=n_samples,
    )


class MiniBatchIterator:
    """Iterator for mini-batch training.
    
    Yields chunks of sample indices for processing datasets larger than memory.
    
    Args:
        n_samples: Total number of samples.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle indices before iteration.
        seed: Random seed for shuffling.
        
    Example:
        >>> iterator = MiniBatchIterator(n_samples=10_000_000, batch_size=100_000)
        >>> for batch_indices in iterator:
        ...     # Load batch data
        ...     X_batch = load_batch(X_mmap, batch_indices)
        ...     grad_batch = grad[batch_indices]
        ...     hess_batch = hess[batch_indices]
        ...     
        ...     # Build and accumulate histogram
        ...     batch_hist = build_histogram(X_batch, grad_batch, hess_batch)
        ...     total_hist += batch_hist
    """
    
    def __init__(
        self,
        n_samples: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        
        self._indices: NDArray | None = None
        self._rng = np.random.default_rng(seed)
    
    @property
    def n_batches(self) -> int:
        """Number of batches per epoch."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate over batch indices."""
        indices = np.arange(self.n_samples, dtype=np.int64)
        
        if self.shuffle:
            self._rng.shuffle(indices)
        
        self._indices = indices
        self._batch_idx = 0
        return self
    
    def __next__(self) -> NDArray[np.int64]:
        """Get next batch of indices."""
        if self._indices is None:
            raise StopIteration
        
        start = self._batch_idx * self.batch_size
        if start >= self.n_samples:
            self._indices = None
            raise StopIteration
        
        end = min(start + self.batch_size, self.n_samples)
        batch_indices = self._indices[start:end]
        self._batch_idx += 1
        
        return batch_indices
    
    def __len__(self) -> int:
        """Number of batches."""
        return self.n_batches


def apply_sampling(
    grad: NDArray,
    hess: NDArray,
    strategy: str | SamplingStrategy = "none",
    *,
    top_rate: float = 0.2,
    other_rate: float = 0.1,
    sample_rate: float = 1.0,
    seed: int | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Apply sampling strategy to gradients and hessians.
    
    Convenience function that applies the specified sampling strategy
    and returns the sampled/weighted gradients and hessians.
    
    Args:
        grad: Gradient array, shape (n_samples,).
        hess: Hessian array, shape (n_samples,).
        strategy: Sampling strategy ("none", "random", "goss").
        top_rate: GOSS top_rate parameter.
        other_rate: GOSS other_rate parameter.
        sample_rate: Random sampling rate.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (indices, weights, sampled_grad, sampled_hess):
        - indices: Selected sample indices
        - weights: Sample weights
        - sampled_grad: Weighted gradients for selected samples
        - sampled_hess: Weighted hessians for selected samples
        
    Example:
        >>> indices, weights, grad_s, hess_s = apply_sampling(
        ...     grad, hess, strategy="goss", top_rate=0.2, other_rate=0.1
        ... )
        >>> # Build histogram with sampled data
        >>> hist = build_histogram(X[:, indices], grad_s, hess_s)
    """
    strategy = SamplingStrategy(strategy)
    
    if strategy == SamplingStrategy.NONE:
        indices = np.arange(len(grad), dtype=np.int64)
        weights = np.ones(len(grad), dtype=np.float32)
        return indices, weights, grad, hess
    
    elif strategy == SamplingStrategy.GOSS:
        result = goss_sample(grad, hess, top_rate, other_rate, seed)
    
    elif strategy == SamplingStrategy.RANDOM:
        result = random_sample(len(grad), sample_rate, seed)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Apply sampling and weighting
    sampled_grad = grad[result.indices] * result.weights
    sampled_hess = hess[result.indices] * result.weights
    
    return result.indices, result.weights, sampled_grad, sampled_hess


# =============================================================================
# Mini-Batch Histogram Accumulation
# =============================================================================

def accumulate_histograms_minibatch(
    X_binned,  # BinnedArray or memory-mapped
    grad: NDArray,
    hess: NDArray,
    batch_size: int,
    n_features: int,
    sample_indices: NDArray | None = None,
    build_fn: Callable | None = None,
) -> tuple[NDArray, NDArray]:
    """Build histograms by accumulating over mini-batches.
    
    This enables training on datasets larger than GPU/CPU memory by
    processing samples in chunks and summing the histograms.
    
    Args:
        X_binned: Binned feature data (can be memory-mapped).
        grad: Full gradient array.
        hess: Full hessian array.
        batch_size: Number of samples per batch.
        n_features: Number of features.
        sample_indices: Optional indices to use (for node-specific histograms).
        build_fn: Function to build histogram for a batch.
                 Signature: (X_batch, grad_batch, hess_batch) -> (hist_grad, hist_hess)
        
    Returns:
        Accumulated (hist_grad, hist_hess) arrays, shape (n_features, 256).
        
    Example:
        >>> # Training on 100M samples with 8GB GPU
        >>> from openboost._core._histogram import build_histogram
        >>> 
        >>> hist_grad, hist_hess = accumulate_histograms_minibatch(
        ...     X_mmap, grad, hess,
        ...     batch_size=100_000,
        ...     n_features=X.shape[1],
        ...     build_fn=build_histogram,
        ... )
    """
    from ._core._histogram import build_histogram as default_build_fn
    
    if build_fn is None:
        build_fn = default_build_fn
    
    # Initialize accumulated histograms
    hist_grad = np.zeros((n_features, 256), dtype=np.float32)
    hist_hess = np.zeros((n_features, 256), dtype=np.float32)
    
    # Determine indices to process
    if sample_indices is not None:
        indices_to_process = sample_indices
    else:
        indices_to_process = np.arange(len(grad), dtype=np.int64)
    
    n_samples = len(indices_to_process)
    
    # Process in batches
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices_to_process[start:end]
        
        # Extract batch data
        # Handle memory-mapped arrays by loading only the batch
        if hasattr(X_binned, 'data'):
            # BinnedArray
            X_batch = X_binned.data[:, batch_indices]
        else:
            # Raw array (possibly memory-mapped)
            X_batch = X_binned[:, batch_indices]
        
        grad_batch = grad[batch_indices]
        hess_batch = hess[batch_indices]
        
        # Build batch histogram and accumulate
        batch_hist_grad, batch_hist_hess = build_fn(X_batch, grad_batch, hess_batch)
        
        hist_grad += batch_hist_grad
        hist_hess += batch_hist_hess
    
    return hist_grad, hist_hess


# =============================================================================
# Memory-Mapped Array Support
# =============================================================================

def create_memmap_binned(
    path: str,
    X: NDArray,
    n_bins: int = 256,
) -> np.memmap:
    """Create memory-mapped binned array for large datasets.
    
    Bins the data and saves to disk as a memory-mapped file,
    enabling training on datasets larger than RAM.
    
    Args:
        path: Path to save the memory-mapped file.
        X: Input features, shape (n_samples, n_features).
        n_bins: Number of bins for quantile binning.
        
    Returns:
        Memory-mapped binned array, shape (n_features, n_samples).
        
    Example:
        >>> # Create once
        >>> X_mmap = create_memmap_binned('data.npy', X_train)
        >>> 
        >>> # Load for training (no copy, uses disk)
        >>> X_mmap = np.memmap('data.npy', mode='r', dtype=np.uint8,
        ...                     shape=(n_features, n_samples))
    """
    from ._array import array as ob_array
    
    # Bin the data
    binned = ob_array(X, n_bins=n_bins, device='cpu')
    
    # Get the binned data
    if hasattr(binned.data, 'copy_to_host'):
        data = binned.data.copy_to_host()
    else:
        data = binned.data
    
    # Create memory-mapped file
    mmap = np.memmap(path, dtype=np.uint8, mode='w+', shape=data.shape)
    mmap[:] = data
    mmap.flush()
    
    return mmap


def load_memmap_binned(
    path: str,
    n_features: int,
    n_samples: int,
) -> np.memmap:
    """Load memory-mapped binned array.
    
    Args:
        path: Path to the memory-mapped file.
        n_features: Number of features.
        n_samples: Number of samples.
        
    Returns:
        Memory-mapped binned array, shape (n_features, n_samples).
    """
    return np.memmap(path, dtype=np.uint8, mode='r', shape=(n_features, n_samples))


__all__ = [
    # Enums and configs
    "SamplingStrategy",
    "GOSSConfig",
    "MiniBatchConfig",
    "SamplingResult",
    # Sampling functions
    "goss_sample",
    "random_sample",
    "apply_sampling",
    # Mini-batch support
    "MiniBatchIterator",
    "accumulate_histograms_minibatch",
    # Memory-mapped support
    "create_memmap_binned",
    "load_memmap_binned",
]
