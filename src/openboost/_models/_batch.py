"""Batch training structures for horizontal kernel fusion (Phase 2).

This module provides data structures and utilities for training multiple
gradient boosting models with different hyperparameters in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ConfigBatch:
    """Batch of hyperparameter configurations for parallel training.
    
    All arrays must have the same length (n_configs).
    
    Attributes:
        max_depths: Maximum tree depth for each config
        reg_lambdas: L2 regularization for each config
        min_child_weights: Minimum hessian sum in leaves for each config
        learning_rates: Learning rate (eta) for each config
        n_rounds: Number of boosting rounds (same for all configs)
        
    Example:
        >>> configs = ConfigBatch.from_grid(
        ...     max_depth=[4, 6, 8],
        ...     reg_lambda=[0.1, 1.0, 10.0],
        ...     learning_rate=[0.05, 0.1, 0.3],
        ... )
        >>> print(f"Training {configs.n_configs} configurations")
    """
    max_depths: NDArray  # (n_configs,) int32
    reg_lambdas: NDArray  # (n_configs,) float32
    min_child_weights: NDArray  # (n_configs,) float32
    learning_rates: NDArray  # (n_configs,) float32
    n_rounds: int = 100
    
    # Internal state for batch training
    _device_arrays: dict = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Validate and convert arrays to correct dtypes."""
        self.max_depths = np.asarray(self.max_depths, dtype=np.int32)
        self.reg_lambdas = np.asarray(self.reg_lambdas, dtype=np.float32)
        self.min_child_weights = np.asarray(self.min_child_weights, dtype=np.float32)
        self.learning_rates = np.asarray(self.learning_rates, dtype=np.float32)
        
        # Validate shapes
        n = len(self.max_depths)
        if not all(len(arr) == n for arr in [
            self.reg_lambdas, self.min_child_weights, self.learning_rates
        ]):
            raise ValueError("All config arrays must have the same length")
    
    @property
    def n_configs(self) -> int:
        """Number of configurations in this batch."""
        return len(self.max_depths)
    
    @classmethod
    def from_grid(
        cls,
        max_depth: Sequence[int] = (6,),
        reg_lambda: Sequence[float] = (1.0,),
        min_child_weight: Sequence[float] = (1.0,),
        learning_rate: Sequence[float] = (0.1,),
        n_rounds: int = 100,
    ) -> ConfigBatch:
        """Create a ConfigBatch from a grid of hyperparameters.
        
        Generates all combinations of the provided hyperparameters.
        
        Args:
            max_depth: List of max_depth values
            reg_lambda: List of reg_lambda values
            min_child_weight: List of min_child_weight values
            learning_rate: List of learning_rate values
            n_rounds: Number of boosting rounds
            
        Returns:
            ConfigBatch with all combinations
            
        Example:
            >>> configs = ConfigBatch.from_grid(
            ...     max_depth=[4, 6],
            ...     reg_lambda=[0.1, 1.0],
            ... )
            >>> configs.n_configs  # 2 × 2 = 4 combinations
            4
        """
        import itertools
        
        # Generate all combinations
        combinations = list(itertools.product(
            max_depth, reg_lambda, min_child_weight, learning_rate
        ))
        
        if not combinations:
            raise ValueError("At least one value required for each hyperparameter")
        
        max_depths = np.array([c[0] for c in combinations], dtype=np.int32)
        reg_lambdas = np.array([c[1] for c in combinations], dtype=np.float32)
        min_child_weights = np.array([c[2] for c in combinations], dtype=np.float32)
        learning_rates = np.array([c[3] for c in combinations], dtype=np.float32)
        
        return cls(
            max_depths=max_depths,
            reg_lambdas=reg_lambdas,
            min_child_weights=min_child_weights,
            learning_rates=learning_rates,
            n_rounds=n_rounds,
        )
    
    @classmethod
    def from_lists(
        cls,
        max_depths: Sequence[int],
        reg_lambdas: Sequence[float],
        min_child_weights: Sequence[float],
        learning_rates: Sequence[float],
        n_rounds: int = 100,
    ) -> ConfigBatch:
        """Create a ConfigBatch from explicit lists of hyperparameters.
        
        Each list must have the same length. Config i uses the i-th element
        from each list.
        
        Args:
            max_depths: List of max_depth values
            reg_lambdas: List of reg_lambda values  
            min_child_weights: List of min_child_weight values
            learning_rates: List of learning_rate values
            n_rounds: Number of boosting rounds
            
        Returns:
            ConfigBatch with specified configurations
        """
        return cls(
            max_depths=np.array(max_depths, dtype=np.int32),
            reg_lambdas=np.array(reg_lambdas, dtype=np.float32),
            min_child_weights=np.array(min_child_weights, dtype=np.float32),
            learning_rates=np.array(learning_rates, dtype=np.float32),
            n_rounds=n_rounds,
        )
    
    def to_device(self):
        """Transfer config arrays to GPU.
        
        Call this once before batch training to avoid repeated transfers.
        """
        from .._backends import is_cuda
        
        if not is_cuda():
            return
        
        if self._device_arrays:
            return  # Already on device
        
        from numba import cuda
        
        self._device_arrays = {
            'max_depths': cuda.to_device(self.max_depths),
            'reg_lambdas': cuda.to_device(self.reg_lambdas),
            'min_child_weights': cuda.to_device(self.min_child_weights),
            'learning_rates': cuda.to_device(self.learning_rates),
        }
    
    def get_device_arrays(self) -> dict:
        """Get GPU arrays for this config batch.
        
        Automatically transfers to device if not already done.
        
        Returns:
            Dict with 'max_depths', 'reg_lambdas', 'min_child_weights', 'learning_rates'
        """
        if not self._device_arrays:
            self.to_device()
        return self._device_arrays
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single configuration as a dict.
        
        Args:
            idx: Configuration index
            
        Returns:
            Dict with hyperparameters for config idx
        """
        return {
            'max_depth': int(self.max_depths[idx]),
            'reg_lambda': float(self.reg_lambdas[idx]),
            'min_child_weight': float(self.min_child_weights[idx]),
            'learning_rate': float(self.learning_rates[idx]),
            'n_rounds': self.n_rounds,
        }
    
    def __iter__(self):
        """Iterate over configurations."""
        for i in range(self.n_configs):
            yield self[i]
    
    def __repr__(self) -> str:
        return (
            f"ConfigBatch(n_configs={self.n_configs}, n_rounds={self.n_rounds}, "
            f"max_depths={self.max_depths.tolist()}, "
            f"reg_lambdas={self.reg_lambdas.tolist()}, ...)"
        )


@dataclass
class BatchTrainingState:
    """State for batch training of multiple models.
    
    Tracks per-config predictions and trees during training.
    
    Attributes:
        n_configs: Number of configurations
        n_samples: Number of training samples
        predictions: Current predictions, shape (n_configs, n_samples)
        trees: List of tree lists, one per config
    """
    n_configs: int
    n_samples: int
    predictions: NDArray  # (n_configs, n_samples) float32
    trees: list = field(default_factory=list)  # List[List[Tree]]
    
    @classmethod
    def create(cls, n_configs: int, n_samples: int) -> BatchTrainingState:
        """Create initial training state.
        
        Args:
            n_configs: Number of configurations
            n_samples: Number of training samples
            
        Returns:
            Initialized BatchTrainingState with zero predictions
        """
        predictions = np.zeros((n_configs, n_samples), dtype=np.float32)
        trees = [[] for _ in range(n_configs)]
        return cls(
            n_configs=n_configs,
            n_samples=n_samples,
            predictions=predictions,
            trees=trees,
        )
    
    def to_device(self):
        """Transfer predictions to GPU."""
        from .._backends import is_cuda
        
        if is_cuda() and not hasattr(self.predictions, '__cuda_array_interface__'):
            from numba import cuda
            self.predictions = cuda.to_device(self.predictions)
    
    def get_predictions(self, config_idx: int) -> NDArray:
        """Get predictions for a specific config.
        
        Args:
            config_idx: Configuration index
            
        Returns:
            Predictions array, shape (n_samples,)
        """
        if hasattr(self.predictions, 'copy_to_host'):
            return self.predictions[config_idx].copy_to_host()
        return self.predictions[config_idx]

