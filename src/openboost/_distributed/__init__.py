"""Distributed training for OpenBoost.

Phase 12: Adds Ray-based distributed training capability.
"""

from typing import Protocol, Any
from numpy.typing import NDArray
import numpy as np


class DistributedContext(Protocol):
    """Protocol for distributed training context."""
    n_workers: int
    rank: int

    def allreduce_histograms(self, local_hist: NDArray) -> NDArray:
        """Sum histograms across all workers."""
        ...
    
    def broadcast_tree(self, tree: Any) -> Any:
        """Broadcast tree from rank 0 to all workers."""
        ...
    
    def partition_data(self, X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        """Get this worker's data shard."""
        ...


__all__ = ["DistributedContext"]
