"""Tests for distributed training (Phase 12)."""

import pytest
import numpy as np
import openboost as ob
from openboost._distributed import DistributedContext


def test_distributed_context_protocol():
    """Verify our mock satisfies protocol."""
    class MockContext:
        n_workers = 1
        rank = 0
        def allreduce_histograms(self, local_hist): pass
        def broadcast_tree(self, tree): pass
        def partition_data(self, X, y): pass
    
    # Protocol check
    assert hasattr(MockContext, 'n_workers')
    assert hasattr(MockContext, 'rank')


def test_gradient_boosting_distributed_init():
    """Test GradientBoosting can be initialized with distributed params."""
    model = ob.GradientBoosting(distributed=True, n_workers=2)
    assert model.distributed
    assert model.n_workers == 2


def test_gradient_boosting_distributed_requires_ray():
    """Test that distributed training raises ImportError without Ray."""
    try:
        import ray
        pytest.skip("Ray is installed, skipping import error test")
    except ImportError:
        pass
    
    model = ob.GradientBoosting(distributed=True)
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    
    with pytest.raises(ImportError):
        model.fit(X, y)
