"""Tests for tree building primitives (Phase 8.1)."""

import numpy as np
import pytest

import openboost as ob


class TestNodeHistograms:
    """Tests for build_node_histograms()."""
    
    def test_single_node_histogram(self):
        """Test building histogram for a single node (root)."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1]
        
        binned = ob.array(X)
        
        # All samples start at root (node 0)
        sample_node_ids = ob.init_sample_node_ids(100)
        
        # MSE gradients
        pred = np.zeros(100, dtype=np.float32)
        grad = (2 * (pred - y)).astype(np.float32)
        hess = np.ones(100, dtype=np.float32) * 2
        
        # Build histogram for root
        histograms = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0]
        )
        
        assert 0 in histograms
        assert histograms[0].node_id == 0
        assert histograms[0].n_samples == 100
        assert histograms[0].hist_grad.shape == (5, 256)
        assert histograms[0].hist_hess.shape == (5, 256)
        
        # Sum of histogram should equal sum of gradients
        np.testing.assert_almost_equal(
            histograms[0].sum_grad, np.sum(grad), decimal=4
        )
    
    def test_multiple_node_histograms(self):
        """Test building histograms for multiple nodes."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        binned = ob.array(X)
        
        # Manually assign samples to different nodes
        sample_node_ids = np.zeros(100, dtype=np.int32)
        sample_node_ids[50:75] = 1
        sample_node_ids[75:] = 2
        
        grad = np.random.randn(100).astype(np.float32)
        hess = np.ones(100, dtype=np.float32)
        
        histograms = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0, 1, 2]
        )
        
        assert len(histograms) == 3
        assert histograms[0].n_samples == 50
        assert histograms[1].n_samples == 25
        assert histograms[2].n_samples == 25
    
    def test_empty_node_histogram(self):
        """Test histogram for node with no samples."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        binned = ob.array(X)
        
        # All samples in node 0
        sample_node_ids = np.zeros(100, dtype=np.int32)
        
        grad = np.random.randn(100).astype(np.float32)
        hess = np.ones(100, dtype=np.float32)
        
        # Request histogram for node 1 (empty)
        histograms = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0, 1]
        )
        
        assert histograms[1].n_samples == 0
        assert histograms[1].sum_grad == 0.0
        assert histograms[1].sum_hess == 0.0


class TestHistogramSubtraction:
    """Tests for subtract_histogram()."""
    
    def test_subtraction_correctness(self):
        """Test that subtraction produces correct sibling histogram."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        binned = ob.array(X)
        
        # All samples at root
        sample_node_ids = np.zeros(100, dtype=np.int32)
        
        grad = np.random.randn(100).astype(np.float32)
        hess = np.ones(100, dtype=np.float32)
        
        # Build parent histogram
        parent_hist = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0]
        )[0]
        
        # Split samples manually
        sample_node_ids[50:] = 1  # Right child
        
        # Build left child histogram (samples 0-49)
        sample_node_ids_left = np.zeros(100, dtype=np.int32)
        sample_node_ids_left[50:] = 999  # Mark right as different node
        left_hist = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids_left, [0]
        )[0]
        
        # Compute right via subtraction
        right_hist = ob.subtract_histogram(parent_hist, left_hist, sibling_node_id=1)
        
        # Build right directly for comparison
        sample_node_ids_right = np.ones(100, dtype=np.int32) * 999
        sample_node_ids_right[50:] = 1
        right_hist_direct = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids_right, [1]
        )[1]
        
        # Should match
        np.testing.assert_array_almost_equal(
            right_hist.hist_grad, right_hist_direct.hist_grad, decimal=4
        )
        np.testing.assert_almost_equal(
            right_hist.sum_grad, right_hist_direct.sum_grad, decimal=4
        )


class TestFindNodeSplits:
    """Tests for find_node_splits()."""
    
    def test_finds_valid_split(self):
        """Test that split finding works."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        y = X[:, 0]  # Target depends on feature 0
        
        binned = ob.array(X)
        sample_node_ids = np.zeros(100, dtype=np.int32)
        
        # MSE gradients
        grad = (2 * (np.zeros(100) - y)).astype(np.float32)
        hess = np.ones(100, dtype=np.float32) * 2
        
        histograms = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0]
        )
        
        splits = ob.find_node_splits(histograms, reg_lambda=1.0)
        
        assert 0 in splits
        assert splits[0].split.is_valid
        assert splits[0].split.feature == 0  # Should split on feature 0
        assert splits[0].left_child == 1
        assert splits[0].right_child == 2
    
    def test_respects_min_child_weight(self):
        """Test that min_child_weight is respected."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float32)  # Small dataset
        binned = ob.array(X)
        
        sample_node_ids = np.zeros(10, dtype=np.int32)
        grad = np.random.randn(10).astype(np.float32)
        hess = np.ones(10, dtype=np.float32) * 0.1  # Very small hessians
        
        histograms = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0]
        )
        
        # With high min_child_weight, no split should be valid
        splits = ob.find_node_splits(histograms, min_child_weight=10.0)
        
        # Either no split or invalid split
        assert len(splits) == 0 or not splits[0].split.is_valid


class TestPartitionSamples:
    """Tests for partition_samples()."""
    
    def test_partition_correctness(self):
        """Test that samples are partitioned correctly."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        binned = ob.array(X)
        
        sample_node_ids = np.zeros(100, dtype=np.int32)
        grad = np.random.randn(100).astype(np.float32)
        hess = np.ones(100, dtype=np.float32)
        
        # Build histogram and find split
        histograms = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0]
        )
        splits = ob.find_node_splits(histograms)
        
        if not splits:
            pytest.skip("No valid split found")
        
        # Partition samples
        new_node_ids = ob.partition_samples(binned.data, sample_node_ids, splits)
        
        # All samples should now be in node 1 or 2
        unique_nodes = np.unique(new_node_ids)
        assert set(unique_nodes) == {1, 2}
        
        # Total samples should be preserved
        assert len(new_node_ids) == 100
    
    def test_partition_respects_split(self):
        """Test that partition follows split rule correctly."""
        # Create simple data where split is predictable
        X = np.zeros((100, 1), dtype=np.float32)
        X[:50, 0] = -1.0  # Will go left
        X[50:, 0] = 1.0   # Will go right
        
        binned = ob.array(X)
        sample_node_ids = np.zeros(100, dtype=np.int32)
        
        # Create gradient that will split at 0
        grad = np.zeros(100, dtype=np.float32)
        grad[:50] = -1.0
        grad[50:] = 1.0
        hess = np.ones(100, dtype=np.float32)
        
        histograms = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0]
        )
        splits = ob.find_node_splits(histograms)
        
        if not splits:
            pytest.skip("No valid split found")
        
        new_node_ids = ob.partition_samples(binned.data, sample_node_ids, splits)
        
        # First 50 samples should be in one node, last 50 in another
        assert len(np.unique(new_node_ids[:50])) == 1
        assert len(np.unique(new_node_ids[50:])) == 1
        assert new_node_ids[0] != new_node_ids[50]


class TestComputeLeafValues:
    """Tests for compute_leaf_values()."""
    
    def test_leaf_value_formula(self):
        """Test that leaf values follow Newton-Raphson formula."""
        sample_node_ids = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        grad = np.array([1.0, 2.0, 3.0, -1.0, -2.0], dtype=np.float32)
        hess = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        reg_lambda = 1.0
        
        values = ob.compute_leaf_values(
            grad, hess, sample_node_ids, [0, 1], reg_lambda
        )
        
        # Node 0: sum_grad=6, sum_hess=3 -> value = -6/(3+1) = -1.5
        np.testing.assert_almost_equal(values[0], -1.5, decimal=5)
        
        # Node 1: sum_grad=-3, sum_hess=2 -> value = 3/(2+1) = 1.0
        np.testing.assert_almost_equal(values[1], 1.0, decimal=5)
    
    def test_empty_leaf(self):
        """Test leaf value for empty node."""
        sample_node_ids = np.array([0, 0, 0], dtype=np.int32)
        grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        hess = np.ones(3, dtype=np.float32)
        
        values = ob.compute_leaf_values(
            grad, hess, sample_node_ids, [0, 1], reg_lambda=1.0
        )
        
        # Node 1 is empty
        assert values[1] == 0.0


class TestUtilities:
    """Tests for utility functions."""
    
    def test_get_nodes_at_depth(self):
        """Test node ID calculation for depth."""
        # Depth 0: just root (node 0)
        # Note: using 1-indexed internal nodes, so depth 0 = [-1, 0)? 
        # Actually the formula is: start = 2^d - 1, end = 2^(d+1) - 1
        # Depth 0: [0, 1) = [0]
        # Depth 1: [1, 3) = [1, 2]
        # Depth 2: [3, 7) = [3, 4, 5, 6]
        
        assert ob.get_nodes_at_depth(0) == [0]
        assert ob.get_nodes_at_depth(1) == [1, 2]
        assert ob.get_nodes_at_depth(2) == [3, 4, 5, 6]
    
    def test_get_children(self):
        """Test child node calculation."""
        # Root (0) -> children 1, 2
        assert ob.get_children(0) == (1, 2)
        # Node 1 -> children 3, 4
        assert ob.get_children(1) == (3, 4)
        # Node 2 -> children 5, 6
        assert ob.get_children(2) == (5, 6)
    
    def test_get_parent(self):
        """Test parent node calculation."""
        assert ob.get_parent(1) == 0
        assert ob.get_parent(2) == 0
        assert ob.get_parent(3) == 1
        assert ob.get_parent(4) == 1
        
        with pytest.raises(ValueError):
            ob.get_parent(0)  # Root has no parent


class TestInitSampleNodeIds:
    """Tests for init_sample_node_ids()."""
    
    def test_init_zeros(self):
        """Test that init creates zeros."""
        node_ids = ob.init_sample_node_ids(100, device="cpu")
        
        assert node_ids.shape == (100,)
        assert node_ids.dtype == np.int32
        np.testing.assert_array_equal(node_ids, np.zeros(100, dtype=np.int32))


class TestEndToEndPrimitives:
    """End-to-end test using primitives to build a tree."""
    
    def test_build_depth_2_tree(self):
        """Test building a simple tree using only primitives."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1]  # Linear combination
        
        binned = ob.array(X)
        n_samples = 200
        
        # Initialize
        sample_node_ids = ob.init_sample_node_ids(n_samples, device="cpu")
        grad = (2 * (np.zeros(n_samples) - y)).astype(np.float32)
        hess = np.ones(n_samples, dtype=np.float32) * 2
        
        reg_lambda = 1.0
        
        # === Depth 0 ===
        # Build histogram for root
        hist_d0 = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0]
        )
        
        # Find split for root
        splits_d0 = ob.find_node_splits(hist_d0, reg_lambda=reg_lambda)
        assert 0 in splits_d0, "Root should have valid split"
        
        # Partition samples
        sample_node_ids = ob.partition_samples(binned.data, sample_node_ids, splits_d0)
        
        # === Depth 1 ===
        # Build histograms for nodes 1, 2
        hist_d1 = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [1, 2]
        )
        
        # Find splits
        splits_d1 = ob.find_node_splits(hist_d1, reg_lambda=reg_lambda)
        
        # Partition samples
        if splits_d1:
            sample_node_ids = ob.partition_samples(binned.data, sample_node_ids, splits_d1)
        
        # === Compute leaf values ===
        # Find all leaf nodes (nodes that weren't split)
        all_possible_leaves = [1, 2, 3, 4, 5, 6]  # Up to depth 2
        
        leaf_values = ob.compute_leaf_values(
            grad, hess, sample_node_ids, all_possible_leaves, reg_lambda
        )
        
        # Should have some non-zero leaf values
        non_zero_leaves = [nid for nid, val in leaf_values.items() if val != 0.0]
        assert len(non_zero_leaves) > 0, "Should have at least one non-empty leaf"
        
        # Verify all samples are assigned to leaves
        unique_nodes = np.unique(sample_node_ids)
        assert len(unique_nodes) >= 2, "Should have samples in at least 2 nodes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
