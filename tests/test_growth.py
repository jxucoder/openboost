"""Tests for growth strategies (Phase 8.2)."""

import numpy as np
import pytest

import openboost as ob


class TestGrowthConfig:
    """Tests for GrowthConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ob.GrowthConfig()
        
        assert config.max_depth == 6
        assert config.max_leaves is None
        assert config.min_child_weight == 1.0
        assert config.reg_lambda == 1.0
        assert config.min_gain == 0.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ob.GrowthConfig(
            max_depth=4,
            max_leaves=16,
            min_child_weight=5.0,
            reg_lambda=2.0,
            min_gain=0.1,
        )
        
        assert config.max_depth == 4
        assert config.max_leaves == 16


class TestTreeStructure:
    """Tests for TreeStructure."""
    
    def test_standard_tree_prediction(self):
        """Test prediction with standard tree."""
        # Simple tree: split on feature 0 at threshold 127
        # Left leaf = -1.0, Right leaf = 1.0
        tree = ob.TreeStructure(
            features=np.array([0, -1, -1], dtype=np.int32),
            thresholds=np.array([127, 0, 0], dtype=np.int32),
            values=np.array([0.0, -1.0, 1.0], dtype=np.float32),
            left_children=np.array([1, -1, -1], dtype=np.int32),
            right_children=np.array([2, -1, -1], dtype=np.int32),
            n_nodes=3,
            depth=1,
            n_features=1,
        )
        
        # Create test data: 4 samples
        # Samples 0, 1 have feature 0 <= 127 (go left)
        # Samples 2, 3 have feature 0 > 127 (go right)
        binned = np.array([[100, 127, 128, 200]], dtype=np.uint8)
        
        pred = tree.predict(binned)
        
        assert pred.shape == (4,)
        np.testing.assert_array_almost_equal(pred, [-1.0, -1.0, 1.0, 1.0])
    
    def test_symmetric_tree_prediction(self):
        """Test prediction with symmetric tree."""
        # Depth-2 symmetric tree
        # Level 0: split feature 0 at 127
        # Level 1: split feature 1 at 127
        # Leaves: 00=-2, 01=-1, 10=1, 11=2
        tree = ob.TreeStructure(
            features=np.array([0, 1, 1, -1, -1, -1, -1], dtype=np.int32),
            thresholds=np.array([127, 127, 127, 0, 0, 0, 0], dtype=np.int32),
            values=np.array([0, 0, 0, -2, -1, 1, 2], dtype=np.float32),
            left_children=np.array([1, 3, 5, -1, -1, -1, -1], dtype=np.int32),
            right_children=np.array([2, 4, 6, -1, -1, -1, -1], dtype=np.int32),
            n_nodes=7,
            depth=2,
            n_features=2,
            is_symmetric=True,
            level_features=np.array([0, 1], dtype=np.int32),
            level_thresholds=np.array([127, 127], dtype=np.int32),
        )
        
        # Test data
        binned = np.array([
            [100, 100, 200, 200],  # Feature 0
            [100, 200, 100, 200],  # Feature 1
        ], dtype=np.uint8)
        
        pred = tree.predict(binned)
        
        assert pred.shape == (4,)
        # leaf_id = (f0 > 127) * 2 + (f1 > 127)
        # Sample 0: 0*2 + 0 = 0 -> -2
        # Sample 1: 0*2 + 1 = 1 -> -1
        # Sample 2: 1*2 + 0 = 2 -> 1
        # Sample 3: 1*2 + 1 = 3 -> 2
        np.testing.assert_array_almost_equal(pred, [-2, -1, 1, 2])


class TestLevelWiseGrowth:
    """Tests for LevelWiseGrowth."""
    
    def test_basic_growth(self):
        """Test basic level-wise tree growth."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1]  # Linear target
        
        binned = ob.array(X)
        
        # MSE gradients
        pred = np.zeros(200, dtype=np.float32)
        grad = (2 * (pred - y)).astype(np.float32)
        hess = np.ones(200, dtype=np.float32) * 2
        
        config = ob.GrowthConfig(max_depth=3)
        strategy = ob.LevelWiseGrowth()
        
        tree = strategy.grow(binned.data, grad, hess, config)
        
        assert tree.depth <= 3
        assert tree.n_nodes > 0
        
        # Should be able to predict
        pred = tree.predict(binned.data)
        assert pred.shape == (200,)
    
    def test_respects_max_depth(self):
        """Test that max_depth is respected."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        y = X[:, 0]
        
        binned = ob.array(X)
        grad = (-2 * y).astype(np.float32)
        hess = np.ones(100, dtype=np.float32) * 2
        
        for max_depth in [1, 2, 4]:
            config = ob.GrowthConfig(max_depth=max_depth)
            strategy = ob.LevelWiseGrowth()
            tree = strategy.grow(binned.data, grad, hess, config)
            
            assert tree.depth <= max_depth
    
    def test_training_reduces_loss(self):
        """Test that level-wise tree reduces loss."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1]
        
        binned = ob.array(X)
        
        pred = np.zeros(200, dtype=np.float32)
        initial_loss = np.mean((pred - y) ** 2)
        
        config = ob.GrowthConfig(max_depth=4, reg_lambda=0.1)
        strategy = ob.LevelWiseGrowth()
        
        # Train multiple trees
        for _ in range(5):
            grad = (2 * (pred - y)).astype(np.float32)
            hess = np.ones(200, dtype=np.float32) * 2
            
            tree = strategy.grow(binned.data, grad, hess, config)
            pred = pred + 0.3 * tree.predict(binned.data)
        
        final_loss = np.mean((pred - y) ** 2)
        
        assert final_loss < initial_loss


class TestLeafWiseGrowth:
    """Tests for LeafWiseGrowth."""
    
    def test_basic_growth(self):
        """Test basic leaf-wise tree growth."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1]
        
        binned = ob.array(X)
        
        pred = np.zeros(200, dtype=np.float32)
        grad = (2 * (pred - y)).astype(np.float32)
        hess = np.ones(200, dtype=np.float32) * 2
        
        config = ob.GrowthConfig(max_depth=6, max_leaves=8)
        strategy = ob.LeafWiseGrowth()
        
        tree = strategy.grow(binned.data, grad, hess, config)
        
        assert tree.n_nodes > 0
        
        pred = tree.predict(binned.data)
        assert pred.shape == (200,)
    
    def test_respects_max_leaves(self):
        """Test that max_leaves is approximately respected."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0]
        
        binned = ob.array(X)
        grad = (-2 * y).astype(np.float32)
        hess = np.ones(200, dtype=np.float32) * 2
        
        for max_leaves in [4, 8, 16]:
            config = ob.GrowthConfig(max_depth=10, max_leaves=max_leaves)
            strategy = ob.LeafWiseGrowth()
            tree = strategy.grow(binned.data, grad, hess, config)
            
            # Count actual leaves (nodes that have a parent but no children)
            # A node is a real leaf if: left_children == -1 AND (is root OR parent has children set)
            n_leaves = 0
            for i in range(tree.n_nodes):
                if tree.left_children[i] == -1:
                    if i == 0:  # Root is a leaf
                        n_leaves += 1
                    else:
                        parent = (i - 1) // 2
                        # Check if parent actually split to create this node
                        if tree.features[parent] >= 0:
                            n_leaves += 1
            
            # Should not exceed max_leaves (allow small margin for edge cases)
            assert n_leaves <= max_leaves, f"Got {n_leaves} leaves, expected <= {max_leaves}"


class TestSymmetricGrowth:
    """Tests for SymmetricGrowth."""
    
    def test_basic_growth(self):
        """Test basic symmetric tree growth."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1]
        
        binned = ob.array(X)
        
        pred = np.zeros(200, dtype=np.float32)
        grad = (2 * (pred - y)).astype(np.float32)
        hess = np.ones(200, dtype=np.float32) * 2
        
        config = ob.GrowthConfig(max_depth=3)
        strategy = ob.SymmetricGrowth()
        
        tree = strategy.grow(binned.data, grad, hess, config)
        
        assert tree.is_symmetric
        assert tree.depth <= 3
        assert tree.level_features is not None
        assert len(tree.level_features) == tree.depth
        
        pred = tree.predict(binned.data)
        assert pred.shape == (200,)
    
    def test_symmetric_structure(self):
        """Test that all nodes at same depth use same split."""
        np.random.seed(42)
        X = np.random.randn(200, 3).astype(np.float32)
        y = X[:, 0]

        binned = ob.array(X)
        grad = (-2 * y).astype(np.float32)
        hess = np.ones(200, dtype=np.float32) * 2

        config = ob.GrowthConfig(max_depth=3)
        strategy = ob.SymmetricGrowth()
        tree = strategy.grow(binned.data, grad, hess, config)

        # Check that nodes at same depth have same feature/threshold
        for d in range(tree.depth):
            level_start = 2**d - 1
            level_end = 2**(d+1) - 1

            features_at_level = tree.features[level_start:level_end]
            thresholds_at_level = tree.thresholds[level_start:level_end]

            # All should be the same
            assert len(np.unique(features_at_level)) == 1
            assert len(np.unique(thresholds_at_level)) == 1

    def test_symmetric_non_degenerate_splits(self):
        """Levels must use DISTINCT splits when the data requires them.

        Regression test for the degenerate symmetric builder: summing per-leaf
        histograms into one pooled histogram makes every depth see the root
        statistics and repeat the same split. The target below needs a split
        on feature 0 first and then feature 1, so a correct oblivious tree
        must use two distinct level splits.
        """
        rng = np.random.RandomState(0)
        n = 1000
        X = rng.randn(n, 3).astype(np.float32)
        y = np.where(
            X[:, 0] > 0,
            np.where(X[:, 1] > 0, 3.0, 1.0),
            np.where(X[:, 1] > 0, -1.0, -3.0),
        ).astype(np.float32)

        binned = ob.array(X, n_bins=254)
        grad = (-2 * y).astype(np.float32)
        hess = np.full(n, 2.0, dtype=np.float32)

        config = ob.GrowthConfig(max_depth=3, reg_lambda=1.0)
        tree = ob.SymmetricGrowth().grow(binned.data, grad, hess, config)

        assert tree.depth >= 2
        level_splits = {
            (int(tree.level_features[d]), int(tree.level_thresholds[d]))
            for d in range(tree.depth)
        }
        assert len(level_splits) >= 2, (
            f"symmetric tree repeated one split per level: {level_splits}"
        )
        # Both informative features must appear
        assert {0, 1} <= {int(f) for f in tree.level_features[:tree.depth]}

        # A single depth-2 oblivious tree separates the 4 cells almost exactly
        # (leaf value = -sum_grad/(sum_hess+lambda) with lr=1). The degenerate
        # builder plateaus around MSE ~1.0 here.
        pred = tree.predict(binned.data)
        assert np.mean((pred - y) ** 2) < 0.1

    def test_symmetric_categorical_passthrough(self):
        """SymmetricGrowth must use categorical split info (category sets).

        The category -> target mapping is non-monotonic in code order, so an
        ordinal split cannot separate the groups but a category-set split can.
        """
        rng = np.random.RandomState(42)
        n = 800
        codes = rng.randint(0, 6, size=n)
        cat_map = np.array([2.0, -2.0, -2.0, 2.0, -2.0, 2.0])  # {0,3,5} vs rest
        y = (cat_map[codes] + 0.01 * rng.randn(n)).astype(np.float32)
        X = np.column_stack([
            codes.astype(np.float32),
            rng.randn(n).astype(np.float32),
        ])

        binned = ob.array(X, categorical_features=[0])
        grad = (-2 * y).astype(np.float32)
        hess = np.full(n, 2.0, dtype=np.float32)

        config = ob.GrowthConfig(max_depth=2, reg_lambda=1.0)
        tree = ob.SymmetricGrowth().grow(
            binned.data, grad, hess, config,
            is_categorical=binned.is_categorical,
            n_categories=binned.n_categories,
        )

        # The only informative feature is categorical -> the root level must
        # be a categorical (bitset) split on feature 0.
        assert tree.is_categorical_split is not None
        assert bool(tree.is_categorical_split[0])
        assert int(tree.features[0]) == 0
        bitset = int(tree.cat_bitsets[0])
        left_set = {c for c in range(6) if (bitset >> c) & 1}
        assert left_set in ({0, 3, 5}, {1, 2, 4})

        # Bitset routing must separate the two groups (ordinal cannot).
        pred = tree.predict(binned.data)
        assert np.mean((pred - y) ** 2) < 0.1


class TestGetGrowthStrategy:
    """Tests for get_growth_strategy factory."""
    
    def test_get_levelwise(self):
        """Test getting level-wise strategy."""
        strategy = ob.get_growth_strategy("levelwise")
        assert isinstance(strategy, ob.LevelWiseGrowth)
        
        # Also test aliases
        assert isinstance(ob.get_growth_strategy("level_wise"), ob.LevelWiseGrowth)
        assert isinstance(ob.get_growth_strategy("level-wise"), ob.LevelWiseGrowth)
    
    def test_get_leafwise(self):
        """Test getting leaf-wise strategy."""
        strategy = ob.get_growth_strategy("leafwise")
        assert isinstance(strategy, ob.LeafWiseGrowth)
    
    def test_get_symmetric(self):
        """Test getting symmetric strategy."""
        strategy = ob.get_growth_strategy("symmetric")
        assert isinstance(strategy, ob.SymmetricGrowth)
        
        # Also test alias
        assert isinstance(ob.get_growth_strategy("oblivious"), ob.SymmetricGrowth)
    
    def test_invalid_strategy(self):
        """Test error on invalid strategy name."""
        with pytest.raises(ValueError, match="Unknown growth strategy"):
            ob.get_growth_strategy("invalid")


class TestGrowthComparison:
    """Compare different growth strategies."""
    
    def test_all_strategies_produce_valid_trees(self):
        """Test that all strategies produce valid, predictable trees."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1]
        
        binned = ob.array(X)
        
        pred = np.zeros(200, dtype=np.float32)
        grad = (2 * (pred - y)).astype(np.float32)
        hess = np.ones(200, dtype=np.float32) * 2
        
        strategies = [
            ("levelwise", ob.GrowthConfig(max_depth=4)),
            ("leafwise", ob.GrowthConfig(max_depth=4, max_leaves=16)),
            ("symmetric", ob.GrowthConfig(max_depth=4)),
        ]
        
        for name, config in strategies:
            strategy = ob.get_growth_strategy(name)
            tree = strategy.grow(binned.data, grad, hess, config)
            
            # Should produce valid predictions
            predictions = tree.predict(binned.data)
            
            assert predictions.shape == (200,), f"{name} prediction shape wrong"
            assert not np.any(np.isnan(predictions)), f"{name} produced NaN"
            assert not np.any(np.isinf(predictions)), f"{name} produced Inf"


# =============================================================================
# Phase 9.0: Leaf Value Abstractions
# =============================================================================

class TestScalarLeaves:
    """Tests for ScalarLeaves."""
    
    def test_create_zeros(self):
        """Test creating zero-initialized leaves."""
        leaves = ob.ScalarLeaves.zeros(10)
        
        assert leaves.shape == (10,)
        np.testing.assert_array_equal(leaves.values, np.zeros(10))
    
    def test_getitem(self):
        """Test indexing into leaves."""
        leaves = ob.ScalarLeaves.zeros(10)
        leaves[3] = 1.5
        leaves[7] = -2.0
        
        indices = np.array([3, 7, 0])
        values = leaves[indices]
        
        np.testing.assert_array_almost_equal(values, [1.5, -2.0, 0.0])
    
    def test_setitem(self):
        """Test setting leaf values."""
        leaves = ob.ScalarLeaves.zeros(5)
        leaves[2] = 3.14
        
        assert leaves.values[2] == pytest.approx(3.14)


class TestVectorLeaves:
    """Tests for VectorLeaves."""
    
    def test_create_zeros(self):
        """Test creating zero-initialized vector leaves."""
        leaves = ob.VectorLeaves.zeros(10, n_outputs=3)
        
        assert leaves.shape == (10, 3)
        assert leaves.n_outputs == 3
        np.testing.assert_array_equal(leaves.values, np.zeros((10, 3)))
    
    def test_getitem(self):
        """Test indexing into vector leaves."""
        leaves = ob.VectorLeaves.zeros(10, n_outputs=2)
        leaves[3] = np.array([1.0, 2.0])
        leaves[7] = np.array([-1.0, -2.0])
        
        indices = np.array([3, 7])
        values = leaves[indices]
        
        assert values.shape == (2, 2)
        np.testing.assert_array_almost_equal(values[0], [1.0, 2.0])
        np.testing.assert_array_almost_equal(values[1], [-1.0, -2.0])
    
    def test_setitem(self):
        """Test setting vector leaf values."""
        leaves = ob.VectorLeaves.zeros(5, n_outputs=3)
        leaves[2] = np.array([1.0, 2.0, 3.0])
        
        np.testing.assert_array_almost_equal(leaves.values[2], [1.0, 2.0, 3.0])


class TestTreeStructureLeafAbstraction:
    """Tests for TreeStructure with different leaf types."""
    
    def test_backward_compatibility_with_ndarray(self):
        """Test that plain NDArray values still work."""
        # Create a simple tree structure with plain array
        tree = ob.TreeStructure(
            features=np.array([0, -1, -1], dtype=np.int32),
            thresholds=np.array([128, 0, 0], dtype=np.int32),
            values=np.array([0.0, -1.0, 1.0], dtype=np.float32),  # Plain NDArray
            left_children=np.array([1, -1, -1], dtype=np.int32),
            right_children=np.array([2, -1, -1], dtype=np.int32),
            n_nodes=3,
            depth=1,
            n_features=1,
        )
        
        # get_leaf_values should work
        leaf_ids = np.array([1, 2])
        values = tree.get_leaf_values(leaf_ids)
        np.testing.assert_array_almost_equal(values, [-1.0, 1.0])
    
    def test_with_scalar_leaves(self):
        """Test TreeStructure with ScalarLeaves."""
        leaves = ob.ScalarLeaves.zeros(3)
        leaves[1] = -1.0
        leaves[2] = 1.0
        
        tree = ob.TreeStructure(
            features=np.array([0, -1, -1], dtype=np.int32),
            thresholds=np.array([128, 0, 0], dtype=np.int32),
            values=leaves,  # ScalarLeaves
            left_children=np.array([1, -1, -1], dtype=np.int32),
            right_children=np.array([2, -1, -1], dtype=np.int32),
            n_nodes=3,
            depth=1,
            n_features=1,
        )
        
        # get_leaf_values should work
        leaf_ids = np.array([1, 2])
        values = tree.get_leaf_values(leaf_ids)
        np.testing.assert_array_almost_equal(values, [-1.0, 1.0])
        
        # leaf_values_array should return underlying array
        np.testing.assert_array_almost_equal(tree.leaf_values_array, [0.0, -1.0, 1.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestLeafValuesArrayRegression:
    """leaf_values_array must not mistake a plain ndarray for a LeafValues container.

    LeafValues is a runtime_checkable structural Protocol, so an ndarray
    satisfies isinstance(..., LeafValues); the old code then called
    ndarray.values and raised AttributeError for ndarray-backed trees
    (symmetric/leafwise growth).
    """

    def test_leaf_values_array_symmetric_and_leafwise(self):
        import numpy as np

        import openboost as ob

        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 5)).astype(np.float32)
        y = (X[:, 0] + rng.standard_normal(500) * 0.1).astype(np.float32)
        for growth in ("symmetric", "leafwise", "levelwise"):
            model = ob.GradientBoosting(n_trees=2, max_depth=3, growth=growth)
            model.fit(X, y)
            for tree in model.trees_:
                if hasattr(tree, "leaf_values_array"):
                    vals = tree.leaf_values_array
                    vals = vals() if callable(vals) else vals
                    assert isinstance(vals, np.ndarray)


class TestTreeSnapshotSafety:
    """Trees must survive deepcopy and pickle (EarlyStopping restore_best
    snapshots trees_; GPU device-array caches are not copyable and must be
    dropped by __getstate__, re-uploading lazily afterwards)."""

    def test_deepcopy_and_pickle_all_growth_strategies(self):
        import copy
        import pickle

        import numpy as np

        import openboost as ob

        rng = np.random.default_rng(3)
        X = rng.standard_normal((400, 4)).astype(np.float32)
        y = (X[:, 0] - X[:, 1] + 0.1 * rng.standard_normal(400)).astype(np.float32)
        for growth in ("levelwise", "leafwise", "symmetric"):
            model = ob.GradientBoosting(n_trees=3, max_depth=3, growth=growth)
            model.fit(X, y)
            for tree in model.trees_:
                clone = copy.deepcopy(tree)
                revived = pickle.loads(pickle.dumps(tree))
                np.testing.assert_allclose(clone(X), tree(X), rtol=1e-6)
                np.testing.assert_allclose(revived(X), tree(X), rtol=1e-6)
                # device caches must not survive the round-trip
                for obj in (clone, revived):
                    for attr, val in vars(obj).items():
                        if attr.endswith("_gpu") or attr == "_gpu_arrays":
                            assert val is None, f"{attr} leaked through snapshot"
