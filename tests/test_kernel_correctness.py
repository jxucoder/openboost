"""Kernel-level correctness tests for OpenBoost.

Verifies that the lowest-level computational kernels (histograms, split finding,
leaf values) produce correct results against hand-computed reference values.
These tests catch bugs in the core algorithms that affect all models.
"""

import numpy as np
import pytest

import openboost as ob
from openboost._core._split import compute_leaf_value, find_best_split

# =============================================================================
# Histogram Correctness
# =============================================================================


class TestHistogramCorrectness:
    """Verify histogram building produces correct aggregations."""

    def test_histogram_sum_equals_gradient_sum(self, binned_100x5, mse_grads_100):
        """Sum of histogram gradients must equal sum of input gradients."""
        binned, _ = binned_100x5
        grad, hess = mse_grads_100

        sample_node_ids = ob.init_sample_node_ids(100)
        histograms = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0]
        )

        hist = histograms[0]
        # Sum across all bins for each feature should equal total gradient
        for feat in range(5):
            feat_grad_sum = np.sum(hist.hist_grad[feat, :])
            feat_hess_sum = np.sum(hist.hist_hess[feat, :])
            np.testing.assert_almost_equal(
                feat_grad_sum, np.sum(grad), decimal=4,
                err_msg=f"Feature {feat}: hist grad sum != input grad sum"
            )
            np.testing.assert_almost_equal(
                feat_hess_sum, np.sum(hess), decimal=4,
                err_msg=f"Feature {feat}: hist hess sum != input hess sum"
            )

    def test_histogram_per_bin_counts(self):
        """Hand-crafted data: verify each bin's grad/hess matches manual sum."""
        # Create data where we know exactly which sample goes to which bin
        # 10 samples, 2 features, carefully crafted to land in known bins
        rng = np.random.RandomState(42)
        n_samples = 20
        X = rng.randn(n_samples, 2).astype(np.float32)
        binned = ob.array(X)

        # Known gradients
        grad = np.arange(n_samples, dtype=np.float32)
        hess = np.ones(n_samples, dtype=np.float32)

        sample_node_ids = ob.init_sample_node_ids(n_samples)
        histograms = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0]
        )

        hist = histograms[0]

        # For each feature, verify that the samples in each bin sum correctly
        for feat in range(2):
            bin_values = binned.data[feat, :]  # bin assignment for each sample
            for b in range(256):
                mask = bin_values == b
                expected_grad = np.sum(grad[mask])
                expected_hess = np.sum(hess[mask])
                np.testing.assert_almost_equal(
                    hist.hist_grad[feat, b], expected_grad, decimal=5,
                    err_msg=f"Feature {feat}, bin {b}: grad mismatch"
                )
                np.testing.assert_almost_equal(
                    hist.hist_hess[feat, b], expected_hess, decimal=5,
                    err_msg=f"Feature {feat}, bin {b}: hess mismatch"
                )

    def test_histogram_with_missing_bin_isolated(self):
        """NaN samples must accumulate only in bin 255 (MISSING_BIN)."""
        rng = np.random.RandomState(42)
        n_samples = 50
        X = rng.randn(n_samples, 3).astype(np.float32)
        # Inject NaN in feature 0 for first 10 samples
        X[:10, 0] = np.nan

        binned = ob.array(X)
        grad = np.ones(n_samples, dtype=np.float32)
        hess = np.ones(n_samples, dtype=np.float32)

        sample_node_ids = ob.init_sample_node_ids(n_samples)
        histograms = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0]
        )

        hist = histograms[0]

        # For feature 0: bin 255 should have grad=10 (10 NaN samples * grad=1)
        np.testing.assert_almost_equal(
            hist.hist_grad[0, 255], 10.0, decimal=4,
            err_msg="Missing bin should accumulate exactly NaN samples"
        )
        np.testing.assert_almost_equal(
            hist.hist_hess[0, 255], 10.0, decimal=4,
        )

        # Non-NaN features should have 0 in bin 255
        np.testing.assert_almost_equal(
            hist.hist_grad[1, 255], 0.0, decimal=4,
            err_msg="Feature without NaN should have 0 in missing bin"
        )

    def test_histogram_constant_feature(self):
        """A constant feature should have all samples in a single bin."""
        n_samples = 50
        X = np.zeros((n_samples, 2), dtype=np.float32)
        X[:, 0] = 5.0  # Constant
        X[:, 1] = np.arange(n_samples, dtype=np.float32)  # Varying

        binned = ob.array(X)
        grad = np.ones(n_samples, dtype=np.float32) * 3.0
        hess = np.ones(n_samples, dtype=np.float32)

        sample_node_ids = ob.init_sample_node_ids(n_samples)
        histograms = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0]
        )

        hist = histograms[0]

        # Feature 0 (constant): exactly one bin should have all grad/hess
        nonzero_bins = np.sum(hist.hist_hess[0, :] > 0)
        assert nonzero_bins == 1, f"Constant feature should have 1 non-zero bin, got {nonzero_bins}"
        np.testing.assert_almost_equal(
            np.sum(hist.hist_grad[0, :]), 3.0 * n_samples, decimal=4
        )

    def test_histogram_subtraction(self, binned_100x5, mse_grads_100):
        """Parent histogram - left child histogram = right child histogram."""
        binned, _ = binned_100x5
        grad, hess = mse_grads_100

        sample_node_ids = ob.init_sample_node_ids(100)

        # Build parent histogram
        parent_hists = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0]
        )
        parent = parent_hists[0]

        # Do a split to create children
        splits = ob.find_node_splits(parent_hists)
        if splits and 0 in splits and splits[0].split.is_valid:
            new_node_ids = ob.partition_samples(binned.data, sample_node_ids, splits)
            left_id = splits[0].left_child
            right_id = splits[0].right_child

            child_hists = ob.build_node_histograms(
                binned.data, grad, hess, new_node_ids, [left_id, right_id]
            )

            if left_id in child_hists and right_id in child_hists:
                left = child_hists[left_id]
                right = child_hists[right_id]

                # Parent = left + right
                np.testing.assert_almost_equal(
                    parent.hist_grad, left.hist_grad + right.hist_grad,
                    decimal=4, err_msg="Parent grad != left + right"
                )
                np.testing.assert_almost_equal(
                    parent.hist_hess, left.hist_hess + right.hist_hess,
                    decimal=4, err_msg="Parent hess != left + right"
                )


# =============================================================================
# Split Finding Correctness
# =============================================================================


class TestSplitFindingCorrectness:
    """Verify split finding selects the optimal split."""

    def test_split_gain_formula_exact(self):
        """Verify split gain matches the formula: left_score + right_score - parent_score."""
        # Construct a histogram with known values
        n_features = 2
        hist_grad = np.zeros((n_features, 256), dtype=np.float32)
        hist_hess = np.zeros((n_features, 256), dtype=np.float32)

        # Feature 0: bins 0-9 have grad=1, hess=1 each; bins 10-19 have grad=-1, hess=1
        for b in range(10):
            hist_grad[0, b] = 1.0
            hist_hess[0, b] = 1.0
        for b in range(10, 20):
            hist_grad[0, b] = -1.0
            hist_hess[0, b] = 1.0

        # Feature 1: spread evenly (poor split)
        for b in range(20):
            hist_grad[1, b] = 0.0
            hist_hess[1, b] = 1.0

        total_grad = float(np.sum(hist_grad[0]))  # 0.0
        total_hess = float(np.sum(hist_hess[0]))  # 20.0

        reg_lambda = 1.0
        split = find_best_split(
            hist_grad, hist_hess, total_grad, total_hess,
            reg_lambda=reg_lambda, min_child_weight=0.0,
        )

        assert split.feature == 0, f"Should split on feature 0, got {split.feature}"

        # Manual gain computation for the best split on feature 0 at threshold=9
        # Left: grad=10, hess=10 -> score = 10^2/(10+1) = 100/11
        # Right: grad=-10, hess=10 -> score = (-10)^2/(10+1) = 100/11
        # Parent: grad=0, hess=20 -> score = 0^2/(20+1) = 0
        # Gain = 100/11 + 100/11 - 0 = 200/11 ≈ 18.18
        expected_gain = 100.0 / 11.0 + 100.0 / 11.0 - 0.0
        np.testing.assert_almost_equal(
            split.gain, expected_gain, decimal=3,
            err_msg=f"Gain should be {expected_gain}, got {split.gain}"
        )

    def test_split_selects_optimal_feature(self):
        """Feature with highest gain should be selected."""
        n_features = 3
        hist_grad = np.zeros((n_features, 256), dtype=np.float32)
        hist_hess = np.zeros((n_features, 256), dtype=np.float32)

        # Feature 0: weak split
        hist_grad[0, :5] = 0.1
        hist_hess[0, :5] = 1.0
        hist_grad[0, 5:10] = -0.1
        hist_hess[0, 5:10] = 1.0

        # Feature 1: NO split possible (constant)
        hist_grad[1, 0] = 0.0
        hist_hess[1, 0] = 10.0

        # Feature 2: strong split (large gradient difference)
        hist_grad[2, :5] = 5.0
        hist_hess[2, :5] = 1.0
        hist_grad[2, 5:10] = -5.0
        hist_hess[2, 5:10] = 1.0

        total_grad = float(np.sum(hist_grad[0]))
        total_hess = float(np.sum(hist_hess[0]))

        split = find_best_split(
            hist_grad, hist_hess, total_grad, total_hess,
            reg_lambda=1.0, min_child_weight=0.0,
        )

        assert split.feature == 2, f"Should pick feature 2 (strongest), got {split.feature}"

    def test_split_min_child_weight_enforcement(self):
        """Splits that violate min_child_weight should be rejected."""
        n_features = 1
        hist_grad = np.zeros((n_features, 256), dtype=np.float32)
        hist_hess = np.zeros((n_features, 256), dtype=np.float32)

        # Only one sample in bin 0, rest in bin 1
        hist_grad[0, 0] = 5.0
        hist_hess[0, 0] = 0.5  # Below min_child_weight=1.0
        hist_grad[0, 1] = -5.0
        hist_hess[0, 1] = 10.0

        total_grad = 0.0
        total_hess = 10.5

        split = find_best_split(
            hist_grad, hist_hess, total_grad, total_hess,
            reg_lambda=1.0, min_child_weight=1.0,
        )

        # Split at threshold=0 would put hess=0.5 in left, violating min_child_weight=1.0
        # Should either find no split or a different threshold
        if split.is_valid and split.threshold == 0:
            pytest.fail("Split at threshold=0 should be rejected (hess=0.5 < min_child_weight=1.0)")


# =============================================================================
# Leaf Value Correctness
# =============================================================================


class TestLeafValueCorrectness:
    """Verify leaf value computation follows Newton-Raphson formula."""

    def test_newton_raphson_formula(self):
        """leaf_value = -sum_grad / (sum_hess + lambda)."""
        # Case 1: simple
        val = compute_leaf_value(sum_grad=6.0, sum_hess=3.0, reg_lambda=1.0)
        expected = -6.0 / (3.0 + 1.0)  # -1.5
        np.testing.assert_almost_equal(val, expected, decimal=10)

        # Case 2: negative gradient
        val = compute_leaf_value(sum_grad=-3.0, sum_hess=2.0, reg_lambda=1.0)
        expected = 3.0 / (2.0 + 1.0)  # 1.0
        np.testing.assert_almost_equal(val, expected, decimal=10)

        # Case 3: zero gradient
        val = compute_leaf_value(sum_grad=0.0, sum_hess=5.0, reg_lambda=1.0)
        np.testing.assert_almost_equal(val, 0.0, decimal=10)

        # Case 4: large lambda
        val = compute_leaf_value(sum_grad=10.0, sum_hess=2.0, reg_lambda=100.0)
        expected = -10.0 / (2.0 + 100.0)  # -0.098...
        np.testing.assert_almost_equal(val, expected, decimal=10)

    def test_l1_soft_thresholding_below_threshold(self):
        """When |sum_grad| <= reg_alpha, leaf value should be 0."""
        val = compute_leaf_value(sum_grad=0.5, sum_hess=5.0, reg_lambda=1.0, reg_alpha=1.0)
        np.testing.assert_almost_equal(val, 0.0, decimal=10)

        val = compute_leaf_value(sum_grad=-0.3, sum_hess=5.0, reg_lambda=1.0, reg_alpha=0.5)
        np.testing.assert_almost_equal(val, 0.0, decimal=10)

    def test_l1_soft_thresholding_above_threshold(self):
        """When |sum_grad| > reg_alpha, apply soft-thresholding."""
        # Positive gradient above threshold
        val = compute_leaf_value(sum_grad=2.0, sum_hess=3.0, reg_lambda=1.0, reg_alpha=0.5)
        expected = -(2.0 - 0.5) / (3.0 + 1.0)  # -0.375
        np.testing.assert_almost_equal(val, expected, decimal=10)

        # Negative gradient above threshold
        val = compute_leaf_value(sum_grad=-2.0, sum_hess=3.0, reg_lambda=1.0, reg_alpha=0.5)
        expected = -(-2.0 + 0.5) / (3.0 + 1.0)  # 0.375
        np.testing.assert_almost_equal(val, expected, decimal=10)


# =============================================================================
# Partition Correctness
# =============================================================================


class TestPartitionCorrectness:
    """Verify sample partitioning preserves counts and is consistent."""

    def test_partition_conserves_samples(self, binned_100x5, mse_grads_100):
        """After partitioning, n_left + n_right = n_total."""
        binned, _ = binned_100x5
        grad, hess = mse_grads_100

        sample_node_ids = ob.init_sample_node_ids(100)

        histograms = ob.build_node_histograms(
            binned.data, grad, hess, sample_node_ids, [0]
        )
        splits = ob.find_node_splits(histograms)

        if splits and 0 in splits and splits[0].split.is_valid:
            new_node_ids = ob.partition_samples(binned.data, sample_node_ids, splits)
            left_id = splits[0].left_child
            right_id = splits[0].right_child

            n_left = np.sum(new_node_ids == left_id)
            n_right = np.sum(new_node_ids == right_id)

            assert n_left + n_right == 100, (
                f"Partition should conserve samples: {n_left} + {n_right} != 100"
            )
            assert n_left > 0, "Left child should have at least 1 sample"
            assert n_right > 0, "Right child should have at least 1 sample"

    def test_partition_deterministic(self, binned_100x5, mse_grads_100):
        """Same data should produce same partition."""
        binned, _ = binned_100x5
        grad, hess = mse_grads_100

        results = []
        for _ in range(2):
            sample_node_ids = ob.init_sample_node_ids(100)
            histograms = ob.build_node_histograms(
                binned.data, grad, hess, sample_node_ids, [0]
            )
            splits = ob.find_node_splits(histograms)
            if splits and 0 in splits and splits[0].split.is_valid:
                new_node_ids = ob.partition_samples(binned.data, sample_node_ids, splits)
                results.append(new_node_ids.copy())

        if len(results) == 2:
            np.testing.assert_array_equal(results[0], results[1])

    def test_tree_depth_matches_max_depth(self, regression_100x5):
        """Trees must respect max_depth constraint."""
        X, y = regression_100x5
        binned = ob.array(X)
        grad = (2 * (np.zeros(100, dtype=np.float32) - y)).astype(np.float32)
        hess = np.ones(100, dtype=np.float32) * 2

        for depth in [1, 2, 3, 4, 5]:
            tree = ob.fit_tree(binned, grad, hess, max_depth=depth)
            assert tree.depth <= depth, f"Tree depth {tree.depth} > max_depth {depth}"


# =============================================================================
# End-to-End Algorithmic Correctness
# =============================================================================


class TestAlgorithmicCorrectness:
    """End-to-end correctness of the boosting algorithm."""

    def test_boosting_monotonic_loss_decrease(self, regression_200x10):
        """Loss should decrease every round (for reasonable settings)."""
        X, y = regression_200x10
        binned = ob.array(X)
        pred = np.zeros(200, dtype=np.float32)

        losses = []
        for _ in range(5):
            loss = float(np.mean((pred - y) ** 2))
            losses.append(loss)
            grad = (2 * (pred - y)).astype(np.float32)
            hess = np.ones(200, dtype=np.float32) * 2
            tree = ob.fit_tree(binned, grad, hess, max_depth=4)
            pred = pred + 0.1 * tree(binned)

        # Each subsequent loss should be lower
        for i in range(1, len(losses)):
            assert losses[i] < losses[i - 1], (
                f"Loss should decrease monotonically: round {i}: {losses[i]} >= {losses[i-1]}"
            )

    def test_converges_to_mean_for_constant_target(self):
        """For constant y, predictions should converge to that constant."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5).astype(np.float32)
        y = np.full(100, 3.7, dtype=np.float32)

        model = ob.GradientBoosting(n_trees=50, max_depth=2, learning_rate=0.3)
        model.fit(X, y)
        pred = model.predict(X)

        # Predictions should be very close to 3.7
        np.testing.assert_allclose(pred, 3.7, atol=0.1,
                                   err_msg="Predictions should converge to constant target value")

    def test_single_split_tree_matches_manual(self):
        """A depth-1 tree with simple data should produce predictable splits."""
        # Feature 0 clearly splits the target
        X = np.array([
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ], dtype=np.float32)
        y = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float32)

        binned = ob.array(X)
        grad = (2 * (np.zeros(4, dtype=np.float32) - y)).astype(np.float32)  # [-2, -2, 2, 2] * -1 = [2, 2, -2, -2]
        hess = np.ones(4, dtype=np.float32) * 2

        tree = ob.fit_tree(binned, grad, hess, max_depth=1)

        # Should split on feature 0
        assert tree.n_nodes >= 3, "Depth-1 tree should have at least 3 nodes (root + 2 leaves)"
        assert tree.depth == 1

        # Predictions for left vs right should have opposite signs
        pred = tree(binned)
        assert pred[0] * pred[2] < 0 or np.abs(pred[0] - pred[2]) > 0.1, (
            "Left and right predictions should differ for this clear split"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
