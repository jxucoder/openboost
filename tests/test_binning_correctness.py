"""BinnedArray correctness tests for OpenBoost.

Verifies that data binning (quantization to uint8) is correct,
consistent between fit and transform, and handles edge cases.
"""

import numpy as np
import pytest

import openboost as ob


class TestBinningConsistency:
    """Verify that binning is consistent between fit and transform."""

    def test_transform_matches_training_bins(self):
        """Re-binning training data with transform should reproduce original bins."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5).astype(np.float32)

        binned = ob.array(X)
        # Transform the same data using training bin edges
        re_binned = binned.transform(X)

        np.testing.assert_array_equal(
            binned.data, re_binned.data,
            err_msg="Re-binning training data should produce identical bins"
        )

    def test_transform_preserves_shape(self):
        """Transform output should have correct shape."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 5).astype(np.float32)
        X_test = rng.randn(50, 5).astype(np.float32)

        binned = ob.array(X_train)
        test_binned = binned.transform(X_test)

        assert test_binned.n_samples == 50
        assert test_binned.n_features == 5
        assert test_binned.data.shape == (5, 50)
        assert test_binned.data.dtype == np.uint8

    def test_transform_out_of_range_values(self):
        """Values outside training range should be clipped to valid bins."""
        X_train = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        X_test = np.array([[-10.0], [10.0]], dtype=np.float32)

        binned = ob.array(X_train)
        test_binned = binned.transform(X_test)

        # Should be valid bin values (not 255 since no NaN)
        assert np.all(test_binned.data < 255), "Out-of-range values should not be missing bin"
        assert np.all(test_binned.data >= 0), "Bins should be non-negative"

    def test_transform_preserves_corrected_top_bin(self):
        """The largest and above-range values stay in the final numeric bin."""
        X_train = np.array([[1.0], [1.0], [2.0], [2.0]], dtype=np.float32)
        X_test = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)

        binned = ob.array(X_train, n_bins=2)
        transformed = binned.transform(X_test)

        np.testing.assert_array_equal(transformed.data[0], [0, 0, 1, 1])

    def test_transform_without_version_uses_legacy_routing(self):
        """Directly unpickled pre-version metadata keeps its old top-bin rule."""
        X = np.array([[1.0], [1.0], [2.0], [2.0]], dtype=np.float32)
        binned = ob.array(X, n_bins=2)
        del binned.binning_version

        transformed = binned.transform(X)

        assert transformed.binning_version == 1
        np.testing.assert_array_equal(transformed.data[0], 0)


class TestBinEdges:
    """Verify bin edge properties."""

    def test_bin_edges_monotonic(self):
        """Bin edges should be strictly increasing per feature."""
        rng = np.random.RandomState(42)
        X = rng.randn(500, 5).astype(np.float32)
        binned = ob.array(X)

        for feat_idx, edges in enumerate(binned.bin_edges):
            edges_arr = np.array(edges)
            if len(edges_arr) > 1:
                diffs = np.diff(edges_arr)
                assert np.all(diffs > 0), (
                    f"Feature {feat_idx}: bin edges not strictly increasing"
                )

    def test_bin_count_respects_max(self):
        """Number of bins should not exceed 254 (bin 255 is reserved)."""
        rng = np.random.RandomState(42)
        X = rng.randn(1000, 3).astype(np.float32)
        binned = ob.array(X)

        # No sample should have bin 255 (no NaN in this data)
        assert np.max(binned.data) < 255, "Max bin should be < 255 when no NaN"


class TestMissingValues:
    """Verify NaN handling in binning."""

    def test_nan_maps_to_missing_bin(self):
        """NaN values should be binned as 255 (MISSING_BIN)."""
        X = np.array([
            [1.0, np.nan],
            [2.0, 3.0],
            [np.nan, 4.0],
        ], dtype=np.float32)

        binned = ob.array(X)

        # Feature 0, sample 2 should be 255
        assert binned.data[0, 2] == 255, "NaN in feature 0, sample 2 should be bin 255"
        # Feature 1, sample 0 should be 255
        assert binned.data[1, 0] == 255, "NaN in feature 1, sample 0 should be bin 255"
        # Non-NaN values should not be 255
        assert binned.data[0, 0] != 255, "Non-NaN value should not be bin 255"
        assert binned.data[0, 1] != 255, "Non-NaN value should not be bin 255"

    def test_no_nan_means_no_missing_bin(self):
        """Without NaN, no sample should land in bin 255."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5).astype(np.float32)

        binned = ob.array(X)

        assert np.all(binned.data != 255), "No bin 255 when no NaN in data"

    def test_all_nan_feature(self):
        """A feature that is all NaN should have all samples in bin 255."""
        X = np.array([
            [1.0, np.nan],
            [2.0, np.nan],
            [3.0, np.nan],
        ], dtype=np.float32)

        binned = ob.array(X)

        assert np.all(binned.data[1, :] == 255), "All-NaN feature should be all bin 255"


class TestBinningEdgeCases:
    """Edge cases for binning."""

    def test_constant_feature(self):
        """Constant feature should produce valid binning."""
        X = np.ones((50, 2), dtype=np.float32)
        X[:, 1] = np.arange(50, dtype=np.float32)  # Feature 1 varies

        binned = ob.array(X)

        # Feature 0 (constant) should have all samples in the same bin
        unique_bins = np.unique(binned.data[0, :])
        assert len(unique_bins) == 1, f"Constant feature should have 1 bin, got {len(unique_bins)}"

    def test_two_unique_values(self):
        """Two distinct values should produce two bins."""
        X = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)

        binned = ob.array(X, n_bins=2)

        unique_bins = np.unique(binned.data[0, :])
        assert len(unique_bins) == 2, f"Two values should produce 2 bins, got {len(unique_bins)}"

    def test_low_cardinality_values_keep_distinct_top_bin(self):
        """Quantile binning must not merge the highest ordinal level."""
        X = np.repeat(
            np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
            repeats=4,
            axis=0,
        )

        binned = ob.array(X, n_bins=4)

        assert np.unique(binned.data[0]).tolist() == [0, 1, 2, 3]

    def test_top_bin_with_missing_values(self):
        """The NaN path preserves both numeric bins plus the missing bin."""
        X = np.array([[1.0], [1.0], [2.0], [2.0], [np.nan]], dtype=np.float32)

        binned = ob.array(X, n_bins=2)

        np.testing.assert_array_equal(binned.data[0], [0, 0, 1, 1, 255])

    def test_top_bin_enables_binary_tree_split(self):
        """A depth-one tree can split the two values retained by binning."""
        X = np.array([[1.0], [1.0], [2.0], [2.0]], dtype=np.float32)
        y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        binned = ob.array(X, n_bins=2)

        tree = ob.fit_tree(
            binned,
            -y,
            np.ones_like(y),
            max_depth=1,
            reg_lambda=0.0,
        )

        assert tree.features[0] == 0
        np.testing.assert_array_equal(tree(binned), y)

    def test_very_large_values(self):
        """Large values should not cause overflow."""
        X = np.array([[1e10, -1e10], [1e15, -1e15]], dtype=np.float32)

        binned = ob.array(X)

        assert binned.data.dtype == np.uint8
        assert np.all(np.isfinite(binned.data.astype(float)))

    def test_single_sample(self):
        """Single sample should bin correctly."""
        X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        binned = ob.array(X)

        assert binned.n_samples == 1
        assert binned.n_features == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
