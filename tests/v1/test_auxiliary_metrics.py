import numpy as np
import pytest
from benchmarks.v1.auxiliary import (
    classification,
    normal_pit,
    paired_interval,
    structure_errors,
    survival,
)
from benchmarks.v1.preprocessing import censoring_support


def test_weighted_auc_ties_and_per_class_counts():
    r = classification("A2", [0, 1, 0, 1], [0.1, 0.5, 0.5, 0.9], weight=[1, 2, 3, 4])
    assert r["binary_auc"] == pytest.approx(21 / 24)
    assert r["classes"]["1"]["weight"] == 6
    assert r["classes"]["0"]["rows"] == 2


def test_absent_class_is_explicit_not_zero_auc():
    r = classification("A2", [0, 0], [0.1, 0.2])
    assert r["binary_auc"] is None
    assert r["classes"]["1"]["recall"] is None


def test_normal_pit_mass_is_conserved():
    r = normal_pit([0.0, 0.0], [[0.0, 1.0], [0.0, 2.0]], weight=[1, 3])
    assert sum(r["pit_decile_mass"]) == 1
    assert r["pit_decile_mass"][5] == 1


def test_ipcw_censored_row_does_not_count_as_death():
    support = censoring_support([1, 2, 3, 4], [1, 0, 1, 1])
    support["grid"] = [2.0]
    p = np.column_stack([np.full(3, np.log(2)), np.ones(3)])
    r = survival([1, 2, 4], [1, 0, 1], p, support)
    assert r["ipcw_brier"] == pytest.approx([(0.25 + 0.25 / (2 / 3)) / 3])
    assert r["contributing_rows"] == [2]


def test_event_censor_tie_uses_frozen_right_continuous_G():
    support = censoring_support([1, 1, 2, 3], [1, 0, 1, 1])
    support["grid"] = [1.0]
    r = survival([1, 2], [1, 1], [[0.0, 1.0], [0.0, 1.0]], support)
    assert r["ipcw_brier"] == pytest.approx([0.375])
    assert r["harrell_c"] == 0.5


def test_grid_outside_censoring_support_is_not_clipped():
    support = censoring_support([1, 2, 3], [1, 1, 0])
    support["grid"] = [3.0]
    with pytest.raises(ValueError, match="support"):
        survival([1, 2], [1, 1], [[0.0, 1.0], [0.0, 1.0]], support)


def test_long_test_followup_needs_no_extrapolated_G_for_earlier_grid():
    support = censoring_support([1, 2, 3], [1, 1, 1])
    support["grid"] = [1.0]
    r = survival([1, 100], [1, 1], [[0.0, 1.0], [0.0, 1.0]], support)
    assert r["ipcw_brier"] == pytest.approx([0.25])


def test_no_comparable_pairs_and_unsupported_weights_are_explicit():
    support = censoring_support([1, 2, 3], [1, 1, 1])
    support["grid"] = [1.0]
    r = survival([2, 3], [0, 0], [[0.0, 1.0], [1.0, 1.0]], support)
    assert r["harrell_c"] is None and r["comparable_pairs"] == 0
    with pytest.raises(ValueError, match="unit"):
        survival([2, 3], [0, 0], [[0.0, 1.0], [1.0, 1.0]], support, weight=[1, 2])


def test_structural_support_and_empty_strata():
    r = structure_errors([1, 2, 4], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 1, 2)
    assert r["below"]["rmse"] is None
    assert r["above"] == {"rows": 1, "rmse": 2.0}


def test_exact_paired_interval_preserves_constant_difference_and_rng():
    np.random.seed(41)
    before = np.random.get_state()
    r = paired_interval([2, 3, 4, 5, 6], [1, 2, 3, 4, 5])
    assert r["percentile95"] == [1.0, 1.0]
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])
