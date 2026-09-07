"""Independent likelihood differences, including a retained-input counterexample."""

from decimal import Decimal

import numpy as np
import pytest
from benchmarks.v1.analyze_normal_neighbors import analyze

from .reference.device_normal import geometry
from .reference.normal_acceptance import compare_precisions, loss_difference


@pytest.mark.parametrize("mean,expected", [(0, 0), (1, 0.5), (-2, 2)])
def test_analytic_mean_difference_and_reverse(mean, expected):
    before, after = np.zeros((1, 2)), np.array([[mean, 0]])
    args = ([0], [[0, 0]], [1])
    assert loss_difference(before, after, *args) == Decimal(expected)
    assert loss_difference(after, before, *args) == -Decimal(expected)


def test_weights_offsets_and_zero_weight_row_are_used_once():
    before, after = np.zeros((3, 2)), np.array([[1, 0], [1, 0], [50, 0]])
    target, offset, weight = [0, 2, 0], [[0, 0], [1, 0], [0, 0]], [1, 3, 0]
    assert loss_difference(before, after, target, offset, weight) == Decimal("-0.25")
    assert loss_difference(before, after, target, offset, np.array(weight) * 7) == Decimal("-0.25")
    order = [2, 0, 1]
    args = [np.asarray(a)[order] for a in (before, after, target, offset, weight)]
    assert loss_difference(*args) == Decimal("-0.25")


def test_log_scale_offset_and_joint_update():
    # y = initial mean, then dmean=1, dlogscale=.25, with log-scale offset=.5.
    before, after = [[0, 0]], [[1, 0.25]]
    delta = loss_difference(before, after, [2], [[2, 0.5]], [3])
    assert float(delta) == pytest.approx(0.25 + np.exp(-1.5) / 2, abs=1e-15)
    result = compare_precisions(before, after, [2], [[2, 0.5]], [3])
    assert result["estimates_agree"] and result["signs"] == [1, 1]


def test_tiny_true_improvement_is_retained_when_full_losses_are_equal():
    before, after = np.array([[2.0**-30, 0]]), np.zeros((1, 2))
    args = ([0], np.zeros((1, 2)), [1])
    assert geometry(before, *args)[0] == geometry(after, *args)[0]
    assert loss_difference(before, after, *args) == Decimal.from_float(-(2.0**-61))
    assert compare_precisions(before, after, *args)["signs"] == [-1, -1]


@pytest.mark.parametrize(
    "bad", ["shape", "target", "offset", "negative", "zero", "nan", "precision"]
)
def test_invalid_diagnostic_inputs_are_explicit(bad):
    before, after, target, offset, weight = np.zeros((1, 2)), np.zeros((1, 2)), [0], [[0, 0]], [1]
    precision = 80
    if bad == "shape":
        before = np.zeros(2)
    elif bad == "target":
        target = [[0]]
    elif bad == "offset":
        offset = [[0]]
    elif bad == "negative":
        weight = [-1]
    elif bad == "zero":
        weight = [0]
    elif bad == "nan":
        after[0, 1] = np.nan
    else:
        precision = 20
    with pytest.raises(ValueError):
        loss_difference(before, after, target, offset, weight, precision=precision)


def test_recorded_device_base_neighbors_reverse_float64_improvement_sign():
    report = analyze()
    assert report["device_execution"] is False
    assert len(report["neighbors"]) == 4
    for row in report["neighbors"]:
        assert row["high_precision"]["estimates_agree"]
        assert row["high_precision"]["signs"] == [1, 1]
        assert (row["float64_nll_difference"] < 0) == (row["channel"] == 0)
    assert report["neighbors"][0]["float64_nll_difference"] == -np.spacing(report["float64_nll"])
