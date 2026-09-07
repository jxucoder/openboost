"""Public objective comparison contracts, independent of rounded full losses."""

import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from openboost import LossChange, NumericData, Problem
from openboost.objectives import Normal

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / "benchmarks/v1/evidence/normal-comparison-092/study.json").read_text())[
    "cases"
]


def problem_for(target, offset, weight):
    data = NumericData(np.zeros((len(target), 1)), np.arange(len(target)), ("x",))
    return Problem(
        data,
        np.asarray(target).reshape(-1, 1),
        data.row_ids,
        offset=offset,
        weight=weight,
        raw_width=2,
    )


@pytest.mark.parametrize(
    "name", ["run7/forward/channel0/alpha4.0/training", "analytic/tiny-improvement"]
)
def test_public_normal_comparison_resolves_changes_hidden_by_full_loss_rounding(name):
    study = json.loads(
        (ROOT / "benchmarks/v1/evidence/normal-comparison-092/study.json").read_text()
    )
    case = next(row for row in study["cases"] if row["id"] == name)
    values = {key: np.asarray(record["values"]) for key, record in case["inputs"].items()}
    p = problem_for(values["target"], values["offset"], values["weight"])
    result = Normal.compare(p, values["before"], values["after"])
    assert result.status == case["comparison"]["status"]
    assert result.improves() == (name == "analytic/tiny-improvement")


@pytest.mark.parametrize("case", CASES, ids=[row["id"] for row in CASES])
def test_public_cpu_operation_encloses_every_independent_stored_input_case(case):
    arrays = {key: np.asarray(value["values"]) for key, value in case["inputs"].items()}
    p = problem_for(arrays["target"], arrays["offset"], arrays["weight"])
    before, after = arrays["before"], arrays["after"]
    copies = before.copy(), after.copy()
    result = Normal.compare(p, before, after)
    assert isinstance(result, LossChange)
    assert result.status == case["comparison"]["status"]
    assert result.reason == case["comparison"]["reason"]
    if result.lower is not None:
        reference = case["high_precision"]
        expected = [reference[f"decimal{digits}"] for digits in (60, 100)]
        if "higher_precision" in reference:
            expected += [reference["higher_precision"][f"decimal{digits}"] for digits in (160, 220)]
        for value in expected:
            assert (
                Decimal.from_float(result.lower)
                <= Decimal(value)
                <= Decimal.from_float(result.upper)
            )
        assert result.lower <= result.estimate <= result.upper
        assert result.uncertainty >= 0
    else:
        assert result.estimate is result.uncertainty is None
    np.testing.assert_array_equal(before, copies[0])
    np.testing.assert_array_equal(after, copies[1])


def test_comparison_does_not_read_loss_or_geometry_callbacks(monkeypatch):
    p = problem_for([0], [[0, 0]], [1])

    def forbidden(*args, **kwargs):
        raise AssertionError("reporting metrics and gradient callbacks are independent")

    monkeypatch.setattr(Normal, "loss", forbidden)
    monkeypatch.setattr(Normal, "geometry", forbidden)
    assert Normal.compare(p, [[1, 0]], [[0, 0]]).improves()


@pytest.mark.parametrize(
    "kind",
    ["nan", "shape", "scale-overflow", "scale-underflow", "residual-overflow", "wrong-problem"],
)
def test_every_row_is_validated_before_unchanged_or_unresolved(kind):
    p = problem_for([0, 0], np.zeros((2, 2)), [1, 0])
    before, after = np.zeros((2, 2)), np.zeros((2, 2))
    before[0, 1] = after[0, 1] = 33
    if kind == "nan":
        after[1, 0] = np.nan
    elif kind == "shape":
        after = after[:, :1]
    elif kind == "scale-overflow":
        after[1, 1] = 1000
    elif kind == "scale-underflow":
        after[1, 1] = -1000
    elif kind == "residual-overflow":
        after[1, 0] = 1e300
    else:
        p = Problem(p.data, p.target, p.row_ids)
    # Unsupported comparison range on an earlier row must not mask a later
    # invalid zero-weight row.
    with pytest.raises((ValueError, FloatingPointError)):
        Normal.compare(p, before, after)


def test_weights_offsets_and_independent_buffers_cannot_change_the_result():
    before, after = np.array([[0.0, 0], [0, 0]]), np.array([[1.0, 0], [1, 0]])
    for order in ([0, 1], [1, 0]):
        for factor in (1, 7, 2.0**100, 2.0**-100):
            p = problem_for(
                np.array([0, 2])[order],
                np.array([[0, 0], [1, 0]])[order],
                np.array([1, 3])[order] * factor,
            )
            result = Normal.compare(p, before[order], after[order])
            assert result.improves() and result.lower <= -0.25 <= result.upper


@pytest.mark.parametrize(
    "changes",
    [
        {"lower": None},
        {"upper": None},
        {"lower": 2},
        {"upper": float("inf")},
        {"lower": float("nan")},
        {"lower": True},
        {"reason": ""},
        {"method": ""},
        {"unchanged": 1},
        {"unchanged": True},
    ],
)
def test_inconsistent_public_comparison_records_fail(changes):
    with pytest.raises(ValueError):
        replace(LossChange(-1, -0.5, "author-method", "bounded"), **changes)


def test_record_is_immutable_and_thresholds_are_strict():
    result = LossChange(-1, -0.5, "author-method", "bounded")
    assert result.status == "improvement" and result.improves(0.25)
    assert not result.improves(0.5)
    with pytest.raises(FrozenInstanceError):
        result.lower = 0
    for threshold in (True, -1, float("nan"), float("inf"), "0"):
        with pytest.raises(ValueError):
            result.improves(threshold)
    assert LossChange(None, None, "author-method", "unsupported").status == "unresolved"
    assert LossChange(0, 0, "author-method", "equal-loss").status == "unresolved"
    assert LossChange(0, 0, "author-method", "identical-raw", True).status == "unchanged"
