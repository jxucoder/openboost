"""Distinguishing numerical tests of 092-A; no production or CUDA execution."""

import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from benchmarks.v1.normal_acceptance_trace import unpack

from .reference.normal_acceptance import compare_precisions, loss_difference
from .reference.normal_comparison import Interval, UnresolvedArithmetic, compare, exponential

ROOT = Path(__file__).resolve().parents[2]


def assert_oracle_enclosed(result, args):
    oracle = compare_precisions(*args)
    values = [Decimal(oracle[f"decimal{digits}"]) for digits in (60, 100)]
    if not oracle["estimates_agree"]:
        # Quadratic-scale cancellation consumes ~32 digits of the original-row
        # oracle. Increase its precision, never tune the binary64 interval policy.
        values += [loss_difference(*args, precision=p) for p in (160, 220)]
        with localcontext() as context:
            context.prec = 240
            assert abs(values[-1] - values[-2]) <= max(map(abs, values[-2:])) * Decimal("1e-40")
        assert [int(v > 0) - int(v < 0) for v in values] == [oracle["signs"][0]] * len(values)
    assert result.lower is not None and result.upper is not None
    for exact in values:
        assert Decimal.from_float(result.lower) <= exact <= Decimal.from_float(result.upper)
    return oracle


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_actual_failed_run7_trials_have_bounded_correct_sign(order):
    trace = json.loads(
        (
            ROOT / "benchmarks/v1/evidence/cuda-acceptance-091" / f"normal/acceptance/{order}.json"
        ).read_text()
    )
    changed, unchanged = 0, 0
    for step in trace["steps"]:
        for trial in step["trials"]:
            for dataset in ("training", "validation"):
                data = trace["inputs"][dataset]
                args = (
                    unpack(step["before"][f"{dataset}_raw"]),
                    unpack(trial["proposal"][f"{dataset}_raw"]),
                    unpack(data["target"]).ravel(),
                    unpack(data["offset"]),
                    unpack(data["weight"]),
                )
                result = compare(*args)
                oracle = assert_oracle_enclosed(result, args)
                if dataset == "training":
                    same = np.array_equal(args[0], args[1])
                    assert result.status == ("unchanged" if same else "worsening")
                    assert not result.improves()
                    changed += not same
                    unchanged += same
                    assert oracle["signs"] == ([0, 0] if same else [1, 1])
                elif result.status != "unchanged":
                    # A candidate can worsen training while improving validation.
                    assert result.improves() == (oracle["signs"] == [-1, -1])
    assert (changed, unchanged) == ((2, 0) if order == "forward" else (5, 3))


def test_tiny_true_improvement_survives_and_threshold_is_strict():
    args = ([[2.0**-30, 0]], [[0, 0]], [0], [[0, 0]], [1])
    result = compare(*args)
    assert_oracle_enclosed(result, args)
    assert result.status == "improvement" and result.improves()
    assert result.improves(2.0**-62)
    assert not result.improves(2.0**-61)
    assert loss_difference(*args) == Decimal.from_float(-(2.0**-61))


@pytest.mark.parametrize("channel", [0, 1, "joint"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_clear_changes_offsets_weights_permutation_and_rescaling(channel, dtype):
    before = np.zeros((3, 2), dtype=dtype)
    after = np.array([[1, 0.25], [1, 0.25], [100, -1]], dtype=dtype)
    if channel != "joint":
        after[:, 1 - channel] = before[:, 1 - channel]
    target, offset, weight = (
        np.array([0, 2, 0]),
        np.array([[0, 0.125], [1, 0.25], [0, 0]]),
        [1, 3, 0],
    )
    args = before, after, target, offset, np.array(weight)
    for order in ([0, 1, 2], [2, 0, 1]):
        for scale in (1, 7, 2.0**-200, 2.0**200):
            permuted = [np.asarray(a)[order] for a in args]
            permuted[-1] = permuted[-1] * scale
            result = compare(*permuted)
            oracle = assert_oracle_enclosed(result, permuted)
            assert result.status == ("improvement" if oracle["signs"][0] < 0 else "worsening")
            reverse = (permuted[1], permuted[0], *permuted[2:])
            assert_oracle_enclosed(compare(*reverse), reverse)


def test_cancellation_returns_unresolved_without_inventing_an_epsilon():
    # At the scale optimum, this nonzero stored update has an O(dl^2) loss change.
    args = ([[0, 0]], [[0, 2.0**-52]], [1], [[0, 0]], [1])
    result = compare(*args)
    oracle = assert_oracle_enclosed(result, args)
    assert oracle["signs"] == [1, 1]
    assert result.status == "unresolved" and result.reason == "contains_zero"
    assert not result.improves()


def test_unchanged_is_about_stored_raw_not_equal_weighted_losses():
    before = np.zeros((2, 2), dtype=np.float32)
    before[0, 0] = 1
    after = before.copy()
    after[0, 0] += np.float32(2.0**-30)
    assert compare(before, after, [0, 0], np.zeros((2, 2)), [1, 0]).status == "unchanged"
    after[1, 0] = 1
    result = compare(before, after, [0, 0], np.zeros((2, 2)), [1, 0])
    assert result.status == "unresolved" and result.lower == result.upper == 0


@pytest.mark.parametrize("bad", ["shape", "target", "offset", "negative", "zero", "nan"])
def test_invalid_inputs_are_not_reported_unchanged(bad):
    args = [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2), np.zeros((2, 2)), np.array([1.0, 0])]
    if bad == "shape":
        args[0] = np.zeros(2)
    elif bad == "target":
        args[2] = np.zeros((2, 1))
    elif bad == "offset":
        args[3] = np.zeros((2, 1))
    elif bad == "negative":
        args[4][1] = -1
    elif bad == "zero":
        args[4][:] = 0
    else:
        args[1][1, 0] = np.nan
    with pytest.raises(ValueError):
        compare(*args)


@pytest.mark.parametrize("kind", ["old", "new", "offset", "overflow"])
def test_zero_weight_outside_support_remains_unresolved(kind):
    args = [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2), np.zeros((2, 2)), [1, 0]]
    if kind == "overflow":
        args[0][1, 0] = 1e300
    else:
        args[{"old": 0, "new": 1, "offset": 3}[kind]][1, 1] = 33
    result = compare(*args)
    assert result.status == "unresolved"
    assert result.reason == ("arithmetic_range" if kind == "overflow" else "exponent_range")
    assert result.estimate is result.uncertainty is result.lower is result.upper is None


def test_basic_intervals_enclose_exact_rational_arithmetic_including_subnormals():
    rng = np.random.default_rng(9201)
    values = [0.0, 1.0, -1.0, 2.0**-1074, -(2.0**-1074), 2.0**-1022, 2.0**1023]
    values += [
        math.ldexp(float(rng.uniform(-1, 1)), int(rng.integers(-1000, 1000))) for _ in range(30)
    ]
    for a, b in zip(values, reversed(values), strict=True):
        x, y = Interval.point(a), Interval.point(b)
        fa, fb = Fraction(a), Fraction(b)
        for operation in ("add", "sub", "mul", "truediv"):
            if operation == "truediv" and b == 0:
                continue
            exact = getattr(fa, f"__{operation}__")(fb)
            try:
                bound = getattr(x, f"__{operation}__")(y)
            except UnresolvedArithmetic:
                assert abs(exact) > Fraction(2.0**1023)
                continue
            assert Fraction(bound.lower) <= exact <= Fraction(bound.upper)


@pytest.mark.parametrize("minus_one", [False, True])
def test_derived_exponential_enclosures_against_correctly_rounded_decimal(minus_one):
    values = np.linspace(-64, 64, 129).tolist() + [-(2.0**-52), 2.0**-52, 0.0]
    for value in values:
        bound = exponential(Interval.point(value), minus_one=minus_one)
        with localcontext() as context:
            context.prec = 100
            exact = Decimal.from_float(value).exp() - int(minus_one)
        assert Decimal.from_float(bound.lower) <= exact <= Decimal.from_float(bound.upper)
    with pytest.raises(UnresolvedArithmetic, match="exponent_range"):
        exponential(Interval.point(math.nextafter(64, math.inf)), minus_one=minus_one)


def test_seeded_original_row_inputs_validate_whole_expression_enclosure():
    rng = np.random.default_rng(9202)
    for dtype in (np.float32, np.float64):
        for exponent in (0, -10, -30, -50):
            for _ in range(4):
                before = rng.normal(size=(7, 2)).astype(dtype)
                after = (before + 2.0**exponent * rng.normal(size=(7, 2))).astype(dtype)
                args = (
                    before,
                    after,
                    rng.normal(size=7).astype(dtype),
                    rng.normal(size=(7, 2)).astype(dtype),
                    rng.uniform(0.1, 3, 7).astype(dtype),
                )
                result = compare(*args)
                assert_oracle_enclosed(result, args)
                if result.improves():
                    assert loss_difference(*args) < 0
