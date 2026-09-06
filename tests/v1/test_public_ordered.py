"""Ordered development package against independent two-parameter references."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from openboost import NumericData, Problem, RunContext, recipes
from openboost.binning import Binning
from openboost.objectives import Normal, diagonal_direction
from openboost.runtime import initialize
from openboost.tree import depthwise
from tests.v1.reference.coupled import formula as reference_formula
from tests.v1.reference.coupled import normal as reference_normal
from tests.v1.reference.coupled import step
from tests.v1.test_public_extensions import ROOT, load


def extension():
    return load(ROOT / "ordered_updates/src/ob_ordered_updates/__init__.py", "ordered")


def problem(family):
    x = NumericData([[0], [1], [2], [3], [4], [np.nan]], np.arange(6), ("feature",))
    return Problem(
        x,
        [[0.5], [1], [2], [3], [5], [8]],
        x.row_ids,
        weight=[1, 0, 2, 1, 3, 1],
        raw_width=2,
        offset=np.column_stack((np.arange(6) / 20, np.arange(6) / 50)),
        structure={"x": np.linspace(0.2, 2, 6)[:, None]} if family == "formula" else {},
    )


@pytest.mark.parametrize(
    "family,mode", [("normal", "natural"), ("normal", "ordinary"), ("formula", "full")]
)
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_three_rounds_match_independent_reference(family, mode, order):
    mod = extension()
    p = problem(family)
    kwargs = {"mode": mode} if family == "normal" else {}
    fit = getattr(mod, family)(
        p, p, context=RunContext("ordered", 7), rounds=3, bins=6, order=order, **kwargs
    )
    binned = Binning.fit(p.data, bins=6).transform(p.data)
    bins = np.where(binned.missing.T, np.nan, binned.codes.T)
    raw = np.broadcast_to(fit.state.model.base, p.offset.shape).copy()
    objective = (
        reference_normal
        if family == "normal"
        else partial(reference_formula, x=p.structure["x"][:, 0])
    )

    def offset_objective(values, target, *, weight):
        return objective(values + p.offset, target, weight=weight)

    accepted_count = 0
    for sweep in fit.steps:
        assert tuple(s.channel for s in sweep) == order
        for actual in sweep:
            expected = step(
                bins,
                raw,
                p.target[:, 0],
                offset_objective,
                weight=p.weight,
                mode="ordinary" if mode == "ordinary" else "full",
                damping=0.1 if family == "formula" else 0.0,
                channels=(actual.channel,),
                rates=tuple(0.1 * 0.5**j for j in range(6)),
            )
            np.testing.assert_allclose(actual.before.train_raw, raw, atol=1e-10)
            np.testing.assert_allclose(actual.after.train_raw, expected.raw_after, atol=1e-10)
            assert actual.coefficients == tuple(t[0] for t in expected.trials)
            assert (actual.after is not actual.before) == expected.accepted
            accepted_count += expected.accepted
            raw = np.array(expected.raw_after)
        assert sweep[1].before is sweep[0].after
    assert fit.state.version == accepted_count
    assert fit.stop.completed_rounds == 3
    assert fit.state.best_score == pytest.approx(
        min(
            [fit.steps[0][0].before.best_score] + [s.after.best_score for r in fit.steps for s in r]
        )
    )


def test_reversed_order_is_a_different_algorithm():
    mod = extension()
    for family in ("normal", "formula"):
        p = problem(family)
        forward = getattr(mod, family)(p, p, context=RunContext("order", 3), order=(0, 1))
        reverse = getattr(mod, family)(p, p, context=RunContext("order", 3), order=(1, 0))
        assert not np.allclose(forward.state.train_raw, reverse.state.train_raw)
        joint = getattr(recipes, family)(p, p, context=RunContext("order", 3))
        assert not np.allclose(forward.state.train_raw, joint.state.train_raw)


@pytest.mark.parametrize("failure", ["reverse", "nan"])
def test_rejected_candidates_preserve_state_best_and_rng(failure):
    mod = extension()
    p = problem("normal")
    context = RunContext("reject", 19)
    state = initialize(context, p, p, Normal.base(p), score=Normal.loss)
    binned = Binning.fit(p.data, bins=6).transform(p.data)
    seen = []

    def geometry(problem, raw):
        seen.append(raw)
        return Normal.geometry(problem, raw)

    def loss(problem, raw):
        if failure == "nan" and not np.array_equal(raw, state.train_raw):
            return np.nan
        return Normal.loss(problem, raw)

    def learner(data, fields):
        tree = depthwise(data, fields, max_depth=1)
        return replace(tree, value=-tree.value) if failure == "reverse" else tree

    final, steps = mod.sweep(
        state, binned, geometry=geometry, direction=diagonal_direction, loss=loss, learner=learner
    )
    assert final is state
    assert all(s.before is state and s.after is state for s in steps)
    assert all(len(s.coefficients) == 6 for s in steps)
    assert all(raw is state.train_raw for raw in seen)
    if failure == "nan":
        assert all(s.failures == ("ValueError",) * 6 for s in steps)
    np.testing.assert_array_equal(
        context.rng(0, "leaf", "sample").random(5), final.context.rng(0, "leaf", "sample").random(5)
    )


def test_stopping_observes_outer_rounds_not_parameter_commits():
    mod = extension()
    p = problem("normal")
    fit = mod.normal(p, p, context=RunContext("stop", 1), rounds=10, patience=2, min_delta=100)
    assert fit.stop.completed_rounds == 2 and fit.stop.reason == "patience"
    assert len(fit.steps) == 2 and fit.state.version == 4
    assert fit.state.best_score < fit.stop.reference_score


def test_rejected_first_parameter_does_not_prevent_second_acceptance():
    mod = extension()
    p = problem("normal")
    state = initialize(RunContext("partial", 1), p, p, Normal.base(p), score=Normal.loss)
    binned = Binning.fit(p.data, bins=6).transform(p.data)
    calls = []

    def learner(data, fields):
        tree = depthwise(data, fields)
        calls.append(fields)
        return replace(tree, value=-tree.value) if len(calls) == 1 else tree

    final, steps = mod.sweep(
        state,
        binned,
        geometry=Normal.geometry,
        direction=diagonal_direction,
        loss=Normal.loss,
        learner=learner,
    )
    assert steps[0].after is state and steps[1].before is state
    assert final.version == 1 and steps[1].after is final
    assert len(steps[0].coefficients) == 6


def test_nonfinite_first_trial_can_recover_with_smaller_step():
    mod = extension()
    p = problem("normal")
    state = initialize(RunContext("recover", 2), p, p, Normal.base(p), score=Normal.loss)
    binned = Binning.fit(p.data, bins=6).transform(p.data)
    seen = []

    def loss(problem, raw):
        seen.append(raw)
        return np.nan if len(seen) == 1 else Normal.loss(problem, raw)

    final, steps = mod.sweep(
        state, binned, geometry=Normal.geometry, direction=diagonal_direction, loss=loss
    )
    assert steps[0].coefficients == (0.1, 0.05)
    assert steps[0].failures == ("ValueError", None)
    assert final.version == 2 and steps[1].before is steps[0].after


@pytest.mark.parametrize("order", [(0,), (0, 0), (False, True), (0, 2)])
def test_invalid_order_rejected_at_zero_rounds(order):
    p = problem("normal")
    with pytest.raises(ValueError, match="permutation"):
        extension().normal(p, p, context=RunContext("bad", 1), rounds=0, order=order)
