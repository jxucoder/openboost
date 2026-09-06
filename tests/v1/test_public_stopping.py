"""Independent stopping oracles and public recipe integration."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import ClassSchema, NumericData, Problem, RunContext, recipes
from openboost.binning import Binning, PreparedData
from openboost.objectives import Normal, Squared
from openboost.runs import RunSpec, run_many
from openboost.stopping import StopState


def test_threshold_reference_ties_and_budget():
    state = StopState.start(10.0, rounds=8, patience=2, min_delta=1.0)
    # Neither strict improvement nor the cumulative threshold is confused with
    # best-model selection: 9 ties the threshold, 8.5 crosses it, 8 does not.
    expected = [(9.0, 10.0, 1), (8.5, 8.5, 0), (8.0, 8.5, 1), (8.0, 8.5, 2)]
    for i, (score, reference, stale) in enumerate(expected, 1):
        previous = state
        state = state.observe(score)
        assert state.completed_rounds == i
        assert state.reference_score == reference
        assert state.stale_rounds == stale
        assert previous.completed_rounds == i - 1
    assert state.reason == "patience"
    with pytest.raises(ValueError, match="finished"):
        state.observe(7.0)
    assert StopState.start(1, rounds=0).reason == "budget"
    assert StopState.start(1, rounds=1, patience=1).observe(1).reason == "patience"
    assert StopState.start(1, rounds=1).observe(2).reason == "budget"


@pytest.mark.parametrize(
    "options",
    [
        {"rounds": -1},
        {"rounds": True},
        {"rounds": 1, "patience": 0},
        {"rounds": 1, "patience": True},
        {"rounds": 1, "patience": 1, "min_delta": -1},
        {"rounds": 1, "patience": 1, "min_delta": float("nan")},
        {"rounds": 1, "min_delta": 1},
    ],
)
def test_invalid_policy(options):
    with pytest.raises(ValueError):
        StopState.start(1, **options)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), True])
def test_invalid_observations(score):
    with pytest.raises(ValueError):
        StopState.start(score, rounds=2)
    initial = StopState.start(1, rounds=2)
    with pytest.raises(ValueError):
        initial.observe(score)
    assert initial.completed_rounds == 0


def problem():
    x = NumericData([[0], [1], [2], [3], [4], [5]], np.arange(6), ("x",))
    return Problem(x, [[1], [2], [3], [5], [7], [9]], x.row_ids)


@pytest.mark.parametrize(
    "name",
    [
        "squared",
        "normal",
        "formula",
        "binary",
        "multiclass",
        "ranking",
        "quantile",
        "poisson",
        "gamma",
        "tweedie",
        "aft",
        "multi_squared",
    ],
)
def test_every_recipe_observes_rejected_or_accepted_rounds_and_zero_budget(name):
    p = problem()
    if name in ("normal", "formula"):
        p = replace(
            p,
            raw_width=2,
            offset=np.zeros((6, 2)),
            structure={"x": np.arange(1, 7)[:, None]} if name == "formula" else {},
        )
    elif name in ("binary", "multiclass"):
        k = 2 if name == "binary" else 3
        p = replace(
            p,
            target=(np.arange(6) % k)[:, None],
            classes=ClassSchema(tuple(range(k))),
            raw_width=1 if k == 2 else k,
            offset=np.zeros((6, 1 if k == 2 else k)),
        )
    elif name == "ranking":
        p = replace(
            p,
            target=(np.arange(6) % 3)[:, None],
            structure={"query": [[0], [0], [0], [1], [1], [1]]},
        )
    elif name == "poisson":
        p = replace(p, structure={"exposure": np.ones((6, 1))})
    elif name == "aft":
        p = replace(p, target=np.column_stack((p.target, p.target)), target_kind="event_right")
    elif name == "multi_squared":
        p = replace(
            p,
            target=np.column_stack((p.target, p.target * 2)),
            raw_width=2,
            offset=np.zeros((6, 2)),
        )
    recipe = getattr(recipes, name)
    context = RunContext(name, 3)
    initial = recipe(p, p, context=context, rounds=0, patience=2)
    assert initial.stop.reason == "budget" and initial.steps == ()
    fit = recipe(p, p, context=context, rounds=7, patience=2, learning_rate=0)
    assert fit.stop.reason == "patience" and fit.stop.completed_rounds == 2
    assert len(fit.steps) == 2
    assert fit.stop.last_score == initial.state.best_score
    assert fit.state.best_model.identity == initial.state.best_model.identity
    np.testing.assert_array_equal(fit.state.train_raw, initial.state.train_raw)
    with pytest.raises(ValueError, match="patience"):
        recipe(p, p, context=context, rounds=0, patience=0)


def test_validation_not_training_controls_stopping_and_strict_best_ignores_delta():
    p = problem()
    v = replace(p, target=p.target[::-1])
    context = RunContext("validation", 7)
    fit = recipes.squared(p, v, context=context, rounds=9, patience=2)
    assert fit.stop.reason == "patience" and fit.stop.completed_rounds == 2
    assert fit.state.version == 2
    assert fit.steps[-1].loss_after < fit.steps[0].loss_before
    assert fit.state.best_model.terms == ()
    assert Squared.loss(v, fit.state.validation_raw) > fit.state.best_score
    improving = recipes.squared(p, p, context=context, rounds=9, patience=2, min_delta=100)
    assert improving.stop.completed_rounds == 2
    assert improving.state.best_model.identity == improving.state.model.identity
    assert improving.state.best_score < improving.stop.reference_score


def test_backtracking_trials_do_not_consume_patience_or_mutate_accepted_state():
    p = problem()
    context = RunContext("reject", 12)
    initial = recipes.squared(p, p, context=context, rounds=0)
    fit = recipes.squared(
        p,
        p,
        context=context,
        rounds=8,
        patience=2,
        learning_rate=0,
        step="backtracking",
        max_trials=6,
    )
    assert fit.stop.completed_rounds == 2 and fit.state.version == 0
    assert [len(s.coefficients) for s in fit.steps] == [6, 6]
    assert not any(s.accepted for s in fit.steps)
    assert fit.state.identity == initial.state.identity
    np.testing.assert_array_equal(
        context.rng(2, "leaf", "draw").random(4), fit.state.context.rng(2, "leaf", "draw").random(4)
    )


@pytest.mark.parametrize("count", [1, 8, 32])
def test_shared_runs_different_validation_stops_reorder_retry_and_failures(count, monkeypatch):
    p = problem()
    prepared = PreparedData(p.data, 4)
    specs = []
    expected = {}
    expected_rounds = {}
    for i in range(count):
        train = p if i % 2 == 0 else replace(p, raw_width=2, offset=np.zeros((6, 2)))
        valid = replace(train, target=train.target[::-1])
        recipe, loss = (
            (recipes.squared, Squared.loss) if i % 2 == 0 else (recipes.normal, Normal.loss)
        )
        context = RunContext(f"job-{i}", 42)
        patience = 1 + i % 3
        baseline = recipe(train, valid, context=context, rounds=0, bins=4).state.best_score
        # Independent scan of a fixed-budget trace; stopping cannot alter the
        # algorithm's accepted prefix. No StopState is used for this oracle.
        scores = []
        for end in range(1, 7):
            prefix = recipe(train, valid, context=context, rounds=end, bins=4)
            scores.append(loss(valid, prefix.state.validation_raw))
        stale, best, end = 0, baseline, 6
        for j, score in enumerate(scores, 1):
            stale = 0 if score < best else stale + 1
            best = min(best, score)
            if stale == patience:
                end = j
                break
        expected[context.run_id] = recipe(train, valid, context=context, rounds=end, bins=4)
        expected_rounds[context.run_id] = end
        specs.append(
            RunSpec(
                context,
                train,
                valid,
                recipe,
                {"rounds": 6, "bins": 4, "patience": patience},
                prepared,
            )
        )
    if count > 1:
        assert len(set(expected_rounds.values())) > 1
    bad = RunSpec(
        RunContext("bad", 1),
        p,
        p,
        recipes.squared,
        {"rounds": 6, "bins": 4, "patience": 0},
        prepared,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("shared preparation must not refit")

    monkeypatch.setattr(Binning, "fit", forbidden)
    first = run_many([bad, *specs])
    assert first[0].error_type == "ValueError" and first[0].result is None
    groups = [
        first[1:],
        run_many(reversed(specs)),
        [out for group in (specs[::2], specs[1::2]) for out in run_many(group)],
        run_many(specs),
    ]
    stops = {out.run_id: out.result.stop for out in first[1:]}
    for outcomes in groups:
        for out in outcomes:
            assert out.error_type is None
            ref = expected[out.run_id]
            assert out.result.state.identity == ref.state.identity
            assert out.result.stop.completed_rounds == expected_rounds[out.run_id]
            assert out.result.stop == stops[out.run_id]
            np.testing.assert_array_equal(out.result.state.validation_raw, ref.state.validation_raw)
            assert out.result.state.best_model.identity == ref.state.best_model.identity
    for left, right in zip(first[1:], first[2:], strict=False):
        assert not np.shares_memory(left.result.state.train_raw, right.result.state.train_raw)


def test_nonfinite_validation_failure_is_retained_and_isolated():
    p = problem()
    bad_valid = replace(p, target=np.full((6, 1), 1e200))
    jobs = [
        RunSpec(
            RunContext("bad-metric", 1), p, bad_valid, recipes.squared, {"rounds": 5, "patience": 1}
        ),
        RunSpec(RunContext("good-metric", 1), p, p, recipes.squared, {"rounds": 2, "patience": 1}),
    ]
    with np.errstate(over="raise", invalid="raise"):
        outcomes = run_many(jobs)
    assert outcomes[0].error_type == "FloatingPointError" and outcomes[0].result is None
    assert outcomes[1].error_type is None and outcomes[1].result.stop.reason == "budget"
