"""CPU consumers distinguish objective evidence from rounded reporting scores."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import LossChange, RunContext, recipes
from openboost.artifacts import ConstantTerm
from openboost.objectives import Normal
from openboost.runtime import initialize, preview_raw, propose, propose_terms, resolve
from openboost.stopping import StopState
from openboost.tree import depthwise
from tests.v1.test_loss_change import CASES, problem_for


def zero_problem():
    return problem_for([0], [[0, 0]], [1])


def test_equal_reported_loss_can_advance_best_and_patience():
    p = zero_problem()
    state = initialize(RunContext("tiny", 1), p, p, [2.0**-30, 0], score=Normal.loss)
    candidate = propose(state, [-2.0**-30, 0])
    raw, _ = preview_raw(state, candidate)
    score = Normal.loss(p, raw)
    assert score == state.best_score
    change = Normal.compare(p, state.train_raw, raw)
    assert change.improves()
    updated = resolve(state, candidate, accept=True, score=Normal.loss, compare=Normal.compare)
    assert updated.best_model is updated.model
    assert updated.best_model is not state.best_model
    assert updated.best_score == state.best_score == score
    stop = StopState.start(score, rounds=3, patience=1).observe_change(score, change)
    assert stop.completed_rounds == 1 and stop.stale_rounds == 0 and stop.reason is None
    assert stop.reference_score == score
    assert state.version == 0 and not state.model.terms


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_captured_false_improvement_is_rejected_by_trial_consumer(order):
    case = next(c for c in CASES if c["id"] == f"run7/{order}/channel0/alpha4.0/training")
    arrays = {key: np.array(value["values"]) for key, value in case["inputs"].items()}
    p = problem_for(arrays["target"], arrays["offset"], arrays["weight"])
    before, after = arrays["before"], arrays["after"]
    state = initialize(RunContext(order, 7), p, p, before[0], score=Normal.loss)
    np.testing.assert_array_equal(state.train_raw, before)
    term = ConstantTerm(after[0] - before[0])
    candidate = propose(state, term.value)
    np.testing.assert_array_equal(preview_raw(state, candidate)[0], after)
    assert Normal.loss(p, after) < Normal.loss(p, before)
    updated, coefficients, accepted, failures, changes = recipes._trials(
        state, (term,), Normal.loss, Normal.loss(p, before), 1, "backtracking", 1,
        compare=Normal.compare,
    )
    assert updated is state and not accepted
    assert coefficients == (1,) and failures == (None,)
    assert changes[0].status == "worsening"


def test_current_best_and_patience_use_three_distinct_anchors():
    p = zero_problem()
    state = initialize(RunContext("anchors", 3), p, p, [2, 0], score=Normal.loss)
    stop = StopState.start(state.best_score, rounds=5, patience=5, min_delta=0.2)
    anchor = state.validation_raw
    best_means, reference_means, stale = [], [], []
    for mean in (3, 1.95, 1.97, 1.9, 1.89):
        previous = state
        candidate = propose(state, [mean - state.train_raw[0, 0], 0])
        state = resolve(state, candidate, accept=True, score=Normal.loss, compare=Normal.compare)
        change = Normal.compare(p, anchor, state.validation_raw)
        stop = stop.observe_change(Normal.loss(p, state.validation_raw), change)
        if change.improves(stop.min_delta):
            anchor = state.validation_raw
        best_means.append(state.best_model.predict(p.data)[0, 0])
        reference_means.append(anchor[0, 0])
        stale.append(stop.stale_rounds)
        assert previous.version == state.version - 1
    np.testing.assert_allclose(best_means, [2, 1.95, 1.95, 1.9, 1.89], rtol=0, atol=1e-15)
    np.testing.assert_allclose(reference_means, [2, 2, 2, 2, 1.89], rtol=0, atol=1e-15)
    assert stale == [1, 2, 3, 4, 0]
    assert stop.reason == "budget"


@pytest.mark.parametrize(
    "update,expected_prefixes",
    [
        ("joint", [0, 4, 4, 8, 10]),
        ("forward", [0, 0, 3, 3, 3, 3, 7, 7, 9, 9]),
        ("reverse", [0, 0, 0, 4, 4, 4, 4, 8, 8, 10]),
    ],
)
def test_accepted_noop_preserves_best_prefix_at_each_commit_boundary(update, expected_prefixes):
    p = zero_problem()
    state = initialize(RunContext(update, 3), p, p, [2, 0], score=Normal.loss)
    prefixes = []
    for mean in (3, 1.95, 1.97, 1.9, 1.89):
        delta = np.float32(mean) - np.float32(state.train_raw[0, 0])
        terms = (ConstantTerm([delta, 0]), ConstantTerm([0, 0]))
        if update == "reverse":
            terms = terms[::-1]
        batches = (terms,) if update == "joint" else tuple((term,) for term in terms)
        for batch in batches:
            previous = state
            candidate = propose_terms(state, batch)
            raw, _ = preview_raw(state, candidate)
            unchanged = np.array_equal(raw, state.train_raw)
            state = resolve(state, candidate, accept=True, score=Normal.loss, compare=Normal.compare)
            assert state.version == previous.version + 1
            if unchanged:
                assert state.best_model is previous.best_model
            prefixes.append(len(state.best_model.terms))
    # With target/scale fixed at zero/one, mean squared determines the strict
    # ranking. Joint commits expose only even prefixes; forward's last term is zero.
    assert prefixes == expected_prefixes
    assert len(state.model.terms) == 10
    np.testing.assert_array_equal(state.best_model.predict(p.data), state.train_raw)


@pytest.mark.parametrize("retention", ["full", "summary"])
def test_normal_recipe_replaces_equal_score_patience_anchor_and_retains_evidence(
    monkeypatch, retention,
):
    p = zero_problem()
    v = replace(p, data=replace(p.data, row_ids=[99]), row_ids=[99])
    tiny = 2.0**-30
    # A controlled valid starting predictor isolates the recipe's consumers from
    # base estimation. The objective, candidate evaluation and comparisons are real.
    monkeypatch.setattr(Normal, "base", lambda *a, **k: np.array([2 * tiny, 0]))
    leaves = iter([-tiny, 0, -tiny, 0])

    def learner(binned, fields):
        leaf = next(leaves)
        return depthwise(binned, fields, max_depth=0, leaf=lambda *_: leaf)

    original = Normal.compare
    validation_anchors = []

    def compare(problem, before, after):
        if problem is v:
            validation_anchors.append(before.copy())
        return original(problem, before, after)

    monkeypatch.setattr(Normal, "compare", compare)
    result = recipes.normal(
        p, v, context=RunContext("recipe", 1), rounds=2, patience=1,
        learning_rate=1, learner=learner, retention=retention,
    )
    assert result.state.version == 2 and result.stop.reason == "budget"
    assert result.stop.stale_rounds == 0
    assert result.state.best_model is result.state.model
    np.testing.assert_array_equal(result.state.train_raw, [[0, 0]])
    np.testing.assert_array_equal(np.array(validation_anchors)[:, 0, 0], [2*tiny, 2*tiny, tiny, tiny])
    for item in result.steps:
        values = dict(item.values) if retention == "summary" else vars(item)
        assert values["accepted"] and values["loss_before"] == values["loss_after"]
        assert len(values["comparisons"]) == 1 and values["comparisons"][0].improves()
        assert values["validation_change"].improves()


@pytest.mark.parametrize("policy", ["backtracking", "fixed"])
def test_unresolved_trial_has_explicit_policy_and_truthful_report(policy):
    p = zero_problem()
    state = initialize(RunContext("range", 1), p, p, [0, 33], score=Normal.loss)
    updated, coefficients, accepted, failures, changes = recipes._trials(
        state, (ConstantTerm([1, 0]),), Normal.loss, state.best_score, 1, policy, 3,
        compare=Normal.compare,
    )
    assert accepted == (policy == "fixed")
    assert len(changes) == (1 if accepted else 3)
    assert all(change.status == "unresolved" for change in changes)
    assert all(failure is None for failure in failures)
    assert updated.best_model is state.best_model
    assert updated.version == int(accepted)
    assert len(coefficients) == len(changes)


def test_rejection_and_failed_comparison_preserve_parent_and_rng():
    p = zero_problem()
    state = initialize(RunContext("atomic", 1), p, p, [1, 0], score=Normal.loss)
    proposal = propose(state, [-1, 0])
    rng = state.context.rng(0, "leaf", "sample").random(3)

    def broken(*args):
        raise RuntimeError("comparison dispatch failed")

    assert resolve(state, proposal, accept=False, score=Normal.loss, compare=broken) is state
    with pytest.raises(RuntimeError, match="dispatch"):
        resolve(state, proposal, accept=True, score=Normal.loss, compare=broken)
    with pytest.raises(TypeError, match="LossChange"):
        resolve(state, proposal, accept=True, score=Normal.loss, compare=lambda *a: -1)
    with pytest.raises(TypeError, match="callable"):
        resolve(state, proposal, accept=True, score=Normal.loss, compare=1)
    updated = resolve(state, proposal, accept=True, score=Normal.loss, compare=Normal.compare)
    with pytest.raises(ValueError, match="parent"):
        resolve(updated, proposal, accept=False, score=Normal.loss, compare=Normal.compare)
    assert state.version == 0
    np.testing.assert_array_equal(state.train_raw, [[1, 0]])
    np.testing.assert_array_equal(state.context.rng(0, "leaf", "sample").random(3), rng)


def test_stop_change_requires_evidence_and_strict_threshold():
    stop = StopState.start(1, rounds=4, patience=2, min_delta=0.5)
    tie = LossChange(-1, -0.5, "test", "bounded")
    assert stop.observe_change(-100, tie).stale_rounds == 1
    unresolved = LossChange(None, None, "test", "unsupported")
    assert stop.observe_change(-100, unresolved).reference_score == 1
    with pytest.raises(TypeError, match="LossChange"):
        stop.observe_change(0, -1)
    with pytest.raises(ValueError, match="finite"):
        stop.observe_change(float("nan"), tie)
    finished = StopState.start(0, rounds=0)
    with pytest.raises(ValueError, match="finished"):
        finished.observe_change(0, tie)
