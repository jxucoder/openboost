"""Installed public CPU composition of the joint multiclass consumer contract."""

import numpy as np
import pytest

from openboost.artifacts import TreeTerm
from openboost.binning import Binning
from openboost.comparison import LossChange
from openboost.objectives import Multiclass
from openboost.runtime import RunContext, initialize, preview_raw, propose_terms, resolve
from openboost.stats import newton
from openboost.stopping import StopState
from openboost.tree import depthwise

from .reference.multiclass_comparison import ZERO, Interval, row_change
from .reference.multiclass_recipe import SETTINGS, fit
from .test_device_multiclass_reference import fixture
from .test_multiclass_composition import close, fresh_replay, original_rows, topology


def cpu_compare(problem, before, after):
    """Test-supplied comparison uses interval math at actual float64 CPU snapshots."""
    for raw in (before, after):
        Multiclass.geometry(problem, raw)
    total, mass = ZERO, ZERO
    for a, b, y, off, w in zip(
        before, after, problem.target[:, 0], problem.offset, problem.weight, strict=True
    ):
        total += row_change(a, b, y, off) * Interval.point(w)
        mass += Interval.point(w)
    if np.array_equal(before, after):
        return LossChange(0, 0, "test-independent-interval", "identity", unchanged=True)
    bound = total / mass
    return LossChange(bound.lower, bound.upper, "test-independent-interval", "enclosed")


@pytest.mark.parametrize("width", [2, 3, 5])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("step,rate", SETTINGS)
def test_public_joint_transactions_follow_independent_trajectory(width, depth, step, rate, tmp_path):
    train, validation = fixture(width), fixture(width, validation=True)
    expected = fit(
        original_rows(train), original_rows(validation), depth=depth, step=step, rate=rate
    )
    state = initialize(
        RunContext("111-public-composition", 7), train, validation,
        Multiclass.base(train), score=Multiclass.loss,
    )
    policy = StopState.start(state.best_score, rounds=3, patience=2, min_delta=0.01)
    anchor = state.validation_raw
    binning = Binning(("x",), (np.arange(len(train.target) - 1, dtype=float),))
    binned = binning.transform(train.data)
    for ref in expected["history"]:
        assert policy.reason is None
        _, g, h = Multiclass.geometry(train, state.train_raw)
        learners = []
        for channel in range(width):
            fields = newton(train, g[:, channel], h[:, channel])
            close(fields.values, ref["fields"][channel])
            tree = depthwise(binned, fields, max_depth=depth)
            assert topology(tree) == [n["key"] for n in ref["nodes"][channel]]
            close(tree.value[:, 0], [n["value"] for n in ref["nodes"][channel]])
            learners.append(tree)
        actual_trials = []
        for trial in range(1 if step == "fixed" else 6):
            coefficient = rate * 0.5**trial
            proposal = propose_terms(state, tuple(
                TreeTerm(tree, np.eye(width)[j:j + 1], coefficient)
                for j, tree in enumerate(learners)
            ))
            raw, val = preview_raw(state, proposal)
            Multiclass.loss(validation, val)
            accepted = step == "fixed" or cpu_compare(train, state.train_raw, raw).improves()
            updated = resolve(
                state, proposal, accept=accepted, score=Multiclass.loss, compare=cpu_compare
            )
            if not accepted:
                assert updated is state
            actual_trials.append((coefficient, accepted))
            state = updated
            if accepted:
                break
        assert actual_trials == [(t["coefficient"], t["accepted"]) for t in ref["trials"]]
        change = cpu_compare(validation, anchor, state.validation_raw)
        policy = policy.observe_change(Multiclass.loss(validation, state.validation_raw), change)
        if change.improves(policy.min_delta):
            anchor = state.validation_raw
        assert state.version == ref["version"] and len(state.model.terms) == ref["terms"]
        assert len(state.best_model.terms) == ref["best_terms"]
        assert policy.stale_rounds == ref["stale"]
        close(state.train_raw, ref["raw"])
        close(state.validation_raw, ref["val"])
    assert policy.reason == expected["reason"]
    for name, model, reference in (
        ("final", state.model, expected["val"]), ("best", state.best_model, expected["best"])
    ):
        destination = tmp_path / name
        destination.mkdir()
        restored = fresh_replay(model, validation, destination)
        close(restored.predict(validation.data), reference)
