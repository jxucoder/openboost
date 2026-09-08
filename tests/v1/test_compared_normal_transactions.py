"""Current CPU Normal transactions against the independent comparison oracle.

All ninety original 090 settings retain their geometry/tree/prediction checks.
The immutable full-loss cohort remains in test_device_normal_reference.py and
can be explicitly selected with --include-historical-normal for historical study.
"""

from functools import partial

import numpy as np
import pytest

from openboost import RunContext
from openboost.artifacts import TreeTerm
from openboost.objectives import Normal, diagonal_direction
from openboost.ops import feasible
from openboost.runtime import initialize, preview_raw, propose_terms, resolve
from openboost.stats import least_squares
from openboost.stopping import StopState
from openboost.tree import depthwise

from .reference.compared_normal import rounds as compared_rounds
from .reference.coupled import normal_scores
from .test_device_normal_reference import prepared_fixture


@pytest.mark.parametrize(
    "case,depth,minimum",
    [
        ("weighted", 1, None),
        ("d2", 1, None),
        ("d2", 2, 1),
        ("conflict", 0, None),
        ("conflict", 2, None),
    ],
)
@pytest.mark.parametrize("mode,damping", [("ordinary", 0), ("natural", 0), ("natural", 0.25)])
@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
@pytest.mark.parametrize("fixed,rate", [(True, 0.1), (False, 8.0)])
def test_three_round_compared_public_transactions(
    case, depth, minimum, mode, damping, update, fixed, rate,
):
    train, validation, binned, information = prepared_fixture(case)
    initial, expected = compared_rounds(
        case,
        mode=mode,
        damping=damping,
        update=update,
        depth=depth,
        minimum=minimum,
        fixed=fixed,
        rate=rate,
    )
    context = RunContext("090-normal", 7)
    state = initialize(context, train, validation, Normal.base(train), score=Normal.loss)
    np.testing.assert_allclose(state.model.base, initial, atol=1e-12)
    stop = StopState.start(state.best_score, rounds=3)
    groups = {"joint": ((0, 1),), "forward": ((0,), (1,)), "reverse": ((1,), (0,))}[update]
    cursor = 0
    for _ in range(3):
        for channels in groups:
            ref = expected[cursor]
            cursor += 1
            before = state
            loss, gradient, fisher = Normal.geometry(train, state.train_raw)
            z = diagonal_direction(gradient, fisher, mode=mode, damping=damping)
            for actual, key in ((gradient, "gradient"), (fisher, "fisher"), (z, "direction")):
                np.testing.assert_allclose(actual, ref[key], atol=1e-11)
            learners = []
            for i, k in enumerate(channels):
                fields = least_squares(train, z[:, k])
                fields = fields.add_independent("a", information[:, 0]).add_independent(
                    "b", information[:, 1]
                )
                np.testing.assert_allclose(fields.values, ref["fields"][i], atol=1e-11)
                tree = depthwise(
                    binned,
                    fields,
                    max_depth=depth,
                    legality=partial(feasible, min_information={"a": minimum, "b": minimum})
                    if minimum is not None
                    else feasible,
                )
                keys = [
                    None if feature == -1 else (int(feature), int(threshold), bool(missing))
                    for feature, threshold, missing in zip(
                        tree.feature, tree.threshold, tree.missing_left, strict=True
                    )
                ]
                assert keys == [node["key"] for node in ref["nodes"][i]]
                np.testing.assert_allclose(
                    tree.value[:, 0], [node["value"] for node in ref["nodes"][i]], atol=1e-11
                )
                learners.append((k, tree))
            attempts = []
            for j in range(1 if fixed else 6):
                coefficient = rate * 0.5**j
                terms = tuple(
                    TreeTerm(tree, np.eye(2)[k : k + 1], coefficient) for k, tree in learners
                )
                proposal = propose_terms(state, terms)
                candidate_raw = preview_raw(state, proposal)[0]
                value = Normal.loss(train, candidate_raw)
                change = Normal.compare(train, state.train_raw, candidate_raw)
                accept = fixed or change.improves()
                attempts.append((coefficient, value, "accepted" if accept else "rejected"))
                state = resolve(
                    state, proposal, accept=accept, score=Normal.loss, compare=Normal.compare,
                )
                if accept:
                    break
                assert state is before
            assert [a[::2] for a in attempts] == [a[::2] for a in ref["attempts"]]
            np.testing.assert_allclose(
                [a[1] for a in attempts], [a[1] for a in ref["attempts"]], rtol=1e-10, atol=1e-11
            )
            np.testing.assert_allclose(state.train_raw, ref["raw"], atol=1e-11)
            np.testing.assert_allclose(state.validation_raw, ref["validation_raw"], atol=1e-11)
            np.testing.assert_array_equal(state.model.predict(train.data), state.train_raw)
            assert state.version == ref["version"]
            assert len(state.model.terms) == ref["nterms"]
            assert len(state.best_model.terms) == ref["best_terms"]
            assert state.best_score == pytest.approx(ref["best_score"], abs=1e-11)
            assert Normal.loss(train, state.train_raw) == pytest.approx(ref["loss"], abs=1e-11)
        stop = stop.observe(Normal.loss(validation, state.validation_raw))
    assert stop.completed_rounds == 3 and stop.reason == "budget"
    np.testing.assert_array_equal(
        context.rng(0, "tree", "rows").integers(100, size=8),
        state.context.rng(0, "tree", "rows").integers(100, size=8),
    )
    final = expected[-1]
    nll, crps = normal_scores(
        state.train_raw + train.offset, train.target[:, 0], weight=train.weight
    )
    assert nll == pytest.approx(final["loss"]) and np.isfinite(crps)
