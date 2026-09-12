"""118 public CPU transaction consumers with independent comparison mathematics."""

import subprocess
import sys

import numpy as np
import pytest

from openboost.artifacts import TreeTerm
from openboost.comparison import LossChange
from openboost.objectives import MultiSquared
from openboost.ops import vector_feasible, vector_leaf, vector_score
from openboost.runtime import RunContext, initialize, preview_raw, propose_terms, resolve
from openboost.stats import vector_newton
from openboost.stopping import StopState
from openboost.tree import depthwise

from .reference import multi_squared as ref
from .test_multi_squared_reference import prepared


def exact_compare(problem, before, after):
    exact = ref.change(before, after, problem.target, problem.offset, problem.weight)
    if np.array_equal(before, after):
        return LossChange(0, 0, "test-rational", "identical", unchanged=True)
    value = float(exact)
    return LossChange(
        np.nextafter(value, -np.inf), np.nextafter(value, np.inf), "test-rational", "enclosed"
    )


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("mode", ["independent", "shared", "projected"])
@pytest.mark.parametrize("step,rate", [("fixed", 0.5), ("backtracking", 8)])
def test_joint_cpu_transactions_follow_exact_trajectory(width, mode, step, rate, tmp_path):
    train, validation, binning = prepared(width)
    _, expected = ref.rounds(width, mode, depth=1, step=step, rate=rate)
    state = initialize(
        RunContext("118-cpu", 17),
        train,
        validation,
        MultiSquared.base(train),
        score=MultiSquared.loss,
    )
    for known in expected:
        g = MultiSquared.gradient(train, state.train_raw)
        terms = []
        for channel in range(width) if mode == "independent" else (None,):
            gradient = g if channel is None else g[:, channel : channel + 1]
            leaves = vector_newton(train, gradient, np.ones_like(gradient))
            fields = (
                vector_newton(train, g[:, :1], np.ones_like(g[:, :1]))
                if mode == "projected"
                else leaves
            )
            tree = depthwise(
                binning.transform(train.data),
                fields,
                max_depth=1,
                leaf_fields=leaves,
                scoring=vector_score,
                legality=vector_feasible,
                leaf=vector_leaf,
            )
            mapping = np.eye(width) if channel is None else np.eye(width)[channel : channel + 1]
            terms.append((tree, mapping))
        trials = []
        for attempt in range(1 if step == "fixed" else 6):
            coefficient = rate * 0.5**attempt
            proposal = propose_terms(state, tuple(TreeTerm(t, m, coefficient) for t, m in terms))
            raw, _ = preview_raw(state, proposal)
            accepted = step == "fixed" or exact_compare(train, state.train_raw, raw).improves()
            state = resolve(
                state, proposal, accept=accepted, score=MultiSquared.loss, compare=exact_compare
            )
            trials.append((coefficient, accepted))
            if accepted:
                break
        assert trials == [(coefficient, accept) for coefficient, accept, _ in known["trials"]]
        assert len(state.best_model.terms) == known["best_prefix"]
        np.testing.assert_allclose(state.train_raw, known["raw"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(state.validation_raw, known["validation"], rtol=1e-4, atol=1e-5)
    for name, model in (("final", state.model), ("best", state.best_model)):
        path, inputs, output = (
            tmp_path / (name + ".json"),
            tmp_path / "x.npy",
            tmp_path / (name + ".npy"),
        )
        model.save(path)
        np.save(inputs, validation.data.values)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "\n".join(
                    [
                        "import sys, numpy as np; sys.modules['cupy']=None; sys.modules['numba']=None",
                        "from openboost.artifacts import Model",
                        "from openboost.data import NumericData",
                        "x=np.load(sys.argv[2]); d=NumericData(x,np.arange(len(x)),('a','b'))",
                        "np.save(sys.argv[3],Model.load(sys.argv[1]).predict(d))",
                    ]
                ),
                str(path),
                str(inputs),
                str(output),
            ],
            check=True,
        )
        np.testing.assert_array_equal(np.load(output), model.predict(validation.data))


def test_anchor_fixture_has_distinct_best_patience_and_noop_decisions():
    from openboost.data import NumericData, Problem

    data = NumericData([[0], [0]], [0, 1], ("x",))
    problem = Problem(data, [[0, 0], [2, 0]], data.row_ids, raw_width=2)
    raw = np.tile(np.array([2, 0], np.float32), (2, 1))
    anchor = raw.copy()
    policy = StopState.start(MultiSquared.loss(problem, raw), rounds=3, patience=2, min_delta=0.1)
    changes = []
    for delta in (-0.1, -0.1, 0):
        candidate = ref.mapped(
            raw,
            np.tile(np.array([delta, 0], np.float32), (2, 1)),
            np.eye(2, dtype=np.float32),
            np.float32(1),
        )
        current = exact_compare(problem, anchor, candidate)
        changes.append(current)
        policy = policy.observe_change(MultiSquared.loss(problem, candidate), current)
        if current.improves(0.1):
            anchor = candidate.copy()
        raw = candidate
    assert [c.improves(0.1) for c in changes] == [False, True, False]
    assert changes[-1].unchanged and policy.stale_rounds == 1
