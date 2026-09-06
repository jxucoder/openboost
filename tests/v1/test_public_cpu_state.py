"""Independent hand cases for initial public ownership/state/inference contracts."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from openboost.artifacts import ConstantTerm, Model
from openboost.data import NumericData, Problem
from openboost.runtime import RunContext, initialize, preview, propose, resolve


def problem():
    data = NumericData([[1], [np.nan]], [11, 12], ("x",))
    return Problem(data, [[5], [7]], [11, 12], weight=[1, 3], offset=[[2], [4]])


def score(p, raw):
    return np.dot(p.weight, ((p.with_offset(raw) - p.target) ** 2)[:, 0]) / p.weight.sum()


def test_owned_inputs_cannot_be_mutated_or_reenabled():
    x = np.array([[1.0], [2.0]])
    y = np.array([[3.0], [4.0]])
    data = NumericData(x, [1, 2], ("x",))
    p = Problem(data, y, [1, 2])
    identity = p.identity
    x[:] = 100
    y[:] = 100
    np.testing.assert_array_equal(data.values[:, 0], [1, 2])
    np.testing.assert_array_equal(p.target[:, 0], [3, 4])
    assert p.identity == identity
    for a in [data.values, data.row_ids, p.target, p.weight, p.offset]:
        with pytest.raises(ValueError):
            a.flags.writeable = True


def test_identity_includes_order_content_and_roles():
    a = NumericData([[1], [2]], [1, 2], ("x",))
    b = NumericData([[1], [2]], [2, 1], ("x",))
    assert a.identity != b.identity
    assert a.identity == NumericData([[1], [2]], [1, 2], ("x",)).identity
    assert (
        Problem(a, [[1], [2]], [1, 2]).identity
        != Problem(a, [[1], [2]], [1, 2], weight=[1, 2]).identity
    )
    with pytest.raises(ValueError, match="row order"):
        Problem(a, [[1], [2]], [2, 1])


def test_rejection_commit_best_and_offset_once():
    p = problem()
    initial = initialize(RunContext("one", 3), p, p, [0], score=score)
    first = propose(initial, [6], coefficient=0.5)
    np.testing.assert_array_equal(
        preview(initial, first).predict(p.data, offset=p.offset), [[5], [7]]
    )
    assert (
        resolve(initial, first, accept=False, score=lambda *_: pytest.fail("rejection scored"))
        is initial
    )
    accepted = resolve(initial, first, accept=True, score=score)
    assert accepted.best_score == 0 and accepted.version == 1
    np.testing.assert_array_equal(initial.train_raw, [[0], [0]])
    np.testing.assert_array_equal(accepted.train_raw, [[3], [3]])
    worse = resolve(accepted, propose(accepted, [2]), accept=True, score=score)
    np.testing.assert_array_equal(worse.train_raw, [[5], [5]])
    np.testing.assert_array_equal(p.with_offset(worse.train_raw), [[7], [9]])
    assert worse.best_model is accepted.model and worse.best_score == 0
    assert score(p, worse.validation_raw) == 4  # original weights applied once
    with pytest.raises(ValueError, match="stale"):
        resolve(accepted, first, accept=True, score=score)


def test_foreign_and_divergent_proposals_and_keyed_rng():
    p = problem()
    a = initialize(RunContext("a", 3), p, p, [0], score=score)
    b = initialize(RunContext("b", 3), p, p, [0], score=score)
    proposal = propose(a, [1])
    with pytest.raises(ValueError, match="foreign"):
        resolve(b, proposal, accept=True, score=score)
    x = resolve(a, proposal, accept=True, score=score)
    y = resolve(a, propose(a, [2]), accept=True, score=score)
    with pytest.raises(ValueError, match="foreign"):
        resolve(y, propose(x, [1]), accept=True, score=score)
    expected = a.context.rng(1, "rows", "sample").integers(1000, size=8)
    resolve(a, proposal, accept=False, score=score)
    np.testing.assert_array_equal(
        expected, a.context.rng(1, "rows", "sample").integers(1000, size=8)
    )
    assert not np.array_equal(expected, b.context.rng(1, "rows", "sample").integers(1000, size=8))


def test_invalid_score_is_atomic():
    p = problem()
    a = initialize(RunContext("a", 1), p, p, [0], score=score)
    with pytest.raises(ValueError, match="finite"):
        resolve(a, propose(a, [2]), accept=True, score=lambda *_: np.nan)
    assert a.version == 0 and not a.model.terms
    with pytest.raises(ValueError):
        propose(a, [1, 2])


def test_fresh_process_inference_and_artifact_corruption(tmp_path):
    model = Model(("x",), [1, 2], (ConstantTerm([2, 4], 0.5),))
    path = tmp_path / "model.json"
    model.save(path)
    code = """import sys, numpy as np
from openboost.artifacts import Model
from openboost.data import NumericData
m=Model.load(sys.argv[1])
x=NumericData([[float('nan')],[3]],[11,12],('x',))
np.testing.assert_array_equal(m.predict(x,offset=[[1,2],[3,4]]),[[3,6],[5,8]])
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-c", code, str(path)], check=True, env=env, timeout=30)
    raw = json.loads(path.read_text())
    raw["terms"][0]["value"] = [1]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        Model.load(path)
    raw["format"] = "unknown"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="schema"):
        Model.load(path)


@pytest.mark.parametrize("kwargs", [dict(device="cuda"), dict(seed=-1), dict(run_id="")])
def test_explicit_runtime_rejection(kwargs):
    with pytest.raises(ValueError):
        RunContext(**(dict(run_id="a", seed=1) | kwargs))


def test_weight_semantics_and_separate_raw_caches():
    p = problem()
    assert score(p, np.array([[0.0], [1.0]])) == 5.25
    valid = Problem(NumericData([[9]], [99], ("x",)), [[3]], [99])
    state = initialize(RunContext("two-datasets", 7), p, valid, [1], score=score)
    next_state = resolve(state, propose(state, [2]), accept=True, score=score)
    np.testing.assert_array_equal(next_state.train_raw, [[3], [3]])
    np.testing.assert_array_equal(next_state.validation_raw, [[3]])
    assert not np.shares_memory(next_state.train_raw, next_state.validation_raw)
    assert next_state.best_score == 0


def test_vector_proposal_is_atomic_and_owned():
    p = Problem(NumericData([[1]], [1], ("x",)), [[2, 4]], [1])

    def metric(p, raw):
        return float(np.sum((raw - p.target) ** 2))

    state = initialize(RunContext("vector", 7), p, p, [0, 0], score=metric)
    values = np.array([2.0, 4.0])
    proposal = propose(state, values)
    values[:] = 100
    result = resolve(state, proposal, accept=True, score=metric)
    np.testing.assert_array_equal(result.train_raw, [[2, 4]])
    assert result.version == 1 and result.best_score == 0
    with pytest.raises(ValueError):
        propose(result, [0, np.nan])
    np.testing.assert_array_equal(result.train_raw, [[2, 4]])


@pytest.mark.parametrize("change", ["unknown", "duplicate", "nan", "wrong_width"])
def test_corrupt_artifacts_fail_closed(tmp_path, change):
    model = Model(("x",), [0], (ConstantTerm([1]),))
    path = tmp_path / "m.json"
    model.save(path)
    record = json.loads(path.read_text())
    if change == "unknown":
        record["hidden"] = 1
    elif change == "nan":
        record["base"] = [float("nan")]
    elif change == "wrong_width":
        record["terms"][0]["value"] = [1, 2]
    path.write_text(json.dumps(record))
    if change == "duplicate":
        path.write_text(path.read_text().replace('"format":', '"base": [1], "format":'))
    with pytest.raises(ValueError):
        Model.load(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(weight=[0, 0]),
        dict(weight=[1, -1]),
        dict(offset=[1, 2]),
        dict(row_ids=[11.0, 12.0]),
        dict(target=[[np.nan], [1]]),
    ],
)
def test_problem_rejects_invalid_roles(kwargs):
    p = problem()
    args = dict(data=p.data, target=[[1], [2]], row_ids=[11, 12]) | kwargs
    with pytest.raises(ValueError):
        Problem(**args)


def test_keyed_stream_does_not_consume_global_rng():
    before = np.random.get_state()
    RunContext("isolated", 3).rng(0, "tree", "rows").normal(size=10)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
