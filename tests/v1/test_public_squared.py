"""Complete public squared recipe compared with independent reference traces."""

import numpy as np

from openboost import NumericData, Problem, RunContext
from openboost.binning import Binning
from openboost.recipes import squared
from tests.v1.reference.tree import boost_squared


def test_two_round_trace_matches_reference():
    data = NumericData([[0], [1], [2], [3], [4], [np.nan]], np.arange(6), ("x",))
    p = Problem(data, [[-4], [-2], [0], [1], [3], [5]], data.row_ids, weight=[1, 0, 2, 1, 3, 1])
    result = squared(p, p, context=RunContext("squared", 3), rounds=3, bins=6)
    b = Binning.fit(data, bins=6).transform(data)
    expected = boost_squared(
        np.where(b.missing.T, np.nan, b.codes.T), p.target[:, 0], weight=p.weight, rounds=3
    )
    assert result.state.version == 3
    assert result.state.model.base[0] == expected.base
    for actual, ref in zip(result.steps, expected.steps, strict=True):
        np.testing.assert_allclose(actual.gradient, ref.gradient)
        np.testing.assert_allclose(actual.raw_before[:, 0], ref.raw_before)
        np.testing.assert_allclose(actual.raw_after[:, 0], ref.raw_after)
        np.testing.assert_allclose(
            [actual.loss_before, actual.loss_after], [ref.loss_before, ref.loss_after]
        )
    np.testing.assert_allclose(
        result.state.train_raw[:, 0], expected.predict(np.where(b.missing.T, np.nan, b.codes.T))
    )


def test_offsets_weights_and_validation_best_are_separate():
    from openboost.objectives import Squared

    x = NumericData([[0], [1], [2], [3]], [1, 2, 3, 4], ("x",))
    offset = np.array([[10], [20], [30], [40]])
    p = Problem(x, [[6], [18], [32], [44]], x.row_ids, weight=[1, 2, 3, 4], offset=offset)
    shifted = Problem(x, p.target - offset, x.row_ids, weight=p.weight)
    valid_x = NumericData([[0], [3]], [91, 92], ("x",))
    valid = Problem(valid_x, [[100], [-100]], valid_x.row_ids)
    a = squared(p, valid, context=RunContext("offset", 1), rounds=3, learning_rate=1)
    b = squared(shifted, valid, context=RunContext("shifted", 1), rounds=3, learning_rate=1)
    np.testing.assert_allclose(a.state.train_raw, b.state.train_raw)
    np.testing.assert_allclose(a.state.model.predict(x, offset=offset), b.state.train_raw + offset)
    assert a.state.best_model is not a.state.model
    candidates = [s.loss_after for s in a.steps]
    assert candidates[-1] < candidates[0]
    assert a.state.best_score <= Squared.loss(valid, a.state.validation_raw)
    assert a.state.train_raw.shape == (4, 1) and a.state.validation_raw.shape == (2, 1)


def test_backtracking_reuses_learner_and_full_rejection_is_atomic():
    from openboost.tree import depthwise

    x = NumericData([[0], [1], [2], [3]], [1, 2, 3, 4], ("x",))
    p = Problem(x, [[-3], [-3], [3], [3]], x.row_ids)
    calls = []

    def fit(binned, fields):
        calls.append(1)
        return depthwise(binned, fields)

    result = squared(
        p,
        p,
        context=RunContext("line-search", 1),
        rounds=1,
        learning_rate=16,
        step="backtracking",
        learner=fit,
    )
    assert len(calls) == 1 and result.steps[0].coefficients == (16, 8, 4, 2)
    assert result.steps[0].accepted and result.state.version == 1
    assert result.state.model.terms[0].coefficient == 2
    rejected = squared(
        p,
        p,
        context=RunContext("rejected", 1),
        rounds=2,
        learning_rate=1024,
        step="backtracking",
        max_trials=6,
        learner=fit,
    )
    assert len(calls) == 3
    assert not rejected.state.model.terms and rejected.state.version == 0
    assert rejected.state.best_model is rejected.state.model
    for step in rejected.steps:
        assert not step.accepted and len(step.coefficients) == 6
        assert step.raw_before is step.raw_after
        assert step.loss_before == step.loss_after


def test_mapped_tree_terms_atomic_and_fresh_process_persistence(tmp_path):
    import subprocess
    import sys

    from openboost.artifacts import ConstantTerm, Model, TreeTerm
    from openboost.runtime import initialize, propose_terms, resolve
    from openboost.stats import newton
    from openboost.tree import depthwise

    x = NumericData([[0], [1], [np.nan]], [1, 2, 3], ("x",))
    scalar = Problem(x, [[0], [0], [0]], x.row_ids)
    b = Binning.fit(x, bins=2).transform(x)
    tree = depthwise(b, newton(scalar, [-2, 0, 2], [1, 1, 1]))
    p = Problem(x, [[1, 2], [3, 4], [5, 6]], x.row_ids)

    def metric(p, raw):
        return float(np.sum((p.with_offset(raw) - p.target) ** 2))

    initial = initialize(RunContext("mapped", 2), p, p, [1, 2], score=metric)
    mapping = np.array([[1.0, -2.0]])
    term = TreeTerm(tree, mapping, 0.5)
    mapping[:] = 99
    proposal = propose_terms(initial, (ConstantTerm([2, 3]), term))
    assert resolve(initial, proposal, accept=False, score=metric) is initial
    final = resolve(initial, proposal, accept=True, score=metric)
    assert final.version == 1 and len(final.model.terms) == 2
    expected = np.array([3, 5]) + 0.5 * tree.predict(x) @ np.array([[1, -2]])
    np.testing.assert_allclose(final.train_raw, expected)
    path = tmp_path / "ensemble.json"
    final.model.save(path)
    restored = Model.load(path)
    assert restored.identity == final.model.identity
    np.testing.assert_allclose(restored.predict(x), expected)
    code = """import json, sys
from openboost import NumericData
from openboost.artifacts import Model
x = NumericData([[-100], [100], [float('nan')]], [11, 12, 13], ('x',))
print(json.dumps(Model.load(sys.argv[1]).predict(x, offset=[[1, 2], [3, 4], [5, 6]]).tolist()))
"""
    import json

    output = subprocess.check_output([sys.executable, "-c", code, str(path)], text=True)
    unseen = NumericData([[-100], [100], [np.nan]], [11, 12, 13], ("x",))
    np.testing.assert_allclose(
        json.loads(output), final.model.predict(unseen, offset=[[1, 2], [3, 4], [5, 6]])
    )


def test_nested_tree_artifact_corruption(tmp_path):
    import json

    import pytest

    from openboost.artifacts import Model

    x = NumericData([[0], [1]], [1, 2], ("x",))
    p = Problem(x, [[-2], [2]], x.row_ids)
    model = squared(p, p, context=RunContext("persist", 1), rounds=1).state.model
    for field, bad in (("mapping", [[1, 2]]), ("coefficient", float("nan")), ("kind", "unknown")):
        record = model.record()
        record["terms"][0][field] = bad
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError):
            Model.load(path)
    record = model.record()
    record["terms"][0]["learner"]["left"][0] = 0
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="cycle"):
        Model.load(path)


def test_zero_rounds_still_validate_options_and_target():
    import pytest

    x = NumericData([[0], [1]], [1, 2], ("x",))
    p = Problem(x, [[-2], [2]], x.row_ids)
    for kwargs in (
        {"rounds": -1},
        {"max_depth": -1},
        {"bins": 0},
        {"reg_lambda": -1},
        {"step": "unknown"},
        {"max_trials": 7},
        {"learning_rate": np.nan},
        {"learner": lambda *_: None, "max_depth": 3},
    ):
        with pytest.raises(ValueError):
            squared(p, p, context=RunContext("invalid", 1), **({"rounds": 0} | kwargs))
    wide = Problem(x, [[1, 2], [3, 4]], x.row_ids)
    with pytest.raises(ValueError, match="scalar"):
        squared(wide, wide, context=RunContext("wide", 1), rounds=0)


def test_direct_proposal_owns_term_sequence():
    import pytest

    from openboost.artifacts import ConstantTerm
    from openboost.runtime import Proposal

    source = [ConstantTerm([1])]
    proposal = Proposal("parent", source)
    source.clear()
    assert len(proposal.terms) == 1
    with pytest.raises(ValueError):
        Proposal("parent", [])
