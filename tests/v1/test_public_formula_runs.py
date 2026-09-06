"""Formula geometry and heterogeneous execution against independent references."""

import numpy as np

from openboost import NumericData, Problem


def test_structure_is_separate_owned_and_identity_bound():
    x = NumericData([[0], [1]], [1, 2], ("feature",))
    role = np.array([[1.0], [2.0]])
    p = Problem(x, [[1], [2]], x.row_ids, raw_width=2, structure={"x": role})
    role[:] = 99
    np.testing.assert_array_equal(p.structure["x"], [[1], [2]])
    assert p.data.values.shape == (2, 1)


import json
import subprocess
import sys
from functools import partial

import pytest

from openboost import RunContext
from openboost.artifacts import Model
from openboost.binning import NumericBinning
from openboost.objectives import Formula, Normal, Squared, full_direction
from openboost.recipes import formula, normal, squared
from openboost.runs import RunSpec, run_many
from tests.v1.reference.coupled import directions, formula_base, formula_predict, step
from tests.v1.reference.coupled import formula as ref_formula


def fixture():
    x = NumericData([[0], [1], [2], [3], [np.nan], [5]], np.arange(6), ("feature",))
    return Problem(
        x,
        [[0.5], [1], [2], [2.5], [3], [4]],
        x.row_ids,
        raw_width=2,
        weight=[1, 0, 2, 1, 3, 1],
        structure={"x": np.linspace(0.2, 2, 6)[:, None]},
    )


def test_three_rounds_and_full_geometry_match_independent_reference():
    p = fixture()
    result = formula(p, p, context=RunContext("formula", 7), rounds=3, bins=6)
    base = formula_base(p.target[:, 0], weight=p.weight)
    np.testing.assert_allclose(result.state.model.base, base)
    raw = np.broadcast_to(base, (6, 2)).copy()
    b = NumericBinning.fit(p.data, bins=6).transform(p.data)
    bins = np.where(b.missing.T, np.nan, b.codes.T)
    objective = partial(ref_formula, x=p.structure["x"][:, 0])
    for actual in result.steps:
        loss, gradient, metric = objective(raw, p.target[:, 0], weight=p.weight)
        np.testing.assert_allclose(actual.gradient, gradient)
        np.testing.assert_allclose(actual.metric, metric)
        np.testing.assert_allclose(actual.direction, directions(gradient, metric, damping=0.1))
        ref = step(
            bins,
            raw,
            p.target[:, 0],
            objective,
            weight=p.weight,
            damping=0.1,
            rates=tuple(0.1 * 0.5**j for j in range(6)),
        )
        assert actual.accepted == ref.accepted
        assert actual.coefficients == tuple(t[0] for t in ref.trials)
        np.testing.assert_allclose(actual.raw_before, raw)
        np.testing.assert_allclose(actual.raw_after, ref.raw_after)
        np.testing.assert_allclose([actual.loss_before, actual.loss_after], [loss, ref.loss_after])
        raw = np.asarray(ref.raw_after)
    np.testing.assert_allclose(
        Formula.predict(raw, p.structure["x"])[:, 0],
        formula_predict(raw, p.structure["x"][:, 0])[0],
    )
    assert result.steps[-1].loss_after < result.steps[0].loss_before


def test_structure_offset_identity_and_rank_deficiency():
    p = fixture()
    shifted = Problem(
        p.data,
        p.target,
        p.row_ids,
        raw_width=2,
        weight=p.weight,
        offset=np.full((6, 2), 0.2),
        structure=p.structure,
    )
    raw = np.broadcast_to(Formula.base(p), (6, 2))
    expected = ref_formula(raw + 0.2, p.target[:, 0], p.structure["x"][:, 0], weight=p.weight)
    got = Formula.geometry(shifted, raw)
    for actual, ref in zip(got, expected, strict=True):
        np.testing.assert_allclose(actual, ref)
    assert p.identity != shifted.identity
    with pytest.raises(ValueError, match="positive definite"):
        full_direction(got[1], got[2], damping=0)
    for objective in (Squared, Normal):
        with pytest.raises(ValueError):
            objective.validate(p)
    with pytest.raises(TypeError):
        p.structure["extra"] = np.ones((6, 1))
    with pytest.raises(ValueError):
        p.structure["x"].flags.writeable = True
    changed = Problem(
        p.data,
        p.target,
        p.row_ids,
        raw_width=2,
        weight=p.weight,
        structure={"x": p.structure["x"] + 1},
    )
    assert changed.identity != p.identity and changed.data.identity == p.data.identity


def test_formula_roundtrip_requires_explicit_inference_structure(tmp_path):
    p = fixture()
    model = formula(p, p, context=RunContext("persist", 1), rounds=2).state.model
    path = tmp_path / "formula.json"
    model.save(path)
    assert Model.load(path).identity == model.identity
    code = """import json, sys
from openboost import NumericData
from openboost.artifacts import Model
from openboost.objectives import Formula
x = NumericData([[-10], [10], [float('nan')]], [10, 11, 12], ('feature',))
print(json.dumps(Formula.predict(Model.load(sys.argv[1]).predict(x), [[0.1], [1], [5]]).tolist()))
"""
    output = subprocess.check_output([sys.executable, "-c", code, str(path)], text=True)
    x = NumericData([[-10], [10], [np.nan]], [10, 11, 12], ("feature",))
    np.testing.assert_allclose(
        json.loads(output), Formula.predict(model.predict(x), [[0.1], [1], [5]])
    )


@pytest.mark.parametrize("count", [1, 2, 8])
def test_heterogeneous_runs_match_independent_and_reordered_execution(count):
    p = fixture()
    scalar = Problem(p.data, p.target, p.row_ids, weight=p.weight)
    distribution = Problem(p.data, p.target, p.row_ids, weight=p.weight, raw_width=2)
    jobs = [(squared, scalar), (normal, distribution), (formula, p)]
    specs = tuple(
        RunSpec(
            RunContext(f"run-{i}", 9),
            jobs[i % 3][1],
            jobs[i % 3][1],
            jobs[i % 3][0],
            {"rounds": i % 3, "bins": 6},
        )
        for i in range(count)
    )
    results = run_many(specs)
    reversed_results = {r.run_id: r for r in run_many(reversed(specs))}
    for spec, outcome in zip(specs, results, strict=True):
        independent = spec.recipe(spec.train, spec.validation, context=spec.context, **spec.options)
        assert outcome.error_type is None
        assert outcome.result.state.identity == independent.state.identity
        assert (
            outcome.result.state.identity == reversed_results[outcome.run_id].result.state.identity
        )
        assert len(outcome.result.steps) == spec.options["rounds"]
        assert outcome.result.state.train_raw.shape[1] == spec.train.raw_width
        for other in results:
            if other is not outcome:
                assert not np.shares_memory(
                    outcome.result.state.train_raw, other.result.state.train_raw
                )


def test_failed_run_continues_and_duplicate_ids_fail_before_execution():
    p = fixture()
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("injected failure")

    specs = (
        RunSpec(RunContext("failed", 1), p, p, fail),
        RunSpec(RunContext("good", 1), p, p, formula, {"rounds": 1}),
    )
    outcomes = run_many(specs)
    assert outcomes[0].result is None and outcomes[0].error_type == "RuntimeError"
    assert outcomes[1].result.state.version == 1
    with pytest.raises(ValueError, match="unique"):
        run_many((specs[0], specs[0]))
    assert len(calls) == 1
    with pytest.raises(ValueError):
        run_many(specs, execution="fused")
    foreign = RunSpec(RunContext("foreign", 2), p, p, lambda *a, **kw: outcomes[1].result)
    assert run_many([foreign])[0].error_type == "ValueError"


def test_invalid_structure_and_options_fail():
    p = fixture()
    with pytest.raises(ValueError):
        Problem(p.data, p.target, p.row_ids, structure={"x": [[1]]})
    for structure in ({}, {"x": np.zeros((6, 1))}, {"wrong": np.ones((6, 1))}):
        bad = Problem(p.data, p.target, p.row_ids, raw_width=2, structure=structure)
        with pytest.raises(ValueError):
            formula(bad, bad, context=RunContext("invalid", 1), rounds=0)
    with pytest.raises(ValueError):
        RunSpec(RunContext("invalid", 1), p, p, formula, {"context": "override"})
    options = {"rounds": 0}
    spec = RunSpec(RunContext("owned", 1), p, p, formula, options)
    options["rounds"] = 100
    assert spec.options["rounds"] == 0
