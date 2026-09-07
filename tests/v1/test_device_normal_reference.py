"""Normal 090 fixtures checked before K=2 device construction."""

from functools import partial

import numpy as np
import pytest

from openboost import NumericData, Problem, RunContext
from openboost.artifacts import TreeTerm
from openboost.binning import Binning
from openboost.objectives import Normal, diagonal_direction
from openboost.ops import candidates, feasible, histogram, score
from openboost.runtime import initialize, preview_raw, propose_terms, resolve
from openboost.stats import RowFields, least_squares
from openboost.stopping import StopState
from openboost.tree import depthwise

from .reference.coupled import normal_scores
from .reference.device_normal import base, direction, fixture, geometry, rounds, trial
from .reference.device_splits import enumerate_candidates


def prepared_fixture(case):
    f = fixture(case)
    names = tuple(f"f{i}" for i in range(f["x"].shape[1]))
    problems = []
    for prefix, start in (("", 101), ("validation_", 301)):
        data = NumericData(f[prefix + "x"], start + 7 * np.arange(len(f[prefix + "x"])), names)
        problems.append(
            Problem(
                data,
                f[prefix + "target"][:, None],
                data.row_ids,
                weight=f[prefix + "weight"],
                offset=f[prefix + "offset"],
                raw_width=2,
            )
        )
    cuts = tuple(
        np.arange(int(np.nanmax(col))) + 0.5 if np.any(~np.isnan(col)) else np.array([])
        for col in f["x"].T
    )
    return *problems, Binning(names, cuts).transform(problems[0].data), f["information"]


def test_offset_base_is_stationary_before_scale_floor():
    f = fixture("weighted")
    initial = base(f["target"], f["offset"], f["weight"])
    raw = np.broadcast_to(initial, f["offset"].shape)
    _, gradient, fisher = geometry(raw, f["target"], f["offset"], f["weight"])
    np.testing.assert_allclose(f["weight"] @ gradient, [0, 0], atol=1e-12)
    assert np.all(fisher > 0)


@pytest.mark.parametrize("case", ["weighted", "d2", "conflict"])
def test_geometry_finite_differences_and_fisher_are_independent(case):
    train, _, _, _ = prepared_fixture(case)
    f = fixture(case)
    initial = base(f["target"], f["offset"], f["weight"])
    np.testing.assert_allclose(Normal.base(train), initial, atol=1e-12)
    raw = np.broadcast_to(initial, f["offset"].shape).copy()
    raw[:, 0] += np.arange(len(raw)) / 7
    raw[:, 1] -= np.arange(len(raw)) / 20
    loss, gradient, fisher = geometry(raw, f["target"], f["offset"], f["weight"])
    actual = Normal.geometry(train, raw)
    assert actual[0] == pytest.approx(loss, abs=1e-12)
    np.testing.assert_allclose(actual[1], gradient, atol=1e-12)
    np.testing.assert_allclose(actual[2], fisher, atol=1e-12)
    # Derivatives of each individual likelihood, including zero-weight rows.
    for row in range(len(raw)):
        for k in range(2):
            values = []
            for sign in (-1, 1):
                perturbed = raw[row : row + 1].copy()
                perturbed[0, k] += sign * 1e-5
                values.append(
                    geometry(
                        perturbed, f["target"][row : row + 1], f["offset"][row : row + 1], [1]
                    )[0]
                )
            assert (values[1] - values[0]) / 2e-5 == pytest.approx(
                gradient[row, k], rel=1e-7, abs=1e-9
            )
    np.testing.assert_array_equal(fisher[:, 1], 2)
    # Observed log-scale Hessian is 2*r^2*p, which is generally different.
    assert not np.allclose(fisher[:, 1], 2 * (1 - gradient[:, 1]))
    for mode, damping in (("ordinary", 0), ("natural", 0), ("natural", 0.25)):
        expected = direction(gradient, fisher, mode=mode, damping=damping)
        np.testing.assert_allclose(
            diagonal_direction(*actual[1:], mode=mode, damping=damping), expected, atol=1e-12
        )
    nll, crps = normal_scores(raw + f["offset"], f["target"], weight=f["weight"])
    assert nll == pytest.approx(loss, abs=1e-12) and np.isfinite(crps)


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
def test_three_round_public_transactions(case, depth, minimum, mode, damping, update, fixed, rate):
    train, validation, binned, information = prepared_fixture(case)
    initial, expected = rounds(
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
                value = Normal.loss(train, preview_raw(state, proposal)[0])
                accept = fixed or value < loss
                attempts.append((coefficient, value, "accepted" if accept else "rejected"))
                state = resolve(state, proposal, accept=accept, score=Normal.loss)
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


def test_frozen_nontrivial_decisions():
    for mode in ("ordinary", "natural"):
        coefficients = (8, 4, 2)
        step = rounds("weighted", mode=mode, rate=8, count=1, depth=1)[1][0]
        assert tuple(a[0] for a in step["attempts"]) == coefficients
        assert tuple(a[2] for a in step["attempts"]) == ("rejected",) * (len(coefficients) - 1) + (
            "accepted",
        )
        assert step["version"] == 1 and step["nterms"] == 2
        assert step["best_terms"] == 0
    plain = rounds("d2", count=1)[1][0]
    cohort = rounds("d2", minimum=1, count=1)[1][0]
    assert [n[0]["key"] for n in plain["nodes"]] == [(0, 0, False)] * 2
    assert [n[0]["key"] for n in cohort["nodes"]] == [(0, 1, False)] * 2
    results = [
        rounds("weighted", update=update, mode="ordinary")[1][-1]
        for update in ("joint", "forward", "reverse")
    ]
    assert [s["version"] for s in results] == [3, 6, 6]
    for i, left in enumerate(results):
        for right in results[i + 1 :]:
            assert not np.allclose(left["raw"], right["raw"])
    joint = rounds("weighted", update="joint")[1][-1]["raw"]
    reverse = rounds("weighted", update="reverse")[1][-1]["raw"]
    # With no damping, the natural mean direction cancels precision. The
    # algorithms agree mathematically here, up to float64 evaluation rounding.
    np.testing.assert_allclose(joint, reverse, rtol=0, atol=1e-14)


def test_actual_invalid_trial_recovery_and_full_rejection():
    # Deliberately off-optimum accepted state: first scale increase is invalid,
    # third is finite and lowers actual NLL. This is a float64 domain fixture.
    raw, offset, weight = np.zeros((2, 2)), np.zeros((2, 2)), np.ones(2)
    target = np.full(2, 100.0)
    delta = np.tile([100, 5000], (2, 1))
    after, attempts = trial(raw, delta, target, offset, weight, rate=0.2)
    assert [a[2] for a in attempts] == ["invalid", "invalid", "accepted"]
    assert tuple(a[0] for a in attempts) == (0.2, 0.1, 0.05)
    assert geometry(after, target, offset, weight)[0] < geometry(raw, target, offset, weight)[0]
    rejected, attempts = trial(raw, np.zeros_like(raw), target, offset, weight)
    assert rejected is raw and [a[2] for a in attempts] == ["rejected"] * 6
    np.testing.assert_array_equal(raw, 0)


def test_initial_floor_and_zero_weight_invalid_row_are_explicit():
    offset = np.array([[0.25, 0.125], [-0.25, -0.125]])
    target = 2 + offset[:, 0]
    np.testing.assert_allclose(base(target, offset, [1, 0], minimum_scale=0.01), [2, np.log(0.01)])
    with pytest.raises((ValueError, OverflowError)):
        geometry(np.array([[0, 0], [0, -1000.0]]), [0, 0], np.zeros((2, 2)), [1, 0])


def test_preserved_zero_weight_split_ambiguity():
    # Captured G/H inputs from round 2, mean first, natural damping .25 at
    # 48a1386. This records a numerical limitation, not a passing CUDA oracle.
    train, _, binned, _ = prepared_fixture("weighted_ties")
    gradient = np.array(
        [
            0,
            0.042141210703726076,
            0.07697524756812128,
            0.044312313692044324,
            0,
            2.0064142014852084,
            -2.0121120875434078,
            0.7221702433864421,
        ]
    )
    # Build through the public field boundary without applying weight again.
    fields = RowFields(
        train.identity,
        train.data.identity,
        ("gradient", "curvature"),
        np.column_stack((gradient, train.weight)),
        ("training", "training"),
    )
    keys = ((0, 0, True), (0, 3, False))
    by_key = {c.key: c for c in candidates(histogram(binned, fields, np.arange(len(gradient))))}
    first, second = (by_key[k] for k in keys)
    assert abs(score(first) - score(second)) <= 2 * np.spacing(score(first))
    # Empty objective mass on row 0 allows equal mathematical scores with
    # different full-row partitions. Do not call these prediction-equivalent.
    options, _ = enumerate_candidates(train.data.values, fields.values, range(len(gradient)))
    rows = {c["key"]: c["rows"] for c in options}
    assert rows[keys[0]] == ([0, 1, 6], [2, 3, 4, 5, 7])
    assert rows[keys[1]] == ([0, 2, 3, 4, 5, 7], [1, 6])
    first_leaf = -first.left[0] / (first.left[1] + 1)
    second_leaf = -second.left[0] / (second.left[1] + 1)
    assert abs(first_leaf - second_leaf) > 0.5


def test_ordered_zero_mean_rejects_before_scale_accepts():
    data = NumericData([[0], [0]], [101, 301], ("x",))
    problem = Problem(data, [[-1], [1]], data.row_ids, raw_width=2)
    state = initialize(RunContext("090-partial", 7), problem, problem, [0, 1], score=Normal.loss)
    binned = Binning(("x",), (np.array([]),)).transform(data)
    raw = np.array([[0.0, 1.0], [0.0, 1.0]])
    for k in (0, 1):
        before = state
        _, g, h = geometry(raw, [-1, 1], np.zeros((2, 2)), [1, 1])
        z = direction(g, h)
        delta = np.zeros_like(raw)
        delta[:, k] = sum(z[:, k]) / 3  # Original rows, lambda=1.
        expected, attempts = trial(raw, delta, [-1, 1], np.zeros((2, 2)), [1, 1])
        _, actual_g, actual_h = Normal.geometry(problem, state.train_raw)
        actual_z = diagonal_direction(actual_g, actual_h)
        tree = depthwise(binned, least_squares(problem, actual_z[:, k]), max_depth=0)
        actual_attempts = []
        loss = Normal.loss(problem, state.train_raw)
        for j in range(6):
            coefficient = 0.1 * 0.5**j
            proposal = propose_terms(state, (TreeTerm(tree, np.eye(2)[k : k + 1], coefficient),))
            accept = Normal.loss(problem, preview_raw(state, proposal)[0]) < loss
            state = resolve(state, proposal, accept=accept, score=Normal.loss)
            actual_attempts.append(coefficient)
            if accept:
                break
        assert actual_attempts == [a[0] for a in attempts]
        np.testing.assert_allclose(state.train_raw, expected, atol=1e-12)
        assert (state is before) == (k == 0)
        raw = expected
    assert state.version == 1 and len(state.model.terms) == 1
    assert state.best_model is state.model


def test_zero_round_oracle_does_not_construct_terms():
    initial, steps = rounds("weighted", count=0)
    assert initial.shape == (2,) and steps == []


def test_reference_runs_without_production_imports():
    import subprocess
    import sys
    from pathlib import Path

    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'openboost' or fullname.startswith('openboost.'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
from tests.v1.reference.device_normal import rounds
for update in ('joint', 'forward', 'reverse'):
    base, steps = rounds('weighted', update=update, rate=8)
    assert len(base) == 2 and steps[-1]['version'] > 0
assert not any(n == 'openboost' or n.startswith('openboost.') for n in sys.modules)
print('independent-normal-ok')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(Path(__file__).resolve().parents[2])],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "independent-normal-ok"
