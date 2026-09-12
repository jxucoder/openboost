"""Normal exact learner and joint/ordered acceptance consumers."""

import numpy as np
import pytest

from openboost import NumericData, Problem, RunContext
from openboost.recipes import normal


def flat(result):
    return tuple(
        sub for outer in result.steps for sub in (outer if isinstance(outer, tuple) else (outer,))
    )


@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
def test_update_modes_have_explicit_substeps_and_one_outer_observation(update):
    x = NumericData([[0], [1], [2], [3]], [0, 1, 2, 3], ("x",))
    p = Problem(x, [[-3], [-1], [1], [3]], x.row_ids, raw_width=2)
    result = normal(
        p, p, context=RunContext("129-" + update, 1), rounds=2, update=update, step="fixed"
    )
    groups = {"joint": ((0, 1),), "forward": ((0,), (1,)), "reverse": ((1,), (0,))}[update]
    assert (
        len(result.steps) == 2
        and len(flat(result)) == 2 * len(groups)
        and result.stop.completed_rounds == 2
    )
    assert [step.channels for step in flat(result)] == list(groups) * 2
    assert [step.round_index for step in flat(result)] == [i for i in range(2) for _ in groups]
    assert result.state.version == 2 * len(groups)
    for i, step in enumerate(flat(result)):
        assert step.before_version == i and step.after_version == i + 1
        assert (step.validation_change is not None) == ((i + 1) % len(groups) == 0)


from functools import partial

from openboost.binning import PreparedData
from openboost.newton_order import leaf, rank
from openboost.tree import best_first, depthwise, symmetric

from .reference.device_normal import fixture
from .reference.exact_growth import fit as exact_tree
from .reference.exact_normal import run as reference_run


def prepared_case(case):
    f = fixture(case)
    names = tuple("f" + str(i) for i in range(f["x"].shape[1]))
    problems = []
    for prefix, start in [("", 101), ("validation_", 301)]:
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
    prepared = PreparedData(problems[0].data, 4)
    for prefix, problem in zip(("", "validation_"), problems, strict=True):
        b = prepared.binned.binning.transform(problem.data)
        f[prefix + "x"] = [
            [None if b.missing[q, i] else int(b.codes[q, i]) for q in range(len(names))]
            for i in range(len(problem.data.values))
        ]
    return *problems, prepared, f


def assert_tree(tree, nodes):
    keys = [
        None if f == -1 else (int(f), int(t), bool(m))
        for f, t, m in zip(tree.feature, tree.threshold, tree.missing_left, strict=True)
    ]
    assert keys == [n["key"] for n in nodes]
    np.testing.assert_array_equal(tree.left, [n["left"] for n in nodes])
    np.testing.assert_array_equal(tree.right, [n["right"] for n in nodes])
    np.testing.assert_allclose(
        tree.value[:, 0], [float(n["value"]) for n in nodes], rtol=1e-7, atol=1e-9
    )


@pytest.mark.parametrize("case", ["weighted", "weighted_ties", "d2", "conflict"])
@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
@pytest.mark.parametrize("mode,damping", [("ordinary", 0), ("natural", 0), ("natural", 0.25)])
@pytest.mark.parametrize("grow", [depthwise, best_first, symmetric])
@pytest.mark.parametrize("fixed,rate", [(True, 0.1), (False, 8.0)])
def test_complete_three_round_independent_exact_normal_trajectory(
    case, update, mode, damping, grow, fixed, rate
):
    train, valid, prepared, f = prepared_case(case)
    fitted = []

    def learner(b, fields):
        tree = grow(b, fields, max_depth=2, ordering=rank, field_leaf=leaf)
        # Separate stored-input structural check: no tolerance for candidate choices.
        expected = exact_tree(f["x"], fields.values, policy=grow.__name__, depth=2)
        assert_tree(tree, expected)
        np.testing.assert_array_equal(tree.value[:, 0], [float(n["value"]) for n in expected])
        fitted.append(tree)
        return tree

    result = normal(
        train,
        valid,
        context=RunContext("129", 1),
        rounds=3,
        bins=4,
        prepared=prepared,
        update=update,
        mode=mode,
        damping=damping,
        step="fixed" if fixed else "backtracking",
        learning_rate=rate,
        learner=learner,
    )
    expected = reference_run(
        f, policy=grow.__name__, update=update, mode=mode, damping=damping, fixed=fixed, rate=rate
    )
    assert len(flat(result)) == len(expected["steps"])
    cursor = 0
    for actual, wanted in zip(flat(result), expected["steps"], strict=True):
        assert (
            actual.round_index,
            actual.channels,
            actual.before_version,
            actual.after_version,
        ) == (wanted["round"], wanted["channels"], wanted["before_version"], wanted["version"])
        assert actual.accepted == wanted["accepted"]
        assert actual.coefficients == tuple(a[0] for a in wanted["attempts"])
        assert [failure is not None for failure in actual.failures] == [
            label == "invalid" for _, label in wanted["attempts"]
        ]
        np.testing.assert_allclose(
            [actual.loss_before, actual.loss_after],
            [wanted["loss_before"], wanted["loss"]],
            rtol=1e-6,
            atol=1e-8,
        )
        for actual_array, key in [
            (actual.gradient, "gradient"),
            (actual.fisher_diagonal, "fisher"),
            (actual.direction, "direction"),
            (actual.raw_before, "before"),
            (actual.raw_after, "raw"),
        ]:
            np.testing.assert_allclose(actual_array, wanted[key], rtol=1e-6, atol=1e-8)
        for node in wanted["nodes"]:
            assert_tree(fitted[cursor], node)
            cursor += 1
        assert (actual.validation_change is None) == (wanted["validation_change"] is None)
    assert result.stop.completed_rounds == 3 and result.stop.reason == "budget"
    assert len(result.state.best_model.terms) == expected["best_terms"]
    np.testing.assert_allclose(
        result.state.validation_raw, expected["validation"], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        result.state.best_model.predict(valid.data), expected["best"], rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
@pytest.mark.parametrize("retention", ["full", "summary"])
def test_normal_result_preserves_one_payload_per_outer_round(update, retention):
    from openboost.results import validate_result

    p, v, prepared, _ = prepared_case("weighted")
    context = RunContext("129-result", 1)
    result = normal(
        p,
        v,
        context=context,
        bins=4,
        prepared=prepared,
        rounds=2,
        update=update,
        retention=retention,
    )
    assert validate_result(result, context=context, train=p, validation=v) is result


@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
@pytest.mark.parametrize("regularization,minimum,penalty", [(0.0, 0.0, 0.0), (2.5, 1.0, 0.25)])
def test_default_normal_uses_exact_public_operations_with_configured_parameters(
    update, regularization, minimum, penalty, monkeypatch
):
    from openboost import ops

    p, v, prepared, f = prepared_case("weighted_ties")
    original = ops.histogram

    def guard(data, fields, rows=None):
        assert rows is not None and len(rows) == 0, (
            "default Normal performed a floating leaf/split reduction"
        )
        return original(data, fields, rows)

    monkeypatch.setattr(ops, "histogram", guard)
    options = dict(
        context=RunContext("129-default", 1),
        rounds=2,
        bins=4,
        prepared=prepared,
        update=update,
        mode="natural",
        damping=0.25,
        step="fixed",
    )
    actual = normal(
        p, v, reg_lambda=regularization, min_child_h=minimum, split_penalty=penalty, **options
    )
    custom = normal(
        p,
        v,
        learner=partial(
            depthwise,
            ordering=partial(
                rank, reg_lambda=regularization, min_child_h=minimum, split_penalty=penalty
            ),
            field_leaf=partial(leaf, reg_lambda=regularization),
        ),
        **options,
    )
    assert actual.state.identity == custom.state.identity and actual.stop == custom.stop


@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
@pytest.mark.parametrize("retention", ["full", "summary"])
@pytest.mark.parametrize("patience", [2, 5])
def test_ordered_best_prefix_and_patience_use_different_anchors(
    update, retention, patience, monkeypatch
):
    from openboost.objectives import Normal
    from openboost.results import validate_result

    data = NumericData([[0]], [1], ("x",))
    other = NumericData([[0]], [2], ("x",))
    p = Problem(data, [[0]], data.row_ids, raw_width=2)
    v = Problem(other, [[0]], other.row_ids, raw_width=2)
    monkeypatch.setattr(Normal, "base", lambda *a, **k: np.array([2.0, 0.0]))
    groups = {"joint": ((0, 1),), "forward": ((0,), (1,)), "reverse": ((1,), (0,))}[update]
    schedule = [(outer, k) for outer in range(5) for channels in groups for k in channels]
    means = [3.0, 1.95, 1.97, 1.9, 1.89]
    current = 2.0
    calls = []

    def learner(b, fields):
        nonlocal current
        outer, k = schedule[len(calls)]
        calls.append((outer, k))
        value = means[outer] - current if k == 0 else 0.0
        if k == 0:
            current += value
        return depthwise(b, fields, max_depth=0, leaf=lambda *_: value)

    original = Normal.compare
    anchors = []

    def compare(problem, before, after):
        if problem is v:
            anchors.append(before[0, 0])
        return original(problem, before, after)

    monkeypatch.setattr(Normal, "compare", compare)
    context = RunContext("129-anchors", 1)
    result = normal(
        p,
        v,
        context=context,
        rounds=5,
        update=update,
        step="fixed",
        learning_rate=1,
        patience=patience,
        min_delta=0.2,
        learner=learner,
        retention=retention,
    )
    validate_result(result, context=context, train=p, validation=v)
    completed = 2 if patience == 2 else 5
    assert result.stop.completed_rounds == completed and result.stop.reason == (
        "patience" if patience == 2 else "budget"
    )
    assert result.stop.stale_rounds == (2 if patience == 2 else 0)
    assert len(calls) == 2 * completed and len(result.state.model.terms) == 2 * completed
    assert len(result.state.best_model.terms) == 2 * completed - (1 if update == "forward" else 0)
    assert anchors[len(groups) :: len(groups) + 1] == [2.0] * completed
    records = [dict(s.values) if retention == "summary" else vars(s) for s in flat(result)]
    changes = [r["validation_change"] for r in records if r["validation_change"] is not None]
    assert [c.improves(0.2) for c in changes] == (
        [False, False] if patience == 2 else [False] * 4 + [True]
    )
    assert all(r["accepted"] for r in records)
    for r in records:
        if r["channels"] == (1,):
            assert r["comparisons"][0].unchanged


@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
@pytest.mark.parametrize("retention", ["full", "summary"])
def test_rejected_zero_channel_keeps_state_and_later_channel_can_accept(
    update, retention, monkeypatch
):
    from openboost.objectives import Normal

    data = NumericData([[0], [0]], [1, 2], ("x",))
    p = Problem(data, [[-1], [1]], data.row_ids, raw_width=2)
    monkeypatch.setattr(Normal, "base", lambda *a, **k: np.array([0.0, 1.0]))
    result = normal(
        p,
        p,
        context=RunContext("129-reject", 1),
        rounds=1,
        update=update,
        max_depth=0,
        retention=retention,
    )
    assert result.state.version == 1 and result.stop.completed_rounds == 1
    records = [dict(s.values) if retention == "summary" else vars(s) for s in flat(result)]
    assert [r["accepted"] for r in records] == {
        "joint": [True],
        "forward": [False, True],
        "reverse": [True, False],
    }[update]
    assert [len(r["coefficients"]) for r in records] == {
        "joint": [1],
        "forward": [6, 1],
        "reverse": [1, 6],
    }[update]
    for r in records:
        if not r["accepted"]:
            assert r["before_version"] == r["after_version"]
            assert all(c.unchanged for c in r["comparisons"])
            if retention == "full":
                assert r["raw_before"] is r["raw_after"]
    assert result.state.best_model is result.state.model


@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
@pytest.mark.parametrize("retention", ["full", "summary"])
def test_reporting_ties_still_update_ordered_best_and_patience(update, retention, monkeypatch):
    from openboost.objectives import Normal

    data = NumericData([[0]], [1], ("x",))
    p = Problem(data, [[0]], data.row_ids, raw_width=2)
    tiny = 2.0**-30
    monkeypatch.setattr(Normal, "base", lambda *a, **k: np.array([2 * tiny, 0.0]))
    schedule = iter([k for _ in range(2) for k in ((1, 0) if update == "reverse" else (0, 1))])

    def learner(b, fields):
        value = -tiny if next(schedule) == 0 else 0
        return depthwise(b, fields, max_depth=0, leaf=lambda *_: value)

    result = normal(
        p,
        p,
        context=RunContext("129-tiny", 1),
        rounds=2,
        patience=1,
        learning_rate=1,
        update=update,
        learner=learner,
        retention=retention,
    )
    assert (
        result.stop.reason == "budget"
        and result.stop.stale_rounds == 0
        and result.state.version == 2
    )
    np.testing.assert_array_equal(result.state.train_raw, [[0.0, 0.0]])
    assert result.state.best_model is result.state.model
    for item in flat(result):
        r = dict(item.values) if retention == "summary" else vars(item)
        assert r["loss_before"] == r["loss_after"]
        if r["validation_change"] is not None:
            assert r["validation_change"].improves()
        if r["accepted"]:
            assert r["comparisons"][-1].improves()


@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
def test_zero_rounds_validate_and_do_not_fit(update):
    p, v, prepared, _ = prepared_case("weighted")

    def forbidden(*a, **k):
        raise AssertionError("zero-round tree fitting")

    result = normal(
        p, v, context=RunContext("129-zero", 1), rounds=0, update=update, learner=forbidden
    )
    assert not result.steps and not result.state.model.terms and result.stop.reason == "budget"
    for kwargs in (
        {"update": "bad"},
        {"mode": "bad"},
        {"damping": -1},
        {"min_child_h": -1},
        {"reg_lambda": -1},
        {"split_penalty": float("nan")},
        {"max_depth": -1},
    ):
        with pytest.raises(ValueError):
            normal(p, v, context=RunContext("bad", 1), rounds=0, **kwargs)


@pytest.mark.parametrize("case", ["weighted_ties", "conflict"])
@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
@pytest.mark.parametrize("grow", [depthwise, best_first, symmetric])
def test_final_best_and_offsets_replay_in_fresh_core_only_process(case, update, grow, tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    import openboost
    from openboost.objectives import Normal
    from openboost.results import validate_result

    p, v, prepared, _ = prepared_case(case)
    context = RunContext("129-fresh", 1)
    options = dict(
        context=context,
        rounds=3,
        bins=4,
        prepared=prepared,
        update=update,
        mode="natural",
        damping=0.25,
        learner=partial(grow, ordering=rank, field_leaf=leaf),
    )
    result = normal(p, v, **options)
    summary = normal(p, v, retention="summary", **options)
    assert result.state.identity == summary.state.identity and result.stop == summary.stop
    validate_result(summary, context=context, train=p, validation=v)
    for full, small in zip(flat(result), flat(summary), strict=True):
        r = dict(small.values)
        assert r["channels"] == full.channels and r["after_version"] == full.after_version
        assert (
            r["comparisons"] == full.comparisons
            and r["validation_change"] == full.validation_change
        )
        assert not any(isinstance(value, np.ndarray) for value in r.values())
    nfeatures = len(p.data.feature_names)
    x = np.linspace(-10, 10, 4 * nfeatures).reshape(4, nfeatures)
    x[-1] = np.nan
    unseen = NumericData(x, [900, 800, 700, 600], p.data.feature_names)
    offset = np.column_stack(([0.25, -0.5, 0.75, 1.0], [-0.125, 0.25, -0.375, 0.5]))
    paths = []
    expected = []
    for name, model in [("final", result.state.model), ("best", result.state.best_model)]:
        path = tmp_path / (name + ".json")
        model.save(path)
        paths.append(str(path))
        raw = model.predict(unseen, offset=offset)
        expected.append(
            dict(
                raw=raw.tolist(),
                parameters=Normal.parameters(raw).tolist(),
                identity=model.identity,
            )
        )
    payload = dict(
        paths=paths,
        x=x.tolist(),
        ids=unseen.row_ids.tolist(),
        names=unseen.feature_names,
        offset=offset.tolist(),
        installed="OPENBOOST_FRESH_CPU_PYTHON" in os.environ,
        local_import_root=str(Path(openboost.__file__).parent.parent),
    )
    code = """
import json,sys
from pathlib import Path
p=json.loads(sys.argv[1])
if not p['installed']: sys.path.insert(0,p['local_import_root'])
for name in ('cupy','numba','openboost.device','openboost.device_inputs','openboost.device_runs',
             'openboost.device_runtime','openboost.device_recipes','openboost.recipes'):
    sys.modules[name]=None
import openboost
from openboost import NumericData,ops,tree,newton_order
from openboost.artifacts import Model
from openboost.objectives import Normal
if p['installed']:
    import importlib.metadata
    assert 'site-packages' in Path(openboost.__file__).parts
    assert {d.metadata['Name']:d.version for d in importlib.metadata.distributions()} == {'numpy':'2.3.5','openboost':'1.0.0.dev0'}
def forbidden(*a,**k): raise AssertionError('inference attempted training')
ops.histogram=newton_order.rank=newton_order.leaf=newton_order.choose=forbidden
tree.depthwise=tree.best_first=tree.symmetric=Normal.geometry=Normal.base=forbidden
x=NumericData(p['x'],p['ids'],p['names'])
result=[]
for path in p['paths']:
    m=Model.load(path); raw=m.predict(x,offset=p['offset'])
    result.append(dict(raw=raw.tolist(),parameters=Normal.parameters(raw).tolist(),identity=m.identity))
print(json.dumps(result))
"""
    done = subprocess.run(
        [
            os.environ.get("OPENBOOST_FRESH_CPU_PYTHON", sys.executable),
            "-I",
            "-c",
            code,
            json.dumps(payload),
        ],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout) == expected


@pytest.mark.parametrize("grow", [depthwise, best_first, symmetric])
@pytest.mark.parametrize("update", ["joint", "forward", "reverse"])
@pytest.mark.parametrize("mode", ["ordinary", "natural"])
def test_normal_custom_cohort_constraints_use_independent_original_mass(grow, update, mode):
    p, v, prepared, f = prepared_case("d2")
    fitted = []

    def learner(b, fields):
        enriched = fields.add_independent("a", f["information"][:, 0]).add_independent(
            "b", f["information"][:, 1]
        )
        tree = grow(
            b, enriched, ordering=partial(rank, min_information={"a": 1, "b": 1}), field_leaf=leaf
        )
        expected = exact_tree(
            f["x"], enriched.values, policy=grow.__name__, information_minima={2: 1, 3: 1}
        )
        assert_tree(tree, expected)
        fitted.append(tree)
        return tree

    result = normal(
        p,
        v,
        context=RunContext("129-cohort", 1),
        rounds=3,
        bins=4,
        prepared=prepared,
        mode=mode,
        update=update,
        learner=learner,
        step="fixed",
    )
    expected = reference_run(
        f, mode=mode, policy=grow.__name__, update=update, minimum=1, fixed=True
    )
    np.testing.assert_allclose(result.state.train_raw, expected["raw"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        result.state.validation_raw, expected["validation"], rtol=1e-6, atol=1e-8
    )
    for actual, nodes in zip(
        fitted, (n for s in expected["steps"] for n in s["nodes"]), strict=True
    ):
        assert_tree(actual, nodes)
