"""Exact growth decisions, including gains erased by binary64 policy reductions."""

from dataclasses import replace
from fractions import Fraction as F
from functools import partial

import numpy as np
import pytest

from openboost import NumericData
from openboost.binning import Binning
from openboost.newton_order import rank
from openboost.stats import RowFields
from openboost.tree import best_first, depthwise, symmetric

from .reference.exact_growth import fit, predict
from .test_exact_newton_order import prepared

POLICIES = (depthwise, best_first, symmetric)


def make(x, g, h):
    data = NumericData(x, np.arange(len(x)), tuple(f"f{i}" for i in range(len(x[0]))))
    b = Binning.fit(data, bins=16).transform(data)
    fields = RowFields(
        "p",
        data.identity,
        ("gradient", "curvature"),
        np.column_stack((g, h)),
        ("training", "training"),
    )
    codes = [
        [None if b.missing[f, i] else int(b.codes[f, i]) for f in range(len(x[0]))]
        for i in range(len(x))
    ]
    return b, fields, codes


def matches(tree, wanted, x, raw, *, categorical=()):
    assert len(tree.value) == len(wanted)
    for i, node in enumerate(wanted):
        key = (
            None
            if tree.feature[i] == -1
            else (int(tree.feature[i]), int(tree.threshold[i]), bool(tree.missing_left[i]))
        )
        assert key == node["key"]
        assert (tree.left[i], tree.right[i]) == (node["left"], node["right"])
        np.testing.assert_allclose(tree.value[i], [float(node["value"])], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        tree.predict(raw)[:, 0], predict(wanted, x, categorical=categorical), rtol=1e-14, atol=1e-14
    )


@pytest.mark.parametrize("grow", POLICIES)
@pytest.mark.parametrize("stored", [False, True])
@pytest.mark.parametrize("depth,leaves", [(0, 1), (1, 8), (2, 3), (2, 4), (3, 8)])
def test_captured_normal_and_policy_budgets_match_exact_rows(grow, stored, depth, leaves):
    b, fields, x, exact = prepared(stored=stored)
    wanted = fit(x, exact, policy=grow.__name__, depth=depth, leaves=leaves)
    tree = grow(b, fields, max_depth=depth, max_leaves=leaves, ordering=rank)
    matches(tree, wanted, x, b.data)


@pytest.mark.parametrize("grow", [depthwise, best_first])
@pytest.mark.parametrize("power", [54, 100, 1074])
def test_leaf_budget_compares_exact_gains_between_nodes(grow, power):
    tiny = F(1, 2**power)
    b, fields, x = make(
        [[0, 0], [0, 1], [1, 0], [1, 1], [1, 0]], [-7.5, -0.5, 0.5, 7.5, -tiny], [1, 1, 1, 1, tiny]
    )
    left = rank(b, fields, [0, 1])[0].gain
    right = rank(b, fields, [2, 3, 4])[0].gain
    assert right > left and float(right) == float(left)
    wanted = fit(x, fields.values, policy=grow.__name__, leaves=3)
    assert wanted[1]["key"] is None and wanted[2]["key"] == (1, 0, False)
    tree = grow(b, fields, max_leaves=3, ordering=rank)
    matches(tree, wanted, x, b.data)


@pytest.mark.parametrize("power", [58, 100, 1074])
@pytest.mark.parametrize("direction", [-1, 3])
def test_symmetric_exact_signed_sum_controls_whether_the_layer_exists(power, direction):
    tiny = F(1, 2**power)
    b, fields, x = make(
        [[0, 0], [0, 1], [1, 0], [1, 1], [1, 0]],
        [-4, -4, 1, 3, direction * tiny],
        [1, 1, 1, 1, tiny],
    )
    ordering = partial(rank, reg_lambda=0, split_penalty=0.5)
    left = ordering(b, fields, [0, 1])[0].gain
    right = ordering(b, fields, [2, 3, 4])[0].gain
    assert left < 0 < right and float(left) + float(right) == 0
    assert (left + right > 0) == (direction == -1)
    wanted = fit(x, fields.values, policy="symmetric", leaves=4, regularization=0, penalty=0.5)
    assert len(wanted) == (7 if direction == -1 else 3)
    from openboost.ops import newton_leaf

    tree = symmetric(
        b, fields, max_leaves=4, ordering=ordering, leaf=partial(newton_leaf, reg_lambda=0)
    )
    matches(tree, wanted, x, b.data)


@pytest.mark.parametrize("grow", POLICIES)
def test_fraction_gain_need_not_fit_binary64(grow):
    b, fields, x = make([[0], [1]], [-1e200, 1e200], [1, 1])
    tree = grow(b, fields, ordering=rank)
    matches(tree, fit(x, fields.values, policy=grow.__name__), x, b.data)


@pytest.mark.parametrize("grow", POLICIES)
@pytest.mark.parametrize(
    "fault",
    [
        "not_callable",
        "scoring",
        "legality",
        "duplicate",
        "record",
        "foreign_rows",
        "foreign_fields",
        "foreign_data",
        "config",
    ],
)
def test_bad_ordering_bindings_fail(grow, fault):
    b, fields, _, _ = prepared()
    options = {}
    ordering = rank
    if fault == "not_callable":
        ordering = 1
    elif fault in ("scoring", "legality"):
        options[fault] = lambda _: 1
    elif fault == "duplicate":

        def ordering(d, f, r):
            return (rank(d, f, r)[0],) * 2
    elif fault == "record":

        def ordering(d, f, r):
            return (rank(d, f, r)[0].candidate,)
    elif fault == "foreign_rows":

        def ordering(d, f, r):
            return rank(d, f, [0, 1, 6])
    elif fault == "foreign_fields":

        def ordering(d, f, r):
            return rank(d, replace(f, values=f.values * np.array([2, 1])), r)
    elif fault == "foreign_data":
        other = Binning(b.data.feature_names, (np.array([0.5]), np.array([0.5]))).transform(b.data)

        def ordering(d, f, r):
            return rank(other, f, r)
    elif fault == "config":

        def ordering(d, f, r):
            return rank(d, f, r, reg_lambda=1 if len(r) == 8 else 2)

    with pytest.raises(ValueError):
        grow(b, fields, ordering=ordering, **options)


@pytest.mark.parametrize("grow", POLICIES)
@pytest.mark.parametrize("depth,leaves", [(1, 2), (2, 3), (3, 5), (4, 8)])
def test_weighted_original_rows_match_all_policy_control_flows(grow, depth, leaves):
    rng = np.random.default_rng(1928)
    x = rng.integers(0, 4, size=(24, 3)).astype(float)
    x[::5, 1] = np.nan
    weight = rng.integers(0, 4, size=24)
    b, fields, codes = make(x, rng.normal(size=24) * weight, rng.uniform(0.1, 2, size=24) * weight)
    wanted = fit(
        codes,
        fields.values,
        policy=grow.__name__,
        depth=depth,
        leaves=leaves,
        regularization=2.5,
        penalty=0.125,
    )
    from openboost.ops import newton_leaf

    tree = grow(
        b,
        fields,
        max_depth=depth,
        max_leaves=leaves,
        ordering=partial(rank, reg_lambda=2.5, split_penalty=0.125),
        leaf=partial(newton_leaf, reg_lambda=2.5),
    )
    matches(tree, wanted, codes, b.data)


@pytest.mark.parametrize("grow", POLICIES)
def test_filtered_reordered_records_and_separate_vector_leaf_fields(grow):
    b, fields, _, _ = prepared(stored=True)
    baseline = grow(b, fields, ordering=rank)
    counts = []

    def ordered(data, split_fields, rows):
        counts.append(tuple(rows))
        # Filter allowed features and return records in the opposite order.
        return tuple(
            reversed([r for r in rank(data, split_fields, rows) if r.candidate.feature == 0])
        )

    leaves = RowFields(
        fields.problem_identity,
        fields.data_identity,
        ("u", "v"),
        np.column_stack((np.arange(8), np.arange(8) ** 2)),
        ("independent", "independent"),
    )
    tree = grow(b, fields, ordering=ordered, leaf_fields=leaves, leaf=lambda s, n: s)
    assert len(counts) == len(set(counts)) and counts[0] == tuple(range(8))
    assert set(tree.feature) <= {-1, 0} and tree.output_width == 2
    assert baseline.output_width == 1
    pending = [(0, np.arange(8))]
    while pending:
        i, rows = pending.pop()
        np.testing.assert_array_equal(tree.value[i], leaves.values[rows].sum(axis=0))
        if tree.feature[i] != -1:
            f, t = tree.feature[i], tree.threshold[i]
            mask = np.where(b.missing[f, rows], tree.missing_left[i], b.codes[f, rows] <= t)
            pending.extend(((tree.left[i], rows[mask]), (tree.right[i], rows[~mask])))


@pytest.mark.parametrize("grow", POLICIES)
def test_mixed_category_unknown_and_missing_growth_matches_original_rows(grow):
    from openboost import MixedData

    x = [["a", 0], ["b", 1], ["c", 0], ["unknown", 2], [None, 1], ["a", 2], ["b", 0], ["c", 2]]
    data = MixedData(x, np.arange(8), ("cat", "x"), ("categorical", "numeric"))
    b = Binning(
        data.feature_names, (np.array([]), np.array([0.5, 1.5])), (("a", "b", "c"), None)
    ).transform(data)
    fields = RowFields(
        "p",
        data.identity,
        ("gradient", "curvature"),
        np.column_stack(([-4, 2, -1, 3, -2, 1, -3, 4], np.ones(8))),
        ("training", "training"),
    )
    codes = [[None if b.missing[f, i] else int(b.codes[f, i]) for f in range(2)] for i in range(8)]
    wanted = fit(codes, fields.values, policy=grow.__name__, depth=3, leaves=5, categorical=(0,))
    tree = grow(b, fields, ordering=rank, max_depth=3, max_leaves=5)
    matches(tree, wanted, codes, data, categorical=(0,))


@pytest.mark.parametrize("grow", POLICIES)
def test_three_round_recipe_and_final_best_fresh_inference(grow, tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    import openboost
    from openboost import Problem, RunContext
    from openboost.binning import PreparedData
    from openboost.recipes import squared

    b, _, codes = make([[-2], [-1], [1], [2]], [0] * 4, [1] * 4)
    target = np.array([-6, -2, 2, 6.0])
    weight = np.array([1, 2, 2, 1.0])
    train = Problem(b.data, target[:, None], b.data.row_ids, weight=weight)
    valid_data = NumericData(b.data.values, np.arange(4) + 100, b.data.feature_names)
    valid = Problem(valid_data, -target[:, None], valid_data.row_ids, weight=weight)
    result = squared(
        train,
        valid,
        context=RunContext("exact-" + grow.__name__, 128),
        rounds=3,
        bins=16,
        prepared=PreparedData(b.data, 16),
        learning_rate=0.5,
        learner=partial(grow, max_depth=2, max_leaves=4, ordering=rank),
    )
    raw = np.zeros(4)
    assert result.state.version == 3 and len(result.state.best_model.terms) == 0
    for step, term in zip(result.steps, result.state.model.terms, strict=True):
        fields = np.column_stack(((raw - target) * weight, weight))
        expected = fit(codes, fields, policy=grow.__name__, leaves=4)
        matches(term.learner, expected, codes, b.data)
        np.testing.assert_allclose(step.raw_before[:, 0], raw, rtol=1e-14, atol=1e-14)
        raw = raw + 0.5 * np.array(predict(expected, codes))
        np.testing.assert_allclose(step.raw_after[:, 0], raw, rtol=1e-14, atol=1e-14)
        assert step.accepted and step.coefficients == (0.5,)
    unseen = NumericData([[-10], [0], [10], [np.nan]], [9, 8, 7, 6], b.data.feature_names)
    offset = np.array([[0.25], [-0.5], [0.75], [1.0]])
    paths, predictions = [], []
    for name, model in [("final", result.state.model), ("best", result.state.best_model)]:
        path = tmp_path / (name + ".json")
        model.save(path)
        paths.append(str(path))
        predictions.append(model.predict(unseen, offset=offset).tolist())
    payload = dict(
        paths=paths,
        x=unseen.values.tolist(),
        ids=unseen.row_ids.tolist(),
        names=unseen.feature_names,
        offset=offset.tolist(),
        installed="OPENBOOST_FRESH_CPU_PYTHON" in os.environ,
        local_import_root=str(Path(openboost.__file__).parent.parent),
    )
    code = """
import json, sys
from pathlib import Path
p=json.loads(sys.argv[1])
if not p['installed']:
    sys.path.insert(0, p['local_import_root'])
for name in ('cupy', 'numba', 'openboost.device', 'openboost.device_inputs',
             'openboost.device_runtime', 'openboost.device_runs', 'openboost.device_recipes',
             'openboost.recipes'):
    sys.modules[name] = None
import openboost
from openboost import NumericData, tree, ops, newton_order
from openboost.artifacts import Model
if p['installed']:
    import importlib.metadata
    assert 'site-packages' in Path(openboost.__file__).parts
    assert {d.metadata['Name']: d.version for d in importlib.metadata.distributions()} == {'numpy': '2.3.5', 'openboost': '1.0.0.dev0'}
def forbidden(*a, **k):
    raise AssertionError('fresh inference attempted training or ordering')
tree.depthwise = tree.best_first = tree.symmetric = forbidden
ops.histogram = newton_order.rank = newton_order.choose = forbidden
x = NumericData(p['x'], p['ids'], p['names'])
print(json.dumps([Model.load(path).predict(x, offset=p['offset']).tolist() for path in p['paths']]))
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
    assert json.loads(done.stdout) == predictions
