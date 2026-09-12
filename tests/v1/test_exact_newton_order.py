"""Public exact Newton ordering checked against independent original-row math."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

import openboost
from openboost import MixedData, NumericData
from openboost.binning import Binning
from openboost.newton_order import choose, rank
from openboost.ops import partition
from openboost.stats import RowFields

from .reference.normal_split_order import candidates, captured, winner


def prepared(*, stored=False, power=None, direction=0):
    x, exact = captured(stored=stored)
    if power is not None:
        mass = F(1, 2**power)
        exact[0] = (direction * mass, mass)
    values = np.array([[np.nan if v is None else v for v in row] for row in x], float)
    data = NumericData(values, 101 + 7 * np.arange(8), ("f0", "f1"))
    binned = Binning(data.feature_names, (np.arange(3) + 0.5, np.arange(3) + 0.5)).transform(data)
    fields = RowFields(
        "captured",
        data.identity,
        ("gradient", "curvature"),
        np.array(exact, float),
        ("training", "training"),
    )
    return binned, fields, x, exact


@pytest.mark.parametrize("stored", [False, True])
@pytest.mark.parametrize(
    "rows", [tuple(range(8)), tuple(reversed(range(8))), (0, 1, 6), (2, 3, 4, 5, 7), ()]
)
def test_exact_candidates_and_routing_match_independent_partitions(stored, rows):
    data, fields, x, exact = prepared(stored=stored)
    expected = {c["key"]: c for c in candidates(x, exact, rows) if c["legal"]}
    actual = rank(data, fields, rows)
    assert {r.candidate.key for r in actual} == set(expected)
    for row in actual:
        wanted = expected[row.candidate.key]
        assert row.gain == wanted["gain"] and row.exact_sums == (*wanted["sums"], wanted["parent"])
        assert row.candidate.data_identity == data.identity
        assert (
            tuple(tuple(int(i) for i in side) for side in partition(data, rows, row.candidate))
            == wanted["rows"]
        )
        for array, exact_sum in zip(
            (row.candidate.left, row.candidate.right, row.candidate.parent),
            row.exact_sums,
            strict=True,
        ):
            np.testing.assert_array_equal(array, [float(v) for v in exact_sum])
            assert not array.flags.writeable
    chosen, wanted = choose(actual), winner(expected.values())
    assert (None if chosen is None else chosen.candidate.key) == (
        None if wanted is None else wanted["key"]
    )
    assert choose(reversed(actual)) == chosen


@pytest.mark.parametrize("power", [24, 54, 100, 149, 1074])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_no_epsilon_or_float_cast_can_erase_a_true_non_tie(power, direction):
    data, fields, x, exact = prepared(stored=True, power=power, direction=direction)
    result = choose(rank(data, fields))
    expected = winner(candidates(x, exact, range(8)))
    assert result.candidate.key == expected["key"]
    assert result.gain == expected["gain"]


@pytest.mark.parametrize("limit", ["curvature", "information"])
def test_exact_feasibility_rejects_a_sum_rounded_up_to_the_minimum(limit):
    data = NumericData([[0], [0], [1], [1]], [1, 2, 3, 4], ("x",))
    values = np.column_stack(([-1, -1, 1, 1], [0.5, 0.5, 1, 1], [1.0, 1.0, 1.0, 1.0]))
    column = 1 if limit == "curvature" else 2
    values[:2, column] = [0.5, np.nextafter(0.5, 0.0)]
    assert values[:2, column].sum() == 1.0
    fields = RowFields(
        "p",
        data.identity,
        ("gradient", "curvature", "group"),
        values,
        ("training", "training", "independent"),
    )
    binned = Binning(("x",), (np.array([0.5]),)).transform(data)
    assert choose(rank(binned, fields)) is not None
    config = dict(min_child_h=1) if limit == "curvature" else dict(min_information={"group": 1})
    assert rank(binned, fields, **config) == ()


@pytest.mark.parametrize("regularization,penalty", [(0.0, 0.0), (2.5, 0.25), (1.0, 10.0)])
def test_configuration_uses_exact_stored_scalar_values(regularization, penalty):
    data, fields, x, exact = prepared(stored=True)
    result = rank(data, fields, reg_lambda=regularization, split_penalty=penalty)
    for item in result:
        left, right, parent = item.exact_sums

        def score(s):
            return s[0] ** 2 / (2 * (s[1] + F(regularization)))

        assert item.gain == score(left) + score(right) - score(parent) - F(penalty)
    if penalty == 10.0:
        assert result and all(r.gain < 0 for r in result) and choose(result) is None


def test_column_names_and_independent_mass_survive_reordering_without_reweighting():
    data, fields, _, _ = prepared()
    enriched = fields.add_independent("cohort", np.ones(8))
    order = [2, 1, 0]
    reordered = RowFields(
        enriched.problem_identity,
        enriched.data_identity,
        tuple(enriched.names[i] for i in order),
        enriched.values[:, order],
        tuple(enriched.roles[i] for i in order),
    )
    a = rank(data, enriched, min_information={"cohort": 2})
    b = rank(data, reordered, min_information={"cohort": 2})
    assert [(r.candidate.key, r.gain) for r in a] == [(r.candidate.key, r.gain) for r in b]
    assert all(r.exact_sums[2][2] == 8 for r in a)  # Includes both zero-training-weight rows.


def test_categorical_and_unknown_missing_routes_use_actual_original_rows():
    data = MixedData(
        [["a"], ["b"], ["unknown"], [None], ["a"], ["b"]], np.arange(6), ("kind",), ("categorical",)
    )
    binned = Binning(("kind",), (np.array([]),), (("a", "b"),)).transform(data)
    values = np.array([[-3, 1], [2, 1], [1, 0.5], [-1, 0.5], [-2, 2], [3, 2.0]])
    fields = RowFields(
        "p", data.identity, ("gradient", "curvature"), values, ("training", "training")
    )
    for item in rank(binned, fields):
        c = item.candidate
        category = ("a", "b")[c.threshold]
        left = tuple(
            i
            for i, row in enumerate(data.values)
            if (c.missing_left if row[0] not in ("a", "b") else row[0] == category)
        )
        right = tuple(i for i in range(6) if i not in left)
        expected = tuple(
            tuple(sum((F(float(values[i, q])) for i in side), F(0)) for q in (0, 1))
            for side in (left, right, tuple(range(6)))
        )
        assert c.kind == "categorical" and item.exact_sums == expected
        assert tuple(tuple(int(i) for i in side) for side in partition(binned, None, c)) == (
            left,
            right,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "foreign",
        "unweighted",
        "role",
        "missing_name",
        "negative_curvature",
        "duplicate_rows",
        "range",
        "fractional_rows",
        "information_role",
        "information_name",
        "negative_lambda",
        "nan_minimum",
        "infinite_penalty",
    ],
)
def test_invalid_identity_roles_rows_and_parameters_fail_before_a_choice(fault):
    data, fields, _, _ = prepared()
    config, rows = {}, None
    if fault == "foreign":
        fields = replace(fields, data_identity="foreign")
    elif fault == "unweighted":
        fields = replace(fields, roles=("unweighted", "unweighted"))
    elif fault == "role":
        fields = replace(fields, roles=("independent", "training"))
    elif fault == "missing_name":
        fields = replace(fields, names=("g", "h"))
    elif fault == "negative_curvature":
        values = fields.values.copy()
        values[0, 1] = -1
        fields = replace(fields, values=values)
    elif fault == "duplicate_rows":
        rows = [0, 0]
    elif fault == "range":
        rows = [8]
    elif fault == "fractional_rows":
        rows = [0.5]
    elif fault == "information_role":
        config = dict(min_information={"gradient": 1})
    elif fault == "information_name":
        config = dict(min_information={"absent": 1})
    elif fault == "negative_lambda":
        config = dict(reg_lambda=-1)
    elif fault == "nan_minimum":
        config = dict(min_child_h=np.nan)
    else:
        config = dict(split_penalty=np.inf)
    with pytest.raises(ValueError):
        rank(data, fields, rows, **config)


def test_choose_rejects_mixed_nodes_or_duplicate_conditions():
    data, fields, _, _ = prepared()
    whole = rank(data, fields)
    child = rank(data, fields, [0, 1, 6])
    with pytest.raises(ValueError):
        choose((*whole, *child))
    with pytest.raises(ValueError):
        choose((whole[0], whole[0]))
    assert choose(()) is None


@pytest.mark.parametrize("change", ["fields", "regularization", "penalty", "minimum"])
def test_same_feature_rows_cannot_mix_different_objective_bindings(change):
    data, fields, _, _ = prepared()
    original = rank(data, fields)
    options = {}
    if change == "fields":
        fields = replace(fields, values=fields.values * np.array([2.0, 1.0]))
    else:
        options = {
            dict(regularization="reg_lambda", penalty="split_penalty", minimum="min_child_h")[
                change
            ]: 0.25
        }
    other = rank(data, fields, **options)
    different_key = next(r for r in other if r.candidate.key != original[0].candidate.key)
    with pytest.raises(ValueError, match="binding"):
        choose((original[0], different_key))


def test_exact_bin_cancellation_can_recover_a_finite_total_without_float_overflow():
    data = NumericData([[0], [0], [0], [0], [1], [1]], np.arange(6), ("x",))
    binned = Binning(("x",), (np.array([0.5]),)).transform(data)
    fields = RowFields(
        "p",
        data.identity,
        ("gradient", "curvature"),
        np.column_stack(([1e308, 1e308, -1e308, -1e308, 1.0, -1.0], np.ones(6))),
        ("training", "training"),
    )
    result = rank(binned, fields)
    assert result and choose(result) is None
    assert all(r.gain == 0 and r.exact_sums[2] == (0, 6) for r in result)


def test_an_unrepresentable_routing_aggregate_fails_explicitly():
    data = NumericData([[0], [0], [1]], [1, 2, 3], ("x",))
    fields = RowFields(
        "p",
        data.identity,
        ("gradient", "curvature"),
        [[1e308, 1], [1e308, 1], [0, 1]],
        ("training", "training"),
    )
    with pytest.raises(ValueError, match="representable"):
        rank(Binning(("x",), (np.array([0.5]),)).transform(data), fields)


def test_fresh_cpu_process_preserves_the_tiny_ordering_without_device_imports():
    data, fields, x, exact = prepared(stored=True, power=1074)
    expected = winner(candidates(x, exact, range(8)))
    python = os.environ.get("OPENBOOST_FRESH_CPU_PYTHON", sys.executable)
    payload = dict(
        x=data.data.values.tolist(),
        fields=fields.values.tolist(),
        installed="OPENBOOST_FRESH_CPU_PYTHON" in os.environ,
        local_import_root=str(Path(openboost.__file__).parent.parent),
    )
    code = """
import json, sys
from pathlib import Path
p = json.loads(sys.argv[1])
if not p['installed']:
    sys.path.insert(0, p['local_import_root'])
for name in ('cupy', 'numba', 'openboost.device', 'openboost.device_inputs',
             'openboost.device_runs', 'openboost.device_runtime', 'openboost.device_recipes',
             'openboost.recipes'):
    sys.modules[name] = None
import numpy as np
import openboost
from openboost import ops, tree
def forbidden(*args, **kwargs):
    raise AssertionError('floating histogram or tree training is not this operation')
ops.histogram = forbidden
tree.depthwise = tree.best_first = tree.symmetric = forbidden
from openboost import NumericData
from openboost.binning import Binning
from openboost.stats import RowFields
from openboost.newton_order import rank, choose
if p['installed']:
    import importlib.metadata
    assert 'site-packages' in Path(openboost.__file__).parts
    assert {d.metadata['Name']: d.version for d in importlib.metadata.distributions()} == {'numpy': '2.3.5', 'openboost': '1.0.0.dev0'}
data = NumericData(p['x'], np.arange(8), ('a', 'b'))
binned = Binning(data.feature_names, (np.arange(3)+.5, np.arange(3)+.5)).transform(data)
fields = RowFields('fresh', data.identity, ('gradient', 'curvature'), p['fields'], ('training', 'training'))
chosen = choose(rank(binned, fields))
print(json.dumps(dict(key=chosen.candidate.key, gain=str(chosen.gain))))
"""
    result = subprocess.run(
        [python, "-I", "-c", code, json.dumps(payload)], text=True, capture_output=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == dict(key=list(expected["key"]), gain=str(expected["gain"]))
