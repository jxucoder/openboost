"""Matched paid-loss problems and persisted two-model composition."""

import json
import subprocess
import sys

import numpy as np
import pytest

from openboost import MixedData, RunContext
from openboost.artifacts import Model
from openboost.composition import FrequencySeverity, paid_loss_problems
from openboost.outputs import poisson_mean, positive_mean
from openboost.recipes import gamma, poisson


def fixture():
    data = MixedData(
        [[0, "a"], [1, "b"], [2, None], [3, "a"], [4, "c"]],
        [10, 11, 12, 13, 14],
        ("x", "c"),
        ("numeric", "categorical"),
    )
    count = np.array([0, 2, 1, 3, 0])
    amount = np.array([0, 6, 2, 15, 0])
    exposure = np.array([0.5, 1, 2, 0.5, 1])
    return data, count, amount, exposure


def trained():
    data, count, amount, e = fixture()
    frequency, severity = paid_loss_problems(data, count, amount, e, weight=[1, 2, 3, 1, 0])
    f = poisson(frequency, frequency, context=RunContext("paid-count", 7), rounds=3)
    s = gamma(severity, severity, context=RunContext("paid-severity", 7), rounds=3)
    assert f.state.version == s.state.version == 3
    return FrequencySeverity(f.state.model, s.state.model), data, e


def test_aggregate_roles_and_weight_units():
    data, count, amount, e = fixture()
    f, s = paid_loss_problems(data, count, amount, e, weight=[1, 2, 3, 1, 0])
    np.testing.assert_array_equal(f.target[:, 0], count)
    np.testing.assert_array_equal(f.structure["exposure"][:, 0], e)
    np.testing.assert_array_equal(s.row_ids, [11, 12, 13])
    np.testing.assert_array_equal(s.target[:, 0], [3, 2, 5])
    np.testing.assert_array_equal(s.weight, [4, 3, 3])


def test_composition_product_exposure_and_policy_order():
    model, data, e = trained()
    offset = np.arange(5.0)[:, None] / 10
    got = model.predict(data, data, e, frequency_offset=offset, severity_offset=-offset)
    f = poisson_mean(model.frequency.predict(data, offset=offset), e)
    s = positive_mean(model.severity.predict(data, offset=-offset))
    np.testing.assert_allclose(got["annualized_mean"], f["rate"] * s)
    np.testing.assert_allclose(got["period_mean"], f["count_mean"] * s)
    doubled = model.predict(data, data, 2 * e, frequency_offset=offset, severity_offset=-offset)
    np.testing.assert_allclose(doubled["period_mean"], 2 * got["period_mean"])
    np.testing.assert_array_equal(doubled["annualized_mean"], got["annualized_mean"])
    order = [4, 2, 1, 3, 0]
    other = MixedData(
        data.values[order], data.row_ids[order], data.feature_names, data.feature_kinds
    )
    reordered = model.predict(other, other, e[order])
    base = model.predict(data, data, e)
    for key in base:
        np.testing.assert_array_equal(reordered[key], base[key][order])
    with pytest.raises(ValueError, match="row IDs"):
        model.predict(data, other, e)


@pytest.mark.parametrize(
    "counts,totals",
    [
        ([0, 2], [1, 3]),
        ([1, 0], [0, 0]),
        ([0.5, 1], [1, 2]),
        ([-1, 1], [1, 2]),
        ([0, 0], [0, 0]),
    ],
)
def test_bad_aggregates_rejected(counts, totals):
    data = MixedData([[1], [2]], [1, 2], ("x",), ("numeric",))
    with pytest.raises(ValueError):
        paid_loss_problems(data, counts, totals, [1, 1])


def test_fresh_process_bundle_and_corruption(tmp_path):
    model, data, e = trained()
    path = tmp_path / "composition.json"
    model.save(path)
    loaded = FrequencySeverity.load(path)
    assert loaded.identity == model.identity
    x = MixedData([[1, "unknown"], [None, None]], [20, 21], data.feature_names, data.feature_kinds)
    code = """import json, sys
from openboost import MixedData
from openboost.composition import FrequencySeverity
x = MixedData([[1,'unknown'],[None,None]], [20,21], ('x','c'), ('numeric','categorical'))
m = FrequencySeverity.load(sys.argv[1])
print(json.dumps({k:v.tolist() for k,v in m.predict(x,x,[0.5,2],
frequency_offset=[[0.1],[0.2]], severity_offset=[[0.2],[0.3]]).items()}))
"""
    got = json.loads(subprocess.check_output([sys.executable, "-c", code, str(path)], text=True))
    expected = model.predict(
        x, x, [0.5, 2], frequency_offset=[[0.1], [0.2]], severity_offset=[[0.2], [0.3]]
    )
    for key in expected:
        np.testing.assert_array_equal(got[key], expected[key])
    for field in ("frequency", "severity"):
        record = model.record()
        record[field]["base"] = [1, 2]
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError):
            FrequencySeverity.load(path)
    path.write_text('{"format":"openboost-frequency-severity-v1","format":"duplicate"}')
    with pytest.raises(ValueError, match="duplicate"):
        FrequencySeverity.load(path)


def test_nested_raw_record_validation_and_declared_dependencies():
    model, data, e = trained()
    restored = Model.from_record(model.frequency.record())
    assert restored.identity == model.frequency.identity
    swapped = FrequencySeverity(model.severity, model.frequency)
    assert swapped.identity != model.identity
    with pytest.raises(ValueError):
        Model.from_record({"format": "old"})
    with pytest.raises(ValueError):
        FrequencySeverity(Model(("x",), [0, 0], ()), model.severity)
