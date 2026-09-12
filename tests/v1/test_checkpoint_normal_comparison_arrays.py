"""Self-contained checkpoint controls extracted from tests/v1/test_normal_comparison_arrays.py.

Source: 91a519dd5227344266a3eb1c86ce21b45acf32c6. Original test/helper bodies are retained;
archive-only trajectory/study replay remains in the original full evidence checkout.
Extraction is not a new numerical validation result.
"""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost._comparison_math import cpu_add, cpu_div, cpu_mul, cpu_row_change
from openboost._normal_comparison_arrays import add, div, mul, rows
from openboost.comparison import _normal_result
from openboost.objectives import Normal

from .test_loss_change import CASES

INTERVALS = [
    (0.0, 0.0),
    (-0.0, -0.0),
    (1.0, 1.0),
    (0.5, 0.5),
    (-1.0, -1.0),
    (-2.0, 3.0),
    (np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0)),
    (-np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0)),
    (1e-300, 2e-300),
    (-1e300, 1e300),
    (1e308, 1e308),
    (-np.inf, np.inf),
    (np.inf, np.inf),
]


def bits(value):
    return np.asarray(value, np.float64).view(np.uint64).tolist()


def retain(tmp_path, name, record):
    root = Path(os.environ.get("OPENBOOST_NORMAL_ARRAY_ARTIFACTS", tmp_path))
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(name.encode()).hexdigest()[:16]
    (root / (key + ".json")).write_text(
        json.dumps(dict(name=name, **record), indent=2, allow_nan=False) + "\n"
    )


@pytest.mark.parametrize(
    "name,scalar,array",
    [("add", cpu_add, add), ("mul", cpu_mul, mul), ("div", cpu_div, div)],
    ids=["add", "mul", "div"],
)
def test_interval_endpoint_bits(name, scalar, array, tmp_path):
    pairs = [(a, b) for a in INTERVALS for b in INTERVALS]
    left, right = (np.asarray([p[k] for p in pairs]) for k in (0, 1))
    expected = np.asarray([scalar(a, b) for a, b in pairs])
    actual = array(left, right)
    retain(
        tmp_path,
        "interval/" + name,
        dict(
            left_bits=bits(left),
            right_bits=bits(right),
            actual_bits=bits(actual),
            scalar_bits=bits(expected),
        ),
    )
    assert bits(actual) == bits(expected)
    assert actual.shape == left.shape and actual.dtype == np.float64
    assert not np.shares_memory(actual, left) and not np.shares_memory(actual, right)


@pytest.mark.parametrize("case", CASES, ids=[row["id"] for row in CASES])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_original_loss_change_rows(case, dtype, tmp_path):
    data = {
        k: np.asarray(v["values"], dtype=dtype).astype(np.float64)
        for k, v in case["inputs"].items()
    }
    before, after, target, offset = (data[k] for k in ("before", "after", "target", "offset"))
    expected = np.asarray(
        [
            cpu_row_change(*old, *new, float(y), *off)
            for old, new, y, off in zip(before, after, target.ravel(), offset, strict=True)
        ]
    )
    actual = rows(before, after, target.ravel(), offset)
    retain(
        tmp_path,
        "rows/" + np.dtype(dtype).name + "/" + case["id"],
        dict(
            inputs={k: v.tolist() for k, v in data.items()},
            actual_bits=bits(actual[:, :2]),
            scalar_bits=bits(expected[:, :2]),
            actual_codes=actual[:, 2].tolist(),
            scalar_codes=expected[:, 2].tolist(),
        ),
    )
    assert bits(actual[:, :2]) == bits(expected[:, :2])
    assert np.array_equal(actual[:, 2], expected[:, 2])


@pytest.mark.parametrize("count", [31, 32, 33, 4095, 4096, 4097])
def test_public_original_row_order_and_chunk_boundary(count, tmp_path, monkeypatch):
    from openboost import _normal_comparison

    calls = []
    original_rows = _normal_comparison.rows

    def observe(*args):
        calls.append(len(args[0]))
        return original_rows(*args)

    monkeypatch.setattr(_normal_comparison, "rows", observe)
    before = np.zeros((count, 2))
    before[::2, 0] = 2.0**-30
    after = np.zeros_like(before)
    weights = np.resize(np.array([1.0, 3.0, 0.0]), count)
    x = NumericData(np.zeros((count, 1)), np.arange(count), ("x",))
    problem = Problem(x, np.zeros((count, 1)), x.row_ids, weight=weights, raw_width=2)
    total = mass = (0.0, 0.0)
    code = 0
    for old, new, y, off, w in zip(
        before, after, problem.target, problem.offset, weights, strict=True
    ):
        lo, hi, status = cpu_row_change(*old, *new, float(y[0]), *off)
        code = max(code, status)
        total = cpu_add(total, cpu_mul((lo, hi), (float(w), float(w))))
        mass = cpu_add(mass, (float(w), float(w)))
    expected = _normal_result(*cpu_div(total, mass), code, False)
    actual = Normal.compare(problem, before, after)
    assert calls == ([count] if count >= 32 else [])
    retain(
        tmp_path,
        "public/" + str(count),
        dict(count=count, actual=asdict(actual), scalar=asdict(expected)),
    )
    assert actual == expected and actual.improves()


def test_mixed_domain_signed_zero_and_first_failure_priority(tmp_path):
    prescribed = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0],
            [2.0**-30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0**-52, 1.0, 0.0, 0.0],
            [1.0, -31.0, 2.0, 31.0, 3.0, -1.0, 0.0],
            [1.0, 32.0, 2.0, 32.0, 3.0, 0.0, 0.0],
            [1.0, -32.0, 2.0, -32.0, 3.0, 0.0, 0.0],
            [1.0, 40.0, 2.0, 1e308, 3.0, 0.0, 0.0],
            [1.0, 1e308, 2.0, 40.0, 3.0, 0.0, 1e308],
            [1e308, 0.0, -1e308, 0.0, 1e308, 1e308, 0.0],
            [np.nextafter(0.0, 1.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 2.0, -1.0, 3.0, 4.0, 0.25],
            [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        np.float64,
    )
    expected = np.array([cpu_row_change(*map(float, r)) for r in prescribed])
    actual = rows(prescribed[:, :2], prescribed[:, 2:4], prescribed[:, 4], prescribed[:, 5:])
    retain(
        tmp_path,
        "mixed-domain",
        dict(
            input_bits=bits(prescribed),
            actual_bits=bits(actual[:, :2]),
            scalar_bits=bits(expected[:, :2]),
            actual_codes=actual[:, 2].tolist(),
            scalar_codes=expected[:, 2].tolist(),
        ),
    )
    assert bits(actual[:, :2]) == bits(expected[:, :2])
    assert np.array_equal(actual[:, 2], expected[:, 2])
    assert actual[0, 0] == actual[0, 1] == 0 and actual[2, 1] < 0


def test_public_domain_validation_precedes_unsupported_and_zero_weight():
    count = 33
    x = NumericData(np.zeros((count, 1)), np.arange(count), ("x",))
    weights = np.zeros(count)
    weights[0] = 1
    problem = Problem(x, np.zeros((count, 1)), x.row_ids, weight=weights, raw_width=2)
    before = np.zeros((count, 2))
    before[0, 1] = 33
    after = before.copy()
    after[-1, 1] = 1000
    with pytest.raises((ValueError, FloatingPointError)):
        Normal.compare(problem, before, after)
