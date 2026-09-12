"""AFT evidence must retain censoring infinities, exact input bytes and JSON cases."""

import base64
import json

import numpy as np

from .aft_artifacts import input_snapshot
from .reference.aft_comparison import CASES
from .test_device_aft_reference import fixture


def test_every_frozen_comparison_is_strict_json_serializable():
    encoded = json.dumps(CASES, allow_nan=False)
    assert json.loads(encoded) == list(CASES)


def test_lossless_input_snapshot_retains_missing_and_right_censoring():
    problem = fixture()
    snapshot = json.loads(json.dumps(input_snapshot(problem), allow_nan=False))
    for name, original in (("features", problem.data.values), ("row_ids", problem.data.row_ids),
                           ("target", problem.target), ("offset", problem.offset), ("weight", problem.weight)):
        item = snapshot[name]
        data = base64.b64decode(item["data_base64"], validate=True)
        restored = np.frombuffer(data, dtype=item["dtype"]).reshape(item["shape"])
        assert restored.dtype == original.dtype and restored.shape == original.shape
        assert data == np.ascontiguousarray(original).tobytes()
    assert np.isposinf(problem.target[:, 1]).any() and np.isnan(problem.data.values).any()


def test_actual_cuda_comparison_host_fixtures_preserve_frozen_inputs():
    from openboost import device_aft

    from .test_device_aft_comparison_cuda import host_problem

    for case in CASES:
        p = host_problem(case["arrays"])
        lower, offset, event = device_aft._host_arrays(p)
        np.testing.assert_array_equal(lower[:, 0], case["arrays"][2])
        np.testing.assert_array_equal(event, case["arrays"][3])
        np.testing.assert_array_equal(offset[:, 0], case["arrays"][4])
        np.testing.assert_array_equal(p.weight, case["arrays"][5])
