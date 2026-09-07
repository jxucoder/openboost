"""090-B Normal components; collect locally, execute only on real CUDA hardware."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost import device_normal as normal
from openboost import device_objectives as operations
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .reference.device_normal import base, direction, fixture
from .reference.normal_precision import (
    ATOL,
    DOMAIN_CASES,
    LOSS_ATOL,
    LOSS_RTOL,
    RTOL,
    stored_geometry,
)
from .test_device_normal_reference import prepared_fixture

pytestmark = pytest.mark.gpu


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def prepare(ops, p, binned):
    return normal.prepare(ops, ops.prepare(binned, p), p)


@pytest.mark.parametrize(
    "name,raw,target,offset,valid", DOMAIN_CASES, ids=[c[0] for c in DOMAIN_CASES]
)
def test_normal_float32_domain_and_failure_cleanup(name, raw, target, offset, valid):
    data = NumericData([[0]], [101], ("x",))
    p = Problem(data, [[target]], data.row_ids, offset=[offset], raw_width=2)
    binned = Binning(("x",), (np.array([]),)).transform(data)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepare(ops, p, binned)
        values = context.upload(np.array([raw], np.float32))
        before, records, handles = (
            context.metrics["live_bytes"],
            set(ops._records),
            set(context._buffers),
        )
        if valid:
            expected = stored_geometry([raw], [target], [offset], [1])
            gradient, fisher = normal.geometry(ops, problem, values)
            close(context.export(gradient), expected[1])
            close(context.export(fisher), expected[2])
            assert np.all(context.export(fisher) > 0)
            assert normal.loss(ops, problem, values) == pytest.approx(
                expected[0], rel=LOSS_RTOL, abs=LOSS_ATOL
            )
        else:
            for operation in (normal.geometry, normal.loss):
                with pytest.raises(ValueError, match="float32 support"):
                    operation(ops, problem, values)
                assert context.metrics["live_bytes"] == before
                assert set(ops._records) == records and set(context._buffers) == handles
            close(context.export(values), [raw])


@pytest.mark.parametrize("case", ["weighted", "d2", "conflict"])
@pytest.mark.parametrize("mode,damping", [("ordinary", 0), ("natural", 0), ("natural", 0.25)])
def test_base_geometry_directions_fields_and_residency(case, mode, damping):
    train, _, binned, information = prepared_fixture(case)
    f = fixture(case)
    initial = base(f["target"], f["offset"], f["weight"])
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepare(ops, train, binned)
        before = dict(context.metrics)
        center = normal.base(ops, problem)
        values = operations.broadcast(ops, center, len(train.target))
        gradient, fisher = normal.geometry(ops, problem, values)
        z = operations.diagonal_direction(ops, gradient, fisher, mode=mode, damping=damping)
        fields = tuple(operations.least_squares(ops, problem.data, z, k) for k in (0, 1))
        loss = normal.loss(ops, problem, values)
        after = dict(context.metrics)
        assert after["upload_bytes"] == before["upload_bytes"]
        assert after["export_bytes"] - before["export_bytes"] == (
            after["validation_export_bytes"] - before["validation_export_bytes"] + 8
        )
        assert after["metric_export_bytes"] - before.get("metric_export_bytes", 0) == 8
        close(context.export(center), initial)
        expected = stored_geometry(
            context.export(values), train.target[:, 0], train.offset, train.weight
        )
        close(context.export(gradient), expected[1])
        close(context.export(fisher), expected[2])
        expected_z = direction(
            expected[1].astype(float), expected[2].astype(float), mode=mode, damping=damping
        )
        close(context.export(z), expected_z)
        assert loss == pytest.approx(expected[0], rel=LOSS_RTOL, abs=LOSS_ATOL)
        for k, field in enumerate(fields):
            assert field.names == ("gradient", "curvature") and field.roles == (
                "training",
                "training",
            )
            close(
                context.export(field.values),
                np.column_stack((-train.weight * expected_z[:, k], train.weight)),
            )
            with pytest.raises(ValueError, match="already applied"):
                ops.apply_weight(field)
        info = context.upload(information[:, 0].astype(np.float32))
        augmented = ops.add_independent(fields[0], "cohort", info, nonnegative=True)
        context.release(info)
        close(context.export(augmented.values)[:, 2], information[:, 0])
        assert augmented.roles[-1] == "independent"
        context.release(gradient)
        context.release(fisher)
        context.release(values)
        close(context.export(z), expected_z)  # Outputs own their storage.
        ops.release(problem)
        close(context.export(augmented.values)[:, 2], information[:, 0])


@pytest.mark.parametrize("mode,damping", [("ordinary", 0), ("natural", 0.25)])
def test_nonpositive_metric_rejected_even_with_damping(mode, damping):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        gradient = context.upload(np.array([[1, 2]], np.float32))
        metric = context.upload(np.array([[0, 2]], np.float32))
        before = context.metrics["live_bytes"]
        with pytest.raises(ValueError):
            operations.diagonal_direction(ops, gradient, metric, mode=mode, damping=damping)
        assert context.metrics["live_bytes"] == before


def test_problem_identity_shape_and_numeric_initialization():
    train, _, binned, _ = prepared_fixture("weighted")
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepare(ops, train, binned)
        with pytest.raises(ValueError, match="forged"):
            normal.base(ops, replace(problem))
        with pytest.raises(ValueError, match="widths"):
            operations.base(ops, problem)
        raw = context.upload(np.zeros((len(train.target), 1), np.float32))
        with pytest.raises(ValueError, match="shape"):
            normal.geometry(ops, problem, raw)
        bad = replace(train, offset=np.tile([0, -1000.0], (len(train.target), 1)))
        bad_problem = prepare(ops, bad, binned)
        before = context.metrics["live_bytes"]
        with pytest.raises(ValueError):
            normal.base(ops, bad_problem)
        assert context.metrics["live_bytes"] == before


def test_actual_sixth_trial_geometry_and_zero_weight_domain():
    data = NumericData([[0], [0]], [101, 301], ("x",))
    train = Problem(data, [[100], [100]], data.row_ids, weight=[1, 0], raw_width=2)
    binned = Binning(("x",), (np.array([]),)).transform(data)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepare(ops, train, binned)
        for j in range(6):
            alpha = 0.2 * 0.5**j
            raw = context.upload(np.tile([alpha * 100, alpha * 5000], (2, 1)).astype(np.float32))
            if j < 5:
                with pytest.raises(ValueError):
                    normal.loss(ops, problem, raw)
            else:
                assert normal.loss(ops, problem, raw) < 5001
            context.release(raw)
        raw = context.upload(np.array([[0, 0], [0, -50]], np.float32))
        with pytest.raises(ValueError):
            normal.loss(ops, problem, raw)


def test_initialization_normalizes_weights_before_offset_precision():
    data = NumericData([[0], [1]], [101, 301], ("x",))
    p = Problem(
        data,
        [[-1], [1]],
        data.row_ids,
        weight=[1e30, 1e30],
        offset=[[0, -345], [0, -345]],
        raw_width=2,
    )
    binned = Binning(("x",), (np.array([0.5]),)).transform(data)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepare(ops, p, binned)
        initial = normal.base(ops, problem)
        close(context.export(initial), [0, 345])
        raw = operations.broadcast(ops, initial, 2)
        assert normal.loss(ops, problem, raw) == pytest.approx(
            0.5 + np.log(2 * np.pi) / 2, rel=LOSS_RTOL, abs=LOSS_ATOL
        )
