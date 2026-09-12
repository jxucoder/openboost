"""Resident AFT components: these cases require actual CUDA, never simulation."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost import device_aft as aft
from openboost import device_objectives as operations
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .reference.device_aft import DOMAIN_CASES, GEOMETRY_ATOL, GEOMETRY_RTOL, base, geometry
from .test_device_aft_reference import fixture

pytestmark = pytest.mark.gpu


def prepared(ops, problem, sigma=1):
    binned = Binning.fit(problem.data, bins=8).transform(problem.data)
    return aft.objective(sigma).prepare(ops, ops.prepare(binned, problem), problem)


def snapshot(ops):
    return set(ops._records), set(ops.execution._buffers), ops.execution.metrics["live_bytes"]


@pytest.mark.parametrize("sigma", [0.5, 0.7, 1, 2])
@pytest.mark.parametrize("all_censored", [False, True])
def test_resident_base_geometry_loss_fields_and_transfers(sigma, all_censored):
    p = fixture(all_censored=all_censored)
    p = replace(p, weight=np.where(np.arange(8) == 4, 0, p.weight))
    objective = aft.objective(sigma)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepared(ops, p, sigma)
        assert problem.sigma == sigma and problem.target_width == 2 and problem.raw_width == 1
        initial = objective.base(ops, problem)
        raw = operations.broadcast(ops, initial, 8)
        expected_base = base(p.target[:, 0], p.offset[:, 0], p.weight)
        np.testing.assert_allclose(context.export(initial), [expected_base], rtol=1e-6, atol=1e-8)
        expected = geometry(np.full(8, expected_base), p.target[:, 0], p.target[:, 0] == p.target[:, 1], p.offset[:, 0], p.weight, sigma)
        before = dict(context.metrics)
        matrix = aft.geometry(ops, problem, raw)
        gradient = objective.gradient(ops, problem, raw)
        fields = objective.fields(ops, problem, raw)
        loss = objective.loss(ops, problem, raw)
        after = dict(context.metrics)
        assert after["upload_bytes"] == before["upload_bytes"]
        assert after["export_bytes"] - before["export_bytes"] == sum(
            after.get(k, 0) - before.get(k, 0) for k in ("validation_export_bytes", "metric_export_bytes")
        )
        assert after["metric_export_bytes"] - before.get("metric_export_bytes", 0) == 8
        assert loss == pytest.approx(expected[0], rel=GEOMETRY_RTOL, abs=GEOMETRY_ATOL)
        np.testing.assert_allclose(context.export(matrix), np.column_stack(expected[1:]), rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)
        assert fields.names == ("gradient", "curvature") and fields.roles == ("training", "training")
        with pytest.raises(ValueError, match="already applied"):
            ops.apply_weight(fields)
        context.release(matrix)
        context.release(raw)
        ops.release(problem)
        np.testing.assert_allclose(context.export(gradient), expected[1], rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)
        np.testing.assert_allclose(context.export(fields.values), np.column_stack(expected[1:]) * p.weight[:, None], rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)


@pytest.mark.parametrize("zero_weight", [False, True])
@pytest.mark.parametrize("name,raw,lower,event,sigma,valid", DOMAIN_CASES, ids=[r[0] for r in DOMAIN_CASES])
def test_all_row_domains_and_atomic_failures(name, raw, lower, event, sigma, valid, zero_weight):
    data = NumericData([[0], [1]], [0, 1], ("x",))
    p = Problem(data, [[lower, lower if event else np.inf], [1, 1]], data.row_ids,
                target_kind="event_right", weight=[0 if zero_weight else 1, 1])
    objective = aft.objective(sigma)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        if name in ("lost_time", "time_overflow"):
            binned = Binning.fit(data, bins=8).transform(data)
            resident = ops.prepare(binned, p)
            before = snapshot(ops)
            with pytest.raises(ValueError, match="float32"):
                objective.prepare(ops, resident, p)
            assert snapshot(ops) == before
            return
        problem = prepared(ops, p, sigma)
        values = context.upload(np.array([[raw], [0]], np.float32))
        before = snapshot(ops)
        calls = (lambda: aft.geometry(ops, problem, values),
                 lambda: objective.gradient(ops, problem, values),
                 lambda: objective.fields(ops, problem, values),
                 lambda: objective.loss(ops, problem, values))
        if not valid:
            for call in calls:
                with pytest.raises(ValueError, match="float32 support"):
                    call()
                assert snapshot(ops) == before
        else:
            expected = geometry([raw, 0], [lower, 1], [event, True], [0, 0], p.weight, sigma)
            actual = context.export(calls[0]())
            # Compare tiny values at their stored boundary; zero cannot pass on atol.
            np.testing.assert_allclose(actual, np.column_stack(expected[1:]), rtol=GEOMETRY_RTOL, atol=0)
            assert objective.loss(ops, problem, values) == pytest.approx(expected[0], rel=GEOMETRY_RTOL, abs=0)


def test_scale_family_identity_and_shape_are_not_interchangeable():
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, objective = fixture(), aft.objective(0.7)
        problem = prepared(ops, p, 0.7)
        before = snapshot(ops)
        for wrong_ops, wrong_problem in ((ops, replace(problem)), (DeviceOperations(context), problem)):
            with pytest.raises(ValueError, match="forged"):
                objective.base(wrong_ops, wrong_problem)
        with pytest.raises(ValueError, match="family"):
            operations.SQUARED.base(ops, problem)
        for operation in ("base", "loss", "gradient", "fields"):
            with pytest.raises(ValueError, match="scale"):
                getattr(aft.objective(1), operation)(ops, problem, *(() if operation == "base" else (None,)))
        assert snapshot(ops) == before
        raw = context.upload(np.zeros((8, 2), np.float32))
        with pytest.raises(ValueError, match="shape"):
            objective.loss(ops, problem, raw)
        with pytest.raises(ValueError, match="identity"):
            objective.prepare(ops, problem.data, replace(p, offset=p.offset + 1))
        ops.release(problem)
        with pytest.raises(ValueError, match="released"):
            objective.base(ops, problem)


@pytest.mark.parametrize("operation", ["base", "gradient", "fields", "loss"])
def test_partial_failure_rolls_back_allocations(operation, monkeypatch):
    with ExecutionContext() as context:
        ops, objective = DeviceOperations(context), aft.objective(0.7)
        problem = prepared(ops, fixture(), 0.7)
        raw = operations.broadcast(ops, objective.base(ops, problem), 8)
        before, launch = snapshot(ops), ops._launch
        def fail(name, *args):
            if name.startswith("aft_") or name == "glm_gradient":
                raise RuntimeError("injected AFT dispatch failure")
            return launch(name, *args)
        with monkeypatch.context() as patch:
            patch.setattr(ops, "_launch", fail)
            with pytest.raises(RuntimeError, match="injected"):
                getattr(objective, operation)(ops, problem, *(() if operation == "base" else (raw,)))
        assert snapshot(ops) == before
        assert np.isfinite(objective.loss(ops, problem, raw))


def test_base_validates_zero_weight_rows_after_broadcast():
    p = fixture()
    offset = p.offset.copy()
    offset[4, 0] = 1000
    p = replace(p, offset=offset, weight=np.where(np.arange(8) == 4, 0, p.weight))
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepared(ops, p)
        before = snapshot(ops)
        with pytest.raises(ValueError, match="float32 support"):
            aft.base(ops, problem)
        assert snapshot(ops) == before
