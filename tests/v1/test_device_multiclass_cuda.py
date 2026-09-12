"""110 resident multiclass contracts; collect locally, execute only on real CUDA."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import device_glm, device_normal
from openboost import device_multiclass as multiclass
from openboost import device_objectives as operations
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .reference.device_multiclass import DOMAIN_CASES, GEOMETRY_RTOL, geometry
from .test_device_glm_cuda import snapshot
from .test_device_multiclass_reference import fixture

pytestmark = pytest.mark.gpu


def prepared(ops, p):
    binned = Binning.fit(p.data, bins=32).transform(p.data)
    return multiclass.prepare(ops, ops.prepare(binned, p), p)


@pytest.mark.parametrize("width", [2, 3, 5])
def test_resident_geometry_fields_metrics_transfers_and_ownership(width):
    p = fixture(width)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepared(ops, p)
        before = dict(context.metrics)
        initial = multiclass.base(ops, problem)
        raw = operations.broadcast(ops, initial, len(p.target))
        g, h = multiclass.geometry(ops, problem, raw)
        fields = [
            multiclass.channel_fields(ops, problem.data, g, h, channel=j) for j in range(width)
        ]
        score = multiclass.loss(ops, problem, raw)
        after = dict(context.metrics)
        assert after["upload_bytes"] == before["upload_bytes"]
        assert after["export_bytes"] - before["export_bytes"] == (
            after["validation_export_bytes"] - before["validation_export_bytes"] + 8
        )
        assert after["metric_export_bytes"] - before.get("metric_export_bytes", 0) == 8
        np.testing.assert_array_equal(context.export(initial), np.zeros(width))
        expected = geometry(np.zeros((len(p.target), width)), p.target[:, 0], p.offset, p.weight)
        assert score == pytest.approx(expected[0], rel=GEOMETRY_RTOL, abs=0)
        for actual, required in ((g, expected[1]), (h, expected[2])):
            np.testing.assert_allclose(context.export(actual), required, rtol=GEOMETRY_RTOL, atol=0)
        np.testing.assert_allclose(context.export(g).sum(axis=1), 0, atol=2e-7)
        # Independently owned fields outlive both their geometry and prepared problem.
        for handle in (g, h, initial, raw):
            context.release(handle)
        ops.release(problem)
        for j, field in enumerate(fields):
            required = np.column_stack((expected[1][:, j], expected[2][:, j])) * p.weight[:, None]
            np.testing.assert_allclose(
                context.export(field.values), required, rtol=GEOMETRY_RTOL, atol=0
            )
            assert field.names == ("gradient", "curvature") and field.roles == (
                "training",
                "training",
            )
            with pytest.raises(ValueError, match="already applied"):
                ops.apply_weight(field)


@pytest.mark.parametrize("name,values,code,valid", DOMAIN_CASES, ids=[c[0] for c in DOMAIN_CASES])
@pytest.mark.parametrize("zero_weight", [False, True])
def test_all_row_domains_and_atomic_failure(name, values, code, valid, zero_weight):
    p = fixture()
    target = p.target.copy()
    target[0] = code
    weight = p.weight.copy()
    weight[0] = int(not zero_weight)
    p = replace(p, target=target, weight=weight, offset=np.zeros_like(p.offset))
    raw = np.zeros_like(p.offset, dtype=np.float32)
    raw[0] = values
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepared(ops, p)
        stored = context.upload(raw)
        before = snapshot(ops)
        if valid:
            expected = geometry(raw, p.target[:, 0], p.offset, p.weight)
            g, h = multiclass.geometry(ops, problem, stored)
            for actual, required in ((g, expected[1]), (h, expected[2])):
                np.testing.assert_allclose(
                    context.export(actual), required, rtol=GEOMETRY_RTOL, atol=0
                )
            assert multiclass.loss(ops, problem, stored) == pytest.approx(
                expected[0], rel=GEOMETRY_RTOL, abs=0
            )
        else:
            for call in (multiclass.geometry, multiclass.loss):
                with pytest.raises(ValueError, match="float32 support"):
                    call(ops, problem, stored)
                assert snapshot(ops) == before
            np.testing.assert_array_equal(context.export(stored), raw)


@pytest.mark.parametrize("case", ["zero_class_weight", "missing_class", "bad_zero_weight_offset"])
def test_base_class_coverage_and_offset_domain(case):
    p = fixture()
    if case == "zero_class_weight":
        p = replace(p, weight=(p.target[:, 0] == 0).astype(float))
    elif case == "missing_class":
        p = replace(p, target=np.zeros_like(p.target))
    else:
        offset = p.offset.copy()
        offset[0] = [120, 0, 0]
        weight = p.weight.copy()
        weight[0] = 0
        p = replace(p, offset=offset, weight=weight)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepared(ops, p)
        before = snapshot(ops)
        if case == "zero_class_weight":
            np.testing.assert_array_equal(context.export(multiclass.base(ops, problem)), [0, 0, 0])
        else:
            with pytest.raises(ValueError):
                multiclass.base(ops, problem)
            assert snapshot(ops) == before


def test_family_identity_shapes_and_lifetime():
    p = fixture(2)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepared(ops, p)
        before = snapshot(ops)
        for other_ops, other in ((ops, replace(problem)), (DeviceOperations(context), problem)):
            with pytest.raises(ValueError, match="forged"):
                multiclass.base(other_ops, other)
        for wrong in (device_normal.objective(), device_glm.binary(), operations.SQUARED):
            with pytest.raises(ValueError):
                wrong.base(ops, problem)
        # The reverse reinterpretation is also forbidden for identical [1,2] widths.
        normal = replace(p, classes=None)
        binned = Binning.fit(normal.data, bins=8).transform(normal.data)
        normal_problem = device_normal.prepare(ops, ops.prepare(binned, normal), normal)
        with pytest.raises(ValueError, match="family"):
            multiclass.base(ops, normal_problem)
        ops.release(normal_problem.data)
        ops.release(normal_problem)
        assert snapshot(ops) == before
        wrong_raw = context.upload(np.zeros((6, 1), np.float32))
        with pytest.raises(ValueError, match="shape"):
            multiclass.loss(ops, problem, wrong_raw)
        raw = operations.broadcast(ops, multiclass.base(ops, problem), 6)
        ops.release(problem)
        with pytest.raises(ValueError, match="released"):
            multiclass.loss(ops, problem, raw)


@pytest.mark.parametrize("fault", ["nonfinite_raw", "upload_overflow", "identity"])
def test_preparation_and_raw_domain_failure_preserves_ownership(fault):
    p = fixture()
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepared(ops, p)
        values = context.upload(np.full((9, 3), np.inf, np.float32))
        before = snapshot(ops)
        with pytest.raises(ValueError):
            if fault == "nonfinite_raw":
                multiclass.geometry(ops, problem, values)
            else:
                wrong = replace(p, offset=p.offset + (1e40 if fault == "upload_overflow" else 1))
                multiclass.prepare(ops, problem.data, wrong)
        assert snapshot(ops) == before


def test_common_shift_and_class_permutation_preserve_geometry():
    p = fixture()
    order = np.array([2, 0, 1])
    q = replace(p, target=np.argsort(order)[p.target.astype(int)], offset=p.offset[:, order])
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem, permuted = prepared(ops, p), prepared(ops, q)
        raw = context.upload(np.zeros((9, 3), np.float32))
        shifted = context.upload(np.full((9, 3), 800, np.float32))
        original = [context.export(a) for a in multiclass.geometry(ops, problem, raw)]
        for record, values, permutation in (
            (problem, shifted, np.arange(3)),
            (permuted, raw, order),
        ):
            actual = multiclass.geometry(ops, record, values)
            for output, expected in zip(actual, original, strict=True):
                np.testing.assert_allclose(
                    context.export(output), expected[:, permutation], rtol=GEOMETRY_RTOL, atol=0
                )
            assert multiclass.loss(ops, record, values) == pytest.approx(
                multiclass.loss(ops, problem, raw), rel=GEOMETRY_RTOL
            )


@pytest.mark.parametrize("operation", ["base", "geometry", "fields", "loss"])
def test_dispatch_failure_cleanup_and_recovery(operation, monkeypatch):
    p = fixture()
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepared(ops, p)
        raw = operations.broadcast(ops, multiclass.base(ops, problem), 9)
        g, h = multiclass.geometry(ops, problem, raw)
        before = snapshot(ops)
        launch = ops._launch

        def fail(name, *args):
            if name.startswith("multiclass_"):
                raise RuntimeError("injected multiclass dispatch failure")
            return launch(name, *args)

        with monkeypatch.context() as patch:
            patch.setattr(ops, "_launch", fail)
            with pytest.raises(RuntimeError, match="injected"):
                if operation == "fields":
                    multiclass.channel_fields(ops, problem.data, g, h, channel=0)
                else:
                    getattr(multiclass, operation)(
                        ops, problem, *(() if operation == "base" else (raw,))
                    )
        assert snapshot(ops) == before
        assert np.isfinite(multiclass.loss(ops, problem, raw))


@pytest.mark.parametrize(
    "fault", ["zero_bound_other_channel", "negative_bound", "gradient_nan", "shape", "channel"]
)
def test_fields_reject_invalid_shared_geometry(fault):
    p = fixture()
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = prepared(ops, p)
        g, h = np.zeros((9, 3), np.float32), np.ones((9, 3), np.float32)
        if fault == "zero_bound_other_channel":
            h[0, 2] = 0
        elif fault == "negative_bound":
            h[0, 2] = -1
        elif fault == "gradient_nan":
            g[0, 2] = np.nan
        elif fault == "shape":
            h = h[:, :1].copy()
        gh, hh = context.upload(g), context.upload(h)
        before = snapshot(ops)
        with pytest.raises(ValueError):
            multiclass.channel_fields(
                ops, problem.data, gh, hh, channel=3 if fault == "channel" else 0
            )
        assert snapshot(ops) == before
