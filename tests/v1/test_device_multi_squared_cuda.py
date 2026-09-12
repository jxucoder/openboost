"""118 resident geometry, diagonal fields and comparison controls; real CUDA only."""

import hashlib
import json
from dataclasses import asdict
from fractions import Fraction

import numpy as np
import pytest

from openboost import device_multi_squared as objective
from openboost import device_objectives as operations
from openboost.binning import Binning
from openboost.data import NumericData, Problem
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext
from openboost.objectives import MultiSquared

from .multi_squared_artifacts import directory, input_snapshot
from .reference import multi_squared as ref
from .test_multi_squared_reference import prepared

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("projected", [False, True])
def test_geometry_and_separate_projected_and_full_fields(width, projected):
    train, _, binning = prepared(width)
    expected_base, steps = ref.rounds(width)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        data = ops.prepare(binning.transform(train.data), train)
        problem = objective.prepare(ops, data, train)
        before = dict(context.metrics)
        base = objective.base(ops, problem)
        raw = operations.broadcast(ops, base, 8)
        g, h = objective.geometry(ops, problem, raw)
        full = operations.vector_fields(ops, data, g, h)
        projection = np.eye(width, dtype=np.float32)[:, :1] if projected else None
        split = operations.vector_fields(ops, data, g, h, projection=projection)
        channels = [operations.channel_fields(ops, data, g, h, channel=k) for k in range(width)]
        value = objective.loss(ops, problem, raw)
        after = dict(context.metrics)
        assert before["upload_bytes"] == after["upload_bytes"]
        assert (
            after["export_bytes"] - before["export_bytes"]
            == after["validation_export_bytes"] - before["validation_export_bytes"] + 8
        )
        np.testing.assert_array_equal(context.export(base), expected_base)
        np.testing.assert_allclose(context.export(g), steps[0]["gradient"], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(context.export(h), np.ones((8, width)))
        weighted = np.column_stack((context.export(g), np.ones((8, width)))) * train.weight[:, None]
        np.testing.assert_array_equal(context.export(full.values), weighted)
        expected_split = weighted[:, [0, width]] if projected else weighted
        np.testing.assert_array_equal(context.export(split.values), expected_split)
        for k, fields in enumerate(channels):
            np.testing.assert_array_equal(
                context.export(fields.values), weighted[:, [k, width + k]]
            )
        assert value == pytest.approx(
            MultiSquared.loss(train, np.tile(expected_base, (8, 1))), rel=1e-6
        )
        with pytest.raises(ValueError, match="already applied"):
            ops.apply_weight(full)
        buffers, records = set(context._buffers), set(ops._records)
        with pytest.raises(ValueError):
            operations.channel_fields(ops, data, g, h, channel=width)
        assert set(context._buffers) == buffers and set(ops._records) == records


@pytest.mark.parametrize("case", ref.CASES, ids=lambda c: c["id"])
def test_comparison_enclosure_reverse_and_snapshot_ownership(case, monkeypatch, tmp_path):
    old, new, target, offset, weight = case["arrays"]
    data = NumericData(np.arange(len(old))[:, None], np.arange(len(old)), ("x",))
    host = Problem(data, target, data.row_ids, offset=offset, weight=weight, raw_width=old.shape[1])
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        device_data = ops.prepare(Binning.fit(data, bins=4).transform(data), host)
        problem = objective.prepare(ops, device_data, host)
        before, after = context.upload(old), context.upload(new)
        buffers, records = set(context._buffers), set(ops._records)
        metrics = dict(context.metrics)
        monkeypatch.setattr(
            objective, "loss", lambda *a: pytest.fail("reporting loss used for comparison")
        )
        comparisons = []
        for a, b, expected in (
            (before, after, ref.change(*case["arrays"])),
            (after, before, -ref.change(*case["arrays"])),
        ):
            result = objective.compare(ops, problem, a, b)
            assert Fraction(result.lower) <= expected <= Fraction(result.upper)
            if expected:
                assert result.status == ("improvement" if expected < 0 else "worsening")
            assert result.unchanged == np.array_equal(old, new)
            comparisons.append(dict(result=asdict(result), exact=str(expected)))
        assert set(context._buffers) == buffers and set(ops._records) == records
        final = dict(context.metrics)
        assert final["upload_bytes"] == metrics["upload_bytes"]
        assert final["comparison_export_bytes"] - metrics.get("comparison_export_bytes", 0) == 64
        assert (
            final["export_bytes"] - metrics["export_bytes"]
            == 64 + final["validation_export_bytes"] - metrics["validation_export_bytes"]
        )
        np.testing.assert_array_equal(context.export(before), old)
        np.testing.assert_array_equal(context.export(after), new)
        report = dict(
            case=case["id"],
            inputs=input_snapshot(host),
            before=old.tolist(),
            after=new.tolist(),
            comparisons=comparisons,
        )
        (
            directory("comparisons", tmp_path)
            / (hashlib.sha256(case["id"].encode()).hexdigest()[:16] + ".json")
        ).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


@pytest.mark.parametrize("method", ["geometry", "loss", "compare"])
@pytest.mark.parametrize("same", [False, True])
def test_zero_weight_invalid_geometry_cannot_hide_behind_identity(method, same):
    old = np.array([[0, 0], [3e38, 3e38]], np.float32)
    offset = np.array([[0, 0], [3e38, 3e38]], np.float32)
    data = NumericData([[0], [1]], [0, 1], ("x",))
    host = Problem(data, np.zeros((2, 2)), data.row_ids, offset=offset, weight=[1, 0], raw_width=2)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem = objective.prepare(
            ops, ops.prepare(Binning.fit(data, bins=4).transform(data), host), host
        )
        before, after = context.upload(old), context.upload(old if same else np.zeros_like(old))
        buffers, records = set(context._buffers), set(ops._records)
        with pytest.raises(ValueError):
            getattr(objective, method)(
                ops, problem, before, after
            ) if method == "compare" else getattr(objective, method)(ops, problem, before)
        assert set(context._buffers) == buffers and set(ops._records) == records


def test_nontrivial_projection_curvature_and_directed_ptx(tmp_path):
    train, _, binning = prepared(2)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        data = ops.prepare(binning.transform(train.data), train)
        problem = objective.prepare(ops, data, train)
        raw = operations.broadcast(ops, objective.base(ops, problem), 8)
        g, h = objective.geometry(ops, problem, raw)
        projection = np.array([[1, -0.5], [2, 1]], np.float32)
        fields = operations.vector_fields(ops, data, g, h, projection=projection)
        expected = (
            np.column_stack(
                (
                    context.export(g).astype(float) @ projection,
                    context.export(h).astype(float) @ (projection**2),
                )
            )
            * train.weight[:, None]
        )
        np.testing.assert_allclose(context.export(fields.values), expected, rtol=1e-6, atol=1e-6)
        changed = context.upload(context.export(raw) + np.float32(0.25))
        objective.compare(ops, problem, raw, changed)
        from openboost import _device_kernels as kernels

        ptx = "\n".join(kernels.multi_squared_compare_rows.inspect_asm().values())
        assert "add.rm.f64" in ptx and "add.rp.f64" in ptx
        assert "mul.rm.f64" in ptx and "mul.rp.f64" in ptx
        reduction_ptx = "\n".join(kernels.glm_compare_reduce.inspect_asm().values())
        assert "div.rm.f64" in reduction_ptx and "div.rp.f64" in reduction_ptx
        (directory("comparisons", tmp_path) / "multi-squared-ptx.json").write_text(
            json.dumps(dict(row_ptx=ptx, reduction_ptx=reduction_ptx), indent=2) + "\n"
        )


@pytest.mark.parametrize("operation", ["geometry", "compare", "fields"])
def test_partial_dispatch_failure_releases_new_outputs(operation, monkeypatch):
    train, _, binning = prepared(2)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        data = ops.prepare(binning.transform(train.data), train)
        problem = objective.prepare(ops, data, train)
        raw = operations.broadcast(ops, objective.base(ops, problem), 8)
        g, h = objective.geometry(ops, problem, raw)
        original = ops._launch
        failing = {
            "geometry": "multi_squared_geometry",
            "compare": "glm_compare_reduce",
            "fields": "projected_diagonal_fields",
        }[operation]

        def dispatch(name, *args):
            if name == failing:
                raise RuntimeError("injected multi-output dispatch failure")
            return original(name, *args)

        monkeypatch.setattr(ops, "_launch", dispatch)
        buffers, records = set(context._buffers), set(ops._records)
        with pytest.raises(RuntimeError, match="injected"):
            if operation == "geometry":
                objective.geometry(ops, problem, raw)
            elif operation == "compare":
                objective.compare(ops, problem, raw, raw)
            else:
                operations.vector_fields(ops, data, g, h, projection=[[1], [0]])
        assert set(context._buffers) == buffers and set(ops._records) == records
