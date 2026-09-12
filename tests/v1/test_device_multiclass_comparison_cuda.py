"""111 comparison enclosures, ownership and lowering; real CUDA required."""

import hashlib
import json
from dataclasses import asdict, replace
from decimal import Decimal

import numpy as np
import pytest

from openboost import ClassSchema, NumericData, Problem
from openboost import device_multiclass as multiclass
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext
from openboost.objectives import Multiclass

from .multiclass_artifacts import directory
from .reference.multiclass_comparison import CASES, direct_difference
from .test_device_glm_cuda import snapshot

pytestmark = pytest.mark.gpu


def prepare(ops, arrays):
    old, new, target, offset, weight = (np.asarray(a, np.float32) for a in arrays)
    data = NumericData(np.arange(len(target))[:, None], np.arange(len(target)), ("x",))
    problem = Problem(
        data,
        target[:, None],
        data.row_ids,
        raw_width=old.shape[1],
        classes=ClassSchema(tuple(f"class-{j}" for j in range(old.shape[1]))),
        offset=offset,
        weight=weight,
    )
    binned = Binning.fit(data, bins=4).transform(data)
    prepared = multiclass.prepare(ops, ops.prepare(binned, problem), problem)
    return prepared, ops.execution.upload(old), ops.execution.upload(new)


def simple():
    return next(c["arrays"] for c in CASES if c["id"] == "k3/tiny-improvement")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_independent_comparison_and_scalar_only_transfers(case, tmp_path):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, case["arrays"])
        owned, before = snapshot(ops), dict(context.metrics)
        result = multiclass.objective().loss_change(ops, p, old, new)
        after = dict(context.metrics)
        assert snapshot(ops) == owned
        assert after["upload_bytes"] == before["upload_bytes"]
        assert after["comparison_export_bytes"] - before.get("comparison_export_bytes", 0) == 32
        assert (
            after["export_bytes"] - before["export_bytes"]
            == 32 + after["validation_export_bytes"] - before["validation_export_bytes"]
        )
        assert after["comparison_calls"] - before.get("comparison_calls", 0) == 1
        exact = direct_difference(*case["arrays"])
        assert Decimal(result.lower) <= exact <= Decimal(result.upper)
        if case["status"] is not None:
            assert result.status == case["status"]
        assert result.unchanged == (case["arrays"][0] == case["arrays"][1])
        np.testing.assert_array_equal(context.export(old), case["arrays"][0])
        np.testing.assert_array_equal(context.export(new), case["arrays"][1])
        reverse = multiclass.compare(ops, p, new, old)
        assert Decimal(reverse.lower) <= -exact <= Decimal(reverse.upper)
        arrays = [np.asarray(a, np.float32) for a in case["arrays"]]
        width = arrays[0].shape[1]
        order = np.roll(np.arange(width), 1)
        permuted_arrays = [
            arrays[0][::-1][:, order],
            arrays[1][::-1][:, order],
            np.argsort(order)[arrays[2][::-1].astype(int)],
            arrays[3][::-1][:, order],
            arrays[4][::-1],
        ]
        other, a, b = prepare(ops, permuted_arrays)
        permuted = multiclass.compare(ops, other, a, b)
        assert Decimal(permuted.lower) <= exact <= Decimal(permuted.upper)
        report = dict(
            case=case,
            comparison=asdict(result),
            reverse=asdict(reverse),
            permuted=asdict(permuted),
            decimal260=str(exact),
            counters={
                k: after.get(k, 0) - before.get(k, 0)
                for k in (
                    "comparison_calls",
                    "comparison_export_bytes",
                    "validation_export_bytes",
                    "export_bytes",
                    "upload_bytes",
                    "kernel_launches",
                    "synchronizations",
                )
            },
        )
        name = hashlib.sha256(case["id"].encode()).hexdigest()[:16] + ".json"
        (directory("comparisons", tmp_path) / name).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )


@pytest.mark.parametrize("fault", ["shape", "nonfinite", "zero-weight-domain", "unselected-class"])
def test_invalid_unchanged_snapshot_cannot_pass(fault):
    arrays = [[[0, 0, 0]] * 2, [[0, 0, 0]] * 2, [0, 1], [[0, 0, 0]] * 2, [1, 0]]
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, arrays)
        bad = np.zeros((2, 2 if fault == "shape" else 3), np.float32)
        if fault == "nonfinite":
            bad[-1, 0] = np.nan
        elif fault == "zero-weight-domain":
            bad[-1, 0] = 120
        elif fault == "unselected-class":
            bad[-1, 2] = -120
        value = context.upload(bad)
        owned = snapshot(ops)
        with pytest.raises(ValueError):
            multiclass.compare(ops, p, value, value)
        assert snapshot(ops) == owned
        assert multiclass.compare(ops, p, old, new).unchanged


def test_foreign_identity_and_released_snapshot():
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, simple())
        owned = snapshot(ops)
        for other_ops, other_p in ((ops, replace(p)), (DeviceOperations(context), p)):
            with pytest.raises(ValueError, match="forged"):
                multiclass.compare(other_ops, other_p, old, new)
        assert snapshot(ops) == owned
        context.release(new)
        with pytest.raises(ValueError, match="released"):
            multiclass.compare(ops, p, old, new)


@pytest.mark.parametrize("failure", ["allocation1", "allocation2", "reduce"])
def test_partial_failures_restore_owned_buffers(failure, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, simple())
        owned = snapshot(ops)
        empty, launch = context._empty, ops._launch
        count = 0

        def allocate(shape, dtype):
            nonlocal count
            if np.dtype(dtype) == np.dtype(np.float64):
                count += 1
                if failure == "allocation" + str(count):
                    raise MemoryError("injected comparison allocation")
            return empty(shape, dtype)

        def dispatch(name, *args):
            if failure == "reduce" and name == "glm_compare_reduce":
                raise RuntimeError("injected comparison reduction")
            return launch(name, *args)

        with monkeypatch.context() as patch:
            patch.setattr(context, "_empty", allocate)
            patch.setattr(ops, "_launch", dispatch)
            with pytest.raises((MemoryError, RuntimeError), match="injected"):
                multiclass.compare(ops, p, old, new)
        assert snapshot(ops) == owned
        assert multiclass.compare(ops, p, old, new).improves()


def test_no_host_or_reporting_fallback(monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, simple())

        def forbidden(*args, **kwargs):
            raise AssertionError("host/reporting fallback forbidden")

        for owner, name in (
            (Multiclass, "geometry"),
            (multiclass, "geometry"),
            (multiclass, "loss"),
        ):
            monkeypatch.setattr(owner, name, forbidden)
        assert multiclass.compare(ops, p, old, new).improves()


def test_actual_ptx_contains_directed_double_operations(tmp_path):
    from openboost import _device_multiclass_comparison as kernels

    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, simple())
        multiclass.compare(ops, p, old, new)
        ptx = "\n".join(kernels.multiclass_compare_rows.inspect_asm().values())
        instructions = [f"{op}.{mode}.f64" for op in ("add", "mul", "div") for mode in ("rm", "rp")]
        report = dict(required_instructions=instructions, ptx=ptx)
        (directory("comparisons", tmp_path) / "multiclass-ptx.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        assert all(instruction in ptx for instruction in instructions)
