"""107 resident comparison cohort; collection is not real CUDA validation."""

import hashlib
import json
from dataclasses import asdict, replace
from decimal import Decimal

import numpy as np
import pytest

from openboost import ClassSchema, NumericData, Problem
from openboost import device_glm as glm
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext
from openboost.objectives import Binary, Poisson

from .glm_artifacts import directory
from .reference.glm_comparison import CASES, direct_difference
from .test_device_glm_cuda import snapshot

pytestmark = pytest.mark.gpu


def prepare(ops, family, arrays):
    old, new, y, o, w, e = (np.asarray(v, np.float32) for v in arrays)
    data = NumericData(np.arange(len(y))[:, None], np.arange(len(y)), ("x",))
    problem = Problem(
        data,
        y[:, None],
        data.row_ids,
        offset=o[:, None],
        weight=w,
        classes=ClassSchema(("no", "yes")) if family == "binary" else None,
        structure={"exposure": e[:, None]} if family == "poisson" else None,
    )
    binned = Binning.fit(data, bins=4).transform(data)
    prepared = getattr(glm, family)().prepare(ops, ops.prepare(binned, problem), problem)
    return prepared, ops.execution.upload(old[:, None]), ops.execution.upload(new[:, None])


def simple(family):
    return next(c for c in CASES if c["id"] == family + "/tiny-improvement")["arrays"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_independent_bounds_and_scalar_only_transfers(case, tmp_path):
    exact = direct_difference(case["family"], *case["arrays"])
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem, old, new = prepare(ops, case["family"], case["arrays"])
        owned, start = snapshot(ops), dict(context.metrics)
        result = getattr(glm, case["family"])().loss_change(ops, problem, old, new)
        finish = dict(context.metrics)
        folder = directory("comparisons", tmp_path)
        report = dict(
            case=case,
            comparison=asdict(result),
            status=result.status,
            decimal220=str(exact),
            counters={
                k: finish.get(k, 0) - start.get(k, 0)
                for k in (
                    "comparison_calls",
                    "comparison_export_bytes",
                    "export_bytes",
                    "upload_bytes",
                    "kernel_launches",
                    "synchronizations",
                )
            },
        )
        name = hashlib.sha256(case["id"].encode()).hexdigest()[:16] + ".json"
        (folder / name).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        assert snapshot(ops) == owned
        assert finish["upload_bytes"] == start["upload_bytes"]
        assert finish["comparison_export_bytes"] - start.get("comparison_export_bytes", 0) == 32
        assert finish["export_bytes"] - start["export_bytes"] == 32 + (
            finish["validation_export_bytes"] - start["validation_export_bytes"]
        )
        assert finish["comparison_calls"] - start.get("comparison_calls", 0) == 1
        assert Decimal(result.lower) <= exact <= Decimal(result.upper)
        if case["status"]:
            assert result.status == case["status"]
        assert result.unchanged == (case["arrays"][0] == case["arrays"][1])
        np.testing.assert_array_equal(context.export(old)[:, 0], case["arrays"][0])
        np.testing.assert_array_equal(context.export(new)[:, 0], case["arrays"][1])


@pytest.mark.parametrize("family", ["binary", "poisson"])
@pytest.mark.parametrize("invalid", ["shape", "nonfinite", "zero-weight-domain"])
def test_invalid_snapshot_cannot_be_unchanged_and_cleans_up(family, invalid):
    arrays = [[0, 0], [0, 0], [0, 1], [0, 0], [1, 0], [1, 1]]
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        problem, old, new = prepare(ops, family, arrays)
        bad = np.zeros((2, 2 if invalid == "shape" else 1), np.float32)
        if invalid == "nonfinite":
            bad[-1, 0] = np.nan
        if invalid == "zero-weight-domain":
            bad[-1, 0] = 120 if family == "binary" else 90
        value = context.upload(bad)
        owned = snapshot(ops)
        with pytest.raises(ValueError):
            glm.compare(ops, problem, value, value, family=family)
        assert snapshot(ops) == owned
        assert glm.compare(ops, problem, old, new, family=family).unchanged


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_family_ownership_and_released_snapshots(family):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, family, simple(family))
        owned = snapshot(ops)
        for wrong_ops, wrong_p in ((ops, replace(p)), (DeviceOperations(context), p)):
            with pytest.raises(ValueError, match="forged"):
                glm.compare(wrong_ops, wrong_p, old, new, family=family)
        with pytest.raises(ValueError, match="family"):
            glm.compare(ops, p, old, new, family="poisson" if family == "binary" else "binary")
        assert snapshot(ops) == owned
        context.release(new)
        with pytest.raises(ValueError, match="released"):
            glm.compare(ops, p, old, new, family=family)


@pytest.mark.parametrize("family", ["binary", "poisson"])
@pytest.mark.parametrize("failure", ["allocation1", "allocation2", "reduce"])
def test_partial_comparison_failures_are_atomic(family, failure, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, family, simple(family))
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
                raise RuntimeError("injected comparison dispatch")
            return launch(name, *args)

        with monkeypatch.context() as patch:
            patch.setattr(context, "_empty", allocate)
            patch.setattr(ops, "_launch", dispatch)
            with pytest.raises((MemoryError, RuntimeError), match="injected"):
                glm.compare(ops, p, old, new, family=family)
        assert snapshot(ops) == owned
        assert glm.compare(ops, p, old, new, family=family).improves()


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_comparison_does_not_call_host_or_report_callbacks(family, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, family, simple(family))

        def forbidden(*args, **kwargs):
            raise AssertionError("host/reporting fallback forbidden")

        for owner, name in (
            (Binary, "geometry"),
            (Poisson, "geometry"),
            (glm, "geometry"),
            (glm, "loss"),
        ):
            monkeypatch.setattr(owner, name, forbidden)
        assert glm.compare(ops, p, old, new, family=family).improves()


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_actual_comparison_ptx_has_directed_double_operations(family, tmp_path):
    from openboost import _device_glm_comparison as kernels

    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, family, simple(family))
        glm.compare(ops, p, old, new, family=family)
        kernel = getattr(kernels, family + "_compare_rows")
        ptx = "\n".join(kernel.inspect_asm().values())
        instructions = (
            "add.rm.f64",
            "add.rp.f64",
            "mul.rm.f64",
            "mul.rp.f64",
            "div.rm.f64",
            "div.rp.f64",
        )
        report = dict(family=family, required_instructions=instructions, ptx=ptx)
        folder = directory("comparisons", tmp_path)
        (folder / (family + "-ptx.json")).write_text(json.dumps(report, indent=2) + "\n")
        for instruction in instructions:
            assert instruction in ptx
