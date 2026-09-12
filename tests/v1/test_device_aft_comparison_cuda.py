"""115 resident AFT comparison cases; collection is not CUDA execution evidence."""

import hashlib
import json
from dataclasses import asdict, replace
from decimal import Decimal

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost import device_aft as aft
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext
from openboost.survival import LogNormalAFT

from .aft_artifacts import directory
from .reference.aft_comparison import CASES, direct_difference
from .reference.device_aft import DOMAIN_CASES
from .test_device_aft_cuda import snapshot

pytestmark = pytest.mark.gpu


def host_problem(arrays):
    _, _, lower, event, offset, weight, _ = arrays
    data = NumericData(np.arange(len(lower))[:, None], np.arange(len(lower)), ("x",))
    lower = np.asarray(lower)
    bounds = np.column_stack((lower, np.where(event, lower, np.inf)))
    return Problem(data, bounds, data.row_ids, target_kind="event_right",
                   offset=np.asarray(offset)[:, None], weight=weight)


def prepare(ops, arrays):
    old, new, _, _, _, _, sigma = arrays
    p = host_problem(arrays)
    binned = Binning.fit(p.data, bins=4).transform(p.data)
    prepared = aft.objective(sigma).prepare(ops, ops.prepare(binned, p), p)
    return prepared, ops.execution.upload(np.asarray(old, np.float32)[:, None]), ops.execution.upload(np.asarray(new, np.float32)[:, None])


def simple():
    return next(c["arrays"] for c in CASES if c["id"] == "1/tiny-censor")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_actual_enclosure_and_scalar_only_transfers(case, tmp_path):
    import cupy as cp

    exact = direct_difference(*case["arrays"])
    with ExecutionContext() as context:
        external = cp.cuda.Stream(non_blocking=True)
        with external:
            ops = DeviceOperations(context)
            p, old, new = prepare(ops, case["arrays"])
            owned, start = snapshot(ops), dict(context.metrics)
            result = aft.objective(p.sigma).loss_change(ops, p, old, new)
            finish = dict(context.metrics)
            report = dict(case=case, comparison=asdict(result), status=result.status, decimal220=str(exact),
                          counters={k: finish.get(k, 0)-start.get(k, 0) for k in
                                    ("comparison_calls", "comparison_export_bytes", "export_bytes", "upload_bytes", "kernel_launches", "synchronizations")})
            filename = hashlib.sha256(case["id"].encode()).hexdigest()[:16]+".json"
            (directory("comparisons", tmp_path)/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            assert snapshot(ops) == owned
            assert finish["upload_bytes"] == start["upload_bytes"]
            assert finish["comparison_export_bytes"]-start.get("comparison_export_bytes", 0) == 32
            assert finish["export_bytes"]-start["export_bytes"] == 32 + finish["validation_export_bytes"]-start["validation_export_bytes"]
            assert finish["comparison_calls"]-start.get("comparison_calls", 0) == 1
            assert Decimal(result.lower) <= exact <= Decimal(result.upper)
            if case["status"]:
                assert result.status == case["status"]
            assert result.unchanged == (case["arrays"][0] == case["arrays"][1])
            np.testing.assert_array_equal(context.export(old)[:, 0], case["arrays"][0])
            np.testing.assert_array_equal(context.export(new)[:, 0], case["arrays"][1])
            assert cp.cuda.get_current_stream().ptr == external.ptr


@pytest.mark.parametrize("case", [c for c in DOMAIN_CASES if not c[-1]], ids=lambda c: c[0])
@pytest.mark.parametrize("zero_weight", [False, True])
def test_invalid_identical_domain_never_shortcuts(case, zero_weight):
    _, raw, lower, event, sigma, _ = case
    arrays = [[0, raw], [0, raw], [1, lower], [True, event], [0, 0], [1, 0 if zero_weight else 1], sigma]
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        if lower < 1e-45 or lower > 3.4e38:
            with pytest.raises(ValueError):
                prepare(ops, arrays)
            return  # Host upload rejection has separate 114 allocation controls.
        p, old, new = prepare(ops, arrays)
        owned = snapshot(ops)
        with pytest.raises(ValueError, match="geometry"):
            aft.compare(ops, p, old, new)
        assert snapshot(ops) == owned


@pytest.mark.parametrize("invalid", ["shape", "nonfinite", "forged", "foreign", "scale", "released"])
def test_identity_shape_and_scale_rejection_is_atomic(invalid):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, simple())
        if invalid in ("shape", "nonfinite"):
            bad = np.zeros((1, 2 if invalid == "shape" else 1), np.float32)
            if invalid == "nonfinite":
                bad[0, 0] = np.nan
            new = context.upload(bad)
        if invalid == "released":
            context.release(new)
        owned = snapshot(ops)
        with pytest.raises(ValueError):
            aft.compare(DeviceOperations(context) if invalid == "foreign" else ops,
                        replace(p) if invalid == "forged" else p, old, new,
                        sigma=2 if invalid == "scale" else 1)
        assert snapshot(ops) == owned


@pytest.mark.parametrize("failure", ["allocation1", "allocation2", "reduce"])
def test_partial_comparison_failure_releases_scratch(failure, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, simple())
        owned, empty, launch, count = snapshot(ops), context._empty, ops._launch, 0

        def allocate(shape, dtype):
            nonlocal count
            if np.dtype(dtype) == np.dtype(np.float64):
                count += 1
                if failure == "allocation"+str(count):
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
                aft.compare(ops, p, old, new)
        assert snapshot(ops) == owned
        assert aft.compare(ops, p, old, new).improves()


def test_no_host_or_reporting_fallback(monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, simple())

        def forbidden(*args, **kwargs):
            raise AssertionError("host/reporting fallback forbidden")

        for owner, name in ((LogNormalAFT, "geometry"), (aft, "geometry"), (aft, "loss")):
            monkeypatch.setattr(owner, name, forbidden)
        assert aft.compare(ops, p, old, new).improves()


def test_tail_range_and_invalid_geometry_precedence():
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, [[-1e13], [-1e13+1e6], [1], [False], [0], [1], 1])
        result = aft.compare(ops, p, old, new)
        assert result.reason == "tail_range" and result.status == "unresolved"
        assert aft.compare(ops, p, old, old).unchanged
        p, old, new = prepare(ops, [[-1e13, 15], [-1e13+1e6, 15], [1, 1], [False, False], [0, 0], [1, 0], 1])
        with pytest.raises(ValueError, match="geometry"):
            aft.compare(ops, p, old, new)


def test_actual_ptx_has_directed_double_arithmetic(tmp_path):
    from openboost import _device_aft_comparison as kernels

    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, old, new = prepare(ops, simple())
        aft.compare(ops, p, old, new)
        ptx = "\n".join(kernels.aft_compare_rows.inspect_asm().values())
        instructions = ("add.rm.f64", "add.rp.f64", "mul.rm.f64", "mul.rp.f64", "div.rm.f64", "div.rp.f64")
        report = dict(required_instructions=instructions, ptx=ptx)
        (directory("comparisons", tmp_path)/"aft-ptx.json").write_text(json.dumps(report, indent=2)+"\n")
        for instruction in instructions:
            assert instruction in ptx
