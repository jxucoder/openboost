"""092-B comparison operations: collect locally, run only on real CUDA hardware."""

import hashlib
import json
import os
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost import device_normal as normal
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext
from openboost.objectives import Normal

from .reference.normal_acceptance import compare_precisions, loss_difference
from .test_loss_change import CASES

pytestmark = pytest.mark.gpu


def prepared(ops, target, offset, weight):
    x = NumericData(np.zeros((len(target), 1)), np.arange(len(target)), ("x",))
    p = Problem(
        x, np.asarray(target).reshape(-1, 1), x.row_ids, offset=offset, weight=weight, raw_width=2
    )
    bins = Binning(("x",), (np.array([]),)).transform(x)
    data = ops.prepare(bins, p)
    return p, normal.prepare(ops, data, p)


@pytest.mark.parametrize("case", CASES, ids=[row["id"] for row in CASES])
def test_stored_input_bounds_residency_and_independent_oracle(case, tmp_path):
    arrays = {
        key: np.asarray(value["values"], dtype=np.float32) for key, value in case["inputs"].items()
    }
    before, after, target, offset, weight = (
        arrays[key] for key in ("before", "after", "target", "offset", "weight")
    )
    oracle = compare_precisions(before, after, target, offset, weight)
    expected = [Decimal(oracle[f"decimal{p}"]) for p in (60, 100)]
    if not oracle["estimates_agree"]:
        expected += [
            loss_difference(before, after, target, offset, weight, precision=p) for p in (160, 220)
        ]
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        p, problem = prepared(ops, target, offset, weight)
        old, new = context.upload(before), context.upload(after)
        handles, records, start = set(context._buffers), set(ops._records), dict(context.metrics)
        result = normal.compare(ops, problem, old, new)
        finish = dict(context.metrics)
        assert set(context._buffers) == handles and set(ops._records) == records
        assert finish["live_bytes"] == start["live_bytes"]
        assert finish["upload_bytes"] == start["upload_bytes"]
        assert finish["comparison_export_bytes"] - start.get("comparison_export_bytes", 0) == 32
        assert finish["export_bytes"] - start["export_bytes"] == 32 + (
            finish["validation_export_bytes"] - start["validation_export_bytes"]
        )
        assert finish["comparison_calls"] - start.get("comparison_calls", 0) == 1
        assert normal.objective().loss_change(ops, problem, old, new) == result
        np.testing.assert_array_equal(context.export(old), before)
        np.testing.assert_array_equal(context.export(new), after)
        if result.lower is not None:
            for value in expected:
                assert Decimal.from_float(result.lower) <= value <= Decimal.from_float(result.upper)
            assert result.unchanged == np.array_equal(before, after)
        else:
            assert (
                result.reason == "exponent_range"
                and case["id"] == "analytic/outside-exponent-range"
            )
        if (
            case["id"].startswith("run7/")
            and "/channel0/" in case["id"]
            and case["id"].endswith("/training")
        ):
            assert result.status == "worsening" and not result.improves()
        if case["id"] == "analytic/tiny-improvement":
            assert result.status == "improvement" and result.improves()
        # This CPU operation receives the exact same promoted float32 inputs;
        # its outward bound can be wider than directed device arithmetic.
        cpu = Normal.compare(p, before, after)
        if cpu.status in ("improvement", "worsening", "unchanged"):
            assert result.status == cpu.status
        report = dict(
            case=case["id"],
            inputs={key: value.tolist() for key, value in arrays.items()},
            comparison=dict(
                asdict(result),
                status=result.status,
                estimate=result.estimate,
                uncertainty=result.uncertainty,
            ),
            cpu=asdict(cpu),
            high_precision=oracle,
            additional_precision=[str(v) for v in expected[2:]],
            counters={
                key: finish.get(key, 0) - start.get(key, 0)
                for key in (
                    "comparison_calls",
                    "comparison_export_bytes",
                    "export_bytes",
                    "upload_bytes",
                    "kernel_launches",
                    "validation_export_bytes",
                    "synchronizations",
                )
            },
        )
        folder = Path(os.environ.get("OPENBOOST_COMPARISON_ARTIFACTS", tmp_path))
        folder.mkdir(parents=True, exist_ok=True)
        filename = hashlib.sha256(case["id"].encode()).hexdigest()[:16] + ".json"
        (folder / filename).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


@pytest.mark.parametrize(
    "bad", ["nan", "scale", "precision", "shape", "released", "forged", "foreign"]
)
def test_domain_identity_and_failure_leave_every_owner_unchanged(bad):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        _, problem = prepared(ops, [0, 0], [[0, 0], [0, 0]], [1, 0])
        a = np.array([[0, 33], [0, 0]], np.float32)  # First row has unresolved support.
        b = a.copy()
        if bad == "nan":
            b[1, 0] = np.nan
        elif bad == "scale":
            b[1, 1] = 1000
        elif bad == "precision":
            b[1, 1] = -50
        elif bad == "shape":
            b = b[:, :1]
        old, new = context.upload(a), context.upload(b)
        if bad == "released":
            context.release(new)
        elif bad == "forged":
            problem = replace(problem)
        elif bad == "foreign":
            ops = DeviceOperations(context)
        handles, records, live = (
            set(context._buffers),
            set(ops._records),
            context.metrics["live_bytes"],
        )
        with pytest.raises(ValueError):
            normal.compare(ops, problem, old, new)
        assert set(context._buffers) == handles and set(ops._records) == records
        assert context.metrics["live_bytes"] == live


@pytest.mark.parametrize("allocation", [1, 2])
def test_comparison_allocation_failure_cleans_scratch_and_keeps_inputs(allocation, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        _, problem = prepared(ops, [0], [[0, 0]], [1])
        old = context.upload(np.array([[1, 0]], np.float32))
        new = context.upload(np.array([[0, 0]], np.float32))
        original, count = context._empty, 0

        def fail(shape, dtype):
            nonlocal count
            if np.dtype(dtype) == np.dtype(np.float64):
                count += 1
                if count == allocation:
                    raise MemoryError("comparison scratch budget")
            return original(shape, dtype)

        handles, records, live = (
            set(context._buffers),
            set(ops._records),
            context.metrics["live_bytes"],
        )
        monkeypatch.setattr(context, "_empty", fail)
        with pytest.raises(MemoryError, match="scratch budget"):
            normal.compare(ops, problem, old, new)
        assert set(context._buffers) == handles and set(ops._records) == records
        assert context.metrics["live_bytes"] == live
        monkeypatch.setattr(context, "_empty", original)
        assert normal.compare(ops, problem, old, new).improves()


def test_device_comparison_does_not_call_host_objective_or_reporting_callbacks(monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        _, problem = prepared(ops, [0], [[0, 0]], [1])
        old = context.upload(np.array([[1, 0]], np.float32))
        new = context.upload(np.array([[0, 0]], np.float32))

        def forbidden(*args, **kwargs):
            raise AssertionError("no host comparison or metric/geometry callback")

        for owner, name in ((Normal, "compare"), (normal, "loss"), (normal, "geometry")):
            monkeypatch.setattr(owner, name, forbidden)
        assert normal.compare(ops, problem, old, new).improves()


def test_failure_after_row_kernel_releases_scratch_and_propagates(monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        _, problem = prepared(ops, [0], [[0, 0]], [1])
        old = context.upload(np.array([[1, 0]], np.float32))
        new = context.upload(np.array([[0, 0]], np.float32))
        original = ops._launch

        def fail(name, *args):
            if name == "normal_compare_reduce":
                raise RuntimeError("intentional device dispatch failure")
            return original(name, *args)

        handles, records, live = (
            set(context._buffers),
            set(ops._records),
            context.metrics["live_bytes"],
        )
        monkeypatch.setattr(ops, "_launch", fail)
        with pytest.raises(RuntimeError, match="dispatch failure"):
            normal.compare(ops, problem, old, new)
        assert set(context._buffers) == handles and set(ops._records) == records
        assert context.metrics["live_bytes"] == live
        monkeypatch.setattr(ops, "_launch", original)
        assert normal.compare(ops, problem, old, new).improves()
