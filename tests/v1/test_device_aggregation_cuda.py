"""Real-device 078-A checks; no emulation, import skipping, or CPU fallback."""

import json
from dataclasses import replace

import numpy as np
import pytest

from openboost.data import NumericData, Problem
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .reference.device_histogram import NAMES, ROLES, aggregate, fixture, weighted_fields
from .test_device_histogram_reference import prepared_fixture

pytestmark = pytest.mark.gpu


def setup(context, large=False):
    binned, problem, values = prepared_fixture(large)
    ops = DeviceOperations(context)
    data = ops.prepare(binned, problem)
    buffer = context.upload(values)
    fields = ops.fields(data, buffer, names=NAMES, roles=ROLES)
    return ops, data, fields


def check_histogram(context, ops, data, fields, rows, *, large=False):
    codes, missing, bins, values, weight = fixture(large)
    expected = aggregate(codes, missing, bins, weighted_fields(values, weight), rows)
    selected = ops.rows(data, rows)
    before = dict(context.metrics)
    result = ops.histogram(data, fields, selected)
    after = dict(context.metrics)
    # Diagnostic reference exports below are separate from operation execution.
    assert after["upload_bytes"] == before["upload_bytes"]
    assert after["export_bytes"] - before["export_bytes"] == 2 * len(NAMES) * 4
    assert (
        after["validation_export_bytes"] - before["validation_export_bytes"] == 2 * len(NAMES) * 4
    )
    assert after["kernel_launches"] - before["kernel_launches"] == 4
    np.testing.assert_array_equal(context.export(selected.positions), rows)
    for handle, reference in zip((result.sums, result.counts, result.total), expected, strict=True):
        actual = context.export(handle)
        if handle is result.counts:
            assert actual.dtype == np.int64
            np.testing.assert_array_equal(actual, reference)
        else:
            assert actual.dtype == np.float32
            np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
    ops.release(result)
    ops.release(selected)


@pytest.mark.parametrize("rows", [list(range(8)), [6, 0, 3, 7], [], [0, 4], [1, 6]])
def test_f1_f2_named_weighted_routed_histograms(rows):
    with ExecutionContext() as context:
        ops, data, source = setup(context)
        fields = ops.apply_weight(source)
        np.testing.assert_array_equal(
            context.export(fields.values), weighted_fields(fixture()[3], fixture()[4])
        )
        # The independent output does not mutate or depend on the source allocation.
        np.testing.assert_array_equal(context.export(source.values), fixture()[3])
        context.release(source.values)
        check_histogram(context, ops, data, fields, np.asarray(rows, dtype=np.int32))


@pytest.mark.parametrize("stride", [1, 3])
def test_f3_bounded_large_histograms(stride):
    with ExecutionContext() as context:
        ops, data, source = setup(context, True)
        fields = ops.apply_weight(source)
        check_histogram(
            context, ops, data, fields, np.arange(0, 8192, stride, dtype=np.int32), large=True
        )
        assert context.metrics["peak_pool_bytes"] <= 16 * 1024**2
        print("aggregation_metrics=" + json.dumps(dict(context.metrics), sort_keys=True))


def test_d2_public_append_with_renamed_information():
    with ExecutionContext() as context:
        binned, problem, values = prepared_fixture()
        ops = DeviceOperations(context)
        data = ops.prepare(binned, problem)
        initial = ops.fields(data, context.upload(values[:, :2]), names=NAMES[:2], roles=ROLES[:2])
        weighted = ops.apply_weight(initial)
        a = context.upload(values[:, 2])
        b = context.upload(values[:, 3])
        extra = ops.add_independent(weighted, "independent:red", a, nonnegative=True)
        complete = ops.add_independent(extra, "independent:blue", b, nonnegative=True)
        context.release(a)
        context.release(b)
        ops.release(extra)
        assert complete.names[-2:] == ("independent:red", "independent:blue")
        np.testing.assert_array_equal(
            context.export(complete.values), weighted_fields(values, problem.weight)
        )
        check_histogram(context, ops, data, complete, np.array([0, 4], dtype=np.int32))
        with pytest.raises(ValueError, match="nonnegative"):
            ops.add_independent(
                weighted, "bad", context.upload(-np.ones(8, np.float32)), nonnegative=True
            )
        with pytest.raises(ValueError, match="names"):
            ops.add_independent(weighted, "gradient", context.upload(values[:, 2]))


@pytest.mark.parametrize("positions", [[1, 1], [-1], [8]])
def test_bad_rows_rejected_host_and_device(positions):
    with ExecutionContext() as context:
        ops, data, _ = setup(context)
        with pytest.raises(ValueError, match="row positions"):
            ops.rows(data, positions)
        buffer = context.upload(np.array(positions, dtype=np.int32))
        before = context.metrics["live_bytes"]
        with pytest.raises(ValueError, match="row positions"):
            ops.rows(data, buffer)
        assert context.metrics["live_bytes"] == before


def test_resident_rows_borrowed_lifetime_and_empty_view():
    with ExecutionContext() as context:
        ops, data, source = setup(context)
        fields = ops.apply_weight(source)
        buffer = context.upload(np.array([6, 0, 3, 7], np.int32))
        before = context.metrics["export_bytes"]
        rows = ops.rows(data, buffer)
        assert context.metrics["export_bytes"] - before == 4
        result = ops.histogram(data, fields, rows)
        np.testing.assert_array_equal(context.export(result.total), [1, 3, 2, 2])
        ops.release(rows)
        np.testing.assert_array_equal(context.export(buffer), [6, 0, 3, 7])
        empty = ops.rows(data, [])
        assert empty.positions.shape == (0,) and empty.positions.nbytes == 0
        borrowed_empty = ops.rows(data, empty.positions)
        result = ops.histogram(data, fields, borrowed_empty)
        np.testing.assert_array_equal(context.export(result.total), [0, 0, 0, 0])
        context.release(empty.positions)
        with pytest.raises(ValueError, match="released"):
            ops.histogram(data, fields, borrowed_empty)


def test_foreign_forged_stale_identity_and_closed_records():
    with ExecutionContext() as a, ExecutionContext() as b:
        ops, data, source = setup(a)
        other_ops, other, _ = setup(b)
        fields = ops.apply_weight(source)
        rows = ops.rows(data)
        for bad in (other, replace(data)):
            with pytest.raises(ValueError, match="foreign|forged"):
                ops.rows(bad)
        with pytest.raises(ValueError, match="foreign"):
            ops.fields(data, b.upload(np.ones((8, 4), np.float32)), names=NAMES, roles=ROLES)
        with pytest.raises(ValueError, match="foreign"):
            other_ops.histogram(other, fields, rows)
        binned, problem, _ = prepared_fixture()
        second = ops.prepare(binned, problem)
        with pytest.raises(ValueError, match="identity"):
            ops.histogram(second, fields, rows)
        reordered = NumericData(
            problem.data.values, problem.row_ids[::-1], problem.data.feature_names
        )
        bad_problem = Problem(reordered, problem.target, reordered.row_ids, weight=problem.weight)
        with pytest.raises(ValueError, match="identity"):
            ops.prepare(binned, bad_problem)
        ops.release(data)
        with pytest.raises(ValueError, match="released"):
            ops.histogram(data, fields, rows)
    with pytest.raises(RuntimeError, match="closed"):
        ops.rows(data)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_fields_rejected_without_leak(value):
    with ExecutionContext() as context:
        ops, data, _ = setup(context)
        values = fixture()[3]
        values[2, 1] = value
        buffer = context.upload(values)
        before = context.metrics["live_bytes"]
        with pytest.raises(ValueError, match="finite"):
            ops.fields(data, buffer, names=NAMES, roles=ROLES)
        assert context.metrics["live_bytes"] == before


def test_field_schema_and_weight_roles():
    with ExecutionContext() as context:
        ops, data, source = setup(context)
        for names, roles in ((NAMES[:3], ROLES), (("g",) * 4, ROLES), (NAMES, ("unknown",) * 4)):
            with pytest.raises(ValueError, match="names|roles"):
                ops.fields(data, source.values, names=names, roles=roles)
        for values in (
            np.ones((7, 4), np.float32),
            np.ones((8, 4), np.float64),
            np.ones(8, np.float32),
        ):
            with pytest.raises(ValueError, match="float32"):
                ops.fields(data, context.upload(values), names=NAMES, roles=ROLES)
        rows = ops.rows(data)
        with pytest.raises(ValueError, match="weights before"):
            ops.histogram(data, source, rows)
        fields = ops.apply_weight(source)
        with pytest.raises(ValueError, match="already applied"):
            ops.apply_weight(fields)
        with pytest.raises(ValueError, match="row positions"):
            ops.rows(data, [True])
        with pytest.raises(ValueError, match="int32"):
            ops.rows(data, context.upload(np.ones(2, np.int64)))


def test_allocation_failure_rolls_back_partial_outputs():
    with ExecutionContext(max_bytes=8192) as context:
        ops, data, source = setup(context)
        fields = ops.apply_weight(source)
        rows = ops.rows(data)
        # Leave exactly one 512-byte pool block; sums fit, counts/total cannot.
        before = context.metrics["live_bytes"]
        ballast_bytes = 8192 - context.metrics["peak_pool_bytes"] - 512
        ballast = context.upload(np.zeros(ballast_bytes // 4, np.float32))
        live = context.metrics["live_bytes"]
        with pytest.raises(MemoryError):
            ops.histogram(data, fields, rows)
        assert context.metrics["live_bytes"] == live
        context.release(ballast)
        assert context.metrics["live_bytes"] == before
        result = ops.histogram(data, fields, rows)
        np.testing.assert_array_equal(context.export(result.total), [10, 9, 4, 4])


def test_weight_overflow_rejected_and_sources_preserved():
    with ExecutionContext() as context:
        ops, data, _ = setup(context)
        source = context.upload(np.full((8, 1), np.finfo(np.float32).max, np.float32))
        fields = ops.fields(data, source, names=("gradient",), roles=("unweighted",))
        before = context.metrics["live_bytes"]
        with pytest.raises(ValueError, match="finite"):
            ops.apply_weight(fields)
        assert context.metrics["live_bytes"] == before
        np.testing.assert_array_equal(
            context.export(source), np.full((8, 1), np.finfo(np.float32).max, np.float32)
        )


def test_unequal_bins_have_feature_specific_missing_and_zero_padding():
    from openboost.binning import Binning

    data = NumericData([[0, 0], [np.nan, 2], [0, np.nan]], [20, 30, 40], ("a", "b"))
    problem = Problem(data, np.zeros((3, 1)), data.row_ids)
    binned = Binning(data.feature_names, (np.array([]), np.array([0.5, 1.5]))).transform(data)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        prepared = ops.prepare(binned, problem)
        fields = ops.fields(
            prepared,
            context.upload(np.ones((3, 1), np.float32)),
            names=("mass",),
            roles=("independent",),
        )
        result = ops.histogram(prepared, fields, ops.rows(prepared))
        np.testing.assert_array_equal(context.export(result.counts), [[2, 1, 0, 0], [1, 0, 1, 1]])
        np.testing.assert_array_equal(
            context.export(result.sums)[:, :, 0], [[2, 1, 0, 0], [1, 0, 1, 1]]
        )


def test_context_stream_is_preserved_during_device_operations():
    import cupy as cp

    caller = cp.cuda.Stream(non_blocking=True)
    with caller, ExecutionContext() as context:
        ops, data, source = setup(context)
        result = ops.histogram(data, ops.apply_weight(source), ops.rows(data))
        assert cp.cuda.get_current_stream().ptr == caller.ptr
        np.testing.assert_array_equal(context.export(result.total), [10, 9, 4, 4])
