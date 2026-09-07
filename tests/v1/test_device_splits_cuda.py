"""Frozen 078-B real-CUDA acceptance; collection alone proves no device behavior."""

import json
from dataclasses import replace

import numpy as np
import pytest

from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .reference.device_splits import CASES, enumerate_candidates, winner
from .test_device_split_reference import prepared_fixture

pytestmark = pytest.mark.gpu


def setup(context, case="d2", *, values=None, names=None, roles=None):
    binned, problem, reference, positions = prepared_fixture(case)
    ops = DeviceOperations(context)
    data = ops.prepare(binned, problem)
    fields = ops.fields(
        data,
        context.upload(np.asarray(reference.values if values is None else values, np.float32)),
        names=reference.names if names is None else names,
        roles=("training", "training", "independent", "independent") if roles is None else roles,
    )
    rows = ops.rows(data, positions)
    hist = ops.histogram(data, fields, rows)
    return ops, data, fields, rows, hist


def close_float(actual, expected):
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def resident_delta(context, before, decision_bytes):
    after = context.metrics
    assert after["upload_bytes"] == before["upload_bytes"]
    assert after["decision_export_bytes"] - before["decision_export_bytes"] == decision_bytes
    assert after["export_bytes"] - before["export_bytes"] == (
        after["validation_export_bytes"] - before["validation_export_bytes"] + decision_bytes
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("minimum", [0, 1, 2])
def test_exhaustive_candidates_scores_masks_routes_leaves(case, minimum):
    binned, _, reference, positions = prepared_fixture(case)
    expected, total = enumerate_candidates(
        binned.data.values, reference.values, positions, minimum=minimum
    )
    with ExecutionContext() as context:
        ops, data, fields, rows, hist = setup(context, case)
        before = dict(context.metrics)
        batch = ops.candidates(hist)
        scores = ops.newton_scores(batch)
        legal = ops.feasible(batch)
        information = ops.mask_and(
            ops.child_minimum(batch, "a", minimum), ops.child_minimum(batch, "b", minimum)
        )
        constrained = ops.mask_and(legal, information)
        choices = [ops.choose(batch, scores, mask) for mask in (legal, constrained)]
        root_leaf = ops.leaf(hist)
        resident_delta(context, before, 8)

        # All exports below are diagnostics, outside the operation transfer measurement.
        active = context.export(batch.active)
        indices = np.flatnonzero(active)
        assert [batch.key(int(i)) for i in indices] == [c["key"] for c in expected]
        close_float(context.export(hist.total), total)
        close_float(context.export(root_leaf), [-total[0] / (total[1] + 1)])
        sums, counts, gains = (
            context.export(b) for b in (batch.values, batch.counts, scores.values)
        )
        assert counts.dtype == np.int64
        assert active.dtype == np.bool_
        legal_values = context.export(legal.values)
        info_values = context.export(information.values)
        np.testing.assert_array_equal(legal_values[~active], False)
        np.testing.assert_array_equal(info_values[~active], False)
        np.testing.assert_array_equal(gains[~active], 0)
        for i, ref in zip(indices, expected, strict=True):
            close_float(sums[i], ref["sums"])
            np.testing.assert_array_equal(counts[i], ref["counts"])
            close_float(gains[i : i + 1], [ref["gain"]])
            assert legal_values[i] == ref["legal"]
            assert info_values[i] == ref["information"]

        for constrained_choice, choice in enumerate(choices):
            ref = winner(expected, bool(constrained_choice))
            assert (choice.key if choice else None) == (ref["key"] if ref else None)
            if choice is None:
                continue
            before = dict(context.metrics)
            children = ops.partition(rows, choice)
            child_hists = [ops.histogram(data, fields, child) for child in children]
            leaves = [ops.leaf(h) for h in child_hists]
            resident_delta(context, before, 8)
            for side, (child, h, leaf) in enumerate(
                zip(children, child_hists, leaves, strict=True)
            ):
                np.testing.assert_array_equal(context.export(child.positions), ref["rows"][side])
                close_float(context.export(h.total), ref["sums"][side])
                g, curvature = ref["sums"][side, :2]
                close_float(context.export(leaf), [-g / (curvature + 1)])
        assert context.metrics["peak_pool_bytes"] <= 16 * 1024**2
        print("split_metrics=" + json.dumps(dict(case=case, minimum=minimum, **context.metrics)))


@pytest.mark.parametrize("minimum", [0, 1, 2])
def test_d2_reordered_named_information_and_public_composition(minimum):
    binned, _, reference, positions = prepared_fixture("d2")
    expected, _ = enumerate_candidates(
        binned.data.values, reference.values, positions, minimum=minimum
    )
    with ExecutionContext() as context:
        ops, _, _, rows, hist = setup(
            context,
            values=reference.values[:, [3, 1, 2, 0]],
            names=("cohort:blue", "curvature", "cohort:red", "gradient"),
            roles=("independent", "training", "independent", "training"),
        )
        before = dict(context.metrics)
        batch = ops.candidates(hist)
        allowed = ops.feasible(batch)
        for name in ("cohort:red", "cohort:blue"):
            allowed = ops.mask_and(allowed, ops.child_minimum(batch, name, minimum))
        choice = ops.choose(batch, ops.newton_scores(batch), allowed)
        resident_delta(context, before, 4)
        ref = winner(expected, True)
        assert (choice.key if choice else None) == (ref["key"] if ref else None)
        if choice:
            for child, expected_rows in zip(ops.partition(rows, choice), ref["rows"], strict=True):
                np.testing.assert_array_equal(context.export(child.positions), expected_rows)


def test_supplied_scores_ignore_padding_and_support_empty_child_routes():
    with ExecutionContext() as context:
        ops, data, fields, rows, hist = setup(context, "inactive")
        batch = ops.candidates(hist)
        active = context.export(batch.active)
        values = np.full(batch.size, 100, np.float32)
        values[active] = 0
        # The last observed threshold has an empty right child, intentionally enabled
        # by a caller-supplied mask instead of ordinary Newton legality.
        index = next(i for i in range(batch.size) if batch.key(i) == (0, 2, False))
        values[index] = 1
        source = context.upload(values)
        scores = ops.scores(batch, source)
        mask_buffer = context.upload(np.ones(batch.size, bool))
        mask = ops.mask(batch, mask_buffer)
        before = dict(context.metrics)
        choice = ops.choose(batch, scores, mask)
        left, right = ops.partition(rows, choice)
        empty_hist = ops.histogram(data, fields, right)
        leaf = ops.leaf(empty_hist)
        resident_delta(context, before, 12)
        assert choice.key == (0, 2, False)
        np.testing.assert_array_equal(context.export(left.positions), np.arange(4))
        assert context.export(right.positions).shape == (0,)
        close_float(context.export(leaf), [0])
        ops.release(scores)
        ops.release(mask)
        np.testing.assert_array_equal(context.export(source), values)
        np.testing.assert_array_equal(context.export(mask_buffer), True)


def test_penalty_regularization_and_curvature_minimum():
    binned, _, fields, positions = prepared_fixture("d2")
    expected, total = enumerate_candidates(
        binned.data.values, fields.values, positions, reg_lambda=2, split_penalty=3
    )
    with ExecutionContext() as context:
        ops, _, _, _, hist = setup(context)
        batch = ops.candidates(hist)
        scores = ops.newton_scores(batch, reg_lambda=2, split_penalty=3)
        legal = ops.feasible(batch, min_child_h=2)
        indices = np.flatnonzero(context.export(batch.active))
        close_float(context.export(scores.values)[indices], [c["gain"] for c in expected])
        np.testing.assert_array_equal(
            context.export(legal.values)[indices],
            [c["legal"] and np.min(c["sums"][:, 1]) >= 2 for c in expected],
        )
        close_float(context.export(ops.leaf(hist, reg_lambda=2)), [-total[0] / (total[1] + 2)])


@pytest.mark.parametrize(
    "parameter",
    [-1, np.nan, np.inf, 1e100, 10**1000, 1j, "1", True],
    ids=["negative", "nan", "infinity", "overflow", "huge_integer", "complex", "string", "bool"],
)
def test_invalid_parameters_fail_without_leaks(parameter):
    with ExecutionContext() as context:
        ops, _, _, _, hist = setup(context)
        batch = ops.candidates(hist)
        actions = (
            lambda: ops.newton_scores(batch, reg_lambda=parameter),
            lambda: ops.newton_scores(batch, split_penalty=parameter),
            lambda: ops.feasible(batch, min_child_h=parameter),
            lambda: ops.child_minimum(batch, "a", parameter),
            lambda: ops.leaf(hist, reg_lambda=parameter),
        )
        before = context.metrics["live_bytes"]
        for action in actions:
            with pytest.raises(ValueError, match="parameter"):
                action()
            assert context.metrics["live_bytes"] == before


def test_scalar_schema_and_original_row_nonnegativity():
    _, _, reference, _ = prepared_fixture("d2")
    with ExecutionContext() as context:
        # A negative row must fail even when its sibling makes the child total positive.
        values = reference.values.copy()
        values[0, 1], values[1, 1] = -1, 10
        values[0, 2], values[1, 2] = -1, 10
        ops, _, _, _, hist = setup(context, values=values)
        batch = ops.candidates(hist)
        for action in (
            lambda: ops.newton_scores(batch),
            lambda: ops.feasible(batch),
            lambda: ops.leaf(hist),
            lambda: ops.child_minimum(batch, "a", 0),
        ):
            before = context.metrics["live_bytes"]
            with pytest.raises(ValueError, match="nonnegative"):
                action()
            assert context.metrics["live_bytes"] == before
        for name in ("gradient", "absent"):
            with pytest.raises(ValueError, match="independent"):
                ops.child_minimum(batch, name, 0)
        for names, roles in (
            (("g", "h", "a", "b"), ("training",) * 4),
            (reference.names, ("independent",) * 4),
        ):
            other, _, _, _, h = setup(context, names=names, roles=roles)
            with pytest.raises(ValueError, match="named scalar|once-weighted"):
                other.newton_scores(other.candidates(h))


def test_nonfinite_gains_and_undefined_leaf_denominators():
    _, _, reference, _ = prepared_fixture("ties")
    values = reference.values.copy()
    values[:, 0] *= 1e20
    with ExecutionContext() as context:
        ops, _, _, _, hist = setup(context, "ties", values=values)
        batch = ops.candidates(hist)
        before = context.metrics["live_bytes"]
        with pytest.raises(ValueError, match="finite gains"):
            ops.newton_scores(batch)
        assert context.metrics["live_bytes"] == before
        other, _, _, _, zero = setup(context, "zero_curvature")
        before = context.metrics["live_bytes"]
        with pytest.raises(ValueError, match="denominator"):
            other.leaf(zero, reg_lambda=0)
        assert context.metrics["live_bytes"] == before
        close_float(context.export(other.leaf(zero)), [0])


def test_supplied_score_and_mask_validation_and_borrowed_lifetimes():
    with ExecutionContext() as context:
        ops, _, _, _, hist = setup(context)
        batch = ops.candidates(hist)
        for value in (
            np.full(batch.size, np.nan, np.float32),
            np.full(batch.size, np.inf, np.float32),
            np.full(batch.size, -np.inf, np.float32),
            np.ones(batch.size, np.float64),
            np.ones(batch.size + 1, np.float32),
        ):
            source = context.upload(value)
            before = context.metrics["live_bytes"]
            with pytest.raises(ValueError, match="finite|float32"):
                ops.scores(batch, source)
            assert context.metrics["live_bytes"] == before
            context.release(source)
        for value in (np.ones(batch.size, np.int32), np.ones(batch.size + 1, bool)):
            with pytest.raises(ValueError, match="bool mask"):
                ops.mask(batch, context.upload(value))
        source = context.upload(np.ones(batch.size, np.float32))
        scores, mask = ops.scores(batch, source), ops.feasible(batch)
        context.release(source)
        with pytest.raises(ValueError, match="released buffer"):
            ops.choose(batch, scores, mask)
        scores = ops.newton_scores(batch)
        source = context.upload(np.ones(batch.size, bool))
        mask = ops.mask(batch, source)
        context.release(source)
        with pytest.raises(ValueError, match="released buffer"):
            ops.choose(batch, scores, mask)


@pytest.mark.parametrize("released", ["histogram", "fields", "rows", "batch", "scores", "mask"])
def test_released_dependencies_reject_choices(released):
    with ExecutionContext() as context:
        ops, _, fields, rows, hist = setup(context)
        batch = ops.candidates(hist)
        scores, mask = ops.newton_scores(batch), ops.feasible(batch)
        ops.release(
            dict(histogram=hist, fields=fields, rows=rows, batch=batch, scores=scores, mask=mask)[
                released
            ]
        )
        before = context.metrics["live_bytes"]
        with pytest.raises(ValueError, match="released"):
            ops.choose(batch, scores, mask)
        assert context.metrics["live_bytes"] == before


def test_wrong_batch_forged_records_and_route_identity():
    with ExecutionContext() as context, ExecutionContext() as other_context:
        ops, data, _, rows, hist = setup(context)
        batch, other_batch = ops.candidates(hist), ops.candidates(hist)
        scores, mask = ops.newton_scores(batch), ops.feasible(batch)
        other_scores, other_mask = ops.newton_scores(other_batch), ops.feasible(other_batch)
        for action in (
            lambda: ops.choose(batch, other_scores, mask),
            lambda: ops.choose(batch, scores, other_mask),
            lambda: ops.mask_and(mask, other_mask),
            lambda: ops.choose(replace(batch), scores, mask),
            lambda: ops.choose(batch, replace(scores), mask),
            lambda: DeviceOperations(context).feasible(batch),
            lambda: DeviceOperations(other_context).feasible(batch),
        ):
            with pytest.raises(ValueError, match="same candidate batch|foreign"):
                action()
        choice = ops.choose(batch, scores, mask)
        duplicate_rows = ops.rows(data)
        with pytest.raises(ValueError, match="different routed rows"):
            ops.partition(duplicate_rows, choice)
        with pytest.raises(ValueError, match="forged"):
            ops.partition(rows, replace(choice, index=choice.index + 1))
        ops.release(choice)
        with pytest.raises(ValueError, match="released"):
            ops.partition(rows, choice)


def test_candidate_allocation_failure_preserves_sources_and_recovers():
    limit = 16384
    with ExecutionContext(max_bytes=limit) as context:
        ops, _, _, _, hist = setup(context)
        expected = context.export(hist.total)
        # Leave one pool block: candidate values fit, later counts/active cannot.
        ballast_bytes = limit - context.metrics["peak_pool_bytes"] - 512
        ballast = context.upload(np.zeros(ballast_bytes // 4, np.float32))
        before = context.metrics["live_bytes"]
        with pytest.raises(MemoryError):
            ops.candidates(hist)
        assert context.metrics["live_bytes"] == before
        np.testing.assert_array_equal(context.export(hist.total), expected)
        context.release(ballast)
        batch = ops.candidates(hist)
        assert ops.choose(batch, ops.newton_scores(batch), ops.feasible(batch)).key == (0, 0, False)


def test_context_stream_restored_after_split_operations():
    import cupy as cp

    caller = cp.cuda.Stream(non_blocking=True)
    with caller, ExecutionContext() as context:
        ops, _, _, rows, hist = setup(context)
        batch = ops.candidates(hist)
        choice = ops.choose(batch, ops.newton_scores(batch), ops.feasible(batch))
        children = ops.partition(rows, choice)
        ops.leaf(hist)
        assert cp.cuda.get_current_stream().ptr == caller.ptr
        np.testing.assert_array_equal(context.export(children[0].positions), [0])
