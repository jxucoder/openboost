"""117 vector primitive parity and failure contracts; real CUDA only."""

import numpy as np
import pytest

from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .reference import device_vector as ref
from .test_device_vector_reference import prepared

pytestmark = pytest.mark.gpu


def bind(ops, data, context, cpu_fields, *, reordered=False):
    order = list(range(len(cpu_fields.names)))
    if reordered:
        order.reverse()
    buffer = context.upload(np.asarray(cpu_fields.values[:, order], np.float32))
    return ops.fields(
        data,
        buffer,
        names=tuple(cpu_fields.names[i] for i in order),
        roles=tuple(cpu_fields.roles[i] for i in order),
    )


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("rows", [tuple(range(8)), (6, 0, 4, 7), (3,), ()])
@pytest.mark.parametrize("reordered", [False, True])
def test_all_candidates_original_rows_and_leaf(width, rows, reordered):
    source, problem, binned, fields, _ = prepared(width)
    weighted_g = source["g"] * source["weight"][:, None]
    weighted_h = source["h"] * source["weight"][:, None]
    known = ref.candidates(
        source["x"], weighted_g, weighted_h, rows, regularization=2, penalty=0.5, minimum=2
    )
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        data = ops.prepare(binned, problem)
        fields = bind(ops, data, context, fields, reordered=reordered)
        positions = ops.rows(data, rows)
        hist = ops.histogram(data, fields, positions)
        batch = ops.candidates(hist)
        before = dict(context.metrics)
        scores = ops.vector_scores(batch, reg_lambda=2, split_penalty=0.5)
        mask = ops.vector_feasible(batch, min_child_h=2)
        chosen = ops.choose(batch, scores, mask)
        value = ops.vector_leaf(hist, reg_lambda=2)
        after = dict(context.metrics)
        assert after["upload_bytes"] == before["upload_bytes"]
        assert after["export_bytes"] - before["export_bytes"] == sum(
            after[k] - before[k] for k in ("validation_export_bytes", "decision_export_bytes")
        )
        actual_scores, actual_mask = context.export(scores.values), context.export(mask.values)
        sums, counts = context.export(batch.values), context.export(batch.counts)
        active = context.export(batch.active)
        by_key = {batch.key(i): i for i in range(batch.size) if active[i]}
        assert set(by_key) == {c["key"] for c in known}
        assert not actual_mask[~active].any()
        assert not actual_scores[~active].any()
        for c in known:
            i = by_key[c["key"]]
            assert actual_mask[i] == c["legal"]
            np.testing.assert_allclose(actual_scores[i], float(c["gain"]), rtol=1e-4, atol=1e-5)
            np.testing.assert_array_equal(counts[i], [len(s) for s in c["rows"]])
            expected = np.array([[*c["g"][s], *c["h"][s]] for s in range(2)], float)
            np.testing.assert_array_equal(sums[i], expected[:, ::-1] if reordered else expected)
        winner = ref.winner(known)
        assert (chosen.key if chosen else None) == (winner["key"] if winner else None)
        if chosen:
            for child, expected in zip(
                ops.partition(positions, chosen), winner["rows"], strict=True
            ):
                np.testing.assert_array_equal(context.export(child.positions), expected)
        np.testing.assert_allclose(
            context.export(value),
            [
                float(v)
                for v in ref.leaf(ref.total(weighted_g, rows), ref.total(weighted_h, rows), 2)
            ],
            rtol=1e-4,
            atol=1e-5,
        )


@pytest.mark.parametrize("method", ["vector_scores", "vector_feasible", "vector_leaf"])
def test_negative_curvature_outside_routed_rows_fails_atomically(method):
    _, problem, binned, fields, _ = prepared(2)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        data = ops.prepare(binned, problem)
        values = fields.values.copy()
        values[3, -1] = -1  # zero-weight original row, absent from this view
        handle = context.upload(values.astype(np.float32))
        fields = ops.fields(data, handle, names=fields.names, roles=fields.roles)
        hist = ops.histogram(data, fields, ops.rows(data, [0, 1, 2]))
        argument = hist if method == "vector_leaf" else ops.candidates(hist)
        buffers, records = set(context._buffers), set(ops._records)
        with pytest.raises(ValueError):
            getattr(ops, method)(argument)
        assert set(context._buffers) == buffers and set(ops._records) == records
        np.testing.assert_array_equal(context.export(handle), values)


@pytest.mark.parametrize("case", ["ties", "zero_channel"])
def test_tie_once_penalty_and_all_channel_feasibility(case):
    source, problem, binned, fields, _ = prepared(2, case=case)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        data = ops.prepare(binned, problem)
        fields = bind(ops, data, context, fields)
        hist = ops.histogram(data, fields, ops.rows(data))
        batch = ops.candidates(hist)
        scores = ops.vector_scores(batch, split_penalty=3)
        mask = ops.vector_feasible(batch)
        chosen = ops.choose(batch, scores, mask)
        if case == "ties":
            assert chosen.key == (0, 0, False)
            assert context.export(scores.values)[chosen.index] == pytest.approx(22.6, abs=1e-5)
        else:
            assert chosen is None
            with pytest.raises(ValueError, match="denominator"):
                ops.vector_leaf(hist, reg_lambda=0)
            assert np.isfinite(context.export(ops.vector_leaf(hist))).all()


def test_vector_single_channel_matches_scalar_exactly():
    _, problem, binned, fields, _ = prepared(1)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        data = ops.prepare(binned, problem)
        vector = bind(ops, data, context, fields)
        scalar = ops.fields(
            data, vector.values, names=("gradient", "curvature"), roles=("training", "training")
        )
        positions = ops.rows(data)
        vh, sh = (ops.histogram(data, f, positions) for f in (vector, scalar))
        vb, sb = (ops.candidates(h) for h in (vh, sh))
        np.testing.assert_array_equal(
            context.export(ops.vector_scores(vb).values),
            context.export(ops.newton_scores(sb).values),
        )
        np.testing.assert_array_equal(
            context.export(ops.vector_feasible(vb).values), context.export(ops.feasible(sb).values)
        )
        np.testing.assert_array_equal(
            context.export(ops.vector_leaf(vh)), context.export(ops.leaf(sh))
        )


@pytest.mark.parametrize("reordered", [False, True])
def test_weight_applied_once_with_independent_information(reordered):
    source, problem, binned, expected, _ = prepared(2)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        data = ops.prepare(binned, problem)
        values = np.column_stack((source["g"], source["h"], np.arange(8)))
        names, roles = (*expected.names, "cohort"), ("unweighted",) * 4 + ("independent",)
        order = list(reversed(range(5))) if reordered else list(range(5))
        fields = ops.fields(
            data,
            context.upload(values[:, order].astype(np.float32)),
            names=tuple(names[i] for i in order),
            roles=tuple(roles[i] for i in order),
        )
        weighted = ops.apply_weight(fields)
        np.testing.assert_array_equal(
            context.export(weighted.values),
            np.column_stack((expected.values, np.arange(8)))[:, order],
        )
        with pytest.raises(ValueError, match="already applied"):
            ops.apply_weight(weighted)
        hist = ops.histogram(data, weighted, ops.rows(data))
        batch = ops.candidates(hist)
        mask = ops.mask_and(ops.vector_feasible(batch), ops.child_minimum(batch, "cohort", 1))
        assert ops.choose(batch, ops.vector_scores(batch), mask) is not None


@pytest.mark.parametrize("failure", ["gain_overflow", "leaf_denominator", "dispatch"])
def test_operation_failure_preserves_inputs_and_releases_outputs(failure, monkeypatch):
    _, problem, binned, fields, _ = prepared(2)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        data = ops.prepare(binned, problem)
        values = fields.values.copy().astype(np.float32)
        if failure == "gain_overflow":
            values[0, 0] = 1e20
        elif failure == "leaf_denominator":
            values[:, -1] = 0
            values[0, -1] = 3e38
        handle = context.upload(values)
        fields = ops.fields(data, handle, names=fields.names, roles=fields.roles)
        hist = ops.histogram(data, fields, ops.rows(data))
        batch = ops.candidates(hist)
        if failure == "dispatch":
            original = ops._launch

            def fail(name, *args):
                if name == "vector_leaf":
                    raise RuntimeError("injected vector dispatch failure")
                return original(name, *args)

            monkeypatch.setattr(ops, "_launch", fail)
        buffers, records = set(context._buffers), set(ops._records)
        with pytest.raises(RuntimeError if failure == "dispatch" else ValueError):
            if failure == "gain_overflow":
                ops.vector_scores(batch)
            else:
                ops.vector_leaf(hist, reg_lambda=3e38 if failure == "leaf_denominator" else 1)
        assert set(context._buffers) == buffers and set(ops._records) == records
        np.testing.assert_array_equal(context.export(handle), values)
