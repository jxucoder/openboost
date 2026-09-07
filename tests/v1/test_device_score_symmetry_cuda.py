"""089 score arithmetic diagnostics and strict ordering; real CUDA hardware only."""

import hashlib
import json

import numpy as np
import pytest

from openboost import device_objectives as objective
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .reference.device_rounds import rounds
from .test_device_round_reference import prepared_fixture

pytestmark = pytest.mark.gpu

ROOT_KEYS = ((0, 0, True), (0, 3, False))


def root_batch(context):
    train, _, binned, _ = prepared_fixture("weighted")
    ops = DeviceOperations(context)
    data = ops.prepare(binned, train)
    problem = objective.prepare(ops, data, train)
    base = objective.base(ops, problem)
    raw = objective.broadcast(ops, base, data.n_rows)
    fields = objective.fields(ops, problem, raw)
    rows = ops.rows(data)
    histogram = ops.histogram(data, fields, rows)
    return ops, base, fields, histogram, ops.candidates(histogram)


def launch_diagnostic(context, kernel, *args):
    # Diagnostic-only direct kernel access uses the same owned buffers and stream.
    # It adds no production custom-kernel registry or CPU emulation boundary.
    from numba import cuda

    with context._scope():
        stream = cuda.external_stream(context._stream.ptr)
        arrays = [
            cuda.as_cuda_array(a, sync=False) if isinstance(a, context._cp.ndarray) else a
            for a in args
        ]
        kernel[1, 128, stream](*arrays)


def bits(array):
    return np.asarray(array, dtype=np.float32).view(np.uint32).tolist()


def assembly(context, kernel):
    with context._scope():
        result = kernel.inspect_asm()
    assert isinstance(result, dict) and result
    return [
        dict(signature=str(signature), sha256=hashlib.sha256(ptx.encode()).hexdigest(), ptx=ptx)
        for signature, ptx in result.items()
    ]


def test_weighted_root_scores_and_archived_kernel_diagnostics():
    from openboost import _device_kernels

    from .run4_score_kernel import scalar_scores as archived_scores

    with ExecutionContext() as context:
        ops, base, fields, histogram, batch = root_batch(context)
        scores = ops.newton_scores(batch)
        old_output = context._empty((batch.size,), np.float32)
        launch_diagnostic(
            context,
            archived_scores,
            *(
                context._array(b)
                for b in (batch.values, batch.counts, batch.active, histogram.total)
            ),
            0,
            1,
            np.float32(1),
            np.float32(0),
            context._array(old_output),
        )
        allowed = ops.feasible(batch)
        choice = ops.choose(batch, scores, allowed)
        old_choice = ops.choose(batch, ops.scores(batch, old_output), allowed)
        indices = [next(i for i in range(batch.size) if batch.key(i) == key) for key in ROOT_KEYS]
        sums = context.export(batch.values)
        gains = context.export(scores.values)
        old_gains = context.export(old_output)
        diagnostic = dict(
            kind="weighted_root",
            archived_revision="c4157559c5982e3df849ae59129c48dc8e60b454",
            keys=ROOT_KEYS,
            indices=indices,
            base=context.export(base).tolist(),
            weighted_fields=context.export(fields.values).tolist(),
            histogram=context.export(histogram.sums).tolist(),
            parent=context.export(histogram.total).tolist(),
            summaries=sums[indices].tolist(),
            summary_bits=bits(sums[indices]),
            counts=context.export(batch.counts)[indices].tolist(),
            scores=gains[indices].tolist(),
            score_bits=bits(gains[indices]),
            archived_scores=old_gains[indices].tolist(),
            archived_score_bits=bits(old_gains[indices]),
            winner=choice.key if choice else None,
            archived_winner=old_choice.key if old_choice else None,
            corrected_assembly=assembly(context, _device_kernels.scalar_scores),
            archived_assembly=assembly(context, archived_scores),
        )
        # Keep the measured data and generated code even when parity fails below.
        print("score_diagnostic=" + json.dumps(diagnostic), flush=True)
        np.testing.assert_array_equal(sums[indices[0]], sums[indices[1], ::-1])
        assert bits(gains[indices])[0] == bits(gains[indices])[1]
        assert choice.key == rounds("weighted", depth=1)[1][0]["nodes"][0]["key"]
        assert choice.key == ROOT_KEYS[0]
        assert all(
            item["ptx"]
            for key in ("corrected_assembly", "archived_assembly")
            for item in diagnostic[key]
        )
        assert context.metrics["peak_pool_bytes"] <= 16 * 1024**2


@pytest.mark.parametrize("reordered", [False, True])
@pytest.mark.parametrize("regularization,penalty", [(0, 0), (1, 0), (2, 0.125)])
def test_direct_swapped_child_scores(regularization, penalty, reordered):
    from openboost import _device_kernels

    pair = np.array([[-3.222222328186035, 2], [3.222221851348877, 7]], np.float32)
    parent = pair.sum(axis=0, dtype=np.float32)
    summaries = np.stack((pair, pair[::-1]))
    g, h = (1, 0) if reordered else (0, 1)
    if reordered:
        summaries = summaries[:, :, ::-1].copy()
        parent = parent[::-1].copy()
    with ExecutionContext() as context:
        buffers = [
            context.upload(a)
            for a in (summaries, np.ones((2, 2), np.int64), np.ones(2, bool), parent)
        ]
        output = context._empty((2,), np.float32)
        launch_diagnostic(
            context,
            _device_kernels.scalar_scores,
            *(context._array(b) for b in buffers),
            g,
            h,
            np.float32(regularization),
            np.float32(penalty),
            context._array(output),
        )
        actual = context.export(output)
        print(
            "score_diagnostic="
            + json.dumps(
                dict(
                    kind="swapped",
                    regularization=regularization,
                    penalty=penalty,
                    reordered=reordered,
                    scores=actual.tolist(),
                    score_bits=bits(actual),
                )
            ),
            flush=True,
        )
        assert bits(actual)[0] == bits(actual)[1]
        gl, hl, gr, hr = pair.astype(np.float64).ravel()
        expected = (
            0.5
            * (
                gl**2 / (hl + regularization)
                + gr**2 / (hr + regularization)
                - (gl + gr) ** 2 / (hl + hr + regularization)
            )
            - penalty
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("exponent", [-10, 0, 10])
def test_adjacent_ulp_scores_are_strictly_ordered(exponent):
    with ExecutionContext() as context:
        ops, _, _, _, batch = root_batch(context)
        indices = [next(i for i in range(batch.size) if batch.key(i) == key) for key in ROOT_KEYS]
        low = np.float32(2.0**exponent)
        high = np.nextafter(low, np.float32(np.inf))
        values = np.zeros(batch.size, np.float32)
        values[indices] = (low, high)
        allowed = ops.feasible(batch)
        scores = ops.scores(batch, context.upload(values))
        assert ops.choose(batch, scores, allowed).key == ROOT_KEYS[1]
        values[indices] = (high, low)
        scores = ops.scores(batch, context.upload(values))
        assert ops.choose(batch, scores, allowed).key == ROOT_KEYS[0]
        values[indices] = (high, high)
        scores = ops.scores(batch, context.upload(values))
        assert ops.choose(batch, scores, allowed).key == ROOT_KEYS[0]
