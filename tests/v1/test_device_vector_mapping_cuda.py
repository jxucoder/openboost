"""118 general mapped arithmetic and transaction ownership; real CUDA only."""

import numpy as np
import pytest

from openboost import device_multi_squared as objective
from openboost import device_objectives as operations
from openboost import device_tree as trees
from openboost.device import DeviceOperations
from openboost.device_runtime import DeviceRun, DeviceTerm, map_update
from openboost.execution import ExecutionContext

from .reference.multi_squared import mapped
from .test_multi_squared_reference import prepared

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("width,raw_width", [(1, 3), (2, 1), (2, 4), (4, 2)])
@pytest.mark.parametrize("coefficient", [0, 1, -0.5])
def test_mapped_products_and_order_match_exact_binary32(width, raw_width, coefficient):
    raw = (np.arange(3 * raw_width).reshape(3, raw_width) / 8).astype(np.float32)
    prediction = (np.arange(3 * width).reshape(3, width) / 3 - 2).astype(np.float32)
    mapping = (np.arange(width * raw_width).reshape(width, raw_width) / 7 - 1).astype(np.float32)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        source, values = context.upload(raw), context.upload(prediction)
        before = dict(context.metrics)
        result = map_update(ops, source, values, mapping, coefficient)
        after = dict(context.metrics)
        assert after["upload_bytes"] == before["upload_bytes"]
        assert (
            after["export_bytes"] - before["export_bytes"]
            == after["validation_export_bytes"] - before["validation_export_bytes"]
        )
        np.testing.assert_array_equal(
            context.export(result), mapped(raw, prediction, mapping, np.float32(coefficient))
        )
        np.testing.assert_array_equal(context.export(source), raw)
        np.testing.assert_array_equal(context.export(values), prediction)


def constant_tree(run, width):
    raw = run.raw(next(iter(run._states)))
    gradient, curvature = objective.geometry(run.ops, run.problem, raw)
    fields = operations.vector_fields(run.ops, run.data, gradient, curvature)
    value = run.execution.upload(np.arange(1, width + 1, dtype=np.float32) / 4)
    tree = trees.depthwise(
        run.ops,
        run.data,
        fields,
        binning=run.binning,
        max_depth=0,
        output_width=width,
        leaf=lambda *a: value,
    )
    for buffer in (raw, gradient, curvature, value):
        run.execution.release(buffer)
    run.ops.release(fields)
    return tree


@pytest.mark.parametrize("width,raw_width", [(2, 1), (2, 4), (4, 2)])
def test_two_round_general_terms_snapshot_rejection_and_export(width, raw_width):
    train, validation, binning = prepared(raw_width)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        run = DeviceRun(
            ops,
            train,
            validation,
            binning=binning,
            run_id="general",
            seed=17,
            objective=objective.objective(),
            comparison="objective",
        )
        state = run.initialize()
        mapping = (np.arange(width * raw_width).reshape(width, raw_width) / 8 - 0.5).astype(
            np.float32
        )
        for i in range(2):
            raw = run.raw(state)
            before = context.export(raw)
            context.release(raw)
            tree = constant_tree(run, width)
            prediction = np.tile(np.arange(1, width + 1, dtype=np.float32) / 4, (8, 1))
            proposal = run.propose_terms(state, (DeviceTerm(tree, mapping),), coefficient=0.5)
            assert run.resolve(state, proposal, accept=False) is state
            updated = run.resolve(state, proposal, accept=True)
            ops.release(tree)
            run.release(proposal)
            run.release(state)
            state = updated
            actual = run.raw(state)
            np.testing.assert_array_equal(
                context.export(actual), mapped(before, prediction, mapping, np.float32(0.5))
            )
            np.testing.assert_allclose(
                run.export(state).predict(train.data), context.export(actual), rtol=1e-4, atol=1e-5
            )
            context.release(actual)
            assert state.version == i + 1 and state.n_terms == i + 1
        run.close()
        assert not ops._records and not context._buffers


def test_vector_cancellation_and_partial_proposal_failure(monkeypatch):
    train, validation, binning = prepared(2)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        a, b = (
            context.upload(np.zeros((1, 1), np.float32)),
            context.upload(np.array([[2**24, 1, -(2**24)]], np.float32)),
        )
        result = map_update(ops, a, b, np.ones((3, 1), np.float32))
        np.testing.assert_array_equal(context.export(result), [[0]])
        for handle in (a, b, result):
            context.release(handle)
        run = DeviceRun(
            ops,
            train,
            validation,
            binning=binning,
            run_id="failure",
            seed=17,
            objective=objective.objective(),
            comparison="objective",
        )
        state = run.initialize()
        tree = constant_tree(run, 2)
        terms = (DeviceTerm(tree, np.eye(2)),)
        buffers, records = set(context._buffers), set(ops._records)
        original, calls = ops._launch, []

        def dispatch(name, *args):
            if name == "vector_mapped_add_raw":
                calls.append(name)
                if len(calls) == 2:
                    raise RuntimeError("injected validation map failure")
            return original(name, *args)

        monkeypatch.setattr(ops, "_launch", dispatch)
        with pytest.raises(RuntimeError, match="injected"):
            run.propose_terms(state, terms)
        assert len(calls) == 2
        assert set(context._buffers) == buffers and set(ops._records) == records
        assert not run._proposals and run.validate_state(state) is state
        run.close()
        ops.release(tree)
        assert not ops._records and not context._buffers
