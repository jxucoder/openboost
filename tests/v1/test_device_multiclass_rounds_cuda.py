"""110 prescribed joint multiclass rounds; real CUDA required, no emulation."""

import numpy as np
import pytest

from openboost import device_multiclass as multiclass
from openboost import device_tree as trees
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.device_runtime import DeviceRun, DeviceTerm
from openboost.execution import ExecutionContext

from .reference.device_multiclass import METRIC_SCALE, geometry, rounds
from .test_device_glm_rounds_cuda import check_root
from .test_device_multiclass_reference import fixture
from .test_device_runtime_cuda import raw
from .test_multiclass_composition import close, fresh_replay, original_rows

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("width", [2, 3, 5])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_joint_rounds_routes_fields_metrics_and_cpu_only_inference(width, depth, tmp_path):
    import cupy as cp

    train, validation = fixture(width), fixture(width, validation=True)
    initial, expected = rounds(original_rows(train), original_rows(validation), depth=depth)
    binning = Binning(("x",), (np.arange(len(train.target) - 1, dtype=float),))
    with ExecutionContext() as context:
        external = cp.cuda.Stream(non_blocking=True)
        with external:
            ops = DeviceOperations(context)
            run = DeviceRun(
                ops,
                train,
                validation,
                run_id="110",
                seed=7,
                binning=binning,
                objective=multiclass.objective(),
            )
            state = run.initialize()
            np.testing.assert_array_equal(
                raw(context, run, state), np.broadcast_to(initial, train.offset.shape)
            )
            for index, ref in enumerate(expected):
                values = run.raw(state)
                prior = context.export(values)
                g, h = multiclass.geometry(ops, run.problem, values)
                close(context.export(g), ref["gradient"])
                close(context.export(h), ref["bound"])
                terms = []
                for j, nodes in enumerate(ref["trees"]):
                    fields = multiclass.channel_fields(ops, run.data, g, h, channel=j)
                    close(context.export(fields.values), ref["fields"][j])
                    if depth:
                        check_root(ops, run.data, fields, train.data.values, ref["fields"][j])
                    tree = trees.depthwise(ops, run.data, fields, binning=binning, max_depth=depth)
                    assert [None if f == -1 else (f, t, m) for f, t, m, _, _ in tree.topology] == [
                        n["key"] for n in nodes
                    ]
                    close(trees.export(ops, tree).value[:, 0], [n["value"] for n in nodes])
                    terms.append(DeviceTerm(tree, np.eye(width)[j : j + 1]))
                    ops.release(fields)
                # All class trees were fitted from the same parent geometry.
                for handle in (values, g, h):
                    context.release(handle)
                before = dict(context.metrics)
                proposal = run.propose_terms(state, terms, coefficient=0.25)
                assert run.resolve(state, proposal, accept=False) is state
                updated = run.resolve(state, proposal, accept=True)
                after = dict(context.metrics)
                assert after["upload_bytes"] == before["upload_bytes"]
                assert after["export_bytes"] - before["export_bytes"] == sum(
                    after.get(k, 0) - before.get(k, 0)
                    for k in ("validation_export_bytes", "metric_export_bytes")
                )
                assert updated.version == index + 1
                assert updated.n_terms == width * (index + 1)
                np.testing.assert_array_equal(raw(context, run, state), prior)
                run.release(proposal)
                for term in terms:
                    ops.release(term.tree)
                run.release(state)
                state = updated
                actual = raw(context, run, state)
                actual_validation = raw(context, run, state, True)
                close(actual, ref["raw"])
                close(actual_validation, ref["validation_raw"])
                for p, values, score, reference in (
                    (train, actual, state.loss, ref["loss"]),
                    (validation, actual_validation, state.validation_score, ref["score"]),
                ):
                    exact = geometry(values, p.target[:, 0], p.offset, p.weight)[0]
                    assert abs(score - exact) <= METRIC_SCALE * max(1, abs(exact))
                    assert abs(score - reference) <= METRIC_SCALE * max(1, abs(reference))
            model = run.export(state)
            assert model.classes == train.classes
            run.close()
            assert context.metrics["live_bytes"] == 0 and not ops._records
            assert cp.cuda.get_current_stream().ptr == external.ptr
    # Export is an explicit boundary; CPU prediction no longer needs training code.
    restored = fresh_replay(model, validation, tmp_path)
    close(restored.predict(validation.data), actual_validation)
    probabilities = geometry(
        expected[-1]["validation_raw"],
        validation.target[:, 0],
        validation.offset,
        validation.weight,
    )[3]
    close(restored.predict_proba(validation.data, offset=validation.offset), probabilities)
    np.testing.assert_array_equal(
        restored.predict_label(validation.data, offset=validation.offset),
        train.classes.decode(probabilities.argmax(axis=1)),
    )
