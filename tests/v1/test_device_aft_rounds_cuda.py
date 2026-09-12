"""Prescribed AFT CUDA rounds with exact routing, scale-aware export and ownership."""

import numpy as np
import pytest

from openboost import device_aft as aft
from openboost import device_tree as trees
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.device_runtime import DeviceRun
from openboost.execution import ExecutionContext

from .reference.device_aft import METRIC_SCALE, geometry, rounds
from .test_aft_composition import close, fresh_replay
from .test_device_aft_reference import fixture, original_rows
from .test_device_glm_rounds_cuda import check_root
from .test_device_runtime_cuda import raw

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("sigma", [0.5, 0.7, 1, 2])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_prescribed_rounds_routing_metrics_and_saved_scale(sigma, depth, tmp_path):
    import cupy as cp

    train, validation = fixture(), fixture(validation=True)
    initial, expected = rounds(original_rows(train), original_rows(validation), sigma=sigma, depth=depth)
    binning = Binning(("x",), (np.arange(7, dtype=float),))
    with ExecutionContext() as context:
        external = cp.cuda.Stream(non_blocking=True)
        with external:
            ops = DeviceOperations(context)
            run = DeviceRun(ops, train, validation, run_id="114-aft", seed=7,
                            binning=binning, objective=aft.objective(sigma))
            state = run.initialize()
            assert run.problem.sigma == run.validation_problem.sigma == sigma
            close(raw(context, run, state), np.full((8, 1), initial))
            for index, ref in enumerate(expected):
                prior = raw(context, run, state)
                gradient, fields = run.gradient(state), run.fields(state)
                close(context.export(gradient), ref["gradient"])
                close(context.export(fields.values), ref["fields"])
                if depth:
                    check_root(ops, run.data, fields, train.data.values, ref["fields"])
                tree = trees.depthwise(ops, run.data, fields, binning=binning, max_depth=depth)
                assert [None if f == -1 else (f, t, m) for f, t, m, _, _ in tree.topology] == [n["key"] for n in ref["nodes"]]
                close(trees.export(ops, tree).value[:, 0], [n["value"] for n in ref["nodes"]])
                before = dict(context.metrics)
                proposal = run.propose(state, tree, coefficient=0.25)
                assert run.resolve(state, proposal, accept=False) is state
                updated = run.resolve(state, proposal, accept=True)
                after = dict(context.metrics)
                assert after["upload_bytes"] == before["upload_bytes"]
                assert after["export_bytes"] - before["export_bytes"] == sum(
                    after.get(k, 0) - before.get(k, 0) for k in ("validation_export_bytes", "metric_export_bytes")
                )
                np.testing.assert_array_equal(raw(context, run, state), prior)
                run.release(proposal)
                ops.release(tree)
                ops.release(fields)
                context.release(gradient)
                run.release(state)
                state = updated
                assert state.version == state.n_terms == index + 1
                actual, valid_raw = raw(context, run, state), raw(context, run, state, True)
                close(actual[:, 0], ref["raw"])
                close(valid_raw[:, 0], ref["validation_raw"])
                for p, values, score, reference in ((train, actual, state.loss, ref["loss"]),
                                                   (validation, valid_raw, state.validation_score, ref["score"])):
                    exact = geometry(values[:, 0], p.target[:, 0], p.target[:, 0] == p.target[:, 1], p.offset[:, 0], p.weight, sigma)[0]
                    assert abs(score - exact) <= METRIC_SCALE * max(1, abs(exact))
                    assert abs(score - reference) <= METRIC_SCALE * max(1, abs(reference))
            artifact = aft.export(run, state)
            best = aft.export(run, state, best=True)
            assert artifact.sigma == best.sigma == sigma
            assert len(best.model.terms) == state.best_n_terms
            with pytest.raises(ValueError, match="boolean"):
                aft.export(run, state, best=1)
            run.close()
            assert context.metrics["live_bytes"] == 0 and not ops._records
            assert cp.cuda.get_current_stream().ptr == external.ptr
    close(artifact.model.predict(validation.data), valid_raw)
    close(artifact.model.predict(validation.data, offset=validation.offset), valid_raw + validation.offset)
    fresh_replay(artifact, validation, tmp_path)
