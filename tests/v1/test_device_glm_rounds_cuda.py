"""106 prescribed GLM rounds and persistence; real CUDA execution is still required."""

import subprocess
import sys

import numpy as np
import pytest

from openboost import device_glm as glm
from openboost import device_tree as trees
from openboost.artifacts import Model
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.device_runtime import DeviceRun
from openboost.execution import ExecutionContext

from .reference.device_glm import ATOL, METRIC_SCALE, RTOL, geometry, rounds
from .reference.device_splits import enumerate_candidates, winner
from .test_device_glm_reference import original_rows, paired_fixture
from .test_device_runtime_cuda import raw

pytestmark = pytest.mark.gpu


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def check_root(ops, data, fields, x, expected):
    """Every candidate, exact route and both leaves against original-row sums."""
    context = ops.execution
    options, total = enumerate_candidates(x, expected, range(len(x)))
    options = {c["key"]: c for c in options}
    records = []
    try:
        rows = ops.rows(data)
        records.append(rows)
        histogram = ops.histogram(data, fields, rows)
        records.append(histogram)
        close(context.export(histogram.total), total)
        candidates = ops.candidates(histogram)
        records.append(candidates)
        scores, legal = ops.newton_scores(candidates), ops.feasible(candidates)
        records.extend((scores, legal))
        active, counts, values = (
            context.export(h) for h in (candidates.active, candidates.counts, candidates.values)
        )
        gains, allowed = context.export(scores.values), context.export(legal.values)
        seen = set()
        for index in np.flatnonzero(active):
            key = candidates.key(index)
            ref = options[key]
            seen.add(key)
            np.testing.assert_array_equal(counts[index], ref["counts"])
            close(values[index], ref["sums"])
            assert bool(allowed[index]) == ref["legal"]
            assert gains[index] == pytest.approx(ref["gain"], rel=RTOL, abs=ATOL)
        assert seen == set(options)
        selected = ops.choose(candidates, scores, legal)
        if selected is not None:
            records.append(selected)
        ref = winner(list(options.values()))
        assert selected is not None and selected.key == ref["key"]
        children = ops.partition(rows, selected)
        records.extend(children)
        for child, positions, sums in zip(children, ref["rows"], ref["sums"], strict=True):
            np.testing.assert_array_equal(context.export(child.positions), positions)
            hist = ops.histogram(data, fields, child)
            records.append(hist)
            close(context.export(hist.total), sums)
            leaf = ops.leaf(hist)
            close(context.export(leaf), [-sums[0] / (sums[1] + 1)])
            context.release(leaf)
    finally:
        for record in reversed(records):
            ops.release(record)


@pytest.mark.parametrize("family", ["binary", "poisson"])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_prescribed_rounds_route_metrics_and_saved_cpu_inference(family, depth, tmp_path):
    import cupy as cp

    train, validation = paired_fixture(family)
    initial, expected = rounds(family, original_rows(train), original_rows(validation), depth=depth)
    binning = Binning(("x",), (np.arange(7, dtype=float),))
    path, inputs, output = (tmp_path / name for name in ("model.json", "inputs.npz", "output.npz"))
    with ExecutionContext() as context:
        external = cp.cuda.Stream(non_blocking=True)
        with external:
            ops = DeviceOperations(context)
            run = DeviceRun(
                ops,
                train,
                validation,
                run_id="106-" + family,
                seed=7,
                binning=binning,
                objective=getattr(glm, family)(),
            )
            state = run.initialize()
            close(raw(context, run, state), np.full((8, 1), initial))
            for index, ref in enumerate(expected):
                prior = raw(context, run, state)
                gradient, fields = run.gradient(state), run.fields(state)
                close(context.export(gradient), ref["gradient"])
                close(context.export(fields.values), ref["fields"])
                if depth:
                    check_root(ops, run.data, fields, train.data.values, ref["fields"])
                tree = trees.depthwise(ops, run.data, fields, binning=binning, max_depth=depth)
                assert [None if f == -1 else (f, t, m) for f, t, m, _, _ in tree.topology] == [
                    n["key"] for n in ref["nodes"]
                ]
                close(trees.export(ops, tree).value[:, 0], [n["value"] for n in ref["nodes"]])
                before = dict(context.metrics)
                proposal = run.propose(state, tree, coefficient=0.25)
                assert run.resolve(state, proposal, accept=False) is state
                updated = run.resolve(state, proposal, accept=True)
                after = dict(context.metrics)
                assert after["upload_bytes"] == before["upload_bytes"]
                assert after["export_bytes"] - before["export_bytes"] == sum(
                    after.get(k, 0) - before.get(k, 0)
                    for k in ("validation_export_bytes", "metric_export_bytes")
                )
                np.testing.assert_array_equal(raw(context, run, state), prior)
                run.release(proposal)
                ops.release(tree)
                ops.release(fields)
                context.release(gradient)
                run.release(state)
                state = updated
                assert state.version == state.n_terms == index + 1
                actual, actual_validation = raw(context, run, state), raw(context, run, state, True)
                close(actual[:, 0], ref["raw"])
                close(actual_validation[:, 0], ref["validation_raw"])
                for p, values, score, expected_score in (
                    (train, actual, state.loss, ref["loss"]),
                    (validation, actual_validation, state.validation_score, ref["score"]),
                ):
                    exposure = p.structure.get("exposure")
                    exact = geometry(
                        family,
                        values[:, 0],
                        p.target[:, 0],
                        p.offset[:, 0],
                        p.weight,
                        None if exposure is None else exposure[:, 0],
                    )[0]
                    assert abs(score - exact) <= METRIC_SCALE * max(1, abs(exact))
                    assert abs(score - expected_score) <= METRIC_SCALE * max(1, abs(expected_score))
            model = run.export(state)
            assert model.classes == train.classes
            model.save(path)
            run.close()
            assert context.metrics["live_bytes"] == 0 and not ops._records
            assert cp.cuda.get_current_stream().ptr == external.ptr
    restored = Model.load(path)
    assert restored.identity == model.identity and restored.classes == train.classes
    close(restored.predict(validation.data), actual_validation)
    close(
        restored.predict(validation.data, offset=validation.offset),
        actual_validation + validation.offset,
    )
    exposure = validation.structure.get("exposure", np.ones((8, 1)))
    np.savez(inputs, x=validation.data.values, offset=validation.offset, exposure=exposure)
    # A separate process blocks CUDA/training-component imports and uses only the
    # saved public Model. This validates inference after all device state is gone.
    script = """
import sys
for name in ('cupy', 'numba', 'openboost.device_glm', 'openboost.device_runtime'):
    sys.modules[name] = None
import numpy as np
from openboost import NumericData
from openboost.artifacts import Model
m = Model.load(sys.argv[1])
with np.load(sys.argv[2]) as f:
    data = NumericData(f['x'], np.arange(len(f['x'])), m.feature_names)
    raw = m.predict(data, offset=f['offset'])
    if m.classes is not None:
        prediction = m.predict_proba(data, offset=f['offset'])
        labels = m.predict_label(data, offset=f['offset'])
    else:
        prediction = np.exp(raw) * f['exposure']
        labels = ()
    np.savez(sys.argv[3], raw=raw, prediction=prediction, labels=np.asarray(labels))
"""
    subprocess.run([sys.executable, "-c", script, str(path), str(inputs), str(output)], check=True)
    with np.load(output) as replay:
        expected_raw = expected[-1]["validation_raw"][:, None] + validation.offset
        close(replay["raw"], expected_raw)
        if family == "binary":
            probability = 1 / (1 + np.exp(-expected_raw[:, 0]))
            close(replay["prediction"], np.column_stack((1 - probability, probability)))
            assert np.min(np.abs(expected_raw)) > 1e-3
            np.testing.assert_array_equal(
                replay["labels"], train.classes.decode((probability > 0.5).astype(int))
            )
        else:
            close(replay["prediction"], np.exp(expected_raw) * exposure)
