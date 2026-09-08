"""088 resident geometry/tree checks; execute only on real CUDA hardware."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from openboost import device_objectives as objective
from openboost import device_tree as trees
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext
from openboost.objectives import Squared

from .reference.device_rounds import predict, rounds
from .test_device_round_reference import prepared_fixture

pytestmark = pytest.mark.gpu


def setup(context, case="weighted"):
    train, validation, binned, information = prepared_fixture(case)
    ops = DeviceOperations(context)
    data = ops.prepare(binned, train)
    problem = objective.prepare(ops, data, train)
    validation_data = ops.prepare(binned.binning.transform(validation.data), validation)
    validation_problem = objective.prepare(ops, validation_data, validation)
    return ops, data, problem, validation_problem, binned, information, train, validation


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def cohort_minimum(minimum):
    def allowed(ops, batch):
        mask = ops.feasible(batch)
        for name in ("cohort:blue", "cohort:red"):
            mask = ops.mask_and(mask, ops.child_minimum(batch, name, minimum))
        return mask

    return allowed


@pytest.mark.parametrize("case", ["weighted", "d2", "conflict"])
@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("minimum", [None, 0, 1, 2])
def test_resident_objective_tree_and_inference(case, depth, minimum):
    base, expected = rounds(case, depth=depth, minimum=minimum)
    ref = expected[0]
    with ExecutionContext() as context:
        ops, data, problem, validation, binned, info, train, validation_cpu = setup(context, case)
        before = dict(context.metrics)
        scalar = objective.base(ops, problem)
        raw = objective.broadcast(ops, scalar, data.n_rows)
        gradient = objective.gradient(ops, problem, raw)
        fields = objective.fields(ops, problem, raw)
        score = objective.loss(ops, problem, raw)
        after = dict(context.metrics)
        assert after["upload_bytes"] == before["upload_bytes"]
        assert after["export_bytes"] - before["export_bytes"] == (
            after["validation_export_bytes"] - before["validation_export_bytes"] + 8
        )
        close(context.export(scalar), [base])
        close(context.export(gradient), ref["gradient"])
        close(context.export(fields.values), ref["fields"][:, :2])
        assert score == pytest.approx(
            Squared.loss(train, np.full((data.n_rows, 1), base)), rel=1e-3, abs=1e-3
        )
        with pytest.raises(ValueError, match="already applied"):
            ops.apply_weight(fields)
        for column, name in ((1, "cohort:blue"), (0, "cohort:red")):
            buffer = context.upload(np.asarray(info[:, column], np.float32))
            augmented = ops.add_independent(fields, name, buffer, nonnegative=True)
            ops.release(fields)
            context.release(buffer)
            fields = augmented
        before = dict(context.metrics)
        tree = trees.depthwise(
            ops,
            data,
            fields,
            binning=binned.binning,
            max_depth=depth,
            legality=cohort_minimum(minimum) if minimum is not None else None,
        )
        train_prediction = trees.predict(ops, tree, data)
        validation_prediction = trees.predict(ops, tree, validation.data)
        after = dict(context.metrics)
        # Only explicit host topology travels to the device while growing a tree.
        assert after["upload_bytes"] - before["upload_bytes"] == tree.n_nodes * 5 * 4
        assert after["export_bytes"] - before["export_bytes"] == sum(
            after[k] - before[k] for k in ("validation_export_bytes", "decision_export_bytes")
        )
        cpu_tree = trees.export(ops, tree)
        actual = [None if f == -1 else (f, t, m) for f, t, m, _, _ in tree.topology]
        assert actual == [n["key"] for n in ref["nodes"]]
        assert list(cpu_tree.left) == [n["left"] for n in ref["nodes"]]
        assert list(cpu_tree.right) == [n["right"] for n in ref["nodes"]]
        close(cpu_tree.value[:, 0], [n["value"] for n in ref["nodes"]])
        close(context.export(train_prediction)[:, 0], predict(ref["nodes"], train.data.values))
        close(context.export(validation_prediction), cpu_tree.predict(validation_cpu.data))
        close(
            context.export(validation_prediction)[:, 0],
            predict(ref["nodes"], validation_cpu.data.values),
        )
        ops.release(fields)
        ops.release(problem)
        ops.release(data)
        # Tree inference has no dependency on training records or callback scratch.
        close(
            context.export(trees.predict(ops, tree, validation.data)),
            cpu_tree.predict(validation_cpu.data),
        )


def test_supplied_policy_leaf_and_scratch_ownership():
    with ExecutionContext() as context:
        ops, data, problem, _, binned, _, _, _ = setup(context, "d2")
        raw = objective.broadcast(ops, objective.base(ops, problem), data.n_rows)
        fields = objective.fields(ops, problem, raw)
        provided_leaf = context.upload(np.array([7], np.float32))
        callbacks = []
        scratch = []

        def scoring(ops, batch):
            # The invalid empty-child split scores highest. Structural masking must
            # still find the desired nonempty threshold rather than abandon growth.
            values = np.zeros(batch.size, np.float32)
            for i in range(batch.size):
                f, t, m = batch.key(i)
                if (f, t, m) == (0, 1, False):
                    values[i] = 2
                elif t == 5:
                    values[i] = 10
            handle = context.upload(values)
            scratch.append(handle)
            callbacks.append("score")
            return ops.scores(batch, handle)

        def legality(ops, batch):
            callbacks.append("mask")
            return ops.mask(batch, context.upload(np.ones(batch.size, bool)))

        def leaf(ops, histogram):
            callbacks.append("leaf")
            return provided_leaf

        before = context.metrics["live_bytes"]
        tree = trees.depthwise(
            ops,
            data,
            fields,
            binning=binned.binning,
            max_depth=1,
            scoring=scoring,
            legality=legality,
            leaf=leaf,
        )
        assert tree.topology[0][:3] == (0, 1, False)
        assert callbacks.count("leaf") == 3
        assert callbacks.count("score") == callbacks.count("mask") == 1
        close(context.export(provided_leaf), [7])
        for handle in scratch:
            with pytest.raises(ValueError, match="released"):
                context.export(handle)
        snapshot = trees.copy(ops, tree)
        ops.release(tree)
        context.release(provided_leaf)
        close(trees.export(ops, snapshot).value, np.full((3, 1), 7))
        ops.release(snapshot)
        assert context.metrics["live_bytes"] == before - provided_leaf.nbytes


@pytest.mark.parametrize("failure", ["callback", "allocation", "nonfinite", "shape"])
def test_growth_failure_discards_scratch_and_preserves_inputs(failure, monkeypatch):
    with ExecutionContext() as context:
        ops, data, problem, _, binned, _, _, _ = setup(context)
        raw = objective.broadcast(ops, objective.base(ops, problem), data.n_rows)
        fields = objective.fields(ops, problem, raw)
        before, records, handles = (
            context.metrics["live_bytes"],
            set(ops._records),
            set(context._buffers),
        )
        original = context.export(fields.values)

        def broken(ops, histogram):
            context.upload(np.ones(3, np.float32))
            if failure == "callback":
                raise RuntimeError("intentional callback failure")
            if failure == "allocation":
                monkeypatch.setattr(context, "_limit", context.metrics["live_bytes"])
                return context.upload(np.ones(3, np.float32))
            return context.upload(
                np.array([np.nan] if failure == "nonfinite" else [1, 2], np.float32)
            )

        with pytest.raises((RuntimeError, MemoryError, ValueError)):
            trees.depthwise(ops, data, fields, binning=binned.binning, leaf=broken)
        assert set(ops._records) == records
        assert set(context._buffers) == handles
        assert context.metrics["live_bytes"] == before
        close(context.export(fields.values), original)


def test_wrong_binning_forged_records_and_cpu_artifact(tmp_path):
    with ExecutionContext() as context:
        ops, data, problem, validation, binned, _, _, validation_cpu = setup(context)
        raw = objective.broadcast(ops, objective.base(ops, problem), data.n_rows)
        fields = objective.fields(ops, problem, raw)
        tree = trees.depthwise(ops, data, fields, binning=binned.binning)
        shifted = Binning(
            binned.binning.feature_names, tuple(c + 0.25 for c in binned.binning.cuts)
        )
        other = ops.prepare(shifted.transform(validation_cpu.data), validation_cpu)
        with pytest.raises(ValueError, match="binning identity"):
            trees.predict(ops, tree, other)
        for forged in (replace(tree), replace(tree, topology=((-1, -1, False, -1, -1),))):
            with pytest.raises(ValueError, match="forged"):
                trees.predict(ops, forged, validation.data)
        with pytest.raises(ValueError, match="forged"):
            objective.base(ops, replace(problem))
        artifact = trees.export(ops, tree)
        path = tmp_path / "tree.json"
        artifact.save(path)
        x = tmp_path / "x.npy"
        np.save(x, validation_cpu.data.values)
        result = tmp_path / "prediction.npy"
        subprocess.run(
            [
                sys.executable,
                "-c",
                "\n".join(
                    [
                        "import sys; sys.modules['cupy']=None; sys.modules['numba']=None",
                        "import numpy as np",
                        "from openboost.tree import Tree",
                        "from openboost.data import NumericData",
                        "tree = Tree.load(sys.argv[1])",
                        "x = np.load(sys.argv[2])",
                        "data = NumericData(x, np.arange(len(x)), tree.binning.feature_names)",
                        "np.save(sys.argv[3], tree.predict(data))",
                    ]
                ),
                str(path),
                str(x),
                str(result),
            ],
            check=True,
        )
        close(np.load(result), context.export(trees.predict(ops, tree, validation.data)))
        ops.release(tree)
        with pytest.raises(ValueError, match="released"):
            trees.export(ops, tree)
