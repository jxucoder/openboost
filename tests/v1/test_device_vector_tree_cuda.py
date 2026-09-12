"""117 shared vector trees, separate leaf fields, ownership and inference; real CUDA only."""

import subprocess
from dataclasses import replace

import numpy as np
import pytest

from openboost import device_tree as trees
from openboost.binning import Binning
from openboost.data import NumericData, Problem
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .multi_squared_artifacts import fresh_command
from .reference import device_vector as ref
from .test_device_vector_ops_cuda import bind
from .test_device_vector_reference import prepared

pytestmark = pytest.mark.gpu


def grow(ops, data, fields, leaves, binning, width, depth=2, **options):
    return trees.depthwise(
        ops,
        data,
        fields,
        binning=binning,
        output_width=width,
        max_depth=depth,
        leaf_fields=leaves,
        scoring=lambda o, b: o.vector_scores(b, reg_lambda=2, split_penalty=0.5),
        legality=lambda o, b: o.vector_feasible(b, min_child_h=1),
        leaf=options.pop("leaf", lambda o, h: o.vector_leaf(h, reg_lambda=2)),
        **options,
    )


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_shared_topology_projected_splits_full_leaves(width, projected, depth, tmp_path):
    source, problem, binned, split_cpu, leaf_cpu = prepared(width, projected)
    known = ref.tree(source, depth, regularization=2, penalty=0.5, minimum=1)
    validation_x = np.vstack((source["x"][::-1], [[np.nan, np.nan], [-1, 4]]))
    validation_data = NumericData(validation_x, 501 + 9 * np.arange(len(validation_x)), ("a", "b"))
    validation = Problem(validation_data, np.zeros((len(validation_x), 1)), validation_data.row_ids)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        data = ops.prepare(binned, problem)
        other = ops.prepare(binned.binning.transform(validation_data), validation)
        split = bind(ops, data, context, split_cpu, reordered=True)
        leaves = bind(ops, data, context, leaf_cpu)
        buffers, records = set(context._buffers), set(ops._records)
        before = dict(context.metrics)
        tree = grow(ops, data, split, leaves, binned.binning, width, depth)
        assert set(ops._records) - records == {tree}
        assert set(context._buffers) - buffers == set(ops._records[tree][0])
        assert context.metrics["live_bytes"] - before["live_bytes"] == tree.n_nodes * (
            20 + 4 * width
        )
        prediction = trees.predict(ops, tree, other)
        after = dict(context.metrics)
        assert after["upload_bytes"] - before["upload_bytes"] == tree.n_nodes * 20
        assert after["export_bytes"] - before["export_bytes"] == sum(
            after[k] - before[k] for k in ("validation_export_bytes", "decision_export_bytes")
        )
        artifact = trees.export(ops, tree)
        assert tree.output_width == artifact.output_width == width
        assert list(tree.topology) == [
            (*n["key"], n["left"], n["right"]) if n["key"] else (-1, -1, False, -1, -1)
            for n in known
        ]
        np.testing.assert_allclose(
            artifact.value, [[float(v) for v in n["value"]] for n in known], rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            context.export(trees.predict(ops, tree, data)),
            ref.predict(known, source["x"]),
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            context.export(prediction), ref.predict(known, validation_x), rtol=1e-4, atol=1e-5
        )
        copied = trees.copy(ops, tree)
        assert not set(ops._records[copied][0]) & set(ops._records[tree][0])
        for record in (tree, split, leaves, data):
            ops.release(record)
        for handle in (split.values, leaves.values):
            context.release(handle)
        np.testing.assert_array_equal(
            context.export(trees.predict(ops, copied, other)), artifact.predict(validation_data)
        )
        with pytest.raises(ValueError, match="forged"):
            trees.predict(ops, replace(copied, output_width=width + 1), other)
        path, inputs, output = tmp_path / "tree.json", tmp_path / "x.npy", tmp_path / "p.npy"
        artifact.save(path)
        np.save(inputs, validation_x)
        subprocess.run(
            fresh_command(
                "\n".join(
                    [
                        "import sys, numpy as np; sys.modules['cupy']=None; sys.modules['numba']=None",
                        "from openboost.data import NumericData",
                        "from openboost.tree import Tree",
                        "x=np.load(sys.argv[2]); data=NumericData(x, np.arange(len(x)), ('a','b'))",
                        "np.save(sys.argv[3], Tree.load(sys.argv[1]).predict(data))",
                    ]
                ),
                str(path),
                str(inputs),
                str(output),
            ),
            check=True,
        )
        np.testing.assert_array_equal(np.load(output), artifact.predict(validation_data))


@pytest.mark.parametrize("failure", ["shape", "nonfinite", "callback", "pack", "copy"])
def test_vector_tree_failure_rolls_back_all_partial_work(failure, monkeypatch):
    _, problem, binned, fields_cpu, leaves_cpu = prepared(2, True)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        data = ops.prepare(binned, problem)
        fields, leaves = (bind(ops, data, context, f) for f in (fields_cpu, leaves_cpu))
        called = 0

        def leaf(ops, hist):
            nonlocal called
            called += 1
            if called == 2:
                if failure == "callback":
                    raise RuntimeError("injected vector leaf failure")
                if failure in ("shape", "nonfinite"):
                    return context.upload(
                        np.array([1] if failure == "shape" else [1, np.nan], np.float32)
                    )
            return ops.vector_leaf(hist, reg_lambda=2)

        original_launch, original_copy = ops._launch, context.copy

        def launch(name, *args):
            if failure == "pack" and name == "pack_vector_leaf":
                raise RuntimeError("injected pack failure")
            return original_launch(name, *args)

        def copy(handle):
            if failure == "copy" and called == 2:
                raise MemoryError("injected vector allocation failure")
            return original_copy(handle)

        monkeypatch.setattr(ops, "_launch", launch)
        monkeypatch.setattr(context, "copy", copy)
        buffers, records = set(context._buffers), set(ops._records)
        with pytest.raises((ValueError, RuntimeError, MemoryError)):
            grow(ops, data, fields, leaves, binned.binning, 2, leaf=leaf)
        assert set(context._buffers) == buffers and set(ops._records) == records
        np.testing.assert_array_equal(context.export(leaves.values), leaves_cpu.values)


def test_vector_callback_borrowing_nonempty_mask_and_foreign_fields():
    _, problem, binned, fields_cpu, leaves_cpu = prepared(2, True)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        data = ops.prepare(binned, problem)
        fields, leaves = (bind(ops, data, context, f) for f in (fields_cpu, leaves_cpu))
        foreign = ops.prepare(binned, problem)  # same bytes, distinct owned data record
        foreign_fields = bind(ops, foreign, context, leaves_cpu)
        with pytest.raises(ValueError, match="different prepared data"):
            grow(ops, data, fields, foreign_fields, binned.binning, 2)
        provided = context.upload(np.array([7, -3], np.float32))

        def score(ops, batch):
            scores = np.ones(batch.size, np.float32)
            # Missing-left at the final first-feature threshold has an empty right child.
            scores[[batch.key(i) == (0, 2, True) for i in range(batch.size)]] = 100
            return ops.scores(batch, context.upload(scores))

        tree = trees.depthwise(
            ops,
            data,
            fields,
            binning=binned.binning,
            output_width=2,
            max_depth=1,
            leaf_fields=leaves,
            leaf=lambda o, h: provided,
            scoring=score,
            legality=lambda o, b: o.mask(b, context.upload(np.ones(b.size, bool))),
        )
        assert tree.n_nodes == 3 and tree.topology[0][:3] != (0, 2, True)
        context.release(provided)
        np.testing.assert_array_equal(
            context.export(trees.predict(ops, tree, data)), np.tile([7, -3], (data.n_rows, 1))
        )
        wrong_binning = Binning(("a", "b"), ([0.1, 1.5], [0.5]))
        other = ops.prepare(wrong_binning.transform(problem.data), problem)
        with pytest.raises(ValueError, match="binning identity"):
            trees.predict(ops, tree, other)
