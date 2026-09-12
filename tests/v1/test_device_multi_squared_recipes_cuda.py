"""118 default and programmable vector recipe consumers; real CUDA only."""

import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from fractions import Fraction

import numpy as np
import pytest

from openboost import device_multi_squared as objective
from openboost import device_recipes as recipes
from openboost import device_tree as trees
from openboost.binning import Binning
from openboost.data import NumericData, Problem
from openboost.device import DeviceOperations
from openboost.device_runtime import DeviceTerm
from openboost.execution import ExecutionContext
from openboost.multioutput import MultiOutputModel, TargetScale
from openboost.recipes import multi_squared as cpu_recipe
from openboost.runtime import RunContext

from .multi_squared_artifacts import directory, fresh_command, input_snapshot
from .reference import multi_squared as ref
from .test_multi_squared_reference import prepared

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("mode", ["independent", "shared", "projected"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("step,rate", [("fixed", 0.5), ("backtracking", 8)])
def test_two_round_recipe_and_every_actual_comparison(
    width, mode, depth, step, rate, monkeypatch, tmp_path
):
    train, validation, binning = prepared(width)
    _, expected = ref.rounds(width, mode, depth=depth, step=step, rate=rate)
    audit, learned = [], []
    compare, grower = objective.compare, trees.depthwise

    def audited(ops, problem, before, after):
        result = compare(ops, problem, before, after)
        host = train if problem.data.problem_identity == train.identity else validation
        arrays = (
            ops.execution.export(before),
            ops.execution.export(after),
            host.target.astype(np.float32),
            host.offset.astype(np.float32),
            host.weight.astype(np.float32),
        )
        exact = ref.change(*arrays)
        assert Fraction(result.lower) <= exact <= Fraction(result.upper)
        audit.append(
            dict(arrays=[a.tolist() for a in arrays], result=asdict(result), exact=str(exact))
        )
        return result

    def observed(*args, **kwargs):
        tree = grower(*args, **kwargs)
        learned.append((tree.topology, trees.export(args[0], tree).value))
        return tree

    monkeypatch.setattr(objective, "compare", audited)
    monkeypatch.setattr(trees, "depthwise", observed)
    with ExecutionContext() as context:
        result = recipes.multi_squared(
            DeviceOperations(context),
            train,
            validation,
            run_id="118-recipe",
            seed=17,
            binning=binning,
            rounds=2,
            mode="shared" if mode == "projected" else mode,
            projection=np.eye(width)[:, :1] if mode == "projected" else None,
            max_depth=depth,
            step=step,
            learning_rate=rate,
        )
        run, state = result.run, result.state
        assert len(result.steps) == 2
        per_round = width if mode == "independent" else 1
        for i, (actual, known) in enumerate(zip(result.steps, expected, strict=True)):
            assert [(t.coefficient, t.accepted) for t in actual.trials] == [
                (c, a) for c, a, _ in known["trials"]
            ]
            assert actual.after_version == i + 1 and all(t.failure is None for t in actual.trials)
            for (topology, values), (nodes, _) in zip(
                learned[i * per_round : (i + 1) * per_round], known["terms"], strict=True
            ):
                assert [None if f == -1 else (f, t, m) for f, t, m, _, _ in topology] == [
                    n["key"] for n in nodes
                ]
                np.testing.assert_allclose(
                    values, [[float(v) for v in n["value"]] for n in nodes], rtol=1e-4, atol=1e-5
                )
        assert state.n_terms == 2 * per_round and state.best_n_terms == expected[-1]["best_prefix"]
        assert len(run._states) == 1 and not run._proposals
        for configuration, known in (
            ({}, expected[-1]["raw"]),
            ({"validation": True}, expected[-1]["validation"]),
            ({"validation": True, "best": True}, expected[-1]["best"]),
        ):
            raw = run.raw(state, **configuration)
            np.testing.assert_allclose(context.export(raw), known, rtol=1e-4, atol=1e-5)
            context.release(raw)
        model, best = run.export(state), run.export(state, best=True)
        run.close()
        assert not run.ops._records and context.metrics["live_bytes"] == 0
    for name, artifact, known in (
        ("final", model, expected[-1]["validation"]),
        ("best", best, expected[-1]["best"]),
    ):
        path, inputs, output = (
            tmp_path / (name + ".json"),
            tmp_path / "x.npy",
            tmp_path / (name + ".npy"),
        )
        artifact.save(path)
        np.save(inputs, validation.data.values)
        subprocess.run(
            fresh_command(
                "\n".join(
                    [
                        "import sys, numpy as np; sys.modules['cupy']=None; sys.modules['numba']=None",
                        "from openboost.artifacts import Model",
                        "from openboost.data import NumericData",
                        "x=np.load(sys.argv[2]); data=NumericData(x,np.arange(len(x)),('a','b'))",
                        "np.save(sys.argv[3],Model.load(sys.argv[1]).predict(data))",
                    ]
                ),
                str(path),
                str(inputs),
                str(output),
            ),
            check=True,
        )
        np.testing.assert_array_equal(np.load(output), artifact.predict(validation.data))
        np.testing.assert_allclose(np.load(output), known, rtol=1e-4, atol=1e-5)
    identifier = f"{width}/{mode}/{depth}/{step}/{rate}"
    report = dict(
        case=identifier,
        inputs=dict(train=input_snapshot(train), validation=input_snapshot(validation)),
        model=model.record(),
        best=best.record(),
        comparisons=audit,
        steps=[asdict(s) for s in result.steps],
        stop=asdict(result.stop),
    )
    (
        directory("recipes", tmp_path)
        / (hashlib.sha256(identifier.encode()).hexdigest()[:16] + ".json")
    ).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_scaling_constant_target_and_original_unit_fresh_inference(mode, tmp_path):
    train, validation, binning = prepared(2)
    train = replace(train, target=np.column_stack((train.target[:, 0], np.full(8, 7.0))))
    scale = TargetScale.fit(train)
    assert scale.constant == (False, True)
    fitted, valid = scale.transform(train), scale.transform(validation)
    cpu = cpu_recipe(
        fitted,
        valid,
        context=RunContext("cpu-scale", 17),
        mode=mode,
        rounds=2,
        max_depth=1,
        learning_rate=0.5,
        bins=4,
    )
    with ExecutionContext() as context:
        result = recipes.multi_squared(
            DeviceOperations(context),
            fitted,
            valid,
            run_id="scale",
            seed=17,
            mode=mode,
            rounds=2,
            step="fixed",
            learning_rate=0.5,
            max_depth=1,
            bins=4,
        )
        model = MultiOutputModel(result.run.export(result.state), scale)
        result.run.close()
    expected = MultiOutputModel(cpu.state.model, scale).predict(
        validation.data, offset=validation.offset
    )
    np.testing.assert_allclose(
        model.predict(validation.data, offset=validation.offset), expected, rtol=1e-4, atol=1e-5
    )
    path = tmp_path / "scaled.json"
    model.save(path)
    np.testing.assert_array_equal(
        MultiOutputModel.load(path).predict(validation.data, offset=validation.offset),
        model.predict(validation.data, offset=validation.offset),
    )
    inputs, offsets, output = (
        tmp_path / "x.npy",
        tmp_path / "offset.npy",
        tmp_path / "prediction.npy",
    )
    np.save(inputs, validation.data.values)
    np.save(offsets, validation.offset)
    subprocess.run(
        fresh_command(
            "\n".join(
                [
                    "import sys, numpy as np; sys.modules['cupy']=None; sys.modules['numba']=None",
                    "from openboost.multioutput import MultiOutputModel",
                    "from openboost.data import NumericData",
                    "x=np.load(sys.argv[2]); data=NumericData(x,np.arange(len(x)),('a','b'))",
                    "m=MultiOutputModel.load(sys.argv[1]); np.save(sys.argv[4],m.predict(data,offset=np.load(sys.argv[3])))",
                ]
            ),
            str(path),
            str(inputs),
            str(offsets),
            str(output),
        ),
        check=True,
    )
    np.testing.assert_array_equal(
        np.load(output), model.predict(validation.data, offset=validation.offset)
    )
    report = dict(
        case="scaled-" + mode,
        inputs=dict(train=input_snapshot(train), validation=input_snapshot(validation)),
        model=model.record(),
        prediction=np.load(output).tolist(),
    )
    (directory("recipes", tmp_path) / ("scaled-" + mode + ".json")).write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


@pytest.mark.parametrize("failure", ["second_tree", "later_round"])
def test_custom_learner_failure_cleans_partial_geometry_and_terms(failure):
    train, validation, binning = prepared(2)
    calls = 0

    def learner(ops, data, fields, leaves, width):
        nonlocal calls
        calls += 1
        if calls == (2 if failure == "second_tree" else 3):
            raise RuntimeError("injected learner failure")
        return trees.depthwise(ops, data, fields, binning=binning, max_depth=1)

    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with pytest.raises(RuntimeError, match="injected"):
            recipes.multi_squared(
                ops,
                train,
                validation,
                run_id="failure",
                seed=17,
                binning=binning,
                mode="independent",
                learner=learner,
                step="fixed",
                rounds=2,
            )
        assert not ops._records and not context._buffers


def test_joint_driver_separate_anchors_and_noops(monkeypatch):
    data = NumericData([[0], [0]], [0, 1], ("x",))
    host = Problem(data, [[0, 0], [2, 0]], data.row_ids, raw_width=2)
    binning = Binning(("x",), ([],))
    factory = objective.objective
    monkeypatch.setattr(
        objective,
        "objective",
        lambda: replace(
            factory(), base=lambda ops, p: ops.execution.upload(np.array([2.0, 0], np.float32))
        ),
    )
    updates = iter(([-0.1, 0], [-0.1, 0], [0, 0]))

    def learner(run, raw):
        # Generic mapped recipe accepts an independently supplied vector learner.
        fields = run.ops.fields(
            run.data,
            run.execution.upload(np.ones((2, 2), np.float32)),
            names=("gradient", "curvature"),
            roles=("training", "training"),
        )
        value = run.execution.upload(np.array(next(updates), np.float32))
        tree = trees.depthwise(
            run.ops,
            run.data,
            fields,
            binning=binning,
            max_depth=0,
            output_width=2,
            leaf=lambda *a: value,
        )
        return (DeviceTerm(tree, np.eye(2)),)

    with ExecutionContext() as context:
        result = recipes.mapped(
            DeviceOperations(context),
            host,
            host,
            objective=objective.objective(),
            learner=learner,
            run_id="anchors",
            seed=17,
            binning=binning,
            rounds=3,
            step="fixed",
            learning_rate=1,
            patience=2,
            min_delta=0.1,
        )
        assert result.state.version == 3 and result.state.n_terms == 3
        assert result.state.best_n_terms == 2  # accepted no-op does not extend best
        assert not result.steps[0].validation_change.improves(0.1)
        assert result.steps[1].validation_change.improves(0.1)
        assert result.steps[2].validation_change.unchanged
        assert result.stop.stale_rounds == 1
        result.run.close()


@pytest.mark.parametrize("step", ["fixed", "backtracking"])
def test_reporting_tie_cannot_supply_acceptance_or_best(step):
    data = NumericData([[0]], [0], ("x",))
    host = Problem(data, [[1e10, 0]], data.row_ids, raw_width=2)
    binning = Binning(("x",), ([],))
    obj = replace(
        objective.objective(), base=lambda ops, p: ops.execution.upload(np.zeros(2, np.float32))
    )

    def learner(run, raw):
        fields = run.ops.fields(
            run.data,
            run.execution.upload(np.ones((1, 2), np.float32)),
            names=("gradient", "curvature"),
            roles=("training", "training"),
        )
        tree = trees.depthwise(
            run.ops,
            run.data,
            fields,
            binning=binning,
            max_depth=0,
            output_width=2,
            leaf=lambda *a: run.execution.upload(np.array([0, 1e-20], np.float32)),
        )
        return (DeviceTerm(tree, np.eye(2)),)

    with ExecutionContext() as context:
        result = recipes.mapped(
            DeviceOperations(context),
            host,
            host,
            objective=obj,
            learner=learner,
            run_id="tie",
            seed=17,
            binning=binning,
            rounds=1,
            step=step,
            learning_rate=1,
        )
        assert result.state.version == (1 if step == "fixed" else 0)
        assert result.state.best_n_terms == 0
        assert all(
            t.loss == result.state.loss and t.comparison.status == "worsening"
            for t in result.steps[0].trials
        )
        assert len(result.steps[0].trials) == (1 if step == "fixed" else 6)
        result.run.close()


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_zero_rounds_and_nondefault_tree_parameters(mode):
    train, validation, _ = prepared(2)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        zero = recipes.multi_squared(
            ops,
            train,
            validation,
            run_id="zero",
            seed=17,
            mode=mode,
            rounds=0,
            bins=4,
            learner=lambda *a: pytest.fail("zero-round learner called"),
        )
        assert not zero.steps and zero.state.n_terms == 0 and zero.stop.reason == "budget"
        zero.run.close()
        config = dict(
            mode=mode,
            rounds=2,
            max_depth=1,
            reg_lambda=2,
            min_child_h=2,
            split_penalty=0.5,
            learning_rate=0.5,
            step="fixed",
            bins=4,
        )
        cpu = cpu_recipe(train, validation, context=RunContext("cpu-config", 17), **config)
        result = recipes.multi_squared(ops, train, validation, run_id="config", seed=17, **config)
        np.testing.assert_allclose(
            result.run.export(result.state).predict(validation.data),
            cpu.state.model.predict(validation.data),
            rtol=1e-4,
            atol=1e-5,
        )
        result.run.close()
        assert not ops._records and not context._buffers


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_retained_storage_grows_only_with_trees(mode):
    train, validation, binning = prepared(2)
    retained = []
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        for count in (2, 24):
            result = recipes.multi_squared(
                ops,
                train,
                validation,
                run_id=f"retention-{count}",
                seed=17,
                binning=binning,
                mode=mode,
                rounds=count,
                step="fixed",
                max_depth=0,
            )
            assert len(result.run._states) == 1 and not result.run._proposals
            assert len(ops._records) == 4 + result.state.n_terms
            retained.append(context.metrics["live_bytes"])
            result.run.close()
            assert not ops._records and not context._buffers
        assert retained[1] - retained[0] == 22 * (2 * 24 if mode == "independent" else 28)
