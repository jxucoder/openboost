"""107 compared scalar recipe trajectories, anchors, persistence and ownership."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from openboost import ClassSchema, NumericData, Problem
from openboost import device_glm as glm
from openboost import device_recipes as recipes
from openboost import device_tree as trees
from openboost.artifacts import Model
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.device_runtime import DeviceRun
from openboost.execution import ExecutionContext
from openboost.stopping import StopState

from .reference.glm_comparison import direct_difference
from .reference.glm_recipe import SETTINGS, fit
from .test_device_comparison_consumers_cuda import owned_bytes, snapshot
from .test_device_glm_reference import original_rows, paired_fixture
from .test_device_glm_rounds_cuda import close

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("family", ["binary", "poisson"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("step,rate", SETTINGS)
def test_recipe_trajectories_and_each_actual_comparison(
    family, depth, step, rate, monkeypatch, tmp_path
):
    train, validation = paired_fixture(family)
    expected = fit(
        family, original_rows(train), original_rows(validation), depth=depth, step=step, rate=rate
    )
    binning = Binning(("x",), (np.arange(7, dtype=float),))
    audit, learned = [], []
    compare, grow = glm.compare, trees.depthwise

    def audited(ops, problem, before, after, *, family):
        result = compare(ops, problem, before, after, family=family)
        p = train if problem.data.problem_identity == train.identity else validation
        old, new = ops.execution.export(before)[:, 0], ops.execution.export(after)[:, 0]
        arrays = (
            old,
            new,
            p.target[:, 0],
            p.offset[:, 0],
            p.weight,
            p.structure.get("exposure", np.ones((len(old), 1)))[:, 0],
        )
        exact = direct_difference(family, *arrays)
        assert result.lower is not None and Decimal(result.lower) <= exact <= Decimal(result.upper)
        audit.append(
            dict(inputs=[a.tolist() for a in arrays], result=asdict(result), decimal220=str(exact))
        )
        return result

    def observed_grower(ops, data, fields, **configuration):
        tree = grow(ops, data, fields, **configuration)
        learned.append(
            (
                ops.execution.export(fields.values),
                tree.topology,
                trees.export(ops, tree).value[:, 0],
            )
        )
        return tree

    monkeypatch.setattr(glm, "compare", audited)
    monkeypatch.setattr(trees, "depthwise", observed_grower)
    with ExecutionContext() as context:
        result = getattr(recipes, family)(
            DeviceOperations(context),
            train,
            validation,
            run_id="107-" + family,
            seed=7,
            rounds=3,
            patience=2,
            min_delta=0.01,
            binning=binning,
            max_depth=depth,
            learning_rate=rate,
            step=step,
        )
        run, state = result.run, result.state
        assert run.comparison == "objective"
        assert len(result.steps) == len(expected["history"]) == len(learned)
        for actual, ref, (fields, topology, values) in zip(
            result.steps, expected["history"], learned, strict=True
        ):
            close(fields, ref["fields"])
            assert [None if f == -1 else (f, t, m) for f, t, m, _, _ in topology] == [
                n["key"] for n in ref["nodes"]
            ]
            close(values, [n["value"] for n in ref["nodes"]])
            assert [(t.coefficient, t.accepted, t.failure is not None) for t in actual.trials] == [
                (t["coefficient"], t["accepted"], t["failure"]) for t in ref["trials"]
            ]
            assert actual.after_version == ref["terms"]
            assert actual.loss == pytest.approx(ref["loss"], rel=1e-3, abs=1e-3)
            assert actual.validation_score == pytest.approx(ref["score"], rel=1e-3, abs=1e-3)
            assert actual.best_score == pytest.approx(ref["best_score"], rel=1e-3, abs=1e-3)
            assert all(t.comparison is not None for t in actual.trials if t.failure is None)
        assert state.n_terms == expected["terms"] and state.best_n_terms == expected["best_terms"]
        assert (
            result.stop.reason == expected["reason"]
            and result.stop.stale_rounds == expected["stale"]
        )
        close(snapshot(run, state)[:, 0], expected["raw"])
        close(snapshot(run, state, validation=True)[:, 0], expected["val"])
        close(snapshot(run, state, validation=True, best=True)[:, 0], expected["best"])
        model, best = run.export(state), run.export(state, best=True)
        assert model.classes == best.classes == train.classes
        assert context.metrics["live_bytes"] == owned_bytes(run, state)
        assert len(run._states) == 1 and not run._proposals
        run.close()
        assert context.metrics["live_bytes"] == 0 and not run.ops._records
    path, best_path = tmp_path / "model.json", tmp_path / "best.json"
    model.save(path)
    best.save(best_path)
    close(Model.load(path).predict(validation.data)[:, 0], expected["val"])
    close(Model.load(best_path).predict(validation.data)[:, 0], expected["best"])
    script = """
import sys
for name in ('cupy', 'numba', 'openboost.device_recipes', 'openboost.device_glm'):
    sys.modules[name] = None
import numpy as np
from openboost import NumericData
from openboost.artifacts import Model
m = Model.load(sys.argv[1])
x = np.load(sys.argv[2])
np.save(sys.argv[3], m.predict(NumericData(x, np.arange(len(x)), m.feature_names)))
"""
    inputs, output = tmp_path / "x.npy", tmp_path / "prediction.npy"
    np.save(inputs, validation.data.values)
    subprocess.run([sys.executable, "-c", script, str(path), str(inputs), str(output)], check=True)
    close(np.load(output)[:, 0], expected["val"])
    folder = Path(os.environ.get("OPENBOOST_GLM_RECIPE_ARTIFACTS", tmp_path))
    folder.mkdir(parents=True, exist_ok=True)
    identifier = f"{family}/{depth}/{step}/{rate}"
    report = dict(
        case=identifier,
        model=model.record(),
        best=best.record(),
        steps=[asdict(s) for s in result.steps],
        stop=asdict(result.stop),
        comparisons=audit,
    )
    filename = hashlib.sha256(identifier.encode()).hexdigest()[:16] + ".json"
    (folder / filename).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def controlled(monkeypatch, family):
    data = NumericData([[0], [0]], [0, 1], ("x",))
    p = Problem(
        data,
        [[0], [1]],
        data.row_ids,
        classes=ClassSchema(("no", "yes")) if family == "binary" else None,
        structure={"exposure": [[1], [1]]} if family == "poisson" else None,
    )
    binning = Binning(("x",), (np.array([]),))
    factory = getattr(glm, family)
    monkeypatch.setattr(
        glm,
        family,
        lambda **kwargs: replace(
            factory(**kwargs),
            base=lambda ops, problem: ops.execution.upload(np.array([2], np.float32)),
        ),
    )
    means, previous, leaves = [3, 1.95, 1.97, 1.9, 1.89], np.float32(2), []
    for mean in means:
        delta = np.float32(mean) - previous
        leaves.append(delta)
        previous = np.float32(previous + delta)
    values = iter(leaves)

    def learner(ops, data, fields):
        leaf = ops.execution.upload(np.array([next(values)], np.float32))
        return trees.depthwise(
            ops, data, fields, binning=binning, max_depth=0, leaf=lambda *_: leaf
        )

    return p, binning, learner


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_current_best_and_patience_are_distinct_anchors(family, monkeypatch):
    p, binning, learner = controlled(monkeypatch, family)
    observed, observe = [], StopState.observe_change

    def tracked(policy, score, change):
        updated = observe(policy, score, change)
        observed.append(updated.stale_rounds)
        return updated

    monkeypatch.setattr(StopState, "observe_change", tracked)
    with ExecutionContext() as context:
        result = getattr(recipes, family)(
            DeviceOperations(context),
            p,
            p,
            run_id="anchors",
            seed=7,
            binning=binning,
            learner=learner,
            rounds=5,
            step="fixed",
            learning_rate=1,
            patience=5,
            min_delta=0.04 if family == "binary" else 0.7,
        )
        assert observed == [1, 2, 3, 4, 0]
        assert result.state.n_terms == result.state.best_n_terms == 5
        scores = [s.best_score for s in result.steps]
        assert scores[1] == scores[2] and scores[0] > scores[1] > scores[3] > scores[4]
        close(snapshot(result.run, result.state, validation=True, best=True), [[1.89], [1.89]])
        assert context.metrics["live_bytes"] == owned_bytes(result.run, result.state)
        result.run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_zero_rounds_release_patience_anchor(family):
    train, val = paired_fixture(family)
    with ExecutionContext() as context:
        result = getattr(recipes, family)(
            DeviceOperations(context), train, val, run_id="zero", seed=7, rounds=0
        )
        assert result.steps == () and result.state.n_terms == 0 and result.stop.reason == "budget"
        assert context.metrics["live_bytes"] == owned_bytes(result.run, result.state)
        result.run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_equal_reporting_scores_still_replace_patience_anchor(family, monkeypatch):
    p, binning, _ = controlled(monkeypatch, family)
    p = replace(p, weight=[1, 3])
    factory = getattr(glm, family)
    monkeypatch.setattr(
        glm,
        family,
        lambda **kwargs: replace(
            factory(**kwargs),
            base=lambda ops, problem: ops.execution.upload(np.array([0], np.float32)),
        ),
    )
    delta = np.float32(1e-20 if family == "binary" else -1e-20)

    def learner(ops, data, fields):
        leaf = ops.execution.upload(np.array([delta], np.float32))
        return trees.depthwise(
            ops, data, fields, binning=binning, max_depth=0, leaf=lambda *_: leaf
        )

    with ExecutionContext() as context:
        result = getattr(recipes, family)(
            DeviceOperations(context),
            p,
            p,
            run_id="tiny",
            seed=7,
            rounds=2,
            patience=1,
            learning_rate=1,
            learner=learner,
            binning=binning,
        )
        assert result.stop.reason == "budget" and result.stop.stale_rounds == 0
        assert result.state.n_terms == result.state.best_n_terms == 2
        assert result.steps[0].validation_score == result.steps[1].validation_score
        second = result.steps[1].validation_change
        exact = direct_difference(
            family, [delta] * 2, [np.float32(delta + delta)] * 2, [0, 1], [0, 0], [1, 3], [1, 1]
        )
        assert second.improves() and Decimal(second.lower) <= exact <= Decimal(second.upper)
        wrong_anchor = direct_difference(
            family, [0] * 2, [np.float32(delta + delta)] * 2, [0, 1], [0, 0], [1, 3], [1, 1]
        )
        assert not Decimal(second.lower) <= wrong_anchor <= Decimal(second.upper)
        assert context.metrics["live_bytes"] == owned_bytes(result.run, result.state)
        result.run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("family", ["binary", "poisson"])
@pytest.mark.parametrize(
    "failure", ["initial_anchor", "observation_copy", "comparison", "result", "learner"]
)
def test_failed_recipe_releases_every_owned_object(family, failure, monkeypatch):
    train, val = paired_fixture(family)
    binning = Binning(("x",), (np.arange(7, dtype=float),))
    compare, calls, learned = glm.compare, 0, 0

    def injected(ops, problem, old, new, *, family):
        nonlocal calls
        calls += 1
        if calls == 3 and failure in ("comparison", "result"):
            if failure == "result":
                return -1
            ops.execution.upload(np.ones(3, np.float32))
            raise RuntimeError("injected patience comparison")
        return compare(ops, problem, old, new, family=family)

    def learner(ops, data, fields):
        nonlocal learned
        learned += 1
        if failure == "learner" and learned == 2:
            ops.execution.upload(np.ones(3, np.float32))
            raise RuntimeError("injected later learner")
        return trees.depthwise(ops, data, fields, binning=binning, max_depth=1)

    monkeypatch.setattr(glm, "compare", injected)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        if failure in ("initial_anchor", "observation_copy"):
            raw, count = DeviceRun.raw, 0

            def fail_copy(run, record, *, validation=False, best=False):
                nonlocal count
                if validation and not best:
                    count += 1
                    if count == (1 if failure == "initial_anchor" else 2):
                        raise MemoryError("injected snapshot copy")
                return raw(run, record, validation=validation, best=best)

            monkeypatch.setattr(DeviceRun, "raw", fail_copy)
        with pytest.raises((MemoryError, RuntimeError, TypeError)):
            getattr(recipes, family)(
                ops,
                train,
                val,
                run_id="failure",
                seed=7,
                rounds=3,
                binning=binning,
                learner=learner,
                step="fixed",
                learning_rate=0.25,
            )
        assert context.metrics["live_bytes"] == 0 and not ops._records
