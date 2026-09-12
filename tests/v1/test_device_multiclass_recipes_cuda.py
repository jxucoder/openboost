"""111 compared joint multiclass recipe trajectories, anchors, persistence and ownership."""

import hashlib
import json
from dataclasses import asdict, replace
from decimal import Decimal

import numpy as np
import pytest

from openboost import ClassSchema, NumericData, Problem
from openboost import device_multiclass as multiclass
from openboost import device_recipes as recipes
from openboost import device_tree as trees
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.device_runtime import DeviceRun
from openboost.execution import ExecutionContext
from openboost.stopping import StopState

from .multiclass_artifacts import directory, input_snapshot
from .reference.multiclass_comparison import direct_difference
from .reference.multiclass_recipe import SETTINGS, fit
from .test_device_comparison_consumers_cuda import owned_bytes, snapshot
from .test_device_multiclass_reference import fixture
from .test_multiclass_composition import close, fresh_replay, original_rows

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("width", [2, 3, 5])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("step,rate", SETTINGS)
def test_recipe_trajectories_and_each_actual_comparison(
    width, depth, step, rate, monkeypatch, tmp_path
):
    train, validation = fixture(width), fixture(width, validation=True)
    expected = fit(
        original_rows(train), original_rows(validation), depth=depth, step=step, rate=rate
    )
    binning = Binning(("x",), (np.arange(len(train.target) - 1, dtype=float),))
    audit, learned = [], []
    compare, grow = multiclass.compare, trees.depthwise

    def audited(ops, problem, before, after):
        result = compare(ops, problem, before, after)
        p = train if problem.data.problem_identity == train.identity else validation
        old, new = ops.execution.export(before), ops.execution.export(after)
        arrays = (
            old,
            new,
            p.target[:, 0],
            p.offset,
            p.weight,
        )
        exact = direct_difference(*arrays)
        assert result.lower is not None and Decimal(result.lower) <= exact <= Decimal(result.upper)
        audit.append(
            dict(inputs=[a.tolist() for a in arrays], result=asdict(result), decimal260=str(exact))
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

    monkeypatch.setattr(multiclass, "compare", audited)
    monkeypatch.setattr(trees, "depthwise", observed_grower)
    with ExecutionContext() as context:
        result = recipes.multiclass(
            DeviceOperations(context),
            train,
            validation,
            run_id="111-" + str(width),
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
        assert len(result.steps) == len(expected["history"])
        assert len(learned) == width * len(result.steps)
        for index, (actual, ref) in enumerate(zip(result.steps, expected["history"], strict=True)):
            assert actual.channels == tuple(range(width))
            for channel, (fields, topology, values) in enumerate(learned[index * width:(index + 1) * width]):
                close(fields, ref["fields"][channel])
                assert [None if f == -1 else (f, t, m) for f, t, m, _, _ in topology] == [
                    n["key"] for n in ref["nodes"][channel]
                ]
                close(values, [n["value"] for n in ref["nodes"][channel]])
            assert [(t.coefficient, t.accepted, t.failure is not None) for t in actual.trials] == [
                (t["coefficient"], t["accepted"], t["failure"]) for t in ref["trials"]
            ]
            assert actual.after_version == ref["version"]
            assert actual.loss == pytest.approx(ref["loss"], rel=1e-3, abs=1e-3)
            assert actual.validation_score == pytest.approx(ref["score"], rel=1e-3, abs=1e-3)
            assert actual.best_score == pytest.approx(ref["best_score"], rel=1e-3, abs=1e-3)
            assert all(t.comparison is not None for t in actual.trials if t.failure is None)
        assert state.n_terms == expected["terms"] and state.best_n_terms == expected["best_terms"]
        assert (
            result.stop.reason == expected["reason"]
            and result.stop.stale_rounds == expected["stale"]
        )
        close(snapshot(run, state), expected["raw"])
        close(snapshot(run, state, validation=True), expected["val"])
        close(snapshot(run, state, validation=True, best=True), expected["best"])
        model, best = run.export(state), run.export(state, best=True)
        assert model.classes == best.classes == train.classes
        assert context.metrics["live_bytes"] == owned_bytes(run, state)
        assert len(run._states) == 1 and not run._proposals
        run.close()
        assert context.metrics["live_bytes"] == 0 and not run.ops._records
    for name, artifact, reference in (("final", model, expected["val"]), ("best", best, expected["best"])):
        destination = tmp_path / name
        destination.mkdir()
        restored = fresh_replay(artifact, validation, destination)
        close(restored.predict(validation.data), reference)
    folder = directory("recipes", tmp_path)
    identifier = f"{width}/{depth}/{step}/{rate}"
    report = dict(
        case=identifier,
        inputs=dict(train=input_snapshot(train), validation=input_snapshot(validation)),
        model=model.record(),
        best=best.record(),
        steps=[asdict(s) for s in result.steps],
        stop=asdict(result.stop),
        comparisons=audit,
    )
    filename = hashlib.sha256(identifier.encode()).hexdigest()[:16] + ".json"
    (folder / filename).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def controlled(monkeypatch, *, initial=2, weights=None):
    data = NumericData([[0], [0], [0]], [0, 1, 2], ("x",))
    p = Problem(data, [[0], [1], [2]], data.row_ids, raw_width=3,
                classes=ClassSchema(("a", "b", "c")), weight=weights)
    binning = Binning(("x",), (np.array([]),))
    factory = multiclass.objective
    monkeypatch.setattr(multiclass, "objective", lambda: replace(
        factory(), base=lambda ops, problem: ops.execution.upload(
            np.array([initial, 0, 0], np.float32)
        )
    ))
    return p, binning


def sequence_learner(binning, values):
    values = iter(values)

    def learner(ops, data, fields):
        leaf = ops.execution.upload(np.array([next(values)], np.float32))
        return trees.depthwise(ops, data, fields, binning=binning, max_depth=0, leaf=lambda *_: leaf)

    return learner


def test_current_best_and_patience_are_distinct(monkeypatch):
    p, binning = controlled(monkeypatch)
    values, previous = [], np.float32(2)
    for value in (3, 1.95, 1.97, 1.9, 1.89):
        delta = np.float32(value) - previous
        values.extend([delta, 0, 0])
        previous = np.float32(previous + delta)
    observed, observe = [], StopState.observe_change

    def tracked(policy, score, change):
        updated = observe(policy, score, change)
        observed.append(updated.stale_rounds)
        return updated

    monkeypatch.setattr(StopState, "observe_change", tracked)
    with ExecutionContext() as context:
        result = recipes.multiclass(
            DeviceOperations(context), p, p, run_id="anchors", seed=7, binning=binning,
            learner=sequence_learner(binning, values), rounds=5, step="fixed", learning_rate=1,
            patience=5, min_delta=0.047,
        )
        assert observed == [1, 2, 3, 4, 0]
        assert result.state.n_terms == result.state.best_n_terms == 15
        scores = [s.best_score for s in result.steps]
        assert scores[1] == scores[2] and scores[0] > scores[1] > scores[3] > scores[4]
        close(snapshot(result.run, result.state, validation=True, best=True), [[1.89, 0, 0]] * 3)
        assert context.metrics["live_bytes"] == owned_bytes(result.run, result.state)
        result.run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("width", [2, 3, 5])
def test_zero_rounds_release_patience_anchor(width):
    with ExecutionContext() as context:
        result = recipes.multiclass(
            DeviceOperations(context), fixture(width), fixture(width, validation=True),
            run_id="zero", seed=7, rounds=0,
        )
        assert result.steps == () and result.state.n_terms == 0 and result.stop.reason == "budget"
        assert context.metrics["live_bytes"] == owned_bytes(result.run, result.state)
        result.run.close()
        assert context.metrics["live_bytes"] == 0


def test_reporting_ties_replace_both_best_and_patience(monkeypatch):
    p, binning = controlled(monkeypatch, initial=0, weights=[3, 1, 1])
    delta = np.float32(1e-20)
    with ExecutionContext() as context:
        result = recipes.multiclass(
            DeviceOperations(context), p, p, run_id="tiny", seed=7, rounds=2, patience=1,
            learning_rate=1, learner=sequence_learner(binning, [delta, 0, 0] * 2), binning=binning,
        )
        assert result.stop.reason == "budget" and result.stop.stale_rounds == 0
        assert result.state.n_terms == result.state.best_n_terms == 6
        assert result.steps[0].validation_score == result.steps[1].validation_score
        second = result.steps[1].validation_change
        before = np.tile([delta, 0, 0], (3, 1)).astype(np.float32)
        after = before * np.float32(2)
        exact = direct_difference(before, after, p.target[:, 0], p.offset, p.weight)
        assert second.improves() and Decimal(second.lower) <= exact <= Decimal(second.upper)
        wrong = direct_difference(np.zeros_like(before), after, p.target[:, 0], p.offset, p.weight)
        assert not Decimal(second.lower) <= wrong <= Decimal(second.upper)
        assert context.metrics["live_bytes"] == owned_bytes(result.run, result.state)
        result.run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("failure", [
    "initial_anchor", "observation_copy", "comparison", "result", "second_class", "later_round"
])
def test_failed_recipe_releases_every_owned_object(failure, monkeypatch):
    train, val = fixture(), fixture(validation=True)
    binning = Binning(("x",), (np.arange(8, dtype=float),))
    compare, calls, learned = multiclass.compare, 0, 0

    def injected(ops, problem, old, new):
        nonlocal calls
        calls += 1
        if calls == 3 and failure in ("comparison", "result"):
            if failure == "result":
                return -1
            ops.execution.upload(np.ones(3, np.float32))
            raise RuntimeError("injected patience comparison")
        return compare(ops, problem, old, new)

    def learner(ops, data, fields):
        nonlocal learned
        learned += 1
        if (failure == "second_class" and learned == 2) or (
            failure == "later_round" and learned == 4
        ):
            ops.execution.upload(np.ones(3, np.float32))
            raise RuntimeError("injected partial joint learner")
        return trees.depthwise(ops, data, fields, binning=binning, max_depth=1)

    monkeypatch.setattr(multiclass, "compare", injected)
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
            recipes.multiclass(
                ops, train, val, run_id="failure", seed=7, rounds=3, binning=binning,
                learner=learner, step="fixed", learning_rate=0.25,
            )
        assert context.metrics["live_bytes"] == 0 and not ops._records


def test_nondefault_tree_parameters_reach_every_class(monkeypatch):
    seen, grow = [], trees.depthwise

    def observed(ops, data, fields, **configuration):
        seen.append(configuration)
        return grow(ops, data, fields, **configuration)

    monkeypatch.setattr(trees, "depthwise", observed)
    with ExecutionContext() as context:
        result = recipes.multiclass(
            DeviceOperations(context), fixture(), fixture(validation=True), run_id="config", seed=7,
            rounds=1, max_depth=0, reg_lambda=2, min_child_h=0.5, split_penalty=0.125, bins=7,
        )
        assert len(seen) == 3
        assert all((s["max_depth"], s["reg_lambda"], s["min_child_h"], s["split_penalty"])
                   == (0, 2, 0.5, 0.125) for s in seen)
        from .reference.device_multiclass import geometry
        p = fixture()
        _, g, h, _, _ = geometry(np.zeros((9, 3)), p.target[:, 0], p.offset, p.weight)
        model = result.run.export(result.state)
        for channel, term in enumerate(model.terms):
            assert len(term.learner.value) == 1
            expected = -sum(g[:, channel] * p.weight) / (sum(h[:, channel] * p.weight) + 2)
            close(term.learner.value[0, 0], expected)
        assert result.state.n_terms == 3
        result.run.close()
        assert context.metrics["live_bytes"] == 0
