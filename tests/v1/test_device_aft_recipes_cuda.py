"""AFT compared recipes: independent trajectories, three anchors and CPU replay."""

import hashlib
import json
from dataclasses import asdict, replace
from decimal import Decimal

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost import device_aft as aft
from openboost import device_recipes as recipes
from openboost import device_tree as trees
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.device_runtime import DeviceRun
from openboost.execution import ExecutionContext
from openboost.stopping import StopState

from .aft_artifacts import directory, input_snapshot
from .reference.aft_comparison import direct_difference
from .reference.aft_recipe import SETTINGS, fit
from .test_aft_composition import close, fresh_replay
from .test_device_aft_reference import fixture, original_rows
from .test_device_comparison_consumers_cuda import owned_bytes, snapshot

pytestmark = pytest.mark.gpu


def trajectory(sigma, depth, step, rate, monkeypatch, tmp_path, *, all_censored=False):
    train = fixture(all_censored=all_censored)
    validation = fixture(validation=True, all_censored=all_censored)
    expected = fit(original_rows(train), original_rows(validation), sigma=sigma, depth=depth, step=step, rate=rate)
    binning = Binning(("x",), (np.arange(7, dtype=float),))
    audit, learned = [], []
    compare, grow = aft.compare, trees.depthwise

    def audited(ops, problem, before, after, *, sigma=None):
        result = compare(ops, problem, before, after, sigma=sigma)
        p = train if problem.data.problem_identity == train.identity else validation
        old, new = ops.execution.export(before)[:, 0], ops.execution.export(after)[:, 0]
        arrays = [old.tolist(), new.tolist(), p.target[:, 0].tolist(),
                  (p.target[:, 0] == p.target[:, 1]).tolist(), p.offset[:, 0].tolist(), p.weight.tolist(), problem.sigma]
        exact = direct_difference(*arrays)
        assert result.lower is not None and Decimal(result.lower) <= exact <= Decimal(result.upper)
        audit.append(dict(inputs=arrays, result=asdict(result), decimal220=str(exact)))
        return result

    def observed_grower(ops, data, fields, **configuration):
        tree = grow(ops, data, fields, **configuration)
        learned.append((ops.execution.export(fields.values), tree.topology, trees.export(ops, tree).value[:, 0]))
        return tree

    monkeypatch.setattr(aft, "compare", audited)
    monkeypatch.setattr(trees, "depthwise", observed_grower)
    with ExecutionContext() as context:
        result = recipes.aft(DeviceOperations(context), train, validation, sigma=sigma,
                             run_id="115-aft", seed=7, rounds=3, patience=2, min_delta=.01,
                             binning=binning, max_depth=depth, learning_rate=rate, step=step)
        run, state = result.run, result.state
        assert run.comparison == "objective"
        assert len(result.steps) == len(expected["history"]) == len(learned)
        for actual, ref, (fields, topology, values) in zip(result.steps, expected["history"], learned, strict=True):
            close(fields, ref["fields"])
            assert [None if f == -1 else (f, t, m) for f, t, m, _, _ in topology] == [n["key"] for n in ref["nodes"]]
            close(values, [n["value"] for n in ref["nodes"]])
            assert [(t.coefficient, t.accepted, t.failure is not None) for t in actual.trials] == [
                (t["coefficient"], t["accepted"], t["failure"]) for t in ref["trials"]]
            assert actual.after_version == ref["terms"]
            for measured, name in ((actual.loss, "loss"), (actual.validation_score, "score"), (actual.best_score, "best_score")):
                assert measured == pytest.approx(ref[name], rel=1e-3, abs=1e-3)
            assert all(t.comparison is not None for t in actual.trials if t.failure is None)
        assert state.n_terms == expected["terms"] and state.best_n_terms == expected["best_terms"]
        assert result.stop.reason == expected["reason"] and result.stop.stale_rounds == expected["stale"]
        close(snapshot(run, state)[:, 0], expected["raw"])
        actual_val, actual_best = snapshot(run, state, validation=True), snapshot(run, state, validation=True, best=True)
        close(actual_val[:, 0], expected["val"])
        close(actual_best[:, 0], expected["best"])
        model, best = aft.export(run, state), aft.export(run, state, best=True)
        assert model.sigma == best.sigma == sigma
        assert context.metrics["live_bytes"] == owned_bytes(run, state)
        assert len(run._states) == 1 and not run._proposals
        run.close()
        assert context.metrics["live_bytes"] == 0 and not run.ops._records
    close(model.model.predict(validation.data), actual_val)
    close(best.model.predict(validation.data), actual_best)
    for name, artifact in (("final", model), ("best", best)):
        folder = tmp_path/name
        folder.mkdir()
        fresh_replay(artifact, validation, folder)
    identifier = f"aft/{sigma}/{depth}/{step}/{rate}/{all_censored}"
    report = dict(case=identifier, sigma=sigma, inputs=dict(train=input_snapshot(train), validation=input_snapshot(validation)),
                  model=model.record(), best=best.record(), steps=[asdict(s) for s in result.steps],
                  stop=asdict(result.stop), comparisons=audit)
    filename = hashlib.sha256(identifier.encode()).hexdigest()[:16]+".json"
    (directory("recipes", tmp_path)/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


@pytest.mark.parametrize("sigma", [.5, .7, 1, 2])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("step,rate", SETTINGS)
def test_actual_recipe_trajectories_and_all_comparisons(sigma, depth, step, rate, monkeypatch, tmp_path):
    trajectory(sigma, depth, step, rate, monkeypatch, tmp_path)


@pytest.mark.parametrize("sigma", [.7, 2])
def test_all_censored_recipe_and_saved_survival(sigma, monkeypatch, tmp_path):
    trajectory(sigma, 1, "backtracking", .25, monkeypatch, tmp_path, all_censored=True)


def controlled(monkeypatch, *, event=True, initial=2, values=(3, 1.95, 1.97, 1.9, 1.89)):
    data = NumericData([[0], [0]], [0, 1], ("x",))
    p = Problem(data, [[1, 1 if event else np.inf]]*2, data.row_ids, target_kind="event_right")
    binning = Binning(("x",), (np.array([]),))
    factory = aft.objective
    monkeypatch.setattr(aft, "objective", lambda sigma=1: replace(
        factory(sigma), base=lambda ops, problem: ops.execution.upload(np.array([initial], np.float32))))
    previous, leaves = np.float32(initial), []
    for value in values:
        delta = np.float32(value)-previous
        leaves.append(delta)
        previous = np.float32(previous+delta)
    candidates = iter(leaves)

    def learner(ops, data, fields):
        leaf = ops.execution.upload(np.array([next(candidates)], np.float32))
        return trees.depthwise(ops, data, fields, binning=binning, max_depth=0, leaf=lambda *_: leaf)

    return p, binning, learner


def test_current_best_and_patience_have_distinct_anchors(monkeypatch):
    p, binning, learner = controlled(monkeypatch)
    observed, observe, sigma = [], StopState.observe_change, .7

    def tracked(policy, score, change):
        updated = observe(policy, score, change)
        observed.append(updated.stale_rounds)
        return updated

    monkeypatch.setattr(StopState, "observe_change", tracked)
    with ExecutionContext() as context:
        result = recipes.aft(DeviceOperations(context), p, p, sigma=sigma, run_id="anchors", seed=7,
                             binning=binning, learner=learner, rounds=5, step="fixed", learning_rate=1,
                             patience=5, min_delta=.21/sigma**2)
        assert observed == [1, 2, 3, 4, 0]
        assert result.state.n_terms == result.state.best_n_terms == 5
        scores = [s.best_score for s in result.steps]
        assert scores[1] == scores[2] and scores[0] > scores[1] > scores[3] > scores[4]
        close(snapshot(result.run, result.state, validation=True, best=True), [[1.89], [1.89]])
        assert context.metrics["live_bytes"] == owned_bytes(result.run, result.state)
        result.run.close()
        assert context.metrics["live_bytes"] == 0


def test_equal_reporting_scores_replace_best_and_patience(monkeypatch):
    delta = np.float32(1e-20)
    p, binning, learner = controlled(monkeypatch, event=False, initial=0, values=(delta, np.float32(delta+delta)))
    with ExecutionContext() as context:
        result = recipes.aft(DeviceOperations(context), p, p, run_id="tiny", seed=7, rounds=2, patience=1,
                             learning_rate=1, learner=learner, binning=binning)
        assert result.stop.reason == "budget" and result.stop.stale_rounds == 0
        assert result.state.n_terms == result.state.best_n_terms == 2
        assert result.steps[0].validation_score == result.steps[1].validation_score
        second = result.steps[1].validation_change
        exact = direct_difference([delta]*2, [np.float32(delta+delta)]*2, [1, 1], [False, False], [0, 0], [1, 1], 1)
        wrong = direct_difference([0]*2, [np.float32(delta+delta)]*2, [1, 1], [False, False], [0, 0], [1, 1], 1)
        assert second.improves() and Decimal(second.lower) <= exact <= Decimal(second.upper)
        assert not Decimal(second.lower) <= wrong <= Decimal(second.upper)
        assert context.metrics["live_bytes"] == owned_bytes(result.run, result.state)
        result.run.close()
        assert context.metrics["live_bytes"] == 0


def test_stationary_reporting_tie_rejects_all_backtracking_trials(monkeypatch):
    p, binning, learner = controlled(monkeypatch, initial=0, values=(1e-20,))
    with ExecutionContext() as context:
        result = recipes.aft(DeviceOperations(context), p, p, run_id="worsening", seed=7, rounds=2,
                             patience=1, learning_rate=1, learner=learner, binning=binning)
        assert result.stop.reason == "patience" and result.state.n_terms == result.state.best_n_terms == 0
        trials = result.steps[0].trials
        assert len(trials) == 6 and all(not t.accepted and t.failure is None for t in trials)
        assert all(t.loss == result.state.loss and t.comparison.status == "worsening" for t in trials)
        assert context.metrics["live_bytes"] == owned_bytes(result.run, result.state)
        result.run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("sigma", [.5, .7, 1, 2])
def test_zero_rounds_release_patience_and_retain_scale(sigma):
    train, val = fixture(), fixture(validation=True)
    with ExecutionContext() as context:
        result = recipes.aft(DeviceOperations(context), train, val, sigma=sigma, run_id="zero", seed=7, rounds=0)
        assert result.steps == () and result.state.n_terms == 0 and result.stop.reason == "budget"
        assert aft.export(result.run, result.state, best=True).sigma == sigma
        assert context.metrics["live_bytes"] == owned_bytes(result.run, result.state)
        result.run.close()
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("failure", ["initial_anchor", "observation_copy", "comparison", "result", "learner"])
def test_failed_recipe_releases_every_owned_object(failure, monkeypatch):
    train, val = fixture(), fixture(validation=True)
    binning = Binning(("x",), (np.arange(7, dtype=float),))
    compare, calls, learned = aft.compare, 0, 0

    def injected(ops, problem, old, new, *, sigma=None):
        nonlocal calls
        calls += 1
        if calls == 3 and failure in ("comparison", "result"):
            if failure == "result":
                return -1
            ops.execution.upload(np.ones(3, np.float32))
            raise RuntimeError("injected patience comparison")
        return compare(ops, problem, old, new, sigma=sigma)

    def learner(ops, data, fields):
        nonlocal learned
        learned += 1
        if failure == "learner" and learned == 2:
            ops.execution.upload(np.ones(3, np.float32))
            raise RuntimeError("injected later learner")
        return trees.depthwise(ops, data, fields, binning=binning, max_depth=1)

    monkeypatch.setattr(aft, "compare", injected)
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
            recipes.aft(ops, train, val, run_id="failure", seed=7, rounds=3,
                        binning=binning, learner=learner, step="fixed", learning_rate=.25)
        assert context.metrics["live_bytes"] == 0 and not ops._records
