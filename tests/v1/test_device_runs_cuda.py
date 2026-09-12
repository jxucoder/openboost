"""120 sequential reference: real-device M=1/8/32, isolation and detached inference."""

import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from openboost import device_recipes as recipes
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.device_runs import RunResult, run_many
from openboost.execution import ExecutionContext
from openboost.runtime import RunContext
from openboost.stopping import StopState

from .glm_artifacts import input_snapshot
from .multi_squared_artifacts import fresh_command
from .reference.device_runs import run as reference
from .test_device_runs_reference import jobs

pytestmark = pytest.mark.gpu


def artifact_directory(temporary):
    explicit = os.environ.get("OPENBOOST_DEVICE_RUN_ARTIFACTS")
    retained = os.environ.get("OPENBOOST_NORMAL_ARTIFACTS")
    folder = Path(explicit) if explicit else Path(retained) / "sequential-runs" if retained else Path(temporary)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def record(result):
    return dict(model=result.model.record(), best_model=result.best_model.record(),
                state=asdict(result.state), steps=[asdict(s) for s in result.steps],
                stop=asdict(result.stop), reason=result.stop.reason)


def direct(ops, job):
    fit = job.recipe(ops, job.train, job.validation, run_id=job.run_id, seed=job.seed,
                     binning=job.binning, **job.options)
    try:
        result = RunResult(fit.run.export(fit.state), fit.run.export(fit.state, best=True),
                           fit.state, fit.steps, fit.stop)
        for data, validation in ((job.train.data, False), (job.validation.data, True)):
            raw = fit.run.raw(fit.state, validation=validation)
            np.testing.assert_allclose(result.model.predict(data), ops.execution.export(raw),
                                       rtol=1e-4, atol=1e-5)
            ops.execution.release(raw)
        return result
    finally:
        fit.run.close()


def check_reference(job, result):
    expected = reference(int(job.run_id.split("-")[1]))
    for model, data, key in (
        (result.model, job.train.data, "raw"),
        (result.model, job.validation.data, "validation_raw"),
        (result.best_model, job.train.data, "best_raw"),
        (result.best_model, job.validation.data, "best_validation_raw"),
    ):
        np.testing.assert_allclose(model.predict(data)[:, 0], expected[key], rtol=1e-4, atol=1e-5)
    assert result.state.version == result.state.n_terms == expected["version"]
    assert result.state.best_n_terms == expected["best_prefix"]
    assert result.stop.completed_rounds == expected["completed"]
    assert result.stop.stale_rounds == expected["stale"]
    assert result.stop.reason == expected["reason"]
    assert [s.accepted for s in result.steps] == [s["accepted"] for s in expected["steps"]]
    for actual, wanted in zip(result.steps, expected["steps"], strict=True):
        for a, b in ((actual.loss, wanted["loss"]), (actual.validation_score, wanted["score"])):
            assert abs(a - b) <= 1e-3 * max(1, abs(b))
    # Compare every accepted tree's non-tied topology and leaf solve, not just
    # final quality. The shared 088 oracle enumerates actual original rows.
    accepted = [s for s in expected["steps"] if s["accepted"]]
    for term, step in zip(result.model.terms, accepted, strict=True):
        tree = term.learner
        actual = [None if f == -1 else (int(f), int(t), bool(m))
                  for f, t, m in zip(tree.feature, tree.threshold, tree.missing_left, strict=True)]
        assert actual == [n["key"] for n in step["nodes"]]
        np.testing.assert_allclose(tree.value[:, 0], [n["value"] for n in step["nodes"]],
                                   rtol=1e-4, atol=1e-5)


def fresh_inference(records, path):
    script = """
import base64, json, sys
from pathlib import Path
import numpy as np
sys.modules['openboost.device_runs'] = None
from openboost import NumericData
from openboost.artifacts import Model
path = Path(sys.argv[1])
rows = json.loads(path.read_text())['outcomes']
def unpack(row):
    return np.frombuffer(base64.b64decode(row['data_base64'], validate=True),
                         dtype=row['dtype']).reshape(row['shape'])
for row in rows:
    for key in ('model', 'best_model'):
        artifact = path.parent / ('fresh-' + row['run_id'] + '-' + key + '.json')
        artifact.write_text(json.dumps(row['result'][key]))
        model = Model.load(artifact)
        for split in ('train', 'validation'):
            inputs = row['inputs'][split]
            data = NumericData(unpack(inputs['features']), unpack(inputs['row_ids']),
                               tuple(model.feature_names))
            actual = model.predict(data, offset=unpack(inputs['offset']))
            np.testing.assert_array_equal(actual, row['predictions'][key][split])
        artifact.unlink()
assert not any(name in sys.modules and sys.modules[name] is not None
               for name in ('cupy', 'numba', 'openboost.device_runs', 'openboost.device_runtime'))
print(len(rows) * 2)
"""
    path.write_text(json.dumps(records, allow_nan=False, indent=2) + "\n")
    completed = subprocess.run(fresh_command(script, str(path)), check=True,
                               text=True, capture_output=True, timeout=30)
    assert int(completed.stdout.strip()) == 2 * len(records["outcomes"])


@pytest.mark.parametrize("count", [1, 8, 32])
def test_independent_sequential_reversed_regrouped_and_fresh_models(count, monkeypatch, tmp_path):
    specs = jobs(count)
    streams = {}

    def observed(ops, train, validation, **configuration):
        fit = recipes.squared(ops, train, validation, **configuration)
        key = fit.run.run_id
        value = tuple(int(v) for v in fit.run.rng(0, "learner", "rows").integers(0, 2**32, 8))
        assert value == tuple(int(v) for v in RunContext(key, fit.run.seed).rng(0, "learner", "rows").integers(0, 2**32, 8))
        assert streams.setdefault(key, value) == value
        return fit

    specs = tuple(replace(s, recipe=observed) for s in specs)
    with ExecutionContext(max_bytes=8 * 1024**2) as context:
        ops = DeviceOperations(context)
        expected = {s.run_id: direct(ops, s) for s in specs}
        for s in specs:
            check_reference(s, expected[s.run_id])
        assert context.metrics["live_bytes"] == 0
        caller = context.upload(np.array([3, 5], np.float32))
        baseline = context.metrics["live_bytes"]

        def forbidden(*args, **kwargs):
            raise AssertionError("supplied fitted cuts cannot trigger Binning.fit")

        monkeypatch.setattr(Binning, "fit", forbidden)
        completed = {}
        for schedule in (specs, specs[::-1], specs[::2], specs[1::2]):
            outcomes = run_many(ops, schedule)
            assert tuple(o.run_id for o in outcomes) == tuple(s.run_id for s in schedule)
            for outcome in outcomes:
                assert outcome.error_type is None, outcome.error_message
                assert record(outcome.result) == record(expected[outcome.run_id])
                completed[outcome.run_id] = outcome.result
            assert context.metrics["live_bytes"] == baseline
            np.testing.assert_array_equal(context.export(caller), [3, 5])
        assert len(set(streams.values())) == count
        context.release(caller)
        assert context.metrics["live_bytes"] == 0
        metrics = dict(context.metrics)
    retained = []
    for s in specs:
        result = completed[s.run_id]
        retained.append(dict(run_id=s.run_id, seed=s.seed, options=dict(s.options),
                             binning_identity=s.binning.identity, result=record(result),
                             rng=list(streams[s.run_id]),
                             inputs=dict(train=input_snapshot(s.train), validation=input_snapshot(s.validation)),
                             predictions={key: {split: model.predict(p.data, offset=p.offset).tolist()
                                                for split, p in (("train", s.train), ("validation", s.validation))}
                                          for key, model in (("model", result.model), ("best_model", result.best_model))}))
    folder = artifact_directory(tmp_path)
    fresh_inference(dict(execution="sequential", count=count, context_metrics=metrics, outcomes=retained),
                     folder / f"sequential-M{count}.json")


@pytest.mark.parametrize("count", [1, 8, 32])
def test_one_later_failure_retained_and_same_id_retry_is_independent(count, tmp_path):
    specs = jobs(count)
    bad = count // 2
    original = specs[bad]

    def failure(ops, train, validation, **configuration):
        # Complete one real round, then fail after allocating additional caller
        # scratch. Neither a returned fit nor recipe cleanup is available.
        options = {**configuration, "rounds": 1, "patience": None, "min_delta": 0.}
        recipes.squared(ops, train, validation, **options)
        ops.execution.upload(np.ones(7, np.float32))
        raise ValueError("forced later failure")

    with ExecutionContext(max_bytes=8 * 1024**2) as context:
        ops = DeviceOperations(context)
        expected = {s.run_id: record(direct(ops, s)) for s in specs}
        changed = tuple(replace(s, recipe=failure) if i == bad else s for i, s in enumerate(specs))
        actual = run_many(ops, changed)
        assert len(actual) == count
        for i, outcome in enumerate(actual):
            if i == bad:
                assert outcome.result is None and outcome.error_type == "ValueError"
                assert outcome.error_message == "forced later failure"
            else:
                assert outcome.error_type is None, outcome.error_message
                assert record(outcome.result) == expected[outcome.run_id]
        assert context.metrics["live_bytes"] == 0 and not ops._records
        retry = run_many(ops, (original,))[0]
        assert retry.error_type is None and record(retry.result) == expected[original.run_id]
        assert context.metrics["live_bytes"] == 0 and not ops._records
        retained = dict(execution="sequential", count=count, failed_id=original.run_id,
                        expected_fault="one completed round, then scratch allocation and ValueError",
                        inputs_from=f"sequential-M{count}.json", context_metrics=dict(context.metrics),
                        outcomes=[dict(run_id=o.run_id, result=None if o.result is None else record(o.result),
                                       error_type=o.error_type, error_message=o.error_message) for o in actual],
                        retry=dict(run_id=retry.run_id, result=record(retry.result)))
    (artifact_directory(tmp_path) / f"failures-M{count}.json").write_text(
        json.dumps(retained, allow_nan=False, indent=2) + "\n")


@pytest.mark.parametrize("fault", ["object", "forged_state", "unfinished", "missing_steps",
                                 "wrong_id", "wrong_seed", "wrong_problem", "wrong_binning", "budget"])
def test_malformed_results_release_partial_work_and_preserve_neighbor(fault):
    job, neighbor = jobs(2)

    def malformed(ops, train, validation, **configuration):
        if fault == "wrong_id":
            configuration["run_id"] = "foreign"
        elif fault == "wrong_seed":
            configuration["seed"] += 1
        elif fault == "wrong_problem":
            train = replace(train, target=train.target + 1)
        elif fault == "wrong_binning":
            configuration["binning"] = Binning.fit(train.data, bins=2)
        elif fault == "budget":
            configuration["rounds"] = 1
        fit = recipes.squared(ops, train, validation, **configuration)
        if fault == "object":
            return object()
        if fault == "forged_state":
            return replace(fit, state=replace(fit.state))
        if fault == "unfinished":
            return replace(fit, stop=StopState.start(1, rounds=2))
        if fault == "missing_steps":
            return replace(fit, steps=())
        return fit

    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        expected = record(direct(ops, neighbor))
        actual = run_many(ops, (replace(job, recipe=malformed), neighbor))
        assert actual[0].result is None and actual[0].error_type == "ValueError"
        assert actual[1].error_type is None and record(actual[1].result) == expected
        assert context.metrics["live_bytes"] == 0 and not ops._records


def test_returning_caller_owned_run_does_not_close_or_release_it():
    job = jobs(1)[0]
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        fit = recipes.squared(ops, job.train, job.validation, run_id=job.run_id, seed=job.seed,
                              binning=job.binning, **job.options)
        expected = fit.run.export(fit.state).identity
        before = context.metrics["live_bytes"]
        actual = run_many(ops, (replace(job, recipe=lambda *a, **kw: fit),))[0]
        assert actual.error_type == "ValueError" and "caller-owned" in actual.error_message
        assert context.metrics["live_bytes"] == before
        assert fit.run.export(fit.state).identity == expected
        fit.run.close()
        assert context.metrics["live_bytes"] == 0


def test_partial_memory_failure_retains_error_and_cap_then_runs_neighbor():
    job, neighbor = jobs(2)
    cap = 1024**2

    def too_large(ops, train, validation, **kwargs):
        ops.execution.upload(np.ones(17, np.float32))
        ops.execution.upload(np.ones(cap // 4, np.float32))
        raise AssertionError("allocation cap was ignored")

    with ExecutionContext(max_bytes=cap) as context:
        ops = DeviceOperations(context)
        expected = record(direct(ops, neighbor))
        actual = run_many(ops, (replace(job, recipe=too_large), neighbor))
        assert actual[0].error_type == "MemoryError" and actual[0].result is None
        assert actual[1].error_type is None and record(actual[1].result) == expected
        assert context.metrics["peak_live_bytes"] <= cap
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_interrupts_propagate_after_workspace_cleanup(exception):
    job = jobs(1)[0]

    def interrupt(ops, *args, **kwargs):
        ops.execution.upload(np.ones(7, np.float32))
        raise exception("stop")

    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with pytest.raises(exception):
            run_many(ops, (replace(job, recipe=interrupt),))
        assert context.metrics["live_bytes"] == 0 and not ops._records


def test_zero_budget_and_unsupported_target_have_separate_outcomes():
    job = jobs(1)[0]
    zero = replace(job, options={**job.options, "rounds": 0})
    multi = replace(job.train, target=np.column_stack((job.train.target, job.train.target)),
                    raw_width=2, offset=np.zeros((len(job.train.target), 2)))
    unsupported = replace(job, run_id="unsupported", train=multi, validation=multi)
    with ExecutionContext() as context:
        actual = run_many(DeviceOperations(context), (zero, unsupported))
        assert actual[0].error_type is None
        assert actual[0].result.stop.reason == "budget"
        assert actual[0].result.state.version == 0 and not actual[0].result.steps
        assert actual[1].result is None and actual[1].error_type == "ValueError"
        assert context.metrics["live_bytes"] == 0
