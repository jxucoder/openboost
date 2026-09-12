"""124 resident feature sharing: original outcomes, explicit borrowing and isolation."""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from openboost import NumericData, device_inputs, device_recipes
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.device_runs import run_many
from openboost.device_runtime import DeviceRun
from openboost.execution import ExecutionContext
from openboost.runtime import RunContext

from .glm_artifacts import input_snapshot
from .test_device_runs_cuda import check_reference, direct, fresh_inference, record
from .test_device_runs_reference import cpu, jobs

pytestmark = pytest.mark.gpu


def prepare_pair(ops, job):
    return tuple(device_inputs.prepare(ops, job.binning.transform(p.data)) for p in (job.train, job.validation))


def folder(temporary):
    root = Path(os.environ.get("OPENBOOST_NORMAL_ARTIFACTS", temporary)) / "resident-inputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.mark.parametrize("count", [1, 8, 32])
def test_shared_encoding_preserves_direct_reversed_regrouped_and_fresh_models(count, monkeypatch, tmp_path):
    specs = jobs(count)
    with ExecutionContext(max_bytes=8 * 1024**2) as context:
        ops = DeviceOperations(context)
        expected = {s.run_id: direct(ops, s) for s in specs}
        for s in specs:
            check_reference(s, expected[s.run_id])
        assert context.metrics["live_bytes"] == 0
        prepared = prepare_pair(ops, specs[0])
        resident_bytes = sum(f.codes.nbytes + f.missing.nbytes for f in prepared)
        assert context.metrics["live_bytes"] == resident_bytes
        borrowed_codes = tuple(context.export(f.codes) for f in prepared)
        streams = {}

        def observed(ops, train, validation, **config):
            fit = device_recipes.squared(ops, train, validation, **config)
            assert fit.run.prepared_features is prepared
            value = tuple(int(v) for v in fit.run.rng(0, "learner", "rows").integers(0, 2**32, 8))
            assert value == tuple(int(v) for v in RunContext(fit.run.run_id, fit.run.seed).rng(0, "learner", "rows").integers(0, 2**32, 8))
            assert streams.setdefault(fit.run.run_id, value) == value
            return fit

        specs = tuple(replace(s, prepared=prepared, recipe=observed) for s in specs)

        def forbidden(*args, **kwargs):
            raise AssertionError("explicit resident features must not refit, encode or upload via ops.prepare")

        with monkeypatch.context() as patches:
            patches.setattr(Binning, "fit", forbidden)
            patches.setattr(Binning, "transform", forbidden)
            patches.setattr(ops, "prepare", forbidden)
            completed = {}
            for schedule in (specs, specs[::-1], specs[::2], specs[1::2]):
                results = run_many(ops, schedule)
                assert tuple(o.run_id for o in results) == tuple(s.run_id for s in schedule)
                for outcome in results:
                    assert outcome.error_type is None, outcome.error_message
                    assert record(outcome.result) == record(expected[outcome.run_id])
                    completed[outcome.run_id] = outcome.result
                assert context.metrics["live_bytes"] == resident_bytes
                assert set(ops._records) == set(prepared)
                for f, wanted in zip(prepared, borrowed_codes, strict=True):
                    np.testing.assert_array_equal(context.export(f.codes), wanted)
        assert len(set(streams.values())) == count
        metrics = dict(context.metrics)
        assert metrics["feature_prepare_calls"] == 2
        assert metrics["feature_upload_bytes"] == resident_bytes
        assert metrics["feature_bind_calls"] == 6 * count
        assert metrics["feature_bind_upload_bytes"] == 3 * count * sum(4 * f.n_rows for f in prepared)
        for f in prepared:
            ops.release(f)
        assert context.metrics["live_bytes"] == 0 and not ops._records
    retained = []
    for s in specs:
        result = completed[s.run_id]
        retained.append(dict(run_id=s.run_id, seed=s.seed, options=dict(s.options), binning_identity=s.binning.identity,
                             result=record(result), rng=list(streams[s.run_id]),
                             inputs=dict(train=input_snapshot(s.train), validation=input_snapshot(s.validation)),
                             predictions={key: {split: model.predict(p.data, offset=p.offset).tolist()
                                                for split, p in (("train", s.train), ("validation", s.validation))}
                                          for key, model in (("model", result.model), ("best_model", result.best_model))}))
    fresh_inference(dict(execution="sequential-feature-reuse", count=count, resident_bytes=resident_bytes,
                         context_metrics=metrics, final_live_bytes=0, outcomes=retained),
                    folder(tmp_path) / f"shared-M{count}.json")


@pytest.mark.parametrize("count", [1, 8, 32])
def test_later_failure_and_retry_preserve_shared_features_and_neighbors(count, tmp_path):
    specs = jobs(count)
    bad = count // 2

    def failure(ops, train, validation, **config):
        device_recipes.squared(ops, train, validation, **{**config, "rounds": 1, "patience": None, "min_delta": 0.})
        ops.execution.upload(np.ones(7, np.float32))
        raise ValueError("forced later shared failure")

    with ExecutionContext(max_bytes=8 * 1024**2) as context:
        ops = DeviceOperations(context)
        expected = {s.run_id: record(direct(ops, s)) for s in specs}
        prepared = prepare_pair(ops, specs[0])
        specs = tuple(replace(s, prepared=prepared) for s in specs)
        baseline = context.metrics["live_bytes"]
        changed = tuple(replace(s, recipe=failure) if i == bad else s for i, s in enumerate(specs))
        actual = run_many(ops, changed)
        for i, result in enumerate(actual):
            if i == bad:
                assert result.result is None and result.error_type == "ValueError"
                assert result.error_message == "forced later shared failure"
            else:
                assert result.error_type is None, result.error_message
                assert record(result.result) == expected[result.run_id]
        assert context.metrics["live_bytes"] == baseline and set(ops._records) == set(prepared)
        retry = run_many(ops, (specs[bad],))[0]
        assert retry.error_type is None and record(retry.result) == expected[retry.run_id]
        assert context.metrics["live_bytes"] == baseline and set(ops._records) == set(prepared)
        retained = dict(execution="sequential-feature-reuse", count=count, failed_id=specs[bad].run_id,
                        inputs_from=f"shared-M{count}.json", resident_bytes=baseline, context_metrics=dict(context.metrics),
                        outcomes=[dict(run_id=o.run_id, result=None if o.result is None else record(o.result),
                                       error_type=o.error_type, error_message=o.error_message) for o in actual],
                        retry=dict(run_id=retry.run_id, result=record(retry.result)))
        for f in prepared:
            ops.release(f)
        assert context.metrics["live_bytes"] == 0 and not ops._records
        retained["final_live_bytes"] = 0
    (folder(tmp_path) / f"failures-M{count}.json").write_text(json.dumps(retained, allow_nan=False, indent=2) + "\n")


@pytest.mark.parametrize("change", ["target", "weight", "offset"])
def test_changed_problem_bindings_match_cpu_and_own_only_weights(change):
    job = jobs(1)[0]
    original = job
    if change == "target":
        job = replace(job, train=replace(job.train, target=job.train.target + .25))
    elif change == "weight":
        job = replace(job, train=replace(job.train, weight=job.train.weight * 2))
    else:
        job = replace(job, train=replace(job.train, offset=job.train.offset + .25))
    expected = cpu(job)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        prepared = prepare_pair(ops, original)
        baseline, before_upload = context.metrics["live_bytes"], context.metrics["upload_bytes"]
        a = device_inputs.bind(ops, prepared[0], original.train)
        b = device_inputs.bind(ops, prepared[0], job.train)
        assert a.codes is b.codes is prepared[0].codes
        assert a.missing is b.missing is prepared[0].missing
        assert a.weight is not b.weight and a.problem_identity == original.train.identity and b.problem_identity == job.train.identity
        assert context.metrics["upload_bytes"] - before_upload == 2 * 4 * len(job.train.weight)
        np.testing.assert_array_equal(context.export(b.weight), job.train.weight.astype(np.float32))
        ops.release(a)
        ops.release(b)
        assert context.metrics["live_bytes"] == baseline
        actual = run_many(ops, (replace(job, prepared=prepared),))[0]
        assert actual.error_type is None, actual.error_message
        for model, wanted in ((actual.result.model, expected.state.model), (actual.result.best_model, expected.state.best_model)):
            np.testing.assert_allclose(model.predict(job.validation.data), wanted.predict(job.validation.data), rtol=1e-4, atol=1e-5)
        assert context.metrics["live_bytes"] == baseline and set(ops._records) == set(prepared)
        for f in prepared:
            ops.release(f)
        assert context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("fault", ["features", "rows", "cuts", "forged", "released", "ignored", "claimed", "swapped", "validation_weights"])
def test_invalid_borrow_or_ignored_pair_is_retained_without_contaminating_neighbor(fault):
    job, neighbor = jobs(2)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        expected = record(direct(ops, neighbor))
        prepared = prepare_pair(ops, job)
        invalid = prepared
        if fault in ("features", "rows"):
            data = job.train.data
            changed = NumericData(data.values + (.25 if fault == "features" else 0),
                                  data.row_ids + (1 if fault == "rows" else 0), data.feature_names)
            job = replace(job, train=replace(job.train, data=changed, row_ids=changed.row_ids))
        elif fault == "cuts":
            changed = device_inputs.prepare(ops, Binning.fit(job.train.data, bins=2).transform(job.validation.data))
            invalid = (prepared[0], changed)
        elif fault == "forged":
            invalid = (replace(prepared[0]), prepared[1])
        elif fault == "released":
            changed = device_inputs.prepare(ops, job.binning.transform(job.train.data))
            ops.release(changed)
            invalid = (changed, prepared[1])
        elif fault == "swapped":
            invalid = prepared[::-1]
        elif fault == "validation_weights":
            job = replace(job, validation=replace(job.validation, weight=np.full(len(job.validation.weight), 1e-300)))
        else:
            def ignored(ops, train, validation, **config):
                claimed = config.pop("prepared")
                fit = device_recipes.squared(ops, train, validation, **config)
                if fault == "claimed":
                    fit.run._prepared_features = claimed
                return fit
            job = replace(job, recipe=ignored)
        baseline, registered = context.metrics["live_bytes"], set(ops._records)
        outcomes = run_many(ops, (replace(job, prepared=invalid), replace(neighbor, prepared=prepared)))
        assert outcomes[0].result is None and outcomes[0].error_type == "ValueError"
        assert outcomes[1].error_type is None and record(outcomes[1].result) == expected
        assert context.metrics["live_bytes"] == baseline and set(ops._records) == registered
        for features in list(ops._records):
            ops.release(features)
        assert context.metrics["live_bytes"] == 0


def test_other_operations_instance_cannot_borrow_feature_record():
    job = jobs(1)[0]
    with ExecutionContext() as first, ExecutionContext() as second:
        a, b = DeviceOperations(first), DeviceOperations(second)
        prepared = prepare_pair(a, job)
        before = first.metrics["live_bytes"]
        result = run_many(b, (replace(job, prepared=prepared),))[0]
        assert result.error_type == "ValueError" and result.result is None
        assert first.metrics["live_bytes"] == before and second.metrics["live_bytes"] == 0
        for f in prepared:
            a.release(f)


def test_two_live_runs_own_independent_raw_and_close_preserves_shared_features():
    job = jobs(1)[0]
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        prepared = prepare_pair(ops, job)
        baseline = context.metrics["live_bytes"]
        a = DeviceRun(ops, job.train, job.validation, run_id="a", seed=17, prepared=prepared)
        b = DeviceRun(ops, job.train, job.validation, run_id="b", seed=17, prepared=prepared)
        a.initialize()
        sb = b.initialize()
        assert a.data.codes is b.data.codes is prepared[0].codes
        assert a.data.weight is not b.data.weight and a.problem is not b.problem
        value = b.export(sb).record()
        a.close()
        assert b.export(sb).record() == value
        b.close()
        assert context.metrics["live_bytes"] == baseline and set(ops._records) == set(prepared)
        for f in prepared:
            ops.release(f)
        assert context.metrics["live_bytes"] == 0


def test_explicit_caller_release_invalidates_borrower_but_run_cleanup_still_works():
    job = jobs(1)[0]
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        prepared = prepare_pair(ops, job)
        run = DeviceRun(ops, job.train, job.validation, run_id=job.run_id, seed=17, prepared=prepared)
        state = run.initialize()
        ops.release(prepared[0])
        with pytest.raises(ValueError, match="released"):
            run.fields(state)
        run.close()
        ops.release(prepared[1])
        assert not ops._records and context.metrics["live_bytes"] == 0


@pytest.mark.parametrize("fault", ["overflow", "underflow", "cap"])
def test_failed_binding_keeps_features_and_allows_fresh_valid_binding(fault):
    job = jobs(1)[0]
    with ExecutionContext(max_bytes=1024**2) as context:
        ops = DeviceOperations(context)
        prepared = prepare_pair(ops, job)
        baseline = context.metrics["live_bytes"]
        if fault == "cap":
            # This fresh pool has no released blocks; its peak includes allocation rounding.
            scratch = context.upload(np.zeros((1024**2 - context.metrics["peak_pool_bytes"]) // 4, np.float32))
            assert context.metrics["peak_pool_bytes"] == 1024**2
            invalid = job.train
        else:
            invalid = replace(job.train, weight=np.full(len(job.train.weight), 1e300 if fault == "overflow" else 1e-300))
        before = context.metrics["live_bytes"]
        with pytest.raises((ValueError, FloatingPointError, MemoryError)):
            device_inputs.bind(ops, prepared[0], invalid)
        assert context.metrics["live_bytes"] == before and set(ops._records) == set(prepared)
        if fault == "cap":
            context.release(scratch)
        bound = device_inputs.bind(ops, prepared[0], job.train)
        ops.release(bound)
        assert context.metrics["live_bytes"] == baseline
        for f in prepared:
            ops.release(f)
        assert context.metrics["live_bytes"] == 0
