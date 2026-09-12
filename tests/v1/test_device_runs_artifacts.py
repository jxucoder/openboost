"""CPU-only installed prerequisites for the future sequential device packet."""

from dataclasses import replace

import pytest

from openboost.device_recipes import DeviceStep
from openboost.device_runs import RunResult, _one
from openboost.device_runtime import DeviceState

from . import test_device_runs_cuda as gpu
from .glm_artifacts import input_snapshot
from .test_device_glm_reference import fixture
from .test_device_runs_reference import cpu, jobs


class BeforeDevice(Exception):
    pass


def test_every_new_device_case_constructs_host_inputs_before_real_context(monkeypatch, tmp_path):
    def stop(*args, **kwargs):
        raise BeforeDevice

    monkeypatch.setattr(gpu, "ExecutionContext", stop)
    calls = []
    for count in (1, 8, 32):
        calls.append((gpu.test_independent_sequential_reversed_regrouped_and_fresh_models,
                      (count, monkeypatch, tmp_path)))
        calls.append((gpu.test_one_later_failure_retained_and_same_id_retry_is_independent, (count, tmp_path)))
    for fault in ("object", "forged_state", "unfinished", "missing_steps", "wrong_id", "wrong_seed",
                  "wrong_problem", "wrong_binning", "budget"):
        calls.append((gpu.test_malformed_results_release_partial_work_and_preserve_neighbor, (fault,)))
    for function in (gpu.test_returning_caller_owned_run_does_not_close_or_release_it,
                     gpu.test_partial_memory_failure_retains_error_and_cap_then_runs_neighbor,
                     gpu.test_zero_budget_and_unsupported_target_have_separate_outcomes):
        calls.append((function, ()))
    for exception in (KeyboardInterrupt, SystemExit):
        calls.append((gpu.test_interrupts_propagate_after_workspace_cleanup, (exception,)))
    assert len(calls) == 20
    for function, arguments in calls:
        with pytest.raises(BeforeDevice):
            function(*arguments)


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_specialized_target_cannot_enter_raw_scalar_export(family):
    job = jobs(1)[0]
    with pytest.raises(ValueError, match="scalar"):
        _one(None, replace(job, train=fixture(family)))


def test_fresh_inference_helper_replays_cpu_controls_with_training_imports_denied(tmp_path):
    rows = []
    for job in jobs(4):
        fit = cpu(job)
        steps = tuple(DeviceStep(i, s.coefficients, s.accepted, (), s.loss_after,
                                 fit.stop.last_score, fit.state.best_score)
                      for i, s in enumerate(fit.steps))
        state = DeviceState("cpu-helper-control", job.run_id, fit.state.version,
                             steps[-1].loss, fit.stop.last_score, fit.state.best_score,
                             len(fit.state.model.terms), len(fit.state.best_model.terms))
        result = RunResult(fit.state.model, fit.state.best_model, state, steps, fit.stop)
        rows.append(dict(run_id=job.run_id, result=gpu.record(result),
                         inputs=dict(train=input_snapshot(job.train), validation=input_snapshot(job.validation)),
                         predictions={key: {split: model.predict(p.data, offset=p.offset).tolist()
                                            for split, p in (("train", job.train), ("validation", job.validation))}
                                      for key, model in (("model", result.model), ("best_model", result.best_model))}))
    gpu.fresh_inference(dict(execution="CPU helper control only", outcomes=rows), tmp_path / "cpu-control.json")


@pytest.mark.parametrize("selection", ["temporary", "collector", "override"])
def test_artifacts_use_declared_collector_root(selection, monkeypatch, tmp_path):
    monkeypatch.delenv("OPENBOOST_DEVICE_RUN_ARTIFACTS", raising=False)
    monkeypatch.delenv("OPENBOOST_NORMAL_ARTIFACTS", raising=False)
    expected = tmp_path
    if selection != "temporary":
        monkeypatch.setenv("OPENBOOST_NORMAL_ARTIFACTS", str(tmp_path / "normal"))
        expected = tmp_path / "normal/sequential-runs"
    if selection == "override":
        monkeypatch.setenv("OPENBOOST_DEVICE_RUN_ARTIFACTS", str(tmp_path / "explicit"))
        expected = tmp_path / "explicit"
    assert gpu.artifact_directory(tmp_path) == expected
