"""Host scheduling contracts only; these tests do not emulate CUDA execution."""

import subprocess
import sys
from dataclasses import replace

import pytest

from openboost import device_recipes
from openboost.binning import Binning
from openboost.device_recipes import DeviceNormalStep, DeviceStep
from openboost.device_runs import RunSpec, _completed, run_many
from openboost.device_runtime import DeviceState
from openboost.stopping import StopState

from .test_device_round_reference import prepared_fixture


def spec(**kwargs):
    train, validation, binned, _ = prepared_fixture("weighted")
    return RunSpec("120", 7, train, validation, device_recipes.squared,
                   binning=binned.binning, **kwargs)


def test_options_are_owned_and_binning_is_explicit():
    options = {"rounds": 2, "learning_rate": .5}
    job = spec(options=options)
    options["rounds"] = 99
    assert job.options["rounds"] == 2
    with pytest.raises(TypeError):
        job.options["rounds"] = 99
    assert isinstance(job.binning, Binning)


@pytest.mark.parametrize("name", ["ops", "train", "validation", "run_id", "seed", "binning", 1])
def test_reserved_or_untyped_options_cannot_replace_identity(name):
    with pytest.raises(ValueError, match="replace"):
        spec(options={name: None})


@pytest.mark.parametrize("value", [[], {}, object(), lambda: None])
def test_mutable_or_callable_option_requires_explicit_recipe(value):
    with pytest.raises(ValueError, match="immutable"):
        spec(options={"learner": value})


@pytest.mark.parametrize("change", [
    {"run_id": ""}, {"seed": True}, {"seed": -1}, {"train": None},
    {"validation": None}, {"recipe": None}, {"options": None}, {"binning": 2},
])
def test_invalid_spec_metadata_is_rejected(change):
    with pytest.raises(ValueError):
        replace(spec(), **change)


@pytest.mark.parametrize("execution", ["batch", "parallel", "cpu", None])
def test_unsupported_execution_rejected_before_dispatch(execution):
    with pytest.raises(ValueError, match="sequential"):
        run_many(None, (spec(),), execution=execution)


def test_duplicate_ids_and_wrong_specs_fail_before_device_access():
    job = spec()
    with pytest.raises(ValueError, match="unique"):
        run_many(None, (job, replace(job, seed=8)))
    with pytest.raises(ValueError, match="RunSpec"):
        run_many(None, (object(),))
    with pytest.raises(ValueError, match="fallback"):
        run_many(None, (job,))


def summary():
    state = DeviceState("diagnostic", "120", 2, 1., 1., 1., 2, 2)
    stop = StopState.start(2, rounds=2).observe(1.5).observe(1)
    steps = tuple(DeviceStep(i, (.5,), True, (), 1, 1, 1) for i in range(2))
    return state, steps, stop


def test_terminal_summary_accepts_ordered_substeps_and_zero_rounds():
    state, steps, stop = summary()
    _completed(state, steps, stop, "120")
    ordered = tuple(DeviceNormalStep(i, (k,), i, i + 1, (), 1, 1, 1)
                    for i in range(2) for k in range(2))
    _completed(state, ordered, stop, "120")
    _completed(replace(state, version=0, n_terms=0, best_n_terms=0), (),
               StopState.start(2, rounds=0), "120")


@pytest.mark.parametrize("fault", ["foreign", "unfinished", "missing", "order", "extra",
                                 "boolean", "mutable", "unknown"])
def test_malformed_completion_cannot_be_exported(fault):
    state, steps, stop = summary()
    if fault == "foreign":
        state = replace(state, run_id="other")
    elif fault == "unfinished":
        stop = StopState.start(2, rounds=2)
    elif fault == "missing":
        steps = steps[:1]
    elif fault == "order":
        steps = steps[::-1]
    elif fault == "extra":
        steps += (replace(steps[-1], round_index=2),)
    elif fault == "boolean":
        steps = (replace(steps[0], round_index=False), steps[1])
    elif fault == "mutable":
        steps = list(steps)
    else:
        steps = (object(), object())
    with pytest.raises(ValueError):
        _completed(state, steps, stop, "120")


def test_import_has_no_cuda_or_cpu_trainer_dependency():
    # Native device recipes import CPU metadata helpers, but cannot select a CPU
    # trainer as fallback. Block their runtime entry points, not those helpers.
    code = """
import sys
for name in ('cupy', 'numba', 'openboost.recipes', 'openboost.runs'):
    sys.modules[name] = None
from openboost.device_runs import RunSpec, run_many
assert callable(run_many)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
