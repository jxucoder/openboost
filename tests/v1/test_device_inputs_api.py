"""Host contracts for explicit resident feature reuse; no CUDA emulation."""

import inspect
import subprocess
import sys
from dataclasses import replace

import pytest

from openboost import NumericData, device_inputs
from openboost.binning import Binning

from .test_device_runs_api import spec
from .test_device_runs_reference import jobs


def test_public_preparation_is_separate_and_consumers_accept_explicit_pair():
    from openboost import device_inputs
    from openboost.device_recipes import squared
    from openboost.device_runs import RunSpec
    from openboost.device_runtime import DeviceRun

    assert callable(device_inputs.prepare) and callable(device_inputs.bind)
    for consumer in (squared, RunSpec, DeviceRun):
        assert "prepared" in inspect.signature(consumer).parameters



def metadata(data, binning):
    """Metadata-only record; deliberately unregistered and unusable for device work."""
    binned = binning.transform(data)
    return device_inputs.DeviceFeatures(data.identity, binned.identity, binning, len(data.row_ids), None, None)


def pair(job):
    return tuple(metadata(p.data, job.binning) for p in (job.train, job.validation))


@pytest.mark.parametrize("index", range(16))
def test_changed_targets_weights_offsets_remain_safe_feature_reuse(index):
    job = jobs(index + 1)[-1]
    original = jobs(1)[0]
    prepared = pair(original)
    train = replace(job.train, offset=job.train.offset + .25)
    validation = replace(job.validation, offset=job.validation.offset - .5)
    assert device_inputs.check_pair(prepared, train, validation) is original.binning
    assert device_inputs.check_pair(prepared, train, validation, binning=job.binning).identity == original.binning.identity
    result = replace(job, train=train, validation=validation, prepared=prepared)
    assert result.prepared is prepared


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("change", ["features", "rows", "schema", "order"])
def test_changed_feature_or_row_identity_is_rejected(split, change):
    job = jobs(1)[0]
    original = getattr(job, split)
    data = original.data
    values, ids, names = data.values.copy(), data.row_ids.copy(), data.feature_names
    if change == "features":
        values[0, 0] += 1
    elif change == "rows":
        ids += 1
    elif change == "schema":
        names = tuple("different-" + n for n in names)
    else:
        values, ids = values[::-1], ids[::-1]
    data = NumericData(values, ids, names)
    changed = replace(original, data=data, row_ids=data.row_ids)
    with pytest.raises(ValueError, match="identity"):
        device_inputs.check_pair(pair(job), changed if split == "train" else job.train,
                                 changed if split == "validation" else job.validation)


@pytest.mark.parametrize("fault", ["list", "one", "three", "object", "cuts", "explicit", "bins"])
def test_invalid_pair_and_conflicting_preparation_are_rejected(fault):
    job = jobs(1)[0]
    prepared, kwargs = pair(job), {}
    if fault == "list":
        prepared = list(prepared)
    elif fault == "one":
        prepared = prepared[:1]
    elif fault == "three":
        prepared += prepared[:1]
    elif fault == "object":
        prepared = (object(), prepared[1])
    elif fault == "cuts":
        prepared = (prepared[0], metadata(job.validation.data, Binning.fit(job.train.data, bins=2)))
    elif fault == "explicit":
        kwargs["binning"] = Binning.fit(job.train.data, bins=2)
    else:
        kwargs["bins"] = 2
    with pytest.raises(ValueError):
        device_inputs.check_pair(prepared, job.train, job.validation, **kwargs)


@pytest.mark.parametrize("prepared", [[], (), (None, None), (object(),)])
def test_spec_requires_explicit_immutable_pair(prepared):
    with pytest.raises(ValueError):
        replace(spec(), prepared=prepared)


def test_options_cannot_override_prepared_feature_identity():
    with pytest.raises(ValueError, match="replace"):
        spec(options={"prepared": None})


def test_public_module_import_does_not_load_cuda_or_cpu_training():
    code = """
import sys
for name in ('cupy', 'numba', 'openboost.recipes', 'openboost.runs'):
    sys.modules[name] = None
from openboost.device_inputs import prepare, bind, DeviceFeatures
assert callable(prepare) and callable(bind)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
