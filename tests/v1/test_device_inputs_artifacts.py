"""Installed host prerequisites for resident-sharing device fixtures and retention."""

from dataclasses import replace

import numpy as np
import pytest

from . import test_device_inputs_cuda as gpu
from .test_device_runs_reference import jobs


class BeforeDevice(Exception):
    pass


def test_all_resident_device_cases_construct_host_inputs_before_cuda(monkeypatch, tmp_path):
    def stop(*args, **kwargs):
        raise BeforeDevice

    monkeypatch.setattr(gpu, "ExecutionContext", stop)
    calls = []
    for count in (1, 8, 32):
        calls.append((gpu.test_shared_encoding_preserves_direct_reversed_regrouped_and_fresh_models, (count, monkeypatch, tmp_path)))
        calls.append((gpu.test_later_failure_and_retry_preserve_shared_features_and_neighbors, (count, tmp_path)))
    for change in ("target", "weight", "offset"):
        calls.append((gpu.test_changed_problem_bindings_match_cpu_and_own_only_weights, (change,)))
    for fault in ("features", "rows", "cuts", "forged", "released", "ignored", "claimed", "swapped", "validation_weights"):
        calls.append((gpu.test_invalid_borrow_or_ignored_pair_is_retained_without_contaminating_neighbor, (fault,)))
    for function in (gpu.test_other_operations_instance_cannot_borrow_feature_record,
                     gpu.test_two_live_runs_own_independent_raw_and_close_preserves_shared_features,
                     gpu.test_explicit_caller_release_invalidates_borrower_but_run_cleanup_still_works):
        calls.append((function, ()))
    for fault in ("overflow", "underflow", "cap"):
        calls.append((gpu.test_failed_binding_keeps_features_and_allows_fresh_valid_binding, (fault,)))
    assert len(calls) == 24
    for function, arguments in calls:
        with pytest.raises(BeforeDevice):
            function(*arguments)


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_weight_boundary_fixture_is_valid_host_input_before_device_narrowing(scale):
    job = jobs(1)[0]
    problem = replace(job.train, weight=np.full(len(job.train.weight), scale))
    assert np.isfinite(problem.weight).all() and problem.weight.sum() > 0


@pytest.mark.parametrize("collector", [False, True])
def test_resident_artifacts_use_explicit_bounded_collector(collector, monkeypatch, tmp_path):
    monkeypatch.delenv("OPENBOOST_NORMAL_ARTIFACTS", raising=False)
    target = tmp_path
    if collector:
        target = tmp_path / "normal"
        monkeypatch.setenv("OPENBOOST_NORMAL_ARTIFACTS", str(target))
    assert gpu.folder(tmp_path) == target / "resident-inputs"
