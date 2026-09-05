"""The evidence gate must reject apparently successful but incomplete runs."""

import json
import subprocess
import sys

import pytest
from benchmarks.foundation.runner import validate_result


@pytest.fixture
def evidence():
    manifest = {"wheel_sha256": "abc", "source_sha": "def", "source_dirty": False}
    result = {
        "returncode": 0,
        "timed_out": False,
        "junit": '<testsuites><testsuite><testcase name="test_device_interop" />'
        '<testcase name="test_normal_gpu_fit" /></testsuite></testsuites>',
        "checks": {
            "interop": True,
            "native_tree_calls": 4,
            "device_objective_calls": 2,
            "installed_files_verified": 10,
            "dataset_sha256": "123",
        },
        "environment": {"cuda_available": True, "gpu_name": "Tesla T4"},
        "wheel_sha256": "abc",
        "source_sha": "def",
    }
    return manifest, result


def test_complete_result(evidence):
    validate_result(*evidence)


@pytest.mark.parametrize(
    "fault",
    [
        "exit",
        "timeout",
        "missing",
        "skip",
        "failure",
        "duplicate",
        "cuda",
        "wheel",
        "source",
        "fallback",
    ],
)
def test_incomplete_result_rejected(evidence, fault):
    manifest, result = evidence
    if fault == "exit":
        result["returncode"] = 1
    elif fault == "timeout":
        result["timed_out"] = True
    elif fault == "missing":
        result["junit"] = "<testsuites/>"
    elif fault in ("skip", "failure"):
        tag = "skipped" if fault == "skip" else "failure"
        result["junit"] = result["junit"].replace(
            'name="test_device_interop" />', f'name="test_device_interop"><{tag}/></testcase>'
        )
    elif fault == "duplicate":
        result["junit"] = result["junit"].replace("test_normal_gpu_fit", "test_device_interop")
    elif fault == "cuda":
        result["environment"]["cuda_available"] = False
    elif fault == "wheel":
        result["wheel_sha256"] = "wrong"
    elif fault == "source":
        result["source_sha"] = "wrong"
    else:
        result["checks"]["device_objective_calls"] = 0
    with pytest.raises(ValueError):
        validate_result(manifest, result)


def test_cli_returns_nonzero_for_missing_report(tmp_path, evidence):
    manifest, _ = evidence
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    run = subprocess.run(
        [sys.executable, "-m", "benchmarks.foundation.runner", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert "results.json" in run.stderr


def test_correctness_requires_weighted_cases(evidence):
    manifest, result = evidence
    manifest['suite'] = 'correctness'
    with pytest.raises(ValueError, match='missing'):
        validate_result(manifest, result)
    extra = ''.join(f'<testcase name="{name}" />' for name in (
        'test_weighted_newton', 'test_weighted_distribution[normal]',
        'test_weighted_distribution[poisson]',
    ))
    result['junit'] = result['junit'].replace('</testsuite>', extra + '</testsuite>')
    validate_result(manifest, result)


def test_unknown_suite_rejected(evidence):
    manifest, result = evidence
    manifest['suite'] = 'unknown'
    with pytest.raises(ValueError, match='Unknown'):
        validate_result(manifest, result)


def test_boundaries_requires_all_execution_cases(evidence):
    manifest, result = evidence
    manifest['suite'] = 'boundaries'
    names = [
        'test_weighted_newton', 'test_weighted_distribution[normal]',
        'test_weighted_distribution[poisson]', 'test_visible_fallback[custom]',
        'test_visible_fallback[exposure]', 'test_visible_fallback[generic]',
        'test_device_error_rolls_back', 'test_device_sampling_preflight[subsample]',
        'test_device_sampling_preflight[colsample_bytree]',
        'test_eval_callback_persistence[normal]', 'test_eval_callback_persistence[poisson]',
    ]
    for name in names:
        with pytest.raises(ValueError, match='missing'):
            validate_result(manifest, result)
        result['junit'] = result['junit'].replace('</testsuite>', f'<testcase name="{name}" /></testsuite>')
    validate_result(manifest, result)


def test_histogram_suite_requires_device_check(evidence):
    manifest, result = evidence
    manifest['suite'] = 'histograms'
    result['junit'] = result['junit'].replace('</testsuite>', '<testcase name="test_batch_histogram_device_oracle" /></testsuite>')
    with pytest.raises(ValueError, match='histogram'):
        validate_result(manifest, result)
    result['checks']['batch_histograms'] = {'device_arrays': True, 'legacy_download_wrappers_blocked': True, 'cases': [{}, {}, {}]}
    validate_result(manifest, result)


def test_split_suite_requires_routing_evidence(evidence):
    manifest, result = evidence
    manifest['suite'] = 'splits'
    result['junit'] = result['junit'].replace('</testsuite>', '<testcase name="test_batch_histogram_device_oracle" /><testcase name="test_batch_split_routing_oracle" /></testsuite>')
    result['checks']['batch_histograms'] = {'device_arrays': True, 'legacy_download_wrappers_blocked': True, 'cases': [{}, {}, {}]}
    with pytest.raises(ValueError, match='split/routing'):
        validate_result(manifest, result)
    result['checks']['batch_splits'] = {'device_arrays': True, 'routed_child_oracle': True, 'exact_ties_and_gain_boundary': True, 'cases': [{}, {}, {}, {}]}
    validate_result(manifest, result)


def test_leaf_suite_requires_changed_gradient(evidence):
    manifest, result = evidence
    manifest['suite'] = 'leaves'
    extra = ''.join(f'<testcase name="{name}" />' for name in
                    ('test_batch_histogram_device_oracle', 'test_batch_split_routing_oracle', 'test_batch_leaf_rule_oracle'))
    result['junit'] = result['junit'].replace('</testsuite>', extra + '</testsuite>')
    result['checks']['batch_histograms'] = {'device_arrays': True, 'legacy_download_wrappers_blocked': True, 'cases': [{}, {}, {}]}
    result['checks']['batch_splits'] = {'device_arrays': True, 'routed_child_oracle': True, 'exact_ties_and_gain_boundary': True, 'cases': [{}, {}, {}, {}]}
    with pytest.raises(ValueError, match='leaf rule'):
        validate_result(manifest, result)
    result['checks']['batch_leaves'] = {'device_arrays': True, 'row_sum_oracle': True, 'bounded_changes_next_gradient': True, 'cases': [{}, {}, {}], 'two_rounds': [{}, {}]}
    validate_result(manifest, result)
