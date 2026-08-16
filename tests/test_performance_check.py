"""Tests for the CI performance comparison harness."""

import json
from pathlib import Path

from benchmarks.check_performance import (
    check_regression,
    collect_provenance,
    load_baselines,
)


def _result(**overrides):
    result = {
        "fit_time_median": 1.0,
        "predict_time_median": 0.1,
        "peak_memory_mb": 10.0,
        "mse": 0.05,
        "r2": 0.95,
        "n_samples": 5000,
        "n_features": 10,
        "n_trees": 100,
        "max_depth": 6,
    }
    result.update(overrides)
    return result


def test_equal_results_have_no_regression():
    baseline = _result()

    assert check_regression(_result(), baseline) == []


def test_runtime_and_quality_regressions_are_reported():
    baseline = _result()
    current = _result(fit_time_median=1.21, mse=0.061)

    regressions = check_regression(current, baseline)

    assert any("fit_time_median" in item for item in regressions)
    assert any("mse" in item for item in regressions)


def test_load_baselines_uses_explicit_path(tmp_path):
    baseline_path = tmp_path / "parent.json"
    baseline_path.write_text(json.dumps(_result()))

    assert load_baselines(baseline_path) == _result()


def test_provenance_records_source_commit_and_environment():
    provenance = collect_provenance(Path.cwd())

    assert len(provenance["git_commit"]) == 40
    assert provenance["python_version"]
    assert provenance["numpy_version"]
    assert provenance["openboost_version"]
