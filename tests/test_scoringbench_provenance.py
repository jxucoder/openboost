"""Tests for ScoringBench artifact provenance that need no external checkout."""

from pathlib import Path

import pytest
from benchmarks.scoringbench.run import (
    _audit_records,
    _build_parser,
    _ci_state,
    _load_dataset_registry,
    _select_datasets,
    _validate_selected_datasets,
    _working_directory,
)


def test_ci_state_is_none_outside_github_actions(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    assert _ci_state() is None


def test_ci_state_distinguishes_tested_merge_from_source_head(monkeypatch):
    values = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_REPOSITORY": "jxucoder/openboost",
        "GITHUB_REF": "refs/pull/19/merge",
        "GITHUB_SHA": "merge-sha",
        "OPENBOOST_SOURCE_SHA": "head-sha",
        "GITHUB_HEAD_REF": "codex/scoringbench-release-hardening",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    assert _ci_state() == {
        "provider": "github_actions",
        "event_name": "pull_request",
        "repository": "jxucoder/openboost",
        "ref": "refs/pull/19/merge",
        "tested_sha": "merge-sha",
        "source_sha": "head-sha",
        "head_ref": "codex/scoringbench-release-hardening",
        "run_id": "123",
        "run_attempt": "2",
    }


def test_working_directory_contains_upstream_output_and_restores_on_error(tmp_path):
    original = Path.cwd()
    artifact_dir = tmp_path / "artifact"

    with pytest.raises(RuntimeError, match="stop"), _working_directory(artifact_dir):
        assert Path.cwd() == artifact_dir
        Path("datasets.json").write_text("[]\n")
        raise RuntimeError("stop")

    assert Path.cwd() == original
    assert (artifact_dir / "datasets.json").read_text() == "[]\n"


def test_named_quality_shard_validates_only_selected_dataset():
    class Args:
        dataset_index = None
        dataset_name = ["Abalone"]

    registry = [
        {"name": "Abalone", "source": "openml", "id": 183},
        {"name": "large_unused", "source": "openml", "id": 999},
    ]
    validated = []

    def validate(datasets):
        validated.extend(datasets)
        return datasets

    result = _validate_selected_datasets(registry, Args(), validate)

    assert result == [registry[0]]
    assert validated == [registry[0]]


def _complete_record(dataset="example", model="openboost_cpu", fold=0):
    return {
        "dataset": dataset,
        "model": model,
        "fold": fold,
        "crps": 0.2,
        "log_score": 0.4,
        "rmse": 0.5,
        "coverage_90": 0.9,
        "interval_score_90": 1.2,
        "train_time": 2.0,
    }


def test_outcome_audit_accepts_exact_complete_distributional_rows():
    outcome = _audit_records(
        [_complete_record(fold=0), _complete_record(fold=1)],
        [{"name": "example"}],
        ["openboost_cpu"],
        n_folds=2,
        n_repeats=1,
    )

    assert outcome["status"] == "complete"
    assert outcome["expected_rows"] == 2
    assert outcome["valid_rows"] == 2


def test_outcome_audit_publishes_missing_error_and_invalid_metric_rows():
    invalid = _complete_record(model="ngboost", fold=0)
    invalid["log_score"] = float("nan")
    error = _complete_record(fold=1)
    error.update(error="model exploded", error_type="RuntimeError")

    outcome = _audit_records(
        [_complete_record(fold=0), error, invalid],
        [{"name": "example"}],
        ["openboost_cpu", "ngboost"],
        n_folds=2,
        n_repeats=1,
    )

    assert outcome["status"] == "incomplete"
    assert outcome["expected_rows"] == 4
    assert outcome["valid_rows"] == 1
    assert outcome["missing_rows"] == [
        {"dataset": "example", "model": "ngboost", "fold": 1}
    ]
    assert outcome["error_rows"][0]["error"] == "model exploded"
    assert outcome["invalid_metric_rows"][0]["metrics"] == ["log_score"]


def test_load_dataset_registry_accepts_frozen_scoringbench_list(tmp_path):
    path = tmp_path / "datasets.json"
    path.write_text('[{"name": "alpha", "source": "pmlb", "url": "https://example"}]')

    assert _load_dataset_registry(path) == [
        {"name": "alpha", "source": "pmlb", "url": "https://example"}
    ]


def test_stable_strided_shards_cover_registry_exactly_once():
    registry = [{"name": f"dataset_{index}"} for index in range(7)]

    class Args:
        dataset_index = None
        dataset_name = None
        shard_count = 3
        shard_index = 0

    selected = []
    for shard_index in range(Args.shard_count):
        Args.shard_index = shard_index
        selected.extend(_select_datasets(registry, Args()))

    assert sorted(dataset["name"] for dataset in selected) == sorted(
        dataset["name"] for dataset in registry
    )
    assert len(selected) == len({dataset["name"] for dataset in selected})


def test_strong_baseline_defaults_match_scoringbench_registered_budgets():
    args = _build_parser().parse_args(
        ["--models", "openboost_cpu,ngboost,xgboost_quantile,xgblss,catboost_quantile"]
    )

    assert args.models == [
        "openboost_cpu",
        "ngboost",
        "xgboost_quantile",
        "xgblss",
        "catboost_quantile",
    ]
    assert args.n_trees == 500
    assert args.xgboost_rounds == 100
    assert args.xgboost_quantiles == 50
    assert args.xgblss_rounds == 100
    assert args.catboost_rounds == 1000
