"""Tests for ScoringBench artifact provenance that need no external checkout."""

from pathlib import Path

import pytest
from benchmarks.scoringbench.run import (
    _ci_state,
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
