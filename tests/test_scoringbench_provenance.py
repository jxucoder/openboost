"""Tests for ScoringBench artifact provenance that need no external checkout."""

from benchmarks.scoringbench.run import _ci_state


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
