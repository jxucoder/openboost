"""Local protocol checks, including recorded 099 prefixes; no live cancellation."""

import copy
import json
import time
from pathlib import Path

import pytest
from benchmarks.v1.authoring import cancellation_smoke as smoke
from benchmarks.v1.authoring import responses_transport as wire
from benchmarks.v1.authoring.accounting import Controller, Limits
from benchmarks.v1.authoring.responses_background import Background

EVIDENCE = Path(__file__).resolve().parents[2] / "benchmarks/v1/evidence/author-accounting-099"


def test_recorded_active_prefix_cancels_before_another_poll(tmp_path, monkeypatch):
    from benchmarks.v1.authoring import responses_background as bg

    monkeypatch.setattr(bg, "POLL_INTERVAL_S", 0)
    prefix = EVIDENCE / "cancellation/request-001"
    queued = json.loads((prefix / "operation-001-create/response.bin").read_text())
    active = json.loads((prefix / "operation-002-retrieve/response.bin").read_text())
    # Only the queued/active prefix is measured. Cancellation below is a fixture.
    cancelled = json.loads((prefix / "response.json").read_text())
    cancelled["status"] = "cancelled"
    replies = [queued, active, cancelled, cancelled]
    calls = []

    def call(operation, directory, deadline, response_id=None):
        calls.append(operation)
        return copy.deepcopy(replies[len(calls) - 1])

    gate = Controller(
        tmp_path / "attempt",
        model="gpt-5.6-luna",
        reasoning_effort="medium",
        limits=Limits(4096, 4096, 1, 5),
        background=True,
        stop_on_in_progress=True,
        transport=Background(call=call, stop_on_in_progress=True),
    )
    with pytest.raises(RuntimeError, match="answer withheld"):
        gate.request(json.loads((prefix / "request.json").read_text())["input"])
    assert calls == ["create", "retrieve", "cancel", "retrieve"]
    assert gate.record["status"] == "transport_stopped"
    assert gate.record["generated_tokens"] == 104  # Synthetic cancelled usage only.
    assert gate.record["reserved_output_tokens"] == 0
    receipt = json.loads((gate.directory / "request-001/background.json").read_text())
    trigger = receipt["stop_trigger"]
    assert trigger["kind"] == "observed_in_progress"
    assert trigger["operation"] == 2 and trigger["response_id"] == active["id"]
    assert trigger["remaining_work_s"] > 0
    with pytest.raises(RuntimeError, match="closed"):
        gate.request("must not generate")
    assert len(calls) == 4


@pytest.mark.parametrize(
    "outcome",
    [
        "cancelled",
        "missing_usage",
        "completion_race",
        "early_completion",
        "expired_active",
        "wrong_model",
        "retrieval_failure",
        "changed_tier",
    ],
)
def test_actual_controller_and_classifier_keep_stop_outcomes_distinct(
    tmp_path, monkeypatch, outcome
):
    now = [time.monotonic()]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    value = json.loads((EVIDENCE / "cancellation/request-001/response.json").read_text())
    value["usage"] = dict(
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
        output_tokens_details={"reasoning_tokens": 3},
    )
    calls = []

    def call(operation, directory, deadline, response_id=None):
        calls.append(operation)
        reply = copy.deepcopy(value)
        if operation == "create":
            reply["status"] = "completed" if outcome == "early_completion" else "in_progress"
            if outcome == "expired_active":
                now[0] += 6
            if outcome == "wrong_model":
                reply["model"] = "unexpected-fixture-model"
        else:
            if outcome == "retrieval_failure" and operation == "retrieve":
                raise TimeoutError("local fixture final retrieval failure")
            reply["status"] = "completed" if outcome == "completion_race" else "cancelled"
            if outcome == "missing_usage":
                reply["usage"] = None
            if outcome == "changed_tier":
                reply["service_tier"] = "priority"
        return reply

    monkeypatch.setattr(wire, "call", call)
    result = smoke.case(tmp_path)
    assert result["status"] == ("pass" if outcome == "cancelled" else "fail")
    assert result["subsequent_request_blocked"] is True
    assert calls == (
        ["create"] if outcome == "early_completion" else ["create", "cancel", "retrieve"]
    )
    assert result["active_trigger_observed"] is (
        outcome not in ("expired_active", "wrong_model", "early_completion")
    )
    assert result["outcome"]["returned"] is (outcome == "early_completion")
    assert result["generated_tokens"] == (None if outcome == "missing_usage" else 7)
    assert result["reserved_output_tokens"] == (4096 if outcome == "missing_usage" else 0)
    if outcome == "missing_usage":
        assert result["cancellation_observed"] is True
        assert result["controller_status"] == "usage_unknown"
    if outcome == "completion_race":
        assert result["cancellation_observed"] is False
    if outcome == "expired_active":
        assert result["cancellation_observed"] is True
        assert result["controller_status"] == "wall_limit"


def test_active_stop_requires_background_before_creating_an_attempt(tmp_path):
    with pytest.raises(ValueError):
        Controller(
            tmp_path / "attempt",
            model="fixture",
            reasoning_effort="medium",
            limits=Limits(10, 10, 1, 5),
            stop_on_in_progress=True,
        )
    assert not (tmp_path / "attempt").exists()


def test_pending_and_consumed_packets_never_dispatch(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("unapproved packet reached network or preflight")

    monkeypatch.setattr(wire, "call", unexpected)
    monkeypatch.setattr(smoke, "preflight", unexpected)
    for state in ("pending", "consumed"):
        with pytest.raises(ValueError, match="approval"):
            smoke.execute(tmp_path, json.dumps({"authorization": state}).encode())


def test_preflight_binds_sources_and_the_unchanged_predecessor_prompt(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("preflight reached network")

    monkeypatch.setattr(wire, "call", unexpected)
    freeze = dict(
        schema="openboost-active-cancellation-smoke-v1",
        authorization="pending",
        config=copy.deepcopy(smoke.CONFIG),
        prompt_sha256=smoke.digest(smoke.CONFIG["prompt"].encode("ascii")),
        files={name: smoke.digest((smoke.ROOT / name).read_bytes()) for name in smoke.FILES},
    )
    result = smoke.preflight(smoke.ROOT, freeze)
    assert result["network_used"] is False and result["generation_requests_max"] == 1
    assert result["estimated_token_cost_upper_usd"] < 0.01
    freeze["files"][smoke.FILES[0]] = "0" * 64
    with pytest.raises(ValueError, match="source changed"):
        smoke.preflight(smoke.ROOT, freeze)


def test_frozen_output_cannot_reset_a_consumed_run(tmp_path, monkeypatch):
    output = tmp_path / "already-used"
    output.mkdir()
    monkeypatch.setitem(smoke.CONFIG, "output", str(output))
    monkeypatch.setattr(smoke, "preflight", lambda *args: None)
    monkeypatch.setattr(smoke.subprocess, "check_output", lambda *args, **kwargs: b"")

    def unexpected(*args, **kwargs):
        pytest.fail("output reuse reached the request controller")

    monkeypatch.setattr(smoke, "case", unexpected)
    with pytest.raises(FileExistsError):
        smoke.execute(tmp_path, b'{"authorization":"approved"}')
