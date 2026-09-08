"""Local protocol fixtures; no provider usage or independent author evidence."""

import json
import time

import pytest
from benchmarks.v1.authoring import responses_background as bg
from benchmarks.v1.authoring import responses_transport as wire
from benchmarks.v1.authoring.accounting import Controller, Limits
from benchmarks.v1.authoring.responses_background import Background


def reply(status, outputs=None, identity="resp_fixture"):
    return dict(
        object="response",
        id=identity,
        model="fixture-model",
        max_output_tokens=8,
        status=status,
        usage=None
        if outputs is None
        else dict(
            input_tokens=5,
            output_tokens=outputs,
            total_tokens=5 + outputs,
            output_tokens_details={"reasoning_tokens": outputs},
        ),
    )


def test_deadline_cancels_retrieves_and_reconciles_without_releasing_work(tmp_path, monkeypatch):
    calls = []
    now = [time.monotonic()]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    def call(operation, directory, deadline, response_id=None):
        calls.append((operation, response_id))
        if operation == "create":
            payload = json.loads((directory / "request.json").read_text())
            assert payload["background"] is True and payload["store"] is False
            now[0] += 31
            return reply("in_progress")
        if operation == "cancel":
            return reply("cancelled")
        return reply("cancelled", 3)

    background = Background(call=call)
    gate = Controller(
        tmp_path / "attempt",
        model="fixture-model",
        reasoning_effort="medium",
        limits=Limits(10, 8, 3, 30),
        background=True,
        transport=background,
    )
    with pytest.raises(TimeoutError):
        gate.request("fixture input")
    assert calls == [("create", None), ("cancel", "resp_fixture"), ("retrieve", "resp_fixture")]
    assert gate.record["generated_tokens"] == 3
    assert gate.record["reserved_output_tokens"] == 0
    assert gate.record["status"] == "wall_limit"
    with pytest.raises(RuntimeError, match="closed"):
        gate.request("must not generate")


@pytest.mark.parametrize("outcome", ["missing_usage", "race", "wrong_id", "cancel_failure"])
def test_interrupted_poll_has_one_cleanup_and_never_releases_work(tmp_path, outcome):
    calls = []

    def call(operation, directory, deadline, response_id=None):
        calls.append(operation)
        if len(calls) == 1:
            return reply("in_progress")
        if len(calls) == 2:
            raise TimeoutError("fixture poll interruption")
        if operation == "cancel":
            if outcome == "cancel_failure":
                raise RuntimeError("fixture HTTP error")
            return reply("cancelled")
        if outcome == "missing_usage":
            return reply("cancelled")
        if outcome == "wrong_id":
            return reply("cancelled", 3, "resp_someone_else")
        return reply("completed" if outcome == "race" else "cancelled", 3)

    gate = Controller(
        tmp_path / "attempt",
        model="fixture-model",
        reasoning_effort="medium",
        limits=Limits(10, 8, 3, 30),
        background=True,
        transport=Background(call=call),
    )
    with pytest.raises((RuntimeError, ValueError)):
        gate.request("fixture input")
    assert calls == ["create", "retrieve", "cancel", "retrieve"]
    known = outcome in ("race", "cancel_failure")
    assert gate.record["generated_tokens"] == (3 if known else None)
    assert gate.record["reserved_output_tokens"] == (0 if known else 8)
    assert gate.record["status"] == ("transport_stopped" if known else "usage_unknown")
    with pytest.raises(RuntimeError, match="closed"):
        gate.request("must not generate")
    assert len(calls) == 4


def test_unknown_create_id_never_causes_a_second_generation(tmp_path):
    calls = []

    def call(*args):
        calls.append(True)
        raise TimeoutError("fixture lost creation receipt")

    gate = Controller(
        tmp_path / "attempt",
        model="fixture-model",
        reasoning_effort="medium",
        limits=Limits(10, 8, 3, 30),
        background=True,
        transport=Background(call=call),
    )
    with pytest.raises(ValueError):
        gate.request("fixture input")
    assert calls == [True]
    assert gate.record["status"] == "usage_unknown"
    receipt = json.loads((gate.directory / "request-001/background.json").read_text())
    assert receipt["response_id"] is None and receipt["final_status"] is None


def test_background_completion_has_no_cancel_and_counts_once(tmp_path, monkeypatch):
    monkeypatch.setattr(bg, "POLL_INTERVAL_S", 0)
    calls = []

    def call(operation, directory, deadline, response_id=None):
        calls.append(operation)
        return reply("queued") if len(calls) == 1 else reply("completed", 3)

    gate = Controller(
        tmp_path / "attempt",
        model="fixture-model",
        reasoning_effort="medium",
        limits=Limits(10, 8, 3, 30),
        background=True,
        transport=Background(call=call),
    )
    assert gate.request("fixture input")["status"] == "completed"
    assert calls == ["create", "retrieve"]
    assert gate.record["generated_tokens"] == 3 and gate.record["status"] == "ready"


def test_cleanup_has_one_shared_deadline_and_retains_unknown_usage(tmp_path, monkeypatch):
    now = [time.monotonic()]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(bg, "POLL_INTERVAL_S", 0)
    calls = []

    def call(operation, directory, deadline, response_id=None):
        calls.append(operation)
        if operation == "create":
            now[0] += 31
            return reply("in_progress")
        assert operation == "cancel" and deadline <= now[0] + bg.CLEANUP_S
        now[0] += bg.CLEANUP_S
        raise TimeoutError("fixture cleanup timeout")

    gate = Controller(
        tmp_path / "attempt",
        model="fixture-model",
        reasoning_effort="medium",
        limits=Limits(10, 8, 3, 30),
        background=True,
        transport=Background(call=call),
    )
    with pytest.raises(ValueError):
        gate.request("fixture input")
    assert calls == ["create", "cancel"]
    assert gate.record["generated_tokens"] is None
    assert gate.record["reserved_output_tokens"] == 8


@pytest.mark.parametrize("operation", ["retrieve", "cancel"])
def test_https_control_calls_use_only_the_frozen_host_and_response_route(
    tmp_path, monkeypatch, operation
):
    calls = []

    class Connection:
        def __init__(self, host, timeout):
            assert host == "api.openai.com" and timeout == 10

        def request(self, method, path, body, headers):
            assert method == ("GET" if operation == "retrieve" else "POST")
            assert path == "/v1/responses/resp_fixture" + (
                "/cancel" if operation == "cancel" else ""
            )
            assert body is None
            calls.append(True)

        def getresponse(self):
            class Reply:
                status = 200

                def getheader(self, name):
                    return "fixture-request-id"

                def read1(self, size):
                    return b""

            return Reply()

        def close(self):
            pass

    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret-never-real")
    monkeypatch.setattr(wire.http.client, "HTTPSConnection", Connection)
    assert wire.exchange(tmp_path, 10, operation, "resp_fixture") == 0
    assert calls == [True]


def test_response_id_cannot_change_the_control_endpoint(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid route reached the HTTP client")

    monkeypatch.setattr(wire.http.client, "HTTPSConnection", unexpected)
    for identity in ("resp_x/../../models", "https://example.com", "resp_x?query=1", None):
        assert wire.exchange(tmp_path, 10, "cancel", identity) == 1
