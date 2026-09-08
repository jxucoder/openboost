"""Local protocol fixtures are not real model usage or author-cost evidence."""

import json
import os
import subprocess
import sys
import time

import pytest
from benchmarks.v1.authoring import accounting
from benchmarks.v1.authoring import responses_transport as wire
from benchmarks.v1.authoring.accounting import Controller, Limits


def response(used=7, identifier="fixture-1", cap=8):
    return {
        "id": identifier,
        "object": "response",
        "model": "fixture-model",
        "max_output_tokens": cap,
        "status": "completed",
        "output": [],
        "usage": {
            "input_tokens": 1,
            "output_tokens": used,
            "output_tokens_details": {"reasoning_tokens": used},
            "total_tokens": used + 1,
        },
    }


def controller(tmp_path, transport):
    return Controller(
        tmp_path / "attempt",
        model="fixture-model",
        reasoning_effort="medium",
        limits=Limits(generated_tokens=10, request_tokens=8, requests=3, wall_s=30),
        transport=transport,
    )


def test_remaining_cap_is_reserved_before_transport_and_exhaustion_stops_calls(tmp_path):
    calls = []

    def transport(request, directory, deadline):
        saved = json.loads((directory.parent / "ledger.json").read_text())
        assert saved["status"] == "in_flight"
        assert saved["generated_tokens"] is None
        assert saved["reserved_output_tokens"] == request["max_output_tokens"]
        assert json.loads((directory / "request.json").read_text()) == request
        calls.append(request["max_output_tokens"])
        used = 7 if len(calls) == 1 else 3
        return response(used, f"fixture-{len(calls)}", request["max_output_tokens"])

    controller = Controller(
        tmp_path / "attempt",
        model="fixture-model",
        reasoning_effort="medium",
        limits=Limits(generated_tokens=10, request_tokens=8, requests=3, wall_s=30),
        transport=transport,
    )
    controller.request("fixture input")
    controller.request("fixture continuation")
    assert calls == [8, 3]
    assert controller.record["generated_tokens"] == 10
    assert controller.record["status"] == "token_limit"
    assert controller.record["reserved_output_tokens"] == 0
    assert controller.record["transport"] == "injected_protocol_test"
    with pytest.raises(RuntimeError, match="closed"):
        controller.request("must not generate")
    assert calls == [8, 3]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "reasoning_missing",
        "double_count",
        "negative",
        "boolean",
        "fraction",
        "over_cap",
        "model",
        "id",
        "cap",
        "in_progress",
    ],
)
def test_invalid_usage_stays_unknown_and_stops_further_dispatch(tmp_path, change):
    result = response()
    if change == "missing":
        result["usage"] = None
    elif change == "reasoning_missing":
        result["usage"]["output_tokens_details"] = {}
    elif change == "double_count":
        result["usage"]["total_tokens"] += 7
    elif change == "negative":
        result["usage"]["output_tokens"] = -1
    elif change == "boolean":
        result["usage"]["input_tokens"] = True
    elif change == "fraction":
        result["usage"]["output_tokens_details"]["reasoning_tokens"] = 1.5
    elif change == "over_cap":
        result = response(9)
    elif change == "model":
        result["model"] = "unexpected-model"
    elif change == "id":
        result["id"] = ""
    elif change == "cap":
        result["max_output_tokens"] = 99
    elif change == "in_progress":
        result["status"] = "in_progress"
    calls = []

    def transport(*args):
        calls.append(True)
        return result

    gate = controller(tmp_path, transport)
    with pytest.raises(ValueError):
        gate.request("fixture input")
    saved = json.loads((gate.directory / "ledger.json").read_text())
    assert saved["status"] == "usage_unknown" and saved["generated_tokens"] is None
    assert saved["reserved_output_tokens"] == 8
    assert (gate.directory / "request-001/response.json").exists()
    with pytest.raises(RuntimeError, match="closed"):
        gate.request("must not generate")
    assert len(calls) == 1


def test_duplicate_response_id_cannot_reuse_or_double_count_usage(tmp_path):
    gate = controller(tmp_path, lambda *args: response(2))
    gate.request("first")
    with pytest.raises(ValueError, match="duplicate"):
        gate.request("second")
    assert gate.record["observed_output_tokens"] == 2
    assert gate.record["generated_tokens"] is None
    assert gate.record["reserved_output_tokens"] == 8


def test_incomplete_reasoning_only_output_is_counted_without_visible_text(tmp_path):
    result = response(8)
    result.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    gate = controller(tmp_path, lambda *args: result)
    gate.request("first")
    assert gate.record["generated_tokens"] == 8
    assert gate.record["status"] == "ready"
    assert gate.record["requests"][0]["usage"]["reasoning_tokens"] == 8


def test_failed_terminal_response_retains_known_usage_but_closes_controller(tmp_path):
    result = response(4)
    result["status"] = "failed"
    gate = controller(tmp_path, lambda *args: result)
    with pytest.raises(RuntimeError, match="did not complete"):
        gate.request("first")
    assert gate.record["generated_tokens"] == 4
    assert gate.record["status"] == "response_failure"


def test_unknown_interrupted_request_does_not_erase_preceding_measured_usage(tmp_path):
    calls = []

    def transport(*args):
        calls.append(True)
        if len(calls) == 1:
            return response(2)
        raise TimeoutError("deliberate fixture interruption")

    gate = controller(tmp_path, transport)
    gate.request("first")
    with pytest.raises(TimeoutError):
        gate.request("interrupted")
    assert gate.record["observed_output_tokens"] == 2
    assert gate.record["generated_tokens"] is None
    assert gate.record["reserved_output_tokens"] == 8
    with pytest.raises(RuntimeError, match="closed"):
        gate.request("must not retry")
    assert len(calls) == 2


def test_persistence_failure_prevents_network_dispatch_and_closes_in_memory(tmp_path, monkeypatch):
    calls = []
    gate = controller(tmp_path, lambda *args: calls.append(True))

    def fail(*args):
        raise OSError("deliberate write failure")

    monkeypatch.setattr(accounting, "persist", fail)
    with pytest.raises(OSError):
        gate.request("first")
    assert calls == [] and gate.record["status"] == "usage_unknown"
    with pytest.raises(RuntimeError, match="closed"):
        gate.request("must not generate")


def test_request_count_also_bounds_zero_output_responses(tmp_path):
    calls = []

    def transport(*args):
        calls.append(True)
        return response(0, f"fixture-{len(calls)}")

    gate = controller(tmp_path, transport)
    for _ in range(3):
        gate.request("fixture input")
    assert gate.record["status"] == "request_limit"
    assert gate.record["generated_tokens"] == 0  # Observed protocol fixture, not missing usage.
    with pytest.raises(RuntimeError, match="closed"):
        gate.request("must not generate")
    assert len(calls) == 3


def test_deadline_before_dispatch_does_not_invoke_transport(tmp_path):
    calls = []
    gate = controller(tmp_path, lambda *args: calls.append(True))
    gate.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        gate.request("first")
    assert gate.record["status"] == "wall_limit" and calls == []


def test_late_response_is_retained_and_counted_but_never_released(tmp_path):
    def transport(*args):
        gate.deadline = time.monotonic() - 1
        return response(3)

    gate = controller(tmp_path, transport)
    with pytest.raises(TimeoutError, match="after the deadline"):
        gate.request("first")
    assert gate.record["status"] == "wall_limit" and gate.record["generated_tokens"] == 3


def test_completed_usage_survives_failure_to_write_final_ledger(tmp_path, monkeypatch):
    gate = controller(tmp_path, lambda *args: response(3))
    original = accounting.persist

    def fail_final(path, record):
        if path.name == "ledger.json" and record["status"] == "ready":
            raise OSError("deliberate final ledger failure")
        original(path, record)

    monkeypatch.setattr(accounting, "persist", fail_final)
    with pytest.raises(OSError):
        gate.request("first")
    assert gate.record["generated_tokens"] == 3
    assert gate.record["status"] == "controller_failure"
    with pytest.raises(RuntimeError, match="closed"):
        gate.request("must not generate")


def test_existing_attempt_directory_cannot_be_resumed_with_a_fresh_budget(tmp_path):
    gate = controller(tmp_path, lambda *args: response())
    before = (gate.directory / "ledger.json").read_bytes()
    with pytest.raises(FileExistsError):
        controller(tmp_path, lambda *args: response())
    assert (gate.directory / "ledger.json").read_bytes() == before


def test_default_transport_fails_before_http_without_a_credential(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gate = Controller(
        tmp_path / "attempt",
        model="fixture-model",
        reasoning_effort="medium",
        limits=Limits(generated_tokens=10, request_tokens=8, requests=3, wall_s=30),
    )
    with pytest.raises(RuntimeError, match="transport exited"):
        gate.request("fixture input")
    assert gate.record["transport"] == "responses_https"
    assert gate.record["status"] == "usage_unknown"
    assert len(gate.record["requests"]) == 1
    directory = gate.directory / "request-001"
    assert not (directory / "http.json").exists()
    assert json.loads((directory / "transport-error.json").read_text()) == {
        "error_type": "RuntimeError"
    }


@pytest.mark.parametrize("status", [302, 429, 500])
def test_http_redirect_or_failure_is_not_followed_or_retried(tmp_path, monkeypatch, status):
    request = {"input": "fixture", "max_output_tokens": 1}
    (tmp_path / "request.json").write_text(json.dumps(request))
    calls = []

    def supervise(*args):
        calls.append(True)
        (tmp_path / "http.json").write_text(json.dumps({"status": status}))

    monkeypatch.setattr(wire, "supervise", supervise)
    with pytest.raises(RuntimeError, match="no retry"):
        wire.send(request, tmp_path, time.monotonic() + 30)
    assert calls == [True]


@pytest.mark.parametrize("raw", ['{"usage":null,"usage":{}}', '{"value":NaN}'])
def test_transport_rejects_duplicate_keys_and_nonfinite_json(raw):
    with pytest.raises(ValueError):
        wire.decode(raw)


def test_actual_local_transport_process_is_killed_at_deadline(tmp_path):
    pid_file = tmp_path / "pid"
    command = [
        sys.executable,
        "-I",
        "-c",
        f"import os,time; open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(60)",
    ]
    with pytest.raises(subprocess.TimeoutExpired):
        wire.supervise(command, tmp_path, time.monotonic() + 1)
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("interrupted", [False, True])
def test_http_transport_uses_one_fixed_endpoint_and_keeps_auth_out_of_artifacts(
    tmp_path, monkeypatch, interrupted
):
    key = "fixture-secret-never-real"
    calls = []
    request = {"input": "fixture", "max_output_tokens": 3}
    (tmp_path / "request.json").write_text(json.dumps(request))
    raw = json.dumps(response(3)).encode()

    class Connection:
        def __init__(self, host, timeout):
            assert host == "api.openai.com" and timeout == 10

        def request(self, method, path, body, headers):
            assert method == "POST" and path == "/v1/responses"
            assert json.loads(body) == request
            assert headers["Authorization"] == f"Bearer {key}"
            calls.append(True)

        def getresponse(self):
            class Reply:
                status = 200
                reads = 0

                def getheader(self, name):
                    assert name == "x-request-id"
                    return "fixture-request-id"

                def read1(self, size):
                    assert size == 65536
                    self.reads += 1
                    if self.reads == 1:
                        return raw
                    if interrupted:
                        raise TimeoutError(key)
                    return b""

            return Reply()

        def close(self):
            pass

    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setattr(wire.http.client, "HTTPSConnection", Connection)
    assert wire.exchange(tmp_path, 10) == int(interrupted) and len(calls) == 1
    assert (tmp_path / "response.bin").read_bytes() == raw
    assert json.loads((tmp_path / "http.json").read_text())["body_complete"] is not interrupted
    assert not any(key.encode() in p.read_bytes() for p in tmp_path.iterdir())


def test_transport_exception_message_cannot_leak_credentials(tmp_path, monkeypatch):
    key = "fixture-secret-never-real"
    monkeypatch.setenv("OPENAI_API_KEY", key)

    def fail(*args, **kwargs):
        raise RuntimeError(key)

    monkeypatch.setattr(wire.http.client, "HTTPSConnection", fail)
    assert wire.exchange(tmp_path, 10) == 1
    assert json.loads((tmp_path / "transport-error.json").read_text()) == {
        "error_type": "RuntimeError"
    }
    assert not any(key.encode() in p.read_bytes() for p in tmp_path.iterdir())
