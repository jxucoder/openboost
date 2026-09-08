"""Verify the fixed smoke harness with local protocol fixtures, never real usage."""

import copy
import json
import time

import pytest
from benchmarks.v1.authoring import accounting_smoke as smoke
from benchmarks.v1.authoring import responses_transport as wire


def response(request, identity, status, outputs=None):
    return dict(
        object="response",
        service_tier="default",
        id=identity,
        model=request["model"],
        max_output_tokens=request["max_output_tokens"],
        status=status,
        incomplete_details={"reason": "max_output_tokens"} if status == "incomplete" else None,
        usage=None
        if outputs is None
        else dict(
            input_tokens=40,
            output_tokens=outputs,
            total_tokens=40 + outputs,
            output_tokens_details={"reasoning_tokens": outputs},
        ),
    )


def test_exhaustion_uses_real_controller_caps_and_blocks_third_dispatch(tmp_path, monkeypatch):
    caps = []

    def call(operation, directory, deadline, response_id=None):
        assert operation == "create"
        request = json.loads((directory / "request.json").read_text())
        assert request["service_tier"] == "default" and request["tools"] == []
        caps.append(request["max_output_tokens"])
        return response(request, f"resp_fixture{len(caps)}", "incomplete", caps[-1])

    monkeypatch.setattr(wire, "call", call)
    result = smoke.case(tmp_path, "exhaustion")
    assert caps == [128, 64]
    assert result["status"] == "pass" and result["generated_tokens"] == 192
    assert result["subsequent_request_blocked"] is True


@pytest.mark.parametrize("outputs", [None, 2])
def test_cancellation_separates_acknowledgement_from_final_usage(tmp_path, monkeypatch, outputs):
    now = [time.monotonic()]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    calls = []
    saved = []

    def call(operation, directory, deadline, response_id=None):
        calls.append(operation)
        if operation == "create":
            saved.append(json.loads((directory / "request.json").read_text()))
            now[0] += 6
            return response(saved[0], "resp_fixture", "in_progress")
        return response(saved[0], "resp_fixture", "cancelled", outputs)

    monkeypatch.setattr(wire, "call", call)
    result = smoke.case(tmp_path, "cancellation")
    assert calls == ["create", "cancel", "retrieve"]
    assert result["cancellation_observed"] is True
    assert result["status"] == ("pass" if outputs is not None else "fail")
    assert result["generated_tokens"] == outputs


def test_fast_completion_does_not_pass_cancellation_gate(tmp_path, monkeypatch):
    def call(operation, directory, deadline, response_id=None):
        request = json.loads((directory / "request.json").read_text())
        return response(request, "resp_fixture", "completed", 2)

    monkeypatch.setattr(wire, "call", call)
    result = smoke.case(tmp_path, "cancellation")
    assert result["status"] == "fail" and result["cancellation_observed"] is False


def test_early_completion_is_failure_without_speculative_extra_requests(tmp_path, monkeypatch):
    calls = []

    def call(operation, directory, deadline, response_id=None):
        calls.append(True)
        request = json.loads((directory / "request.json").read_text())
        return response(request, f"resp_fixture{len(calls)}", "completed", 1)

    monkeypatch.setattr(wire, "call", call)
    result = smoke.case(tmp_path, "exhaustion")
    assert len(calls) == 2
    assert result["status"] == "fail" and result["subsequent_request_blocked"] is False


def test_changed_service_tier_stops_before_another_generation(tmp_path, monkeypatch):
    calls = []

    def call(operation, directory, deadline, response_id=None):
        calls.append(True)
        request = json.loads((directory / "request.json").read_text())
        value = response(
            request, f"resp_fixture{len(calls)}", "incomplete", request["max_output_tokens"]
        )
        value["service_tier"] = "priority"
        return value

    monkeypatch.setattr(wire, "call", call)
    result = smoke.case(tmp_path, "exhaustion")
    assert len(calls) == 1 and result["status"] == "fail"


def packet(root):
    return dict(
        schema="openboost-accounting-smoke-v1",
        authorization="pending",
        settings=copy.deepcopy(smoke.SETTINGS),
        prompts=list(smoke.PROMPTS),
        prompt_sha256=[smoke.digest(text.encode("ascii")) for text in smoke.PROMPTS],
        files={name: smoke.digest((root / name).read_bytes()) for name in smoke.FILES},
    )


def test_preflight_verifies_every_source_and_uses_no_network(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("preflight reached network or credential transport")

    monkeypatch.setattr(wire, "call", unexpected)
    freeze = packet(smoke.ROOT)
    result = smoke.preflight(smoke.ROOT, freeze)
    assert result["network_used"] is False
    assert result["output_tokens_max"] == 4288
    assert result["estimated_token_cost_upper_usd"] < 0.01
    freeze["files"][smoke.FILES[0]] = "0" * 64
    with pytest.raises(ValueError, match="source changed"):
        smoke.preflight(smoke.ROOT, freeze)


@pytest.mark.parametrize("authorization", ["pending", "consumed"])
def test_pending_or_consumed_packet_cannot_dispatch(tmp_path, authorization, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("unapproved execution reached preflight or HTTP")

    monkeypatch.setattr(smoke, "preflight", unexpected)
    monkeypatch.setattr(wire, "call", unexpected)
    with pytest.raises(ValueError, match="approval"):
        smoke.execute(tmp_path, {"authorization": authorization}, b"{}", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_run_stops_after_failed_case_and_refuses_output_reuse(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "fixture.py").write_text("# Local fixture\n")
    output = tmp_path / "output"
    monkeypatch.setattr(smoke, "FILES", ("fixture.py",))
    monkeypatch.setitem(smoke.SETTINGS, "output", str(output))
    freeze = packet(root)
    freeze["authorization"] = "approved"
    monkeypatch.setattr(smoke, "preflight", lambda *args: None)
    monkeypatch.setattr(smoke.platform, "platform", lambda: "local-protocol-fixture")
    monkeypatch.setattr(
        smoke.subprocess,
        "check_output",
        lambda args, **kw: (b"" if "status" in args else "fixture-revision"),
    )
    calls = []

    def case(*args):
        calls.append(args[1])
        return dict(status="fail", generated_tokens=None)

    monkeypatch.setattr(smoke, "case", case)
    result = smoke.execute(root, freeze, json.dumps(freeze).encode(), output)
    assert result["status"] == "fail" and calls == ["exhaustion"]
    assert (output / "archive.json").exists()
    with pytest.raises(FileExistsError):
        smoke.execute(root, freeze, json.dumps(freeze).encode(), output)
    assert calls == ["exhaustion"]
