"""Sequential, fail-closed request accounting; not a complete author runner."""

import hashlib
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.v1.authoring.responses_transport import TransportStopped, send


@dataclass(frozen=True)
class Limits:
    generated_tokens: int
    request_tokens: int
    requests: int
    wall_s: float

    def __post_init__(self):
        if any(
            type(x) is not int or x <= 0
            for x in (self.generated_tokens, self.request_tokens, self.requests)
        ):
            raise ValueError("positive integer token and request limits required")
        if self.request_tokens > self.generated_tokens or self.generated_tokens > 20_000:
            raise ValueError("request cap must fit the unchanged 20k formal ceiling")
        if (
            type(self.wall_s) not in (int, float)
            or not math.isfinite(self.wall_s)
            or not 0 < self.wall_s <= 1800
        ):
            raise ValueError("wall limit must fit the unchanged 1800-second ceiling")


def persist(path, record):
    """Complete a durable trusted-side record before allowing network dispatch."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def usage(response, cap, model, seen):
    if (
        not isinstance(response, dict)
        or response.get("object") != "response"
        or not isinstance(response.get("id"), str)
        or not response["id"]
        or response["id"] in seen
        or response.get("model") != model
        or response.get("max_output_tokens") != cap
        or response.get("status") not in ("completed", "incomplete", "failed", "cancelled")
    ):
        raise ValueError("missing, duplicate, nonterminal or mismatched response identity/cap")
    counts = response.get("usage")
    if not isinstance(counts, dict):
        raise ValueError("response usage unavailable")
    details = counts.get("output_tokens_details")
    if not isinstance(details, dict):
        raise ValueError("reasoning usage unavailable")
    values = [counts.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")]
    values.append(details.get("reasoning_tokens"))
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError("invalid token counts")
    inputs, outputs, total, reasoning = values
    if total != inputs + outputs or reasoning > outputs or outputs > cap:
        raise ValueError("inconsistent or over-cap token usage")
    return dict(input_tokens=inputs, output_tokens=outputs, reasoning_tokens=reasoning)


class Controller:
    """Trusted text-request boundary. Candidate code must never own this object.

    A fresh directory never resumes an attempt. Future dispatch integration must
    additionally prevent the same attempt from being relaunched in another one.
    Injected transports support protocol tests, not measured provider evidence.
    """

    def __init__(
        self, directory, *, model, reasoning_effort, limits, transport=None, background=False
    ):
        if (
            not isinstance(model, str)
            or not model.strip()
            or not isinstance(reasoning_effort, str)
            or not reasoning_effort.strip()
            or not isinstance(limits, Limits)
            or type(background) is not bool
        ):
            raise ValueError("explicit model, reasoning setting and limits required")
        self.started = time.monotonic()
        self.deadline = self.started + limits.wall_s
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.limits = limits
        self.background = background
        if transport is None and background:
            from benchmarks.v1.authoring.responses_background import Background

            self.transport = Background()
        else:
            self.transport = send if transport is None else transport
        self.lock = threading.Lock()
        self.record = dict(
            schema="openboost-author-request-accounting-v1",
            model=model,
            reasoning_effort=reasoning_effort,
            limits=asdict(limits),
            transport="responses_https" if transport is None else "injected_protocol_test",
            background=background,
            dispatch_ready=False,
            status="ready",
            generated_tokens=None,
            observed_output_tokens=0,
            reserved_output_tokens=0,
            requests=[],
        )
        self._save()

    def _save(self):
        self.record["elapsed_s"] = time.monotonic() - self.started
        persist(self.directory / "ledger.json", self.record)

    def _stop(self, status):
        self.record["status"] = status
        self._save()

    def request(self, text):
        with self.lock:
            if self.record["status"] != "ready":
                raise RuntimeError(f"request controller closed: {self.record['status']}")
            if time.monotonic() >= self.deadline:
                self._stop("wall_limit")
                raise TimeoutError("attempt deadline reached before dispatch")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("nonempty text input required for this slice")
            remaining = self.limits.generated_tokens - self.record["observed_output_tokens"]
            cap = min(self.limits.request_tokens, remaining)
            request = dict(
                model=self.record["model"],
                reasoning={"effort": self.record["reasoning_effort"]},
                input=text,
                max_output_tokens=cap,
                tools=[],
                parallel_tool_calls=False,
                stream=False,
                background=self.background,
                store=False,
            )
            entry = dict(
                sequence=len(self.record["requests"]) + 1,
                status="reserved",
                max_output_tokens=cap,
                usage=None,
            )
            self.record["requests"].append(entry)
            self.record.update(
                status="in_flight", reserved_output_tokens=cap, generated_tokens=None
            )
            directory = self.directory / f"request-{entry['sequence']:03d}"
            try:
                directory.mkdir()
                persist(directory / "request.json", request)
                entry["request_sha256"] = hashlib.sha256(
                    (directory / "request.json").read_bytes()
                ).hexdigest()
                self._save()
                # Persist first, then check the same total deadline in the real transport.
                if time.monotonic() >= self.deadline:
                    self._stop("wall_limit")
                    raise TimeoutError("attempt deadline reached while persisting request")
                stopped = False
                try:
                    response = self.transport(request, directory, self.deadline)
                except TransportStopped as interruption:
                    response = interruption.response
                    stopped = True
                persist(directory / "response.json", response)
                entry["response_sha256"] = hashlib.sha256(
                    (directory / "response.json").read_bytes()
                ).hexdigest()
                seen = {r.get("response_id") for r in self.record["requests"][:-1]}
                counted = usage(response, cap, self.record["model"], seen)
                entry.update(
                    response_id=response["id"], usage=counted, status=response.get("status")
                )
                self.record["observed_output_tokens"] += counted["output_tokens"]
                self.record.update(
                    generated_tokens=self.record["observed_output_tokens"], reserved_output_tokens=0
                )
                terminal = response.get("status") == "completed" or (
                    response.get("status") == "incomplete"
                    and response.get("incomplete_details") == {"reason": "max_output_tokens"}
                )
                if time.monotonic() >= self.deadline:
                    self._stop("wall_limit")
                    raise TimeoutError("response arrived after the deadline")
                if stopped:
                    self._stop("transport_stopped")
                    raise RuntimeError("background transport stopped; answer withheld")
                if not terminal:
                    self._stop("response_failure")
                    raise RuntimeError("response did not complete or reach its output cap")
                if self.record["observed_output_tokens"] == self.limits.generated_tokens:
                    self._stop("token_limit")
                elif len(self.record["requests"]) == self.limits.requests:
                    self._stop("request_limit")
                else:
                    self._stop("ready")
                return response
            except BaseException as error:
                if self.record["status"] not in (
                    "wall_limit",
                    "response_failure",
                    "transport_stopped",
                ):
                    if entry["usage"] is None:
                        self.record.update(status="usage_unknown", generated_tokens=None)
                        entry["status"] = "usage_unknown"
                    else:
                        self.record["status"] = "controller_failure"
                entry["error_type"] = type(error).__name__
                self._save()
                raise
