"""Bounded background polling and one cancellation/reconciliation sequence."""

import time

from benchmarks.v1.authoring import responses_transport as wire
from benchmarks.v1.authoring.accounting import persist

TERMINAL = {"completed", "incomplete", "failed", "cancelled"}
POLL_INTERVAL_S = 1.0
MAX_POLLS = 60
HTTP_TIMEOUT_S = 10.0
CLEANUP_S = 15.0


class ActiveResponseStop(RuntimeError):
    """The trusted caller requested stop on a validated active observation."""


class Background:
    """Trusted transport, not an independent runner or proof of provider expiry.

    There is one create, at most MAX_POLLS reads, then at most one cancellation
    and one final read if stopped. Cleanup never creates another generation.
    An injected call is only a local protocol test.
    """

    def __init__(self, *, call=None, stop_on_in_progress=False):
        if type(stop_on_in_progress) is not bool:
            raise ValueError("stop_on_in_progress must be a boolean")
        self.call = wire.call if call is None else call
        self.stop_on_in_progress = stop_on_in_progress

    def __call__(self, request, directory, deadline):
        if request.get("background") is not True or request.get("store") is not False:
            raise ValueError("background=true and store=false required")
        if wire.decode((directory / "request.json").read_bytes()) != request:
            raise ValueError("persisted request differs from reserved background request")
        started = time.monotonic()
        record = dict(
            schema="openboost-background-operations-v1",
            transport="responses_https" if self.call is wire.call else "injected_protocol_test",
            response_id=None,
            operations=[],
            stopped=False,
            stop_on_in_progress=self.stop_on_in_progress,
            stop_trigger=None,
        )
        response_id = None
        last = None

        def save():
            persist(directory / "background.json", record)

        def operation(name, until):
            if time.monotonic() >= until:
                raise TimeoutError("operation deadline reached")
            entry = dict(operation=name, response_id=response_id, status="reserved")
            record["operations"].append(entry)
            target = directory / f"operation-{len(record['operations']):03d}-{name}"
            target.mkdir()
            persist(target / "operation.json", entry)
            if name == "create":
                persist(target / "request.json", request)
            save()
            try:
                value = self.call(
                    name, target, min(until, time.monotonic() + HTTP_TIMEOUT_S), response_id
                )
                # Save the provider value even if identity/status validation later fails.
                persist(target / "response.json", value)
                entry["status"] = "received"
                save()
                return value
            except BaseException as error:
                entry.update(status="failed", error_type=type(error).__name__)
                save()
                raise

        def validate(value):
            if (
                not isinstance(value, dict)
                or value.get("object") != "response"
                or value.get("id") != response_id
                or value.get("model") != request["model"]
                or value.get("max_output_tokens") != request["max_output_tokens"]
                or value.get("status") not in TERMINAL | {"queued", "in_progress"}
            ):
                raise ValueError("background response identity, cap or status mismatch")
            return value

        try:
            first = operation("create", deadline)
            if isinstance(first, dict):
                wire.route("retrieve", first.get("id"))
                response_id = first["id"]
                record["response_id"] = response_id
                save()
            last = validate(first)
            polls = 0
            while last["status"] not in TERMINAL:
                if self.stop_on_in_progress and last["status"] == "in_progress":
                    now = time.monotonic()
                    if now >= deadline:
                        raise TimeoutError("active observation arrived after the work deadline")
                    record["stop_trigger"] = dict(
                        kind="observed_in_progress",
                        response_id=response_id,
                        operation=len(record["operations"]),
                        elapsed_s=now - started,
                        remaining_work_s=deadline - now,
                    )
                    save()
                    raise ActiveResponseStop("active response stop requested")
                if polls >= MAX_POLLS:
                    raise TimeoutError("background polling allowance exhausted")
                time.sleep(max(0, min(POLL_INTERVAL_S, deadline - time.monotonic())))
                last = validate(operation("retrieve", deadline))
                polls += 1
            return last
        except BaseException as error:
            record.update(stopped=True, stop_type=type(error).__name__)
            # A known ID permits cleanup after an HTTP timeout or malformed poll.
            # A create failure without an ID cannot be confirmed cancelled.
            cleanup_deadline = time.monotonic() + CLEANUP_S
            if response_id is not None:
                for name in ("cancel", "retrieve"):
                    try:
                        last = validate(operation(name, cleanup_deadline))
                    except BaseException as cleanup_error:
                        record[f"{name}_error_type"] = type(cleanup_error).__name__
            record["final_status"] = last.get("status") if last is not None else None
            save()
            raise wire.TransportStopped(last) from None
