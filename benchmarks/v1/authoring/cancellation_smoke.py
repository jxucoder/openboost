"""One separately authorized active-response cancellation observation."""

import argparse
import hashlib
import json
import os
import platform
import ssl
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.v1.authoring.accounting import Controller, Limits, persist
from benchmarks.v1.authoring.responses_background import (
    CLEANUP_S,
    HTTP_TIMEOUT_S,
    MAX_POLLS,
    POLL_INTERVAL_S,
)
from benchmarks.v1.authoring.responses_transport import decode

ROOT = Path(__file__).resolve().parents[3]
FREEZE = ROOT / "v1-sprints/100-cancellation-smoke.json"
PREDECESSOR = "benchmarks/v1/evidence/author-accounting-099"
FILES = (
    "benchmarks/__init__.py",
    "benchmarks/v1/__init__.py",
    "benchmarks/v1/authoring/accounting.py",
    "benchmarks/v1/authoring/responses_transport.py",
    "benchmarks/v1/authoring/responses_background.py",
    "benchmarks/v1/authoring/cancellation_smoke.py",
    "tests/v1/test_author_accounting.py",
    "tests/v1/test_author_background.py",
    "tests/v1/test_author_active_cancellation.py",
    "v1-sprints/100-active-cancellation.md",
    "pyproject.toml",
    "uv.lock",
    f"{PREDECESSOR}/archive-index.json",
    f"{PREDECESSOR}/cancellation/request-001/request.json",
    f"{PREDECESSOR}/cancellation/request-001/response.json",
    f"{PREDECESSOR}/cancellation/request-001/operation-001-create/response.bin",
    f"{PREDECESSOR}/cancellation/request-001/operation-002-retrieve/response.bin",
)
CONFIG = dict(
    model="gpt-5.6-luna",
    reasoning_effort="medium",
    immutable_snapshot=False,
    service_tier="default",
    stop_on_in_progress=True,
    background=True,
    store=False,
    limits=dict(generated_tokens=4096, request_tokens=4096, requests=1, wall_s=5),
    cleanup_s=15.0,
    http_timeout_s=10.0,
    max_polls=60,
    poll_interval_s=1.0,
    input_tokens_review_bound=4096,
    input_usd_per_million_upper_estimate=0.25,
    output_usd_per_million=1.20,
    requested_cost_allowance_usd=0.01,
    retries=0,
    output="/tmp/openboost-author-cancellation-100",
    prompt="Write every integer from 1 through 100000, one per line, without abbreviation. "
    "Do not add an introduction or commentary. Continue until the entire list is written.",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def preflight(root, freeze):
    if (
        freeze.get("schema") != "openboost-active-cancellation-smoke-v1"
        or freeze.get("authorization") not in ("pending", "approved", "consumed")
        or freeze.get("config") != CONFIG
        or freeze.get("prompt_sha256") != digest(CONFIG["prompt"].encode("ascii"))
        or set(freeze.get("files", {})) != set(FILES)
        or (CLEANUP_S, HTTP_TIMEOUT_S, MAX_POLLS, POLL_INTERVAL_S) != (15.0, 10.0, 60, 1.0)
    ):
        raise ValueError("active cancellation packet differs from the fixed configuration")
    for name in FILES:
        path = root / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("frozen sources must be resolved regular paths")
        if digest(path.read_bytes()) != freeze["files"][name]:
            raise ValueError(f"frozen source changed: {name}")
    previous = decode((root / PREDECESSOR / "cancellation/request-001/request.json").read_bytes())
    if (
        previous["input"] != CONFIG["prompt"]
        or previous["model"] != CONFIG["model"]
        or previous["max_output_tokens"] != CONFIG["limits"]["request_tokens"]
    ):
        raise ValueError("model, prompt or output cap differs from the predecessor probe")
    estimate = (
        CONFIG["input_tokens_review_bound"] * CONFIG["input_usd_per_million_upper_estimate"]
        + CONFIG["limits"]["generated_tokens"] * CONFIG["output_usd_per_million"]
    ) / 1_000_000
    if (
        len(CONFIG["prompt"].encode("ascii")) > 1024
        or estimate > CONFIG["requested_cost_allowance_usd"]
    ):
        raise ValueError("prompt or conservative cost estimate exceeds the packet allowance")
    return dict(
        status="preflight_pass",
        authorization=freeze["authorization"],
        source_files=len(FILES),
        generation_requests_max=1,
        output_tokens_max=4096,
        estimated_token_cost_upper_usd=round(estimate, 10),
        network_used=False,
    )


def request(gate, text):
    try:
        gate.request(text)
        return dict(returned=True)
    except Exception as error:
        return dict(returned=False, error_type=type(error).__name__)


def classify(directory, ledger, outcome, blocked):
    def read(path):
        target = directory / path
        return decode(target.read_bytes()) if target.exists() else None

    receipt = read("request-001/background.json") or {}
    response = read("request-001/response.json") or {}
    operations = receipt.get("operations", [])
    trigger = receipt.get("stop_trigger")
    active = False
    if trigger and trigger.get("kind") == "observed_in_progress":
        number = trigger.get("operation")
        if type(number) is int and 1 <= number <= len(operations):
            origin = operations[number - 1]["operation"]
            seen = read(f"request-001/operation-{number:03d}-{origin}/response.json") or {}
            active = (
                seen.get("status") == "in_progress"
                and seen.get("id") == trigger.get("response_id") == receipt.get("response_id")
                and trigger.get("remaining_work_s", 0) > 0
            )
    clean_stop = (
        receipt.get("stopped") is True
        and [op["operation"] for op in operations[-2:]] == ["cancel", "retrieve"]
        and all(op["status"] == "received" for op in operations[-2:])
        and not any(k in receipt for k in ("cancel_error_type", "retrieve_error_type"))
    )
    cancelled = response.get("status") == "cancelled"
    entries = ledger["requests"]
    counted = entries[0]["usage"] if len(entries) == 1 else None
    usage_known = ledger["generated_tokens"] is not None and counted is not None
    bounds_ok = (
        usage_known
        and counted["input_tokens"] <= CONFIG["input_tokens_review_bound"]
        and response.get("service_tier") == "default"
    )
    passed = (
        active
        and clean_stop
        and receipt.get("stop_type") == "ActiveResponseStop"
        and cancelled
        and bounds_ok
        and blocked
        and not outcome["returned"]
        and ledger["reserved_output_tokens"] == 0
        and ledger["status"] in ("transport_stopped", "wall_limit")
    )
    return dict(
        status="pass" if passed else "fail",
        active_trigger_observed=active,
        cancellation_observed=cancelled,
        cleanup_confirmed=clean_stop,
        provider_final_status=response.get("status"),
        final_usage_known=usage_known,
        bounds_ok=bounds_ok,
        outcome=outcome,
        subsequent_request_blocked=blocked,
        controller_status=ledger["status"],
        generated_tokens=ledger["generated_tokens"],
        observed_output_tokens=ledger["observed_output_tokens"],
        reserved_output_tokens=ledger["reserved_output_tokens"],
    )


def case(output):
    gate = Controller(
        output / "attempt",
        model=CONFIG["model"],
        reasoning_effort=CONFIG["reasoning_effort"],
        limits=Limits(**CONFIG["limits"]),
        background=True,
        stop_on_in_progress=True,
    )
    outcome = request(gate, CONFIG["prompt"])
    blocked = False
    if gate.record["status"] != "ready":
        before = len(gate.record["requests"])
        denied = request(gate, "This request must be rejected locally.")
        blocked = not denied["returned"] and len(gate.record["requests"]) == before
    result = classify(gate.directory, gate.record, outcome, blocked)
    persist(gate.directory / "result.json", result)
    return result


def execute(root, raw):
    freeze = decode(raw)
    if freeze.get("authorization") != "approved":
        raise ValueError("this separate live cancellation smoke requires user approval")
    preflight(root, freeze)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root):
        raise ValueError("live cancellation smoke requires a clean committed source tree")
    output = Path(CONFIG["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "freeze.json").write_bytes(raw)
    for name in FILES:
        data = (root / name).read_bytes()
        if digest(data) != freeze["files"][name]:
            raise ValueError("source changed while archiving")
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    started = time.monotonic()
    run = dict(
        schema="openboost-active-cancellation-result-v1",
        status="running",
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        dirty=False,
        freeze_sha256=digest(raw),
        python=sys.version,
        platform=platform.platform(),
        openssl=ssl.OPENSSL_VERSION,
        cpu_count=os.cpu_count(),
        gpu=None,
        started_utc=datetime.now(timezone.utc).isoformat(),
        argv=list(sys.argv),
        independent_author=False,
        generated_tokens=None,
    )
    persist(output / "run.json", run)
    try:
        run["case"] = case(output)
        run.update(status=run["case"]["status"], generated_tokens=run["case"]["generated_tokens"])
    except BaseException as error:
        run.update(status="fail", error_type=type(error).__name__)
        raise
    finally:
        run["elapsed_s"] = time.monotonic() - started
        persist(output / "run.json", run)
        persist(
            output / "archive.json",
            dict(
                files={
                    p.relative_to(output).as_posix(): digest(p.read_bytes())
                    for p in sorted(output.rglob("*"))
                    if p.is_file() and p.name != "archive.json"
                }
            ),
        )
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    raw = FREEZE.read_bytes()
    result = execute(ROOT, raw) if args.execute else preflight(ROOT, decode(raw))
    print(json.dumps(result, indent=2))
    raise SystemExit(int(result["status"] == "fail"))
