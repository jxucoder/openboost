"""Preflight or execute one separately authorized, fixed-input model smoke."""

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
FREEZE = ROOT / "v1-sprints/099-accounting-smoke.json"
FILES = (
    "benchmarks/__init__.py",
    "benchmarks/v1/__init__.py",
    "benchmarks/v1/authoring/accounting.py",
    "benchmarks/v1/authoring/responses_transport.py",
    "benchmarks/v1/authoring/responses_background.py",
    "benchmarks/v1/authoring/accounting_smoke.py",
    "tests/v1/test_author_accounting.py",
    "tests/v1/test_author_background.py",
    "tests/v1/test_author_accounting_smoke.py",
    "v1-sprints/099-background-accounting-smoke.md",
    "pyproject.toml",
    "uv.lock",
)
SETTINGS = dict(
    model="gpt-5.6-luna",
    immutable_snapshot=False,
    reasoning_effort="medium",
    service_tier="default",
    background=True,
    store=False,
    tools=[],
    exhaustion=dict(generated_tokens=192, request_tokens=128, requests=3, wall_s=60),
    cancellation=dict(generated_tokens=4096, request_tokens=4096, requests=1, wall_s=5),
    max_generation_requests=3,
    max_generated_tokens=4288,
    max_prompt_bytes=1024,
    input_tokens_per_response_review_bound=4096,
    input_usd_per_million_upper_estimate=0.25,
    output_usd_per_million=1.20,
    authorized_cost_ceiling_usd=0.05,
    max_polls=60,
    poll_interval_s=1.0,
    http_timeout_s=10.0,
    cleanup_s=15.0,
    retries=0,
    output="/tmp/openboost-author-accounting-099",
)
PROMPTS = [
    "Write the positive integers in order, one per line, starting at 1 and continuing "
    "through 10000. Do not add an introduction, explanation, or conclusion.",
    "List integers one per line from 10001 through 20000. Do not abbreviate or add commentary.",
    "Write every integer from 1 through 100000, one per line, without abbreviation. "
    "Do not add an introduction or commentary. Continue until the entire list is written.",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def preflight(root, freeze):
    """Check the complete local packet without reading credentials or using HTTP."""
    if (
        freeze.get("schema") != "openboost-accounting-smoke-v1"
        or freeze.get("authorization") not in ("pending", "approved", "consumed")
        or freeze.get("settings") != SETTINGS
        or freeze.get("prompts") != PROMPTS
        or freeze.get("prompt_sha256") != [digest(text.encode("ascii")) for text in PROMPTS]
        or set(freeze.get("files", {})) != set(FILES)
        or (MAX_POLLS, POLL_INTERVAL_S, HTTP_TIMEOUT_S, CLEANUP_S) != (60, 1.0, 10.0, 15.0)
    ):
        raise ValueError("accounting smoke configuration differs from the fixed packet")
    for name in FILES:
        path = root / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            # macOS /tmp itself is an OS alias; source trees must still be resolved.
            raise ValueError(f"source path must be resolved without symlinks: {name}")
        if digest(path.read_bytes()) != freeze["files"][name]:
            raise ValueError(f"frozen source changed: {name}")
    sizes = [len(text.encode("ascii")) for text in PROMPTS]
    if max(sizes) > SETTINGS["max_prompt_bytes"]:
        raise ValueError("prompt exceeds the byte limit")
    estimate = (
        3
        * SETTINGS["input_tokens_per_response_review_bound"]
        * SETTINGS["input_usd_per_million_upper_estimate"]
        + SETTINGS["max_generated_tokens"] * SETTINGS["output_usd_per_million"]
    ) / 1_000_000
    if estimate > SETTINGS["authorized_cost_ceiling_usd"]:
        raise ValueError("conservative token-cost estimate exceeds the proposed allowance")
    return dict(
        status="preflight_pass",
        authorization=freeze["authorization"],
        prompt_bytes=sizes,
        estimated_token_cost_upper_usd=estimate,
        generation_requests_max=3,
        output_tokens_max=4288,
        model=SETTINGS["model"],
        immutable_snapshot=False,
        network_used=False,
    )


def request(gate, text):
    try:
        value = gate.request(text)
        return dict(returned=True, response_status=value["status"])
    except Exception as error:
        return dict(returned=False, error_type=type(error).__name__)


def case(root, name):
    gate = Controller(
        root / name,
        model=SETTINGS["model"],
        reasoning_effort=SETTINGS["reasoning_effort"],
        limits=Limits(**SETTINGS[name]),
        background=True,
    )
    outcomes = []
    tiers = []
    for prompt in PROMPTS[:2] if name == "exhaustion" else PROMPTS[2:]:
        outcomes.append(request(gate, prompt))
        entry = gate.record["requests"][-1] if gate.record["requests"] else {}
        usage = entry.get("usage")
        reply_path = root / name / f"request-{len(gate.record['requests']):03d}" / "response.json"
        stored = decode(reply_path.read_bytes()) if reply_path.exists() else None
        tiers.append(stored.get("service_tier") if isinstance(stored, dict) else None)
        if (
            not outcomes[-1]["returned"]
            or usage is None
            or usage["input_tokens"] > SETTINGS["input_tokens_per_response_review_bound"]
            or tiers[-1] != "default"
        ):
            break
    # Exercise rejection only after the controller has closed, never send a
    # speculative extra request to learn whether the token cap was enforced.
    blocked = False
    if gate.record["status"] != "ready":
        count = len(gate.record["requests"])
        denied = request(gate, "This request must be rejected locally.")
        blocked = not denied["returned"] and len(gate.record["requests"]) == count
    record = gate.record
    entries = record["requests"]
    input_bound_ok = bool(entries) and all(
        e["usage"] is not None
        and e["usage"]["input_tokens"] <= SETTINGS["input_tokens_per_response_review_bound"]
        for e in entries
    )
    if name == "exhaustion":
        response_files = sorted((root / name).glob("request-*/response.json"))
        responses = [decode(p.read_bytes()) for p in response_files]
        passed = (
            len(outcomes) == 2
            and all(o["returned"] for o in outcomes)
            and [e["max_output_tokens"] for e in entries] == [128, 64]
            and len(responses) == 2
            and all(
                r["status"] == "incomplete"
                and r.get("incomplete_details") == {"reason": "max_output_tokens"}
                for r in responses
            )
            and record["generated_tokens"] == 192
            and record["status"] == "token_limit"
            and input_bound_ok
            and all(t == "default" for t in tiers)
            and blocked
        )
        cancellation_observed = None
    else:
        receipt_path = root / name / "request-001/background.json"
        receipt = decode(receipt_path.read_bytes()) if receipt_path.exists() else {}
        operations = receipt.get("operations", [])
        cancellation_observed = (
            any(o["operation"] == "cancel" and o["status"] == "received" for o in operations)
            and receipt.get("final_status") == "cancelled"
        )
        passed = (
            cancellation_observed
            and input_bound_ok
            and blocked
            and record["status"] == "wall_limit"
            and not outcomes[0]["returned"]
            and record["generated_tokens"] is not None
            and record["reserved_output_tokens"] == 0
            and all(t == "default" for t in tiers)
        )
    result = dict(
        status="pass" if passed else "fail",
        controller_status=record["status"],
        outcomes=outcomes,
        subsequent_request_blocked=blocked,
        cancellation_observed=cancellation_observed,
        generated_tokens=record["generated_tokens"],
        observed_output_tokens=record["observed_output_tokens"],
        reserved_output_tokens=record["reserved_output_tokens"],
        input_bound_ok=input_bound_ok,
        returned_service_tiers=tiers,
    )
    persist(root / name / "result.json", result)
    return result


def execute(root, freeze, raw, output):
    if freeze.get("authorization") != "approved":
        raise ValueError("this concrete model smoke requires user approval")
    if decode(raw) != freeze:
        raise ValueError("archived freeze differs from execution configuration")
    preflight(root, freeze)
    if output.resolve() != Path(SETTINGS["output"]).resolve():
        raise ValueError("only the frozen one-use output directory is allowed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root):
        raise ValueError("model smoke requires a clean committed source tree")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    (output / "freeze.json").write_bytes(raw)
    for name in FILES:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (root / name).read_bytes()
        if digest(data) != freeze["files"][name]:
            raise ValueError("source changed while archiving")
        target.write_bytes(data)
    run = dict(
        schema="openboost-accounting-smoke-result-v1",
        status="running",
        cases={},
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        dirty=False,
        freeze_sha256=digest(raw),
        started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,
        platform=platform.platform(),
        openssl=ssl.OPENSSL_VERSION,
        cpu_count=os.cpu_count(),
        gpu=None,
        independent_author=False,
        cli=" ".join(sys.argv),
        generated_tokens=None,
    )
    persist(output / "run.json", run)
    try:
        for name in ("exhaustion", "cancellation"):
            run["cases"][name] = case(output, name)
            if run["cases"][name]["status"] != "pass":
                break
        run["status"] = (
            "pass"
            if len(run["cases"]) == 2 and all(c["status"] == "pass" for c in run["cases"].values())
            else "fail"
        )
        values = [c["generated_tokens"] for c in run["cases"].values()]
        if values and all(v is not None for v in values):
            run["generated_tokens"] = sum(values)
    except BaseException as error:
        run.update(status="fail", error_type=type(error).__name__)
        raise
    finally:
        run["elapsed_s"] = time.monotonic() - started
        persist(output / "run.json", run)
        persist(
            output / "archive.json",
            {
                "files": {
                    p.relative_to(output).as_posix(): digest(p.read_bytes())
                    for p in sorted(output.rglob("*"))
                    if p.is_file() and p.name != "archive.json"
                }
            },
        )
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--execute", type=Path, metavar="FROZEN_OUTPUT")
    args = parser.parse_args()
    raw = args.freeze.read_bytes()
    freeze = decode(raw)
    result = execute(ROOT, freeze, raw, args.execute) if args.execute else preflight(ROOT, freeze)
    print(json.dumps(result, indent=2))
    return int(result["status"] == "fail")


if __name__ == "__main__":
    raise SystemExit(main())
