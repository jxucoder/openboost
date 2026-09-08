"""Verify the retained 099 run and derive its result without HTTP or model calls."""

import hashlib
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    def read(name):
        return json.loads((root / name).read_text())

    index = read("archive.json")["files"]
    for name, expected in index.items():
        assert digest(root / name) == expected, name
    freeze = read("freeze.json")
    run = read("run.json")
    assert freeze["authorization"] == "approved"
    assert run["freeze_sha256"] == digest(root / "freeze.json")
    assert run["revision"] == "5c0f31ab1ec1c93936396bd40a2f5be6e02f5d77"
    assert run["dirty"] is False and run["independent_author"] is False
    for name, expected in freeze["files"].items():
        assert digest(root / "source" / name) == expected, name
    rows = []
    operations = Counter()
    ids = set()
    for case in ("exhaustion", "cancellation"):
        ledger = read(f"{case}/ledger.json")
        result = read(f"{case}/result.json")
        assert result == run["cases"][case]
        assert ledger["transport"] == "responses_https"
        assert ledger["generated_tokens"] == result["generated_tokens"]
        assert ledger["reserved_output_tokens"] == 0
        assert result["subsequent_request_blocked"] is True
        case_output = 0
        for entry in ledger["requests"]:
            prefix = f"{case}/request-{entry['sequence']:03d}"
            receipt = read(f"{prefix}/background.json")
            request = read(f"{prefix}/request.json")
            response = read(f"{prefix}/response.json")
            assert request["input"] == freeze["prompts"][len(rows)]
            assert digest(root / prefix / "request.json") == entry["request_sha256"]
            assert digest(root / prefix / "response.json") == entry["response_sha256"]
            assert receipt["transport"] == "responses_https" and receipt["stopped"] is False
            last = None
            statuses = []
            for number, operation in enumerate(receipt["operations"], 1):
                name = operation["operation"]
                operations[name] += 1
                target = f"{prefix}/operation-{number:03d}-{name}"
                http = read(f"{target}/http.json")
                assert operation["status"] == "received" and http["status"] == 200
                assert http["body_complete"] is True
                assert len((root / target / "response.bin").read_bytes()) == http["body_bytes"]
                raw = read(f"{target}/response.bin")
                statuses.append(raw["status"])
                assert raw == read(f"{target}/response.json")
                assert raw["id"] == receipt["response_id"] == response["id"]
                if name == "create":
                    assert read(f"{target}/request.json") == request
                last = raw
            assert last == response
            assert response["id"] not in ids
            ids.add(response["id"])
            assert response["model"] == request["model"] == freeze["settings"]["model"]
            assert response["service_tier"] == request["service_tier"] == "default"
            assert response["max_output_tokens"] == request["max_output_tokens"]
            assert response["store"] is False and response["background"] is True
            assert response["tools"] == request["tools"] == []
            counts = response["usage"]
            inputs, outputs = counts["input_tokens"], counts["output_tokens"]
            reasoning = counts["output_tokens_details"]["reasoning_tokens"]
            assert counts["total_tokens"] == inputs + outputs
            assert 0 <= reasoning <= outputs <= request["max_output_tokens"]
            assert entry["usage"] == dict(
                input_tokens=inputs, output_tokens=outputs, reasoning_tokens=reasoning
            )
            assert counts["input_tokens_details"] == dict(cache_write_tokens=0, cached_tokens=0)
            case_output += outputs
            rows.append(
                dict(
                    case=case,
                    sequence=entry["sequence"],
                    status=response["status"],
                    max_output_tokens=request["max_output_tokens"],
                    input_tokens=inputs,
                    output_tokens=outputs,
                    reasoning_tokens=reasoning,
                    observed_statuses=statuses,
                )
            )
        assert case_output == ledger["generated_tokens"]
    assert operations == Counter(create=3, retrieve=7)
    assert [r["max_output_tokens"] for r in rows] == [128, 64, 4096]
    assert [r["status"] for r in rows] == ["incomplete", "incomplete", "completed"]
    for sequence in (1, 2):
        value = read(f"exhaustion/request-{sequence:03d}/response.json")
        assert value["incomplete_details"] == {"reason": "max_output_tokens"}
    assert run["cases"]["exhaustion"]["status"] == "pass"
    assert run["cases"]["exhaustion"]["controller_status"] == "token_limit"
    assert run["cases"]["cancellation"]["status"] == "fail"
    assert run["cases"]["cancellation"]["cancellation_observed"] is False
    assert run["cases"]["cancellation"]["outcomes"][0]["returned"] is True
    assert read("cancellation/ledger.json")["elapsed_s"] < 5
    totals = {
        key: sum(row[key] for row in rows)
        for key in ("input_tokens", "output_tokens", "reasoning_tokens")
    }
    assert totals["output_tokens"] == run["generated_tokens"] == 296
    assert run["status"] == "fail"
    estimate = (
        Decimal(totals["input_tokens"]) * Decimal("0.25")
        + Decimal(totals["output_tokens"]) * Decimal("1.20")
    ) / Decimal(1_000_000)
    return dict(
        schema="openboost-accounting-observation-v1",
        verdict="fail",
        exhaustion="pass",
        cancellation="not_exercised",
        frozen_cancellation_verdict="fail",
        original_indexed_files=len(index),
        frozen_sources=len(freeze["files"]),
        generation_requests=operations["create"],
        retrieval_requests=operations["retrieve"],
        cancellation_requests=operations["cancel"],
        requests=rows,
        totals=totals,
        elapsed_s=run["elapsed_s"],
        conservative_token_cost_estimate_usd=str(estimate),
        invoice_verified=False,
        allowance="consumed",
        retries=0,
        independent_author=False,
        next_gate="Observe actual cancellation and reconcile final usage under a new frozen allowance",
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = verify(root)
    if "--check" in sys.argv:
        assert result == json.loads((root / "analysis.json").read_text())
        print(
            "Verified 80 original artifact hashes, 12 source hashes and the retained failed verdict"
        )
    else:
        print(json.dumps(result, indent=2))
