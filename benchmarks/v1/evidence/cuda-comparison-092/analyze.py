"""Offline audit of run 8; preserve its failed verdict and derive the fixture prefix."""

import argparse
import hashlib
import json
import struct
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def prefix_derivation(update):
    """Exact single-row mean-squared ranking at the fixture's commit boundaries.

    Target is zero and log-scale is zero, so the common NLL constant cancels.
    This derives the expected prefix from the declared leaves, not a CUDA result.
    """

    def f32(value):
        return struct.unpack("!f", struct.pack("!f", value))[0]

    current = 2.0
    best_loss, best_terms, count = Fraction(2), 0, 0
    result = []
    for mean in (3, 1.95, 1.97, 1.9, 1.89):
        delta = f32(f32(mean) - current)
        values = (0.0, delta) if update == "reverse" else (delta, 0.0)
        for index, value in enumerate(values):
            current = f32(current + value)
            count += 1
            if update == "joint" and index == 0:
                continue
            loss = Fraction.from_float(current) ** 2 / 2
            if loss < best_loss:
                best_loss, best_terms = loss, count
            result.append(dict(terms=count, mean=current, best_terms=best_terms))
    return dict(boundaries=result, final_best_terms=best_terms)


def enclosure(record, evidence, extra):
    if record["lower"] is None:
        return
    lo, hi = Decimal.from_float(record["lower"]), Decimal.from_float(record["upper"])
    for value in [evidence["decimal60"], evidence["decimal100"], *extra]:
        check(lo <= Decimal(value) <= hi, "recorded precision estimate outside enclosure")


def analyze():
    manifest = read("manifest.json")
    for name, expected in manifest["artifacts"].items():
        check(digest((ROOT / name).read_bytes()) == expected, f"raw artifact changed: {name}")
    for name, expected in manifest["sources"].items():
        value = subprocess.check_output(
            ["git", "show", manifest["revision"] + ":" + name], cwd=REPO
        )
        check(digest(value) == expected, f"dispatch source differs from revision: {name}")
    verdict = read("verdict.json")
    cohorts = {}
    for name in ("historical", "revised"):
        cases = list(ET.parse(ROOT / name / "junit.xml").getroot().iter("testcase"))
        failures = []
        for case in cases:
            issues = [child for child in case if child.tag in ("failure", "error", "skipped")]
            if issues:
                failures.append(
                    dict(
                        case=case.get("classname") + "::" + case.get("name"),
                        kind=issues[0].tag,
                        message=issues[0].get("message"),
                    )
                )
        cohorts[name] = dict(cases=len(cases), passed=len(cases) - len(failures), failures=failures)
    operations, trajectories, old_failures = Counter(), Counter(), []
    for path in sorted((ROOT / "revised/comparisons").glob("*.json")):
        item = json.loads(path.read_text())
        comparison = item["comparison"]
        operations[comparison["status"]] += 1
        enclosure(comparison, item["high_precision"], item["additional_precision"])
        if item["case"] in (
            "run7/forward/channel0/alpha4.0/training",
            "run7/reverse/channel0/alpha4.0/training",
        ):
            old_failures.append(dict(case=item["case"], comparison=comparison))
    for path in sorted((ROOT / "revised/trajectories").glob("*.json")):
        item = json.loads(path.read_text())
        for event in item["comparisons"]:
            check("audit_error" not in event, "device trajectory contains an audit error")
            trajectories[event["comparison"]["status"]] += 1
            audit = event["audit"]
            enclosure(event["comparison"], audit["high_precision"], audit["additional_precision"])
    lowering = read("revised/diagnostics/lowering.json")
    for rows in lowering["kernels"].values():
        for row in rows:
            check(digest(row["ptx"].encode()) == row["sha256"], "PTX hash differs")
    replays = {}
    for cohort in ("historical", "revised"):
        files = sorted((ROOT / cohort / "normal").glob("*/cpu-replay.json"))
        for path in files:
            item = json.loads(path.read_text())
            check(
                digest((path.parent / "model.json").read_bytes()) == item["model_sha256"],
                "CPU replay model differs",
            )
            check(
                item["sources"] == manifest["result"]["installed_sources"],
                "CPU replay sources differ",
            )
            check(
                set(item["absent"]) == {"cupy", "numba", "ob_cohort_splits"},
                "CPU replay dependency boundary differs",
            )
        replays[cohort] = len(files)
    return dict(
        revision=manifest["revision"],
        manifest_sha256=digest((ROOT / "manifest.json").read_bytes()),
        raw_artifacts_verified=len(manifest["artifacts"]),
        dispatch_sources_verified=len(manifest["sources"]),
        recorded_status=manifest["status"],
        recorded_verdict_passed=verdict["passed"],
        historical_disagreements_match=verdict["historical"]["expected_disagreements_match"],
        cohorts=cohorts,
        operation_statuses=dict(operations),
        trajectory_comparison_statuses=dict(trajectories),
        old_false_improvement_cases=old_failures,
        directed_double_lowering=lowering["required"],
        cpu_replays=replays,
        fixture_prefix_derivation={
            name: prefix_derivation(name) for name in ("joint", "forward", "reverse")
        },
        dispatch_seconds=(
            datetime.fromisoformat(manifest["finished"])
            - datetime.fromisoformat(manifest["started"])
        ).total_seconds(),
        worker_seconds=manifest["result"]["worker_wall_seconds"],
        scope="Offline artifact and recorded-enclosure audit; fixture derivation is not a new device execution or a passing revised verdict.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = analyze()
    if args.check:
        check(result == read("analysis.json"), "derived analysis differs")
    else:
        (ROOT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "raw_artifacts_verified",
                    "dispatch_sources_verified",
                    "recorded_status",
                    "cpu_replays",
                )
            },
            indent=2,
        )
    )
