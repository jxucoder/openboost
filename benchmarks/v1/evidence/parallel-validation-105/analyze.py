"""Offline audit of run 11; replay exact stored inputs and preserve failed gates."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO))

from benchmarks.v1.cuda_aggregation_preflight import judge_run  # noqa: E402
from benchmarks.v1.performance_evidence import load_inputs, verify_report  # noqa: E402
from benchmarks.v1.validation_judge import judge  # noqa: E402


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def equivalent(left, right):
    """Tolerate only local arithmetic in reported numeric differences, not verdicts."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(v, right[k]) for k, v in left.items())
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            equivalent(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float) and isinstance(right, float):
        return bool(np.isclose(left, right, rtol=1e-12, atol=1e-12))
    return type(left) is type(right) and left == right


def summary(report):
    fits = report.get("fits", [])
    measures = [f["measurement"] for f in fits]
    complete = report["status"] == "complete"
    result = dict(
        status=report["status"],
        completed_fits=len(fits),
        child_seconds=report["child_wall_seconds"],
        child_cap_seconds=report["child_timeout_seconds"],
        first_fit_seconds=measures[0]["fit_seconds"] if fits else None,
        observed_warm_fit_seconds=[m["fit_seconds"] for m in measures[1:]],
        complete_warm_median_seconds=median(m["fit_seconds"] for m in measures[1:])
        if complete
        else None,
        quality=fits[-1]["quality"] if fits else None,
    )
    if fits:
        result.update(
            last_state=measures[-1]["state"],
            last_final_metrics=measures[-1]["final_metrics"],
            post_fit_evidence_seconds=[f["post_fit_seconds"] for f in fits],
            last_cpu_prediction_warm_medians_seconds={
                k: median(v[1:]) for k, v in fits[-1]["cpu_prediction_seconds"].items()
            },
        )
    if complete:
        result["warm_phase_medians_seconds"] = {
            k: median(m[k] for m in measures[1:]) for k in measures[0] if k.endswith("_seconds")
        }
    return result


def analyze():
    manifest, verdict = read(ROOT / "manifest.json"), read(ROOT / "verdict.json")
    protocol, result = manifest["protocol"], manifest["result"]
    check(manifest["dirty"] is False, "dirty execution revision")
    check(
        protocol["authorization"] == protocol["upload_authorization"] == "approved",
        "dispatch approval missing",
    )
    check(len(manifest["sources"]) == protocol["upload_file_count"] == 88, "source closure differs")
    for name, expected in manifest["sources"].items():
        original = subprocess.check_output(
            ["git", "show", manifest["revision"] + ":" + name], cwd=REPO
        )
        check(digest(original) == expected, "dispatch source differs: " + name)
    protocol_name = "v1-sprints/105-validation-run11.json"
    original = json.loads(
        subprocess.check_output(
            ["git", "show", manifest["revision"] + ":" + protocol_name], cwd=REPO
        )
    )
    check(original == protocol, "embedded protocol differs")
    check(
        protocol["frozen_sources"]
        == {p: h for p, h in manifest["sources"].items() if p != protocol_name},
        "prefrozen sources differ",
    )
    # Historical replay must use the executed core and helpers, not later code.
    for name, expected in protocol["frozen_sources"].items():
        if name.startswith(("src/openboost/", "benchmarks/v1/")):
            check(digest((REPO / name).read_bytes()) == expected, "audit source differs: " + name)
    check(
        set(manifest["artifacts"])
        == set(protocol["retained_artifacts"]) | {"pytest.log", "junit.xml", "verdict.json"},
        "raw artifact set differs",
    )
    for name, expected in manifest["artifacts"].items():
        check(digest((ROOT / name).read_bytes()) == expected, "raw artifact changed: " + name)
    check(
        result["retained_artifact_hashes"]
        == {p: manifest["artifacts"][p] for p in protocol["retained_artifacts"]},
        "retained artifact hashes differ",
    )
    check(
        judge_run(result, (ROOT / "junit.xml").read_text(), protocol, manifest["sources"])
        == verdict,
        "verdict rejudgment differs",
    )
    provenance = (
        "installed_sources_match",
        "versions_match",
        "snapshot_sources_match",
        "installed_extensions_match",
        "cpu_environment_built",
        "retained_artifacts_complete",
    )
    check(all(verdict[k] for k in provenance), "source/package/artifact provenance failed")
    check(manifest["status"] == ("pass" if verdict["passed"] else "fail"), "raw status differs")
    expected_cases = [
        p.split("::")[0][:-3].replace("/", ".") + "::" + p.split("::")[1]
        for p in protocol["expected_cases"]
    ]
    check(
        sorted(c["case"] for c in verdict["cases"]) == sorted(expected_cases), "case matrix differs"
    )
    sources = {p: h for p, h in manifest["sources"].items() if p.startswith("src/openboost/")}
    location = ROOT / "normal/validation"
    baseline = read(location / "build.json")
    versions = dict(p.split("==") for p in protocol["packages"])
    check(
        baseline["passed"]
        and baseline["installed"]["sources"] == baseline["sources"] == protocol["baseline_sources"],
        "baseline core differs",
    )
    check(baseline["installed"]["packages"] == versions, "baseline package versions differ")
    pairs, verified_fits = [], 0
    for name, bounds in protocol["measurement_cases"].items():
        case = location / name
        inputs = read(case / "inputs.json")
        train, validation = load_inputs(inputs, inputs["sha256"])
        recipe, rows = name.split("-")
        check(
            inputs["config"] == protocol["config"] and inputs["recipe"] == recipe,
            "input settings differ",
        )
        check(
            train.data.values.shape == (int(rows), 16)
            and len(validation.row_ids) == int(rows) // 5,
            "input shape differs",
        )
        reports = {arm: read(case / f"{arm}.json") for arm in ("baseline", "candidate")}
        if "cpu_seconds" in bounds:
            reports["cpu"] = read(case / "cpu.json")
        for arm, report in reports.items():
            backend = "cpu" if arm == "cpu" else "cuda"
            cap = bounds["cpu_seconds"] if arm == "cpu" else bounds["gpu_seconds_per_arm"]
            check(report["child_timeout_seconds"] == cap, "child cap differs")
            argv = report["child_argv"]
            check(
                argv[argv.index("--input-sha") + 1] == inputs["sha256"],
                "child input binding differs",
            )
            check(
                argv[argv.index("--backend") + 1] == backend and "--profile" not in argv,
                "child mode differs",
            )
            if arm == "baseline":
                check(argv[0] == baseline["python"], "baseline executable differs")
            check(
                report["status"] != "complete" or report["child_exit_code"] == 0,
                "completed child failed",
            )
            # A timeout before initialization retains no fit to replay.
            if "schema" not in report:
                check(
                    report["status"] in ("timeout", "error") and not report.get("fits"),
                    "missing fit provenance",
                )
                continue
            checked = verify_report(
                report,
                inputs,
                inputs["sha256"],
                expected_repetitions=2 if arm == "cpu" else 4,
                expected_sources=protocol["baseline_sources"] if arm == "baseline" else sources,
            )
            verified_fits += checked["verified_fits"]
            check(
                report["environment"]["numpy"] == "2.3.5"
                and "site-packages" in report["environment"]["installed_path"],
                "child installation differs",
            )
            for fit in report["fits"]:
                check(
                    fit["phase"] == ("first" if fit["repeat"] == 0 else "warm"), "fit phase differs"
                )
                if arm != "cpu":
                    check(
                        fit["measurement"]["final_metrics"]["peak_pool_bytes"]
                        <= protocol["config"]["pool_bytes"],
                        "private pool cap exceeded",
                    )
        recalculated = judge(
            reports["baseline"],
            reports["candidate"],
            inputs,
            baseline_sources=protocol["baseline_sources"],
            candidate_sources=sources,
            limit=bounds["candidate_over_baseline_limit"],
            cpu=reports.get("cpu"),
        )
        original = read(case / "judgment.json")
        check(equivalent(recalculated, original), "pair rejudgment differs: " + name)
        summaries = {arm: summary(report) for arm, report in reports.items()}
        pairs.append(
            dict(
                case=name,
                input_sha256=inputs["sha256"],
                arms=summaries,
                judgment=original,
                cpu_over_candidate_warm_ratio=(
                    summaries["cpu"]["complete_warm_median_seconds"]
                    / summaries["candidate"]["complete_warm_median_seconds"]
                    if "cpu" in reports
                    and original["measurement_complete"]
                    and original["quality_passed"]
                    else None
                ),
            )
        )
    profiles = {}
    for arm in ("baseline", "candidate"):
        report = read(location / f"profile-{arm}.json")
        check(
            report["child_timeout_seconds"] == protocol["profile_seconds_per_arm"],
            "profile cap differs",
        )
        value = dict(status=report["status"], child_seconds=report["child_wall_seconds"])
        if report["status"] == "complete":
            check(
                report["child_exit_code"] == 0
                and report["profile"]
                and report["shape"] == [100000, 2],
                "profile mode differs",
            )
            check(
                report["environment"]["core_sources"]
                == (protocol["baseline_sources"] if arm == "baseline" else sources),
                "profile core differs",
            )
            check(
                report["input_sha256"] == read(location / "squared-100000/inputs.json")["sha256"],
                "profile input differs",
            )
            check(
                report["live_bytes_after_release"] == report["final_metrics"]["live_bytes"] == 0,
                "profile leaked storage",
            )
            check(
                set(report["launches_by_name"]) == {"validate_fields"}
                and report["launches_by_name"]["validate_fields"]["calls"]
                == len(report["samples"])
                == 11,
                "profile counts differ",
            )
            check(
                [s["repeat"] for s in report["samples"]] == list(range(11)),
                "profile repetitions differ",
            )
            check(
                all(
                    np.isfinite(s[k]) and s[k] >= 0
                    for s in report["samples"]
                    for k in ("wall_seconds", "stream_interval_ms")
                ),
                "invalid profile time",
            )
            value.update(
                first=report["samples"][0],
                warm_medians={
                    k: median(s[k] for s in report["samples"][1:])
                    for k in ("wall_seconds", "stream_interval_ms")
                },
                launches_by_name=report["launches_by_name"],
                final_metrics=report["final_metrics"],
            )
        profiles[arm] = value
    suites = list(ET.parse(ROOT / "junit.xml").getroot().iter("testsuite"))
    return dict(
        execution_revision=manifest["revision"],
        manifest_sha256=digest((ROOT / "manifest.json").read_bytes()),
        raw_status=manifest["status"],
        raw_verdict=verdict["passed"],
        cases_total=len(verdict["cases"]),
        cases_passed=sum(c["status"] == "pass" for c in verdict["cases"]),
        dispatch_sources_verified=len(manifest["sources"]),
        raw_artifacts_verified=len(manifest["artifacts"]),
        retained_json_bytes=sum((ROOT / p).stat().st_size for p in protocol["retained_artifacts"]),
        retained_completed_fits_replayed=verified_fits,
        baseline_installation_verified=True,
        local_audit_environment=dict(
            os=platform.platform(),
            machine=platform.machine(),
            python=platform.python_version(),
            numpy=np.__version__,
        ),
        dispatch_seconds=(
            datetime.fromisoformat(manifest["finished"])
            - datetime.fromisoformat(manifest["started"])
        ).total_seconds(),
        worker_seconds=result["worker_wall_seconds"],
        junit_seconds=sum(float(s.get("time", 0)) for s in suites),
        pairs=pairs,
        profiles=profiles,
        scope="Exact saved-input/model/score replay and frozen same-algorithm GPU comparison. Synthetic one-seed measurements; operation profile intervals include enqueue/wait gaps. No external-library speed, full Normal conformance or E4 claim.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = analyze()
    if args.check:
        old = read(ROOT / "analysis.json")
        check(
            equivalent(
                {k: v for k, v in result.items() if k != "local_audit_environment"},
                {k: v for k, v in old.items() if k != "local_audit_environment"},
            ),
            "derived analysis differs",
        )
    else:
        (ROOT / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "raw_status",
                    "cases_passed",
                    "cases_total",
                    "retained_completed_fits_replayed",
                    "dispatch_sources_verified",
                    "raw_artifacts_verified",
                )
            },
            indent=2,
        )
    )
