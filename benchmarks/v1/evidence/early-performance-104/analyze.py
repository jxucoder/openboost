"""Offline audit of the frozen checkpoint, preserving incomplete and failed pairs."""

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
from benchmarks.v1.performance_checkpoint import CONFIG, quality, workload  # noqa: E402

from openboost.artifacts import Model  # noqa: E402


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def close(actual, expected, message):
    check(np.allclose(actual, expected, rtol=1e-12, atol=1e-12), message)


def verify_retained_fits(result, recipe, backend):
    """A timed-out child may still preserve finished fits, but not final quality."""
    measurements = result.get("measurements", [])
    for repeat, m in enumerate(measurements):
        check(m["repeat"] == repeat, "retained repeat index differs")
        check(
            m["state"]["version"] == m["stop"]["completed_rounds"] == 20,
            "retained fit lacks twenty rounds",
        )
        check(
            m["state"]["terms"] == (20 if recipe == "squared" else 40),
            "retained fit term count differs",
        )
        if backend == "cuda":
            check(
                m["live_bytes_after_run_close"] == m["final_metrics"]["live_bytes"] == 0,
                "retained fit leaked owned storage",
            )
    return len(measurements)


def verify_complete(result, sources):
    """Check completed CUDA work even when the CPU timeout short-circuits the judge."""
    if result["status"] != "complete":
        return None
    recipe, rows = result["recipe"], result["train_rows"]
    train, validation = workload(rows, recipe)
    check(result["config"] == CONFIG, "child configuration differs")
    check(result["environment"]["core_sources"] == sources, "child installed sources differ")
    check(result["environment"]["numpy"] == "2.3.5", "child NumPy version differs")
    check("site-packages" in result["environment"]["installed_path"], "child is not installed")
    check(result["validation_rows"] == rows // 5, "validation size differs")
    expected = 2 if result["profile"] or result["backend"] == "cpu" else 4
    measurements = result["measurements"]
    check(len(measurements) == expected, "completed child lacks repetitions")
    record_identity = digest(json.dumps(result["model"], sort_keys=True).encode())
    check(result["model_identities"] == [record_identity] * expected, "repeat models differ")
    check(result["replay_exact"], "child model replay failed")
    replay = Model.from_record(result["model"]).predict(validation.data)
    check(np.array_equal(replay, result["validation_raw"]), "offline model replay differs")
    scores = quality(validation, replay, recipe)
    check(scores.keys() == result["quality"].keys(), "quality metric set differs")
    # Generation includes floating transcendental functions. Same-version NumPy
    # does not reproduce the Linux input hashes on every platform. Never replace
    # the remote inputs/metrics or silently loosen their frozen comparison gate.
    regeneration = dict(
        recipe=recipe,
        rows=rows,
        backend=result["backend"],
        local_train_identity=train.identity,
        local_validation_identity=validation.identity,
        train_identity_exact=result["train_identity"] == train.identity,
        validation_identity_exact=result["validation_identity"] == validation.identity,
        prediction_replay_exact=True,
        local_metric_absolute_differences={k: scores[k] - result["quality"][k] for k in scores},
    )
    width = 1 if recipe == "squared" else 2
    for repeat, measurement in enumerate(measurements):
        check(measurement["repeat"] == repeat, "repeat index differs")
        check(measurement["phase"] == ("first" if repeat == 0 else "warm"), "phase differs")
        check(
            np.isfinite(measurement["fit_seconds"]) and measurement["fit_seconds"] > 0,
            "invalid fit time",
        )
        check(measurement["state"]["version"] == 20, "incomplete accepted rounds")
        check(measurement["state"]["terms"] == width * 20, "incomplete accepted terms")
        check(measurement["stop"]["completed_rounds"] == 20, "incomplete stop rounds")
        check(
            measurement["binning_identity"] == measurements[0]["binning_identity"],
            "repeat binning differs",
        )
        if result["backend"] == "cuda":
            check(measurement["live_bytes_after_run_close"] == 0, "run buffers leaked")
            check(measurement["final_metrics"]["live_bytes"] == 0, "context buffers leaked")
            check(
                measurement["final_metrics"]["peak_pool_bytes"] <= CONFIG["pool_bytes"],
                "private pool budget exceeded",
            )
    return regeneration


def verify_qualification(cpu, cuda, original):
    """Recompute eligibility from preserved same-host metrics, never local substitutes.

    Independent metric recomputation occurred in the frozen remote judge. Exact
    input regeneration is reported separately; this is not a portable rerun of it.
    """
    reasons = [
        f"{name}: {child['status']}"
        for name, child in (("cpu", cpu), ("cuda", cuda))
        if child["status"] != "complete"
    ]
    if reasons:
        check(
            original
            == dict(
                measurement_complete=False,
                quality_comparable=False,
                reasons=reasons,
                cpu_over_cuda_warm_fit_ratio=None,
            ),
            "incomplete pair received a different qualification",
        )
        return
    check(
        (cpu["train_identity"], cpu["validation_identity"])
        == (cuda["train_identity"], cuda["validation_identity"]),
        "same-host inputs differ",
    )
    check(
        cpu["measurements"][0]["binning_identity"] == cuda["measurements"][0]["binning_identity"],
        "same-host binning differs",
    )
    deltas = {
        k: abs(cuda["quality"][k] - value) / max(abs(value), 1e-12)
        for k, value in cpu["quality"].items()
    }
    reference = np.asarray(cpu["validation_raw"])
    error = np.sqrt(np.mean((np.asarray(cuda["validation_raw"]) - reference) ** 2, axis=0))
    error /= np.maximum(1, np.std(reference, axis=0))
    if max(deltas.values()) > 0.01:
        reasons.append("task metrics differ by more than 1%")
    if max(error) > 0.01:
        reasons.append("normalized prediction RMSE exceeds 1%")
    check(original["measurement_complete"] is True, "complete pair classification differs")
    check(original["quality_comparable"] == (not reasons), "quality qualification differs")
    check(original["reasons"] == reasons, "quality reasons differ")
    close(original["normalized_prediction_rmse"], error, "prediction comparison differs")
    for key, value in deltas.items():
        close(original["relative_metric_differences"][key], value, "metric difference differs")
    ratio = (
        None
        if reasons
        else (
            median(m["fit_seconds"] for m in cpu["measurements"][1:])
            / median(m["fit_seconds"] for m in cuda["measurements"][1:])
        )
    )
    check(original["cpu_over_cuda_warm_fit_ratio"] == ratio, "qualified ratio differs")


def summarize(result):
    measurements = result.get("measurements", [])
    warm = measurements[1:]
    summary = dict(
        status=result["status"],
        child_wall_seconds=result["child_wall_seconds"],
        child_timeout_seconds=result["child_timeout_seconds"],
        completed_repetitions=len(measurements),
        first_fit_seconds=measurements[0]["fit_seconds"] if measurements else None,
        warm_fit_seconds=[m["fit_seconds"] for m in warm],
        warm_fit_median_seconds=median(m["fit_seconds"] for m in warm) if warm else None,
        quality=result.get("quality"),
    )
    if measurements:
        summary["first_phases_seconds"] = {
            k: measurements[0][k] for k in measurements[0] if k.endswith("_seconds")
        }
        summary["last_fit_state"] = measurements[-1]["state"]
        summary["last_fit_metrics"] = measurements[-1]["final_metrics"]
    if warm:
        summary["warm_phase_medians_seconds"] = {
            k: median(m[k] for m in warm) for k in warm[0] if k.endswith("_seconds")
        }
    if result["status"] == "complete":
        summary["construction_seconds"] = result["construction_seconds"]
        summary["model_load_seconds"] = result["model_load_seconds"]
        summary["cpu_inference_warm_medians_seconds"] = {
            k: median(values[1:]) for k, values in result["cpu_prediction_seconds"].items()
        }
    return summary


def analyze():
    manifest = read(ROOT / "manifest.json")
    verdict = read(ROOT / "verdict.json")
    protocol, result = manifest["protocol"], manifest["result"]
    check(manifest["dirty"] is False, "dirty execution revision")
    check(
        protocol["authorization"] == protocol["upload_authorization"] == "approved",
        "dispatch authorization missing",
    )
    check(len(manifest["sources"]) == protocol["upload_file_count"] == 46, "source closure differs")
    for name, expected in manifest["sources"].items():
        original = subprocess.check_output(
            ["git", "show", manifest["revision"] + ":" + name], cwd=REPO
        )
        check(digest(original) == expected, "dispatch source differs: " + name)
    protocol_name = "v1-sprints/104-performance-run10.json"
    original_protocol = json.loads(
        subprocess.check_output(
            ["git", "show", manifest["revision"] + ":" + protocol_name], cwd=REPO
        )
    )
    check(protocol == original_protocol, "embedded dispatch protocol differs")
    check(
        protocol["frozen_sources"]
        == {p: h for p, h in manifest["sources"].items() if p != protocol_name},
        "prefrozen sources differ",
    )
    # Local replay and rejudging must use the same implementation as the executed packet.
    for name, expected in protocol["frozen_sources"].items():
        if name.startswith(("src/openboost/", "benchmarks/v1/")):
            check(digest((REPO / name).read_bytes()) == expected, "audit source differs: " + name)
    check(
        sorted(result["retained_artifact_hashes"]) == sorted(protocol["retained_artifacts"]),
        "retained case artifacts differ",
    )
    check(
        set(manifest["artifacts"])
        == set(protocol["retained_artifacts"]) | {"pytest.log", "junit.xml", "verdict.json"},
        "raw artifact set differs",
    )
    for name, expected in manifest["artifacts"].items():
        check(digest((ROOT / name).read_bytes()) == expected, "raw artifact changed: " + name)
    for name, expected in result["retained_artifact_hashes"].items():
        check(manifest["artifacts"][name] == expected, "retained artifact hash differs")
    recomputed = judge_run(result, (ROOT / "junit.xml").read_text(), protocol, manifest["sources"])
    check(recomputed == verdict, "raw verdict differs from rejudgment")
    expected_cases = [
        node.replace(".py::", "::").replace("/", ".") for node in protocol["expected_cases"]
    ]
    check(
        sorted(c["case"] for c in verdict["cases"]) == sorted(expected_cases),
        "executed case matrix differs",
    )
    for key in (
        "installed_sources_match",
        "versions_match",
        "snapshot_sources_match",
        "cpu_environment_built",
        "retained_artifacts_complete",
    ):
        check(verdict[key], "provenance failed: " + key)
    check(manifest["status"] == ("pass" if verdict["passed"] else "fail"), "status differs")
    sources = {p: h for p, h in manifest["sources"].items() if p.startswith("src/openboost/")}
    pairs, local_replays, retained_fits = [], [], 0
    for case in protocol["child_budgets"]["cpu_seconds"]:
        location = ROOT / "normal/checkpoint" / case
        cpu, cuda, original = [
            read(location / (name + ".json")) for name in ("cpu", "cuda", "judgment")
        ]
        recipe, rows = case.split("-")
        for backend, child in (("cpu", cpu), ("cuda", cuda)):
            expected_timeout = (
                protocol["child_budgets"]["cpu_seconds"][case] if backend == "cpu" else 50
            )
            check(child["child_timeout_seconds"] == expected_timeout, "child deadline differs")
            for key, value in (("--recipe", recipe), ("--rows", rows), ("--backend", backend)):
                argv = child["child_argv"]
                check(argv[argv.index(key) + 1] == value, "child command differs")
            if child["status"] == "complete":
                check(
                    (child["recipe"], child["train_rows"], child["backend"], child["profile"])
                    == (recipe, int(rows), backend, False),
                    "completed case differs",
                )
            retained_fits += verify_retained_fits(child, recipe, backend)
            replay = verify_complete(child, sources)
            if replay is not None:
                local_replays.append(replay)
        verify_qualification(cpu, cuda, original)
        pairs.append(dict(case=case, cpu=summarize(cpu), cuda=summarize(cuda), judgment=original))
    profile = read(ROOT / "normal/checkpoint/profile.json")
    check(
        profile["child_timeout_seconds"] == protocol["child_budgets"]["profile_seconds"],
        "profile deadline differs",
    )
    if profile["status"] == "complete":
        check(
            (profile["recipe"], profile["train_rows"], profile["backend"], profile["profile"])
            == ("normal", 100_000, "cuda", True),
            "profile case differs",
        )
    replay = verify_complete(profile, sources)
    retained_fits += verify_retained_fits(profile, "normal", "cuda")
    if replay is not None:
        local_replays.append(replay)
    tree = ET.parse(ROOT / "junit.xml")
    suites = list(tree.getroot().iter("testsuite"))
    return dict(
        manifest_sha256=digest((ROOT / "manifest.json").read_bytes()),
        execution_revision=manifest["revision"],
        raw_status=manifest["status"],
        raw_verdict=verdict["passed"],
        cases_passed=sum(c["status"] == "pass" for c in verdict["cases"]),
        cases_total=len(verdict["cases"]),
        dispatch_sources_verified=len(manifest["sources"]),
        raw_artifacts_verified=len(manifest["artifacts"]),
        completed_children_replayed=len(local_replays),
        retained_completed_fits_verified=retained_fits,
        local_audit_environment=dict(
            os=platform.platform(),
            machine=platform.machine(),
            python=platform.python_version(),
            numpy=np.__version__,
        ),
        local_regeneration_audit=local_replays,
        local_inputs_match_remote=all(
            r["train_identity_exact"] and r["validation_identity_exact"] for r in local_replays
        ),
        qualified_pairs=sum(p["judgment"]["quality_comparable"] for p in pairs),
        dispatch_seconds=(
            datetime.fromisoformat(manifest["finished"])
            - datetime.fromisoformat(manifest["started"])
        ).total_seconds(),
        worker_seconds=result["worker_wall_seconds"],
        junit_seconds=sum(float(s.get("time", 0)) for s in suites),
        pairs=pairs,
        separate_profile=dict(summary=summarize(profile), calls=profile.get("profile_calls", [])),
        scope="Synthetic one-seed internal checkpoint. Ratios require completed comparable pairs; "
        "profile timings are separate and inclusive costs overlap. No external speed or E4 claim.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = analyze()
    if args.check:
        # Local input regeneration is an observation of the audit host, not a
        # portable dataset guarantee. Still execute replay on the current host.
        local = {"local_audit_environment", "local_regeneration_audit", "local_inputs_match_remote"}
        recorded = read(ROOT / "analysis.json")
        check(
            {k: v for k, v in result.items() if k not in local}
            == {k: v for k, v in recorded.items() if k not in local},
            "derived analysis differs",
        )
    else:
        (ROOT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "raw_status",
                    "cases_passed",
                    "cases_total",
                    "qualified_pairs",
                    "dispatch_sources_verified",
                    "raw_artifacts_verified",
                    "completed_children_replayed",
                )
            },
            indent=2,
        )
    )
