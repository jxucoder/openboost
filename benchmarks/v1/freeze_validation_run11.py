"""Freeze the pending 105 validation comparison; collect locally without CUDA execution."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.v1.cuda_aggregation_preflight import snapshot_hashes, snapshot_paths
from benchmarks.v1.cuda_validation_preflight import PROTOCOL
from benchmarks.v1.freeze_comparison_run8 import collect, collect_snapshot
from benchmarks.v1.performance_checkpoint import CONFIG

ROOT = Path(__file__).resolve().parents[2]


def budget_seconds(protocol):
    children = (
        sum(
            case.get("cpu_seconds", 0) + 2 * case["gpu_seconds_per_arm"]
            for case in protocol["measurement_cases"].values()
        )
        + 2 * protocol["profile_seconds_per_arm"]
    )
    return children + protocol["bootstrap_seconds"] + protocol["correctness_and_audit_seconds"]


def build_protocol():
    path = ROOT / PROTOCOL
    if path.exists():
        current = json.loads(path.read_text())
        if current["authorization"] != "pending" or current["upload_authorization"] != "pending":
            raise ValueError("only a pending freeze may be regenerated")
        if (ROOT / current["output"]).exists():
            raise ValueError("an attempted run cannot be regenerated")
    previous = json.loads((ROOT / "v1-sprints/092-comparison-run8.json").read_text())
    run10 = json.loads(
        (ROOT / "benchmarks/v1/evidence/early-performance-104/manifest.json").read_text()
    )
    # Bound the repeat suite using measured run-8 durations. The 96-case runtime
    # matrix alone used 51.9 seconds; retain installed D2 and both recipe cohorts.
    excluded = {"test_compared_normal_runtime_cuda.py", "test_comparison_lowering_cost_cuda.py"}
    tests = [
        p for p in previous["cohorts"]["revised"]["test_files"] if Path(p).name not in excluded
    ]
    tests += [
        "tests/v1/test_parallel_validation_cuda.py",
        "tests/v1/test_validation_checkpoint_cuda.py",
    ]
    support = [p for p in previous["support_files"] if "preflight.py" not in p]
    support += [
        "tests/v1/test_device_normal_cuda.py",
        "benchmarks/v1/cuda_validation_preflight.py",
        "benchmarks/v1/performance_checkpoint.py",
        "benchmarks/v1/performance_evidence.py",
        "benchmarks/v1/validation_checkpoint.py",
        "benchmarks/v1/validation_judge.py",
        "benchmarks/v1/build_validation_baseline.py",
        "benchmarks/v1/validation_baseline/device.py",
        "benchmarks/v1/validation_baseline/_device_kernels.py",
    ]
    support = list(dict.fromkeys(p for p in support if p not in tests))
    cases = {
        "squared-10000": dict(
            cpu_seconds=25, gpu_seconds_per_arm=30, candidate_over_baseline_limit=1.1
        ),
        "squared-100000": dict(
            cpu_seconds=50, gpu_seconds_per_arm=95, candidate_over_baseline_limit=0.8
        ),
        "normal-10000": dict(gpu_seconds_per_arm=65, candidate_over_baseline_limit=1.1),
    }
    artifacts = ["normal/validation/build.json"]
    for name, bounds in cases.items():
        files = ["inputs", "baseline", "candidate", "judgment"] + (
            ["cpu"] if "cpu_seconds" in bounds else []
        )
        artifacts += [f"normal/validation/{name}/{file}.json" for file in files]
    artifacts += [f"normal/validation/profile-{arm}.json" for arm in ("baseline", "candidate")]
    expected = collect(tests)
    protocol = dict(
        scope="105 parallel field validation: unchanged-algorithm original/candidate GPU cost with exact input bytes, bounded scalar/Normal/D2 regression. Synthetic engineering gate only; no full Normal, E4, external-library or adoption claim.",
        authorization="pending",
        upload_authorization="pending",
        require_upload_authorization=True,
        upload_destination="Modal",
        app_name="openboost-v1-parallel-validation",
        output="benchmarks/v1/evidence/parallel-validation-105",
        budget=dict(
            run=11,
            previous_runs_consumed=10,
            additional_runs=1,
            gpu="T4",
            function_seconds=900,
            test_seconds=600,
            retries=0,
        ),
        resources=dict(
            cpu=2, memory_mib=8192, gpu="T4", timeout_seconds=900, retries=0, max_containers=1
        ),
        image=previous["image"],
        packages=previous["packages"],
        install_projects=previous["install_projects"],
        normal_cpu_environment=True,
        test_files=tests,
        support_files=support,
        expected_cases=expected,
        cohorts=dict(candidate=dict(test_files=tests, expected_cases=expected)),
        config=dict(CONFIG),
        measurement_cases=cases,
        bootstrap_seconds=30,
        profile_seconds_per_arm=15,
        correctness_and_audit_seconds=85,
        retained_artifacts=sorted(artifacts),
        retained_artifact_bytes=64 * 1024**2,
        baseline_revision=run10["revision"],
        baseline_sources={
            p: h for p, h in run10["sources"].items() if p.startswith("src/openboost/")
        },
        baseline_overlays={
            f"src/openboost/{name}.py": f"benchmarks/v1/validation_baseline/{name}.py"
            for name in ("device", "_device_kernels")
        },
        quality_thresholds=dict(
            max_relative_task_metric_difference=0.01,
            max_normalized_prediction_rmse=0.01,
            baseline_candidate_model="byte-identical",
            baseline_candidate_predictions="exact",
        ),
        compilation_policy="Fresh subprocess and private empty CuPy/driver cache per arm/case. First plus three warm fits in each GPU child; CPU first plus one warm. Fit includes context/binning/training/model export/cleanup and triggered JIT; evidence verification/serialization follows each timed fit. Fixed order CPU (where declared), baseline GPU, candidate GPU; one device allocation, no randomized order or variability confidence claim. Separate 11-call validation operation profile includes host enqueue gaps and blocking flags, not exclusive kernel time.",
        accounting="All declared repetitions and artifacts are mandatory. Measurement/profile child caps total 485 seconds, baseline installation 30 seconds, and 85 seconds reserved for correctness/setup/audit; the shared 600-second pytest and 900-second function caps dominate. No retries. Failures retain partial fits and fail the complete-case gate. Image building precedes the function cap; no guaranteed dollar cost is asserted.",
        regression_scope="431 existing revised cases, 39 new validation cases, and four cost/profile cases. The earlier 96-case compared Normal runtime matrix and two old lowering/cost cases are not repeated. Existing tests keep exact assertions and tolerances. Temporary regression trajectories are not retained as a new full trace archive; JUnit records their outcomes. Seventeen new cost/input/profile JSON files are retained.",
    )
    if budget_seconds(protocol) != 600 or len(expected) != 474 or len(artifacts) != 17:
        raise ValueError("declared case, artifact or deadline accounting differs")
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    sources = snapshot_hashes(ROOT, snapshot_paths(ROOT, PROTOCOL, protocol))
    protocol["upload_file_count"] = len(sources)
    protocol["frozen_sources"] = {p: h for p, h in sources.items() if p != PROTOCOL}
    for target, original in protocol["baseline_overlays"].items():
        if (
            hashlib.sha256((ROOT / original).read_bytes()).hexdigest()
            != protocol["baseline_sources"][target]
        ):
            raise ValueError("baseline overlay does not match run-10 manifest")
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-snapshot", action="store_true")
    args = parser.parse_args()
    protocol = build_protocol()
    if args.collect_snapshot:
        result = collect_snapshot(protocol, protocol_path=PROTOCOL)
        (ROOT / "v1-sprints/105-isolated-collection.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(
        f"Pending: {protocol['upload_file_count']} files, {len(protocol['expected_cases'])} collected cases; no dispatch."
    )
