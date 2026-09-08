"""Construct the pending run-8 freeze and collect its installed, isolated snapshot locally."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from benchmarks.v1.cuda_aggregation_preflight import snapshot_hashes, snapshot_paths
from benchmarks.v1.cuda_comparison_preflight import PROTOCOL

ROOT = Path(__file__).resolve().parents[2]
BINDINGS = "v1-sprints/092-collected-case-bindings.json"


def collect(files, root=ROOT):
    output = subprocess.check_output(
        [sys.executable, "-m", "pytest", *files, "--collect-only", "-o", "addopts=", "-q"],
        cwd=root,
        text=True,
    )
    return [line for line in output.splitlines() if line.startswith("tests/")]


def build_protocol():
    existing = ROOT / PROTOCOL
    if existing.exists():
        current = json.loads(existing.read_text())
        if (
            current.get("authorization") != "pending"
            or current.get("upload_authorization") != "pending"
        ):
            raise ValueError("only a pending freeze may be regenerated")
        if (ROOT / current["output"]).exists():
            raise ValueError("an attempted run cannot be regenerated")
    previous = json.loads((ROOT / "v1-sprints/091-acceptance-run7.json").read_text())
    bindings = json.loads((ROOT / BINDINGS).read_text())
    revised_files = list(
        dict.fromkeys(row["revised_node"].split("::")[0] for row in bindings["cases"])
    )
    revised_files += [
        f"tests/v1/{name}.py"
        for name in (
            "test_device_normal_comparison_cuda",
            "test_device_comparison_consumers_cuda",
            "test_device_comparison_recipe_cuda",
            "test_comparison_lowering_cost_cuda",
        )
    ]
    historical_cases = collect(previous["test_files"])
    assert historical_cases == previous["expected_cases"]
    revised_cases = collect(revised_files)
    assert revised_cases[:383] == [row["revised_node"] for row in bindings["cases"]]
    expected_failures = {
        node: 'assert tuple(coefficients) == tuple(a[0] for a in ref["attempts"])'
        for node in previous["diagnostics"]["expected_unresolved_cases"]
    }
    expected_failures.update(
        {
            node: 'assert context.metrics["live_bytes"] == expected_bytes'
            for node in historical_cases
            if "test_device_normal_recipe_cuda.py::test_recipe_matches_frozen_trajectory_and_retention["
            in node
        }
    )
    study = json.loads(
        (ROOT / "benchmarks/v1/evidence/normal-comparison-092/study.json").read_text()
    )
    artifacts = ["historical/" + p for p in previous["retained_artifacts"]]
    artifacts += ["revised/" + p for p in previous["retained_artifacts"] if "/acceptance/" not in p]
    artifacts += [
        "revised/comparisons/" + hashlib.sha256(row["id"].encode()).hexdigest()[:16] + ".json"
        for row in study["cases"]
    ]
    artifacts += [
        "revised/trajectories/"
        + hashlib.sha256(row["revised_node"].encode()).hexdigest()[:16]
        + ".json"
        for row in bindings["cases"]
        if row["disposition"] != "unchanged-operations"
    ]
    artifacts += ["revised/diagnostics/lowering.json", "revised/diagnostics/cost.json"]
    protocol = {
        key: previous[key]
        for key in (
            "limits",
            "float32",
            "normal_float32",
            "metric_relative_scale",
            "packages",
            "install_projects",
            "normal_cpu_environment",
            "cpu_environment_contract",
        )
    }
    protocol.update(
        scope="092 bounded Normal comparison validation with separately reported original and revised semantic cohorts. No universal boosting, quality, speed or author/adoption claim.",
        authorization="pending",
        upload_authorization="pending",
        require_upload_authorization=True,
        upload_destination="Modal",
        app_name="openboost-v1-normal-comparison",
        output="benchmarks/v1/evidence/cuda-comparison-092",
        budget=dict(
            run=8,
            previous_runs_consumed=7,
            additional_runs=1,
            gpu="T4",
            function_seconds=900,
            test_seconds=600,
            retries=0,
        ),
        resources=dict(
            cpu=2, memory_mib=8192, gpu="T4", timeout_seconds=900, retries=0, max_containers=1
        ),
        image=dict(base="nvidia/cuda:12.6.3-devel-ubuntu22.04", python="3.12"),
        cohorts=dict(
            historical=dict(
                test_files=previous["test_files"],
                expected_cases=historical_cases,
                expected_failures=expected_failures,
                interpretation="Two original full-loss failures and 24 old recipe byte assertions (missing the new owned best snapshot) must remain explicit failures at those exact assertions. Any other disagreement or unexpected pass requires review.",
            ),
            revised=dict(
                test_files=revised_files,
                expected_cases=revised_cases,
                interpretation="All 529 cases must pass without skips; 383 bound requirements plus 117 comparisons, 27 consumers and two lowering/cost checks.",
            ),
        ),
        test_files=list(dict.fromkeys(previous["test_files"] + revised_files)),
        support_files=previous["support_files"]
        + [
            "benchmarks/v1/cuda_comparison_preflight.py",
            "tests/v1/comparison_audit.py",
            "tests/v1/reference/compared_normal.py",
            "tests/v1/reference/normal_comparison.py",
            "tests/v1/test_loss_change.py",
            "benchmarks/v1/evidence/normal-comparison-092/study.json",
        ],
        retained_artifacts=sorted(artifacts),
        retained_artifact_bytes=32 * 1024**2,
        compilation_policy="Separate pytest processes; no explicit prewarm. Instrumented correctness timings include audit exports/high-precision work. Two weighted and two installed-D2 fits are timed without comparison instrumentation, including fixture/context/preparation/training/export and any triggered compilation. Prior revised tests may warm kernels. CPU prediction timed separately. No process-cold or matched-quality speed claim.",
        bindings_sha256=hashlib.sha256((ROOT / BINDINGS).read_bytes()).hexdigest(),
        accounting="385 historical plus 529 revised executions; 236 shared operation checks execute in both processes and are not unique additional requirements. One invocation, shared 600-second test deadline, zero automatic retries. Preserve partial failures and stop for retrospective.",
        historical_source_policy="All original test/reference files remain byte-identical to run 7; current production sources execute both cohorts.",
    )
    assert len(artifacts) == len(set(artifacts)) == 409
    assert len(expected_failures) == 26
    assert len(historical_cases) == 385 and len(revised_cases) == 529
    # The protocol hashes itself only at dispatch; all other uploaded files are frozen here.
    path = ROOT / PROTOCOL
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    sources = snapshot_hashes(ROOT, snapshot_paths(ROOT, PROTOCOL, protocol))
    protocol["upload_file_count"] = len(sources)
    protocol["frozen_sources"] = {p: h for p, h in sources.items() if p != PROTOCOL}
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def collect_snapshot(protocol):
    """Wheel extraction is local only; no environment mutation or CUDA execution."""
    with tempfile.TemporaryDirectory(prefix="openboost-run8-collection-") as directory:
        work = Path(directory).resolve()
        snapshot, installed = work / "snapshot", work / "site-packages"
        paths = snapshot_paths(ROOT, PROTOCOL, protocol)
        for source in paths:
            target = snapshot / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        subprocess.run(
            [
                "uv",
                "build",
                str(snapshot),
                "--wheel",
                "--offline",
                "--out-dir",
                str(work / "wheel"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        (wheel,) = (work / "wheel").glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(installed)
        core = {
            "src/openboost/" + str(p.relative_to(installed / "openboost")): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (installed / "openboost").rglob("*.py")
        }
        assert core == {
            p: h for p, h in protocol["frozen_sources"].items() if p.startswith("src/openboost/")
        }
        results = {}
        for name, cohort in protocol["cohorts"].items():
            # -I ignores cwd/PYTHONPATH; explicitly insert only the copied tests and extracted wheel.
            code = "\n".join(
                [
                    "import sys",
                    f"sys.path[:0] = {[str(installed), str(snapshot)]!r}",
                    "from pathlib import Path",
                    "import openboost, pytest",
                    f"assert Path(openboost.__file__).is_relative_to({str(installed)!r})",
                    f"raise SystemExit(pytest.main({[*cohort['test_files'], '--collect-only', '--rootdir=' + str(snapshot), '-o', 'addopts=', '-q']!r}))",
                ]
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=snapshot,
                text=True,
                capture_output=True,
                check=True,
            )
            cases = [line for line in completed.stdout.splitlines() if line.startswith("tests/")]
            if cases != cohort["expected_cases"]:
                raise ValueError(
                    f"isolated {name} collection differs: {len(cases)} cases; {completed.stdout[:1600]!r}"
                )
            results[name] = dict(cases=cases, collected=len(cases), executed=0)
        return dict(
            scope="Local isolated wheel/snapshot collection only; no CUDA test executed.",
            installed_sources=core,
            snapshot_sources=snapshot_hashes(ROOT, paths),
            python=sys.version,
            cohorts=results,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-snapshot", action="store_true")
    args = parser.parse_args()
    protocol = build_protocol()
    if args.collect_snapshot:
        report = collect_snapshot(protocol)
        (ROOT / "v1-sprints/092-isolated-collection.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print(
        f"Pending freeze: {protocol['upload_file_count']} files, 385 historical + 529 revised cases; no remote dispatch."
    )
