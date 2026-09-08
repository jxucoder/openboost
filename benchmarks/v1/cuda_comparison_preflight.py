"""One separately authorized run-8 invocation, with distinct immutable/revised cohorts."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.v1.cuda_aggregation_preflight import (
    check_dispatch,
    check_frozen_sources,
    judge_junit,
    judge_run,
    retain_artifacts,
    snapshot_hashes,
    snapshot_paths,
)

PROTOCOL = "v1-sprints/092-comparison-run8.json"


def junit_name(node):
    path, name = node.split("::")
    return path[:-3].replace("/", ".") + "::" + name


def judge_history(xml, cohort, exit_code):
    """Keep the literal historical verdict; separately check preregistered disagreements."""
    verdict = judge_junit(xml, cohort["expected_cases"])
    expected = {
        junit_name(node): assertion for node, assertion in cohort["expected_failures"].items()
    }
    required = [junit_name(node) for node in cohort["expected_cases"]]
    try:
        cases = list(ET.fromstring(xml).iter("testcase"))
    except ET.ParseError:
        cases = []
    observed = [c.get("classname", "") + "::" + c.get("name", "") for c in cases]
    matches = (
        bool(required)
        and len(set(required)) == len(required)
        and sorted(observed) == sorted(required)
    )
    for node, case in zip(observed, cases, strict=True):
        failures = case.findall("failure")
        if case.find("skipped") is not None or case.find("error") is not None:
            matches = False
        if node in expected:
            matches &= (
                len(failures) == 1
                and failures[0].get("type", "AssertionError") == "AssertionError"
                and (failures[0].text or "").rstrip().endswith(": AssertionError")
                and any(
                    re.match(r"^>\s*" + re.escape(expected[node]) + r"$", line)
                    for line in (failures[0].text or "").splitlines()
                )
            )
        else:
            matches &= not failures
    matches &= exit_code == (1 if expected else 0)
    verdict["expected_disagreements_match"] = bool(matches)
    verdict["passed"] &= exit_code == 0
    return verdict


def judge_cohorts(result, xmls, protocol, sources):
    """Revised acceptance cannot erase a failed historical assertion."""
    historical = judge_history(
        xmls.get("historical", ""),
        protocol["cohorts"]["historical"],
        result.get("cohorts", {}).get("historical", {}).get("exit_code"),
    )
    revised = judge_run(
        dict(result, exit_code=result.get("cohorts", {}).get("revised", {}).get("exit_code")),
        xmls.get("revised", ""),
        dict(protocol, expected_cases=protocol["cohorts"]["revised"]["expected_cases"]),
        sources,
    )
    return dict(
        scope="Revised comparison contract and exact historical disagreements; not historical all-pass conformance.",
        passed=revised["passed"] and historical["expected_disagreements_match"],
        historical=historical,
        revised=revised,
    )


def check_protocol(protocol):
    """Execution parameters must agree with the reviewable bounded request."""
    if protocol["budget"] != dict(
        run=8,
        previous_runs_consumed=7,
        additional_runs=1,
        gpu="T4",
        function_seconds=900,
        test_seconds=600,
        retries=0,
    ):
        raise ValueError("run-8 budget changed")
    if protocol["resources"] != dict(
        cpu=2, memory_mib=8192, gpu="T4", timeout_seconds=900, retries=0, max_containers=1
    ):
        raise ValueError("run-8 resources changed")
    if protocol["image"] != dict(base="nvidia/cuda:12.6.3-devel-ubuntu22.04", python="3.12"):
        raise ValueError("run-8 image changed")
    if tuple(protocol["cohorts"]) != ("historical", "revised"):
        raise ValueError("run-8 cohort order changed")
    if protocol["retained_artifact_bytes"] != 32 * 1024**2:
        raise ValueError("run-8 artifact limit changed")


def main(output, *, protocol_path=PROTOCOL):
    repo = Path(__file__).resolve().parents[2]
    protocol = json.loads((repo / protocol_path).read_text())
    check_dispatch(repo, output, protocol)
    check_protocol(protocol)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo):
        raise ValueError("clean source required")
    paths = snapshot_paths(repo, protocol_path, protocol)
    sources = snapshot_hashes(repo, paths)
    check_frozen_sources(protocol_path, protocol, sources)

    import modal

    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        dirty=False,
        argv=sys.argv,
        protocol=protocol,
        started=datetime.now(timezone.utc).isoformat(),
        status="running",
        sources=sources,
        requested=protocol["resources"],
    )

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    image = modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    image = image.uv_pip_install(*protocol["packages"], uv_version="0.12.1")
    image = image.env({"PYTHONDONTWRITEBYTECODE": "1"})
    for p in paths:
        image = image.add_local_file(p, "/snapshot/" + str(p.relative_to(repo)), copy=True)
    image = image.uv_pip_install(
        "/snapshot", extra_options="--no-deps --no-build-isolation", uv_version="0.12.1"
    )
    for project in protocol.get("install_projects", []):
        image = image.uv_pip_install(
            "/snapshot/" + project["path"],
            extra_options="--no-deps --no-build-isolation",
            uv_version="0.12.1",
        )
    if protocol.get("normal_cpu_environment"):
        image = image.run_commands("python /snapshot/benchmarks/v1/normal_build_cpu_env.py")
    app = modal.App(protocol.get("app_name", "openboost-v1-normal-comparison"))

    @app.function(
        image=image,
        gpu="T4",
        cpu=2,
        memory=8192,
        timeout=900,
        retries=0,
        max_containers=1,
        serialized=True,
        include_source=False,
    )
    def run():
        import importlib.metadata
        import importlib.util
        import os
        import platform
        import time

        import cupy as cp
        from numba import cuda

        import openboost

        started = time.perf_counter()
        config = json.loads(Path("/snapshot/" + protocol_path).read_text())
        installed = Path(openboost.__file__).parent
        versions = {
            p.split("==")[0]: importlib.metadata.version(p.split("==")[0])
            for p in config["packages"]
        }
        sources = {
            "src/openboost/" + str(p.relative_to(installed)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in installed.rglob("*.py")
        }
        result = dict(
            python=platform.python_version(),
            os=platform.platform(),
            cpu=platform.processor(),
            visible_cpu_count=os.cpu_count(),
            packages=versions,
            installed_path=str(installed),
            numba_cuda_path=cuda.__file__,
            installed_sources=sources,
            gpu=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
            ),
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cuda_driver=cp.cuda.runtime.driverGetVersion(),
        )
        if "install_projects" in config:
            result["installed_extensions"] = {}
            for project in config["install_projects"]:
                installed_extension = Path(
                    importlib.util.find_spec(project["module"]).origin
                ).parent
                if "site-packages" not in installed_extension.parts:
                    raise ValueError("extension must be installed outside snapshot")
                result["installed_extensions"][project["distribution"]] = dict(
                    version=importlib.metadata.version(project["distribution"]),
                    sources={
                        project["source_root"]
                        + "/"
                        + str(p.relative_to(installed_extension)): hashlib.sha256(
                            p.read_bytes()
                        ).hexdigest()
                        for p in installed_extension.rglob("*.py")
                    },
                )
        if config.get("normal_cpu_environment"):
            result["cpu_environment"] = json.loads(
                Path("/opt/openboost-normal-cpu/build.json").read_text()
            )
        if "frozen_sources" in config:
            result["snapshot_sources"] = {
                p: hashlib.sha256(Path("/snapshot", p).read_bytes()).hexdigest()
                for p in (*config["frozen_sources"], protocol_path)
            }
        work = Path("/tmp/openboost-comparison")
        work.mkdir()
        result["cohorts"] = {}
        deadline = time.monotonic() + config["budget"]["test_seconds"]
        for name, cohort in config["cohorts"].items():
            directory = work / name
            directory.mkdir()
            environment = dict(os.environ)
            environment.update(
                OPENBOOST_FRESH_CPU_PYTHON="/opt/openboost-normal-cpu/venv/bin/python",
                OPENBOOST_NORMAL_ARTIFACTS=str(directory / "normal"),
                OPENBOOST_REVISED_NORMAL_ARTIFACTS=str(directory / "normal"),
                OPENBOOST_COMPARISON_ARTIFACTS=str(directory / "comparisons"),
                OPENBOOST_COMPARISON_TRAJECTORIES=str(directory / "trajectories"),
                OPENBOOST_COMPARISON_DIAGNOSTICS=str(directory / "diagnostics"),
            )
            command = [
                sys.executable,
                "-m",
                "pytest",
                *["/snapshot/" + f for f in cohort["test_files"]],
                "--rootdir=/snapshot",
                "-o",
                "addopts=",
                "-q",
                "-s",
                "--junitxml=" + str(directory / "junit.xml"),
            ]
            observation = dict(test_argv=command)
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, 0)
                completed = subprocess.run(
                    command,
                    cwd=directory,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=remaining,
                )
                observation.update(
                    exit_code=completed.returncode, log=completed.stdout + completed.stderr
                )
            except subprocess.TimeoutExpired as error:

                def decode(value):
                    return (
                        value.decode(errors="replace") if isinstance(value, bytes) else value or ""
                    )

                observation.update(
                    exit_code="timeout", log=decode(error.stdout) + decode(error.stderr)
                )
            junit = directory / "junit.xml"
            observation["junit"] = junit.read_text() if junit.exists() else ""
            result["cohorts"][name] = observation
        # Preserve partial declared artifacts even if a cohort fails or times out.
        paths = sorted(work.rglob("*.json"))
        if sum(p.stat().st_size for p in paths) > config["retained_artifact_bytes"]:
            result.update(artifact_error="retained JSON byte limit exceeded", retained_artifacts={})
        else:
            result["retained_artifacts"] = {str(p.relative_to(work)): p.read_text() for p in paths}
        result["worker_wall_seconds"] = time.perf_counter() - started
        return result

    try:
        with modal.enable_output(), app.run():
            result = run.remote()
        manifest["result"] = result
        xmls = {}
        for name, observation in result["cohorts"].items():
            directory = output / name
            directory.mkdir()
            (directory / "pytest.log").write_text(observation.pop("log"))
            xmls[name] = observation.pop("junit")
            (directory / "junit.xml").write_text(xmls[name])
        result["retained_artifact_hashes"] = retain_artifacts(
            output, result.pop("retained_artifacts"), protocol
        )
        verdict = judge_cohorts(result, xmls, protocol, manifest["sources"])
        for name in protocol["cohorts"]:
            (output / name / "verdict.json").write_text(json.dumps(verdict[name], indent=2) + "\n")
        (output / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        report_files = ["verdict.json"] + [
            f"{cohort}/{name}"
            for cohort in protocol["cohorts"]
            for name in ("pytest.log", "junit.xml", "verdict.json")
        ]
        manifest.update(
            status="pass" if verdict["passed"] else "fail",
            image_id=image.object_id,
            artifacts={
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in report_files
            },
        )
        manifest["artifacts"].update(result["retained_artifact_hashes"])
    except Exception as error:
        manifest.update(status="error", error=str(error))
        raise
    finally:
        manifest["finished"] = datetime.now(timezone.utc).isoformat()
        save()
    if manifest["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output.resolve())
