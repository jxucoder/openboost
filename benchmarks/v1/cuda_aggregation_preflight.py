"""One declared 078-A T4 run, with exact case accounting and installed source checks."""

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

PROTOCOL = "v1-sprints/078-aggregation-run2.json"


def judge_junit(xml, expected):
    """A zero exit code cannot hide missing, duplicate, skipped or failed cases."""
    try:
        cases = list(ET.fromstring(xml).iter("testcase"))
    except ET.ParseError:
        return dict(passed=False, reason="missing or invalid JUnit", cases=[])
    required = [
        node.split("::")[0][:-3].replace("/", ".") + "::" + node.split("::")[1] for node in expected
    ]
    observed = [case.get("classname", "") + "::" + case.get("name", "") for case in cases]
    result = [
        dict(
            case=name,
            status="fail"
            if any(case.find(k) is not None for k in ("skipped", "failure", "error"))
            else "pass",
        )
        for name, case in zip(observed, cases, strict=True)
    ]
    passed = (
        bool(required)
        and len(set(required)) == len(required)
        and sorted(observed) == sorted(required)
        and all(r["status"] == "pass" for r in result)
    )
    return dict(
        passed=passed, reason=None if passed else "case matrix or result failure", cases=result
    )


def main(output):
    import modal

    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo):
        raise ValueError("clean source required")
    protocol = json.loads((repo / PROTOCOL).read_text())
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted((repo / "src/openboost").rglob("*.py")) + [
        repo / p
        for p in (
            "pyproject.toml",
            "README.md",
            "LICENSE",
            PROTOCOL,
            "benchmarks/v1/cuda_aggregation_preflight.py",
            "tests/__init__.py",
            "tests/v1/__init__.py",
            "tests/v1/reference/__init__.py",
            "tests/v1/reference/device_histogram.py",
            "tests/v1/test_device_histogram_reference.py",
            *protocol["test_files"],
        )
    ]
    manifest = dict(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        dirty=False,
        argv=sys.argv,
        protocol=protocol,
        started=datetime.now(timezone.utc).isoformat(),
        status="running",
        sources={
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        requested=dict(cpu=2, memory_mib=8192, gpu="T4", timeout_seconds=900, retries=0),
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
    app = modal.App("openboost-v1-cuda-aggregation")

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
        import os
        import platform
        import time

        import cupy as cp
        from numba import cuda

        import openboost

        started = time.perf_counter()
        config = json.loads(Path("/snapshot/" + PROTOCOL).read_text())
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
        work = Path("/tmp/openboost-aggregation")
        work.mkdir()
        command = [
            sys.executable,
            "-m",
            "pytest",
            *["/snapshot/" + f for f in config["test_files"]],
            "--rootdir=/snapshot",
            "-o",
            "addopts=",
            "-q",
            "-s",
            "--junitxml=" + str(work / "junit.xml"),
        ]
        result["test_argv"] = command
        try:
            completed = subprocess.run(
                command, cwd=work, capture_output=True, text=True, timeout=600
            )
            result.update(exit_code=completed.returncode, log=completed.stdout + completed.stderr)
        except subprocess.TimeoutExpired as error:

            def decode(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

            result.update(exit_code="timeout", log=decode(error.stdout) + decode(error.stderr))
        result["junit"] = (work / "junit.xml").read_text() if (work / "junit.xml").exists() else ""
        result["worker_wall_seconds"] = time.perf_counter() - started
        return result

    try:
        with modal.enable_output(), app.run():
            result = run.remote()
        manifest["result"] = result
        for name, key in (("pytest.log", "log"), ("junit.xml", "junit")):
            (output / name).write_text(result.pop(key))
        verdict = judge_junit((output / "junit.xml").read_text(), protocol["expected_cases"])
        expected_sources = {
            k: v for k, v in manifest["sources"].items() if k.startswith("src/openboost/")
        }
        versions = {p.split("==")[0]: p.split("==")[1] for p in protocol["packages"]}
        verdict["installed_sources_match"] = result["installed_sources"] == expected_sources
        verdict["versions_match"] = result["packages"] == versions
        verdict["passed"] = (
            verdict["passed"]
            and result["exit_code"] == 0
            and verdict["installed_sources_match"]
            and verdict["versions_match"]
        )
        (output / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        manifest.update(
            status="pass" if verdict["passed"] else "fail",
            image_id=image.object_id,
            artifacts={
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in ("pytest.log", "junit.xml", "verdict.json")
            },
        )
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
