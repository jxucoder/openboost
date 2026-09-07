"""Allowlisted Linux integration of current protected selection and real recipes."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def probe():
    repo = Path("/snapshot")
    os.chdir(repo)
    # A new snapshot commit is distinct from the coordinator's original revision.
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=OpenBoost Probe",
            "-c",
            "user.email=probe@example.invalid",
            "commit",
            "-qm",
            "Allowlisted source snapshot",
        ],
        check=True,
    )
    root = Path("/tmp/selection-evidence")
    root.mkdir(mode=0o755)
    with (root / "pytest.log").open("wb") as log:
        tested = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/v1/test_current_selection_smoke.py",
                "-o",
                "addopts=",
                "-p",
                "no:cacheprovider",
                "-q",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
    sys.path.insert(0, str(repo))
    from benchmarks.v1.current_selection_smoke import run
    from benchmarks.v1.process_runner import execute

    try:
        report = run(root / "selection", protected=True)
    except Exception:
        import traceback

        (root / "failure.log").write_text(traceback.format_exc())
        return dict(
            passed=False,
            checks=dict(selection_and_replay=False),
            artifacts={
                str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
            },
        )
    checks = dict(
        linux_integration_tests=tested.returncode == 0, selection_and_replay=report["passed"]
    )
    for name, target, operation in (
        ("read_features", root / "selection/evaluator/test-features.npz", "read_bytes()"),
        ("write_protocol", root / "selection/evaluator/protocol.json", 'write_bytes(b"bad")'),
        ("read_completed_model", root / "selection/trial-0/model.bin", "read_bytes()"),
    ):
        before = hashlib.sha256(target.read_bytes()).hexdigest()
        out = root / name
        result = execute(
            [sys.executable, "-c", f"from pathlib import Path; Path({str(target)!r}).{operation}"],
            out,
            timeout_s=5,
            threads=1,
            address_limit_bytes=8 * 1024**3,
            unprivileged=True,
        )
        log = (out / "worker.log").read_text()
        checks[name + "_denied"] = (
            result["status"] == "error"
            and "PermissionError: [Errno 13]" in log
            and str(target) in log
            and hashlib.sha256(target.read_bytes()).hexdigest() == before
        )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        snapshot_revision=report["revision"],
        snapshot_dirty=report["dirty"],
        scope="Synthetic four-round selection with UID permissions; not full search or hostile-code sandbox",
        artifacts={
            str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
        },
    )


def main(output):
    import modal

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    repo = source.parents[2]
    names = [
        "selection_preflight",
        "current_selection_smoke",
        "openboost_worker",
        "openboost_predict",
        "process_runner",
        "selection",
        "preprocessing",
        "quality",
        "quality_report",
        "auxiliary",
        "judge",
        "ranking",
    ]
    paths = (
        sorted((repo / "src/openboost").rglob("*.py"))
        + [source.with_name(name + ".py") for name in names]
        + [
            repo / name
            for name in (
                "benchmarks/__init__.py",
                "benchmarks/v1/__init__.py",
                "tests/v1/test_current_selection_smoke.py",
                "pyproject.toml",
                "README.md",
                "LICENSE",
            )
        ]
    )
    manifest = dict(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        sources={
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        argv=sys.argv,
        requested=dict(cpu=[2, 2], memory_mib=[8192, 8192], function_seconds=180, retries=0),
        status="running",
        modal_version=modal.__version__,
    )

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    app = modal.App("openboost-v1-selection-preflight")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .uv_pip_install("numpy==2.3.5", "pytest==9.0.2", "hatchling==1.27.0", uv_version="0.12.1")
        .env({"PYTHONDONTWRITEBYTECODE": "1"})
    )
    for path in paths:
        image = image.add_local_file(path, "/snapshot/" + str(path.relative_to(repo)), copy=True)
    image = image.uv_pip_install(
        "/snapshot", extra_options="--no-deps --no-build-isolation", uv_version="0.12.1"
    )

    @app.function(
        image=image,
        cpu=(2, 2),
        memory=(8192, 8192),
        timeout=180,
        retries=0,
        max_containers=1,
        serialized=True,
        include_source=False,
    )
    def remote():
        import runpy

        return runpy.run_path("/snapshot/benchmarks/v1/selection_preflight.py")["probe"]()

    try:
        with modal.enable_output(), app.run():
            result = remote.remote()
            manifest["image_id"] = image.object_id
        for name, data in result.pop("artifacts").items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        manifest.update(
            status="complete",
            result=result,
            artifacts={
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )
    except Exception as error:
        manifest.update(status="error", error=str(error))
        raise
    finally:
        save()
    print(json.dumps(result["checks"], indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
