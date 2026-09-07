"""One bounded T4 storage-ownership run; no training or authoring claim."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main(output):
    import modal

    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "status", "--porcelain"]):
        raise ValueError("clean source required")
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted((repo / "src/openboost").rglob("*.py")) + [
        repo / name
        for name in (
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "tests/v1/test_execution_cuda.py",
            "benchmarks/v1/cuda_storage_preflight.py",
        )
    ]
    manifest = dict(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=False,
        argv=sys.argv,
        scope="CUDA storage ownership only; not boosting or E1/E4 exit",
        sources={
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        budget=dict(
            gpu="T4", function_seconds=900, test_seconds=600, retries=0, run=1, allowed_runs=2
        ),
        status="running",
    )

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    image = (
        modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
        .uv_pip_install(
            "numpy==2.3.5",
            "cupy-cuda12x==13.6.0",
            "pytest==9.0.2",
            "hatchling==1.27.0",
            uv_version="0.12.1",
        )
        .env({"PYTHONDONTWRITEBYTECODE": "1"})
    )
    for p in paths:
        image = image.add_local_file(p, "/snapshot/" + str(p.relative_to(repo)), copy=True)
    image = image.uv_pip_install(
        "/snapshot", extra_options="--no-deps --no-build-isolation", uv_version="0.12.1"
    )
    app = modal.App("openboost-v1-cuda-storage")

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
        import platform

        import cupy as cp
        import numpy as np

        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        )
        try:
            tests = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "/snapshot/tests/v1/test_execution_cuda.py",
                    "-o",
                    "addopts=",
                    "-q",
                    "-s",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            status, log = tests.returncode, tests.stdout + tests.stderr
        except subprocess.TimeoutExpired as error:
            status = "timeout"
            log = str(error)
        return dict(
            passed=status == 0,
            exit_code=status,
            log=log,
            python=platform.python_version(),
            os=platform.platform(),
            cupy=cp.__version__,
            numpy=np.__version__,
            gpu=gpu,
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cuda_driver=cp.cuda.runtime.driverGetVersion(),
        )

    try:
        with modal.enable_output(), app.run():
            result = run.remote()
        (output / "pytest.log").write_text(result.pop("log"))
        manifest.update(
            status="complete",
            result=result,
            image_id=image.object_id,
            artifacts={
                "pytest.log": hashlib.sha256((output / "pytest.log").read_bytes()).hexdigest()
            },
        )
    except Exception as error:
        manifest.update(status="error", error=str(error))
        raise
    finally:
        save()
    if not manifest["result"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output.resolve())
