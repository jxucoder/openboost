"""Isolated wheel-only Modal smoke; legacy source-mounted jobs are not loaded."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import modal

from benchmarks.foundation.prepare import BUNDLE, IMAGE, ROOT, sha256
from benchmarks.foundation.runner import validate_result

manifest = json.loads((BUNDLE / "manifest.json").read_text())
if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
    raise RuntimeError("Source is dirty; commit and prepare a new bundle first")
if (
    subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    != manifest["source_sha"]
):
    raise RuntimeError("Bundle is stale; run the prepare command again")
for name, digest in {**manifest["files"], **manifest.get("extension_wheels", {}), manifest["wheel"]: manifest["wheel_sha256"]}.items():
    if Path(name).name != name or sha256(BUNDLE / name) != digest:
        raise RuntimeError(f"Bundle integrity failure: {name}")

app = modal.App("openboost-foundation")
image = modal.Image.from_registry(IMAGE, add_python="3.12").uv_pip_install(
    requirements=[str(BUNDLE / "requirements.txt")],
    extra_options="--require-hashes",
    uv_version="0.12.1",
)
for name in [*manifest["files"], *manifest.get("extension_wheels", {}), manifest["wheel"], "manifest.json"]:
    image = image.add_local_file(BUNDLE / name, f"/opt/foundation/{name}", copy=True)
image = image.uv_pip_install(
    f"/opt/foundation/{manifest['wheel']}",
    extra_options="--no-deps",
    uv_version="0.12.1",
).env(
    {
        "OPENBOOST_BACKEND": "cuda",
        "OMP_NUM_THREADS": "2",
        "NUMBA_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "PYTHONPATH": "",
    }
)

for wheel in manifest.get("extension_wheels", {}):
    image = image.uv_pip_install(f"/opt/foundation/{wheel}", extra_options="--no-deps", uv_version="0.12.1")


@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=8192,
    max_containers=1,
    retries=0,
    timeout=manifest["timeout_s"],
    scaledown_window=2,
    serialized=True,
    include_source=False,
)
def smoke_job():
    import importlib.metadata
    import json
    import os
    import platform
    import subprocess
    import sys
    import time
    from pathlib import Path

    start = time.monotonic()
    directory = Path("/opt/foundation")
    source = json.loads((directory / "manifest.json").read_text())
    environment = {
        "os": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor(),
        "visible_cpu_count": os.cpu_count(),
        "requested_cpu": 2,
        "requested_memory_mib": 8192,
        "host_ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        environment["cpu_model"] = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name")), "not exposed")
    except OSError:
        environment["cpu_model"] = "not exposed"
    try:
        import cupy
        from numba import cuda

        device_name = cuda.get_current_device().name
        environment.update(
            cuda_available=cuda.is_available(),
            gpu_name=device_name.decode() if isinstance(device_name, bytes) else str(device_name),
            cuda_runtime=cupy.cuda.runtime.runtimeGetVersion(),
            cuda_driver=cupy.cuda.runtime.driverGetVersion(),
        )
        environment["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
    except Exception as exc:
        environment["error"] = f"{type(exc).__name__}: {exc}"
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        "pytest.ini",
        *source.get("test_files", ["test_smoke.py"]),
        "--junitxml=junit.xml",
        *(["--maxfail=1"] if source.get("suite") == "baseline" else []),
    ]
    result = {
        "environment": environment,
        "source_sha": source["source_sha"],
        "wheel_sha256": source["wheel_sha256"],
        "argv": argv,
        "timed_out": False,
    }
    try:
        completed = subprocess.run(argv, cwd=directory, capture_output=True, text=True, timeout=source["timeout_s"] - 60)
        result.update(
            returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr
        )
    except subprocess.TimeoutExpired as exc:
        result.update(
            returncode=-1,
            timed_out=True,
            stdout=(exc.stdout or b"").decode(errors="replace"),
            stderr=(exc.stderr or b"").decode(errors="replace"),
        )
    result["junit"] = (
        (directory / "junit.xml").read_text() if (directory / "junit.xml").exists() else ""
    )
    result["checks"] = (
        json.loads((directory / "checks.json").read_text())
        if (directory / "checks.json").exists()
        else {}
    )
    if source.get("suite") == "extensions" and result.get("returncode") == 0:
        try:
            uninstall = subprocess.run(
                ["/.uv/uv", "pip", "uninstall", "--system", "openboost-example-normal-fisher", "openboost-example-bounded-leaves"],
                cwd=directory, capture_output=True, text=True, timeout=30,
            )
            inference = subprocess.run(
                [sys.executable, "check_extension_inference.py"], cwd=directory,
                capture_output=True, text=True, timeout=30,
            )
            result["checks"]["extension_uninstall"] = {
                "uninstall_returncode": uninstall.returncode,
                "inference_returncode": inference.returncode,
                "inference_stdout": inference.stdout,
                "stderr": uninstall.stderr + inference.stderr,
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["checks"]["extension_uninstall"] = {"error": f"{type(exc).__name__}: {exc}"}
    result["remote_function_wall_s"] = time.monotonic() - start
    result["timing_scope"] = (
        "suite execution including environment checks/JIT; excludes image/startup, not billed duration; baseline cell timings have separate scopes"
    )
    return result


def run_suite(suite):
    if manifest.get("suite", "smoke") != suite:
        raise ValueError(f"Prepare --suite {suite} first")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = ROOT / "benchmarks/results/foundation" / run_id
    directory.mkdir(parents=True)
    saved_manifest = {**manifest, "run_id": run_id, "modal_image_id": image.object_id}
    (directory / "manifest.json").write_text(json.dumps(saved_manifest, indent=2) + "\n")
    try:
        result = smoke_job.remote()
    except Exception as exc:
        result = {"returncode": -1, "remote_error": f"{type(exc).__name__}: {exc}"}
        (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        raise
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (directory / "junit.xml").write_text(result.get("junit", ""))
    print(f"Evidence saved to {directory}")
    print(result.get("stdout", ""))
    validate_result(saved_manifest, result)
    print(f"Foundation {suite} passed.")


@app.local_entrypoint()
def foundation_smoke():
    run_suite("smoke")


@app.local_entrypoint()
def foundation_correctness():
    run_suite("correctness")


@app.local_entrypoint()
def foundation_boundaries():
    run_suite("boundaries")


@app.local_entrypoint()
def foundation_baseline():
    run_suite("baseline")


@app.local_entrypoint()
def foundation_histograms():
    run_suite("histograms")


@app.local_entrypoint()
def foundation_splits():
    run_suite("splits")


@app.local_entrypoint()
def foundation_leaves():
    run_suite("leaves")


@app.local_entrypoint()
def foundation_builder():
    run_suite("builder")


@app.local_entrypoint()
def foundation_trainer():
    run_suite("trainer")


@app.local_entrypoint()
def foundation_extensions():
    run_suite("extensions")


@app.local_entrypoint()
def foundation_value():
    run_suite("value")
