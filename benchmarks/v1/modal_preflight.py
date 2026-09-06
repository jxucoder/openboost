"""Allowlisted comparator preflight; uploads no production source or datasets."""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("capability_smoke.py")
LOCK = Path(__file__).with_name("requirements-cuda.txt")
SOURCE_SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
DIRTY = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
HASH = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
LOCK_HASH = hashlib.sha256(LOCK.read_bytes()).hexdigest()
app = modal.App("openboost-v1-comparator-preflight")
image = (
    modal.Image.from_registry(
        "nvidia/cuda@sha256:14c54fad24b376ab78a70e1ef6595a2b7c8cdbf187e4f9b76de99a926fb62460",
        add_python="3.12",
    )
    .uv_pip_install(requirements=[str(LOCK)], extra_options="--require-hashes", uv_version="0.12.1")
    .add_local_file(SOURCE, "/opt/capability_smoke.py", copy=True)
    .env({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
)


# The standard wheel lacks CUDA: rebuild the same hash-locked release explicitly.
image = (
    image.apt_install("build-essential", "libboost-dev")
    .env({"CC": "gcc", "CXX": "g++", "CMAKE_BUILD_PARALLEL_LEVEL": "2"})
    .uv_pip_install(
        requirements=[str(LOCK)],
        extra_options="--require-hashes --reinstall-package lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=75",
        uv_version="0.12.1",
    )
)


@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=8192,
    timeout=1800,
    retries=0,
    max_containers=1,
    serialized=True,
    include_source=False,
)
def preflight():
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import cupy as cp

    results = {}
    code = "import importlib.util,json,sys; spec=importlib.util.spec_from_file_location('smoke','/opt/capability_smoke.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); r=m.run(*sys.argv[1:4]); open(sys.argv[4],'w').write(json.dumps(r))"
    for device in ["cpu", "cuda"]:
        cells = []
        environment = {}
        for library in ["xgboost", "lightgbm", "catboost"]:
            for application in [f"A{i}" for i in range(1, 12)]:
                with tempfile.TemporaryDirectory() as temp:
                    output = Path(temp) / "result.json"
                    try:
                        worker = subprocess.run(
                            [sys.executable, "-c", code, device, library, application, str(output)],
                            capture_output=True,
                            text=True,
                            timeout=90,
                        )
                        if worker.returncode != 0 or not output.exists():
                            cells.append(
                                {
                                    "library": library,
                                    "application": application,
                                    "device": device,
                                    "status": "error",
                                    "exit_code": worker.returncode,
                                    "reason": "native/worker failure",
                                    "log": worker.stdout + worker.stderr,
                                }
                            )
                        else:
                            payload = json.loads(output.read_text())
                            environment = payload["environment"]
                            for cell in payload["cells"]:
                                cell["worker_exit_code"] = worker.returncode
                                cell["worker_log"] = worker.stdout + worker.stderr
                                cells.append(cell)
                    except subprocess.TimeoutExpired:
                        cells.append(
                            {
                                "library": library,
                                "application": application,
                                "device": device,
                                "status": "timeout",
                                "reason": "90-second per-cell wall limit",
                            }
                        )
                print(device, library, application, cells[-1]["status"], flush=True)
        results[device] = {"cells": cells, "environment": environment}
    results.update(
        gpu_inventory=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ),
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        cuda_driver=cp.cuda.runtime.driverGetVersion(),
    )
    return results


@app.local_entrypoint()
def main():
    result = preflight.remote()
    result.update(
        source_sha=SOURCE_SHA,
        dirty=DIRTY,
        source_file_sha256=HASH,
        lock_sha256=LOCK_HASH,
        scope="comparator capability preflight only",
    )
    path = ROOT / "benchmarks/v1/evidence/modal-capabilities-isolated.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for device in ["cpu", "cuda"]:
        print(
            device, [(c["library"], c["application"], c["status"]) for c in result[device]["cells"]]
        )
    if any(
        c["status"] in {"error", "timeout"} for d in ["cpu", "cuda"] for c in result[d]["cells"]
    ):
        raise RuntimeError("preflight contains failed cells; inspect saved results")


@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=8192,
    timeout=600,
    retries=0,
    max_containers=1,
    serialized=True,
    include_source=False,
)
def pyboost_job():
    import tempfile
    import traceback
    from pathlib import Path

    import cupy as cp
    import numpy as np
    import py_boost
    from py_boost import GradientBoosting

    rng = np.random.default_rng(73)
    x = rng.normal(size=(96, 5)).astype(np.float32)
    weights = np.linspace(0.5, 2, 96, dtype=np.float32)
    cells = []
    for outputs in [1, 2]:
        try:
            y = np.column_stack([x[:, 0] + i * x[:, 1] for i in range(outputs)]).astype(np.float32)
            model = GradientBoosting(
                "mse", ntrees=4, lr=0.1, max_depth=2, min_data_in_leaf=2, seed=73, verbose=10
            )
            model.fit(x, y, sample_weight=weights)
            before = model.predict(x)
            with tempfile.TemporaryDirectory() as temp:
                path = str(Path(temp) / "model.json")
                model.dump(path)
                loaded = GradientBoosting("mse")
                loaded.load(path)
                after = loaded.predict(x)
            if before.shape != (96, outputs) or not np.isfinite(before).all():
                raise ValueError("invalid prediction shape/value")
            np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)
            cells.append(
                {
                    "outputs": outputs,
                    "status": "pass",
                    "reload_max_abs_error": float(np.max(np.abs(before - after))),
                }
            )
        except Exception:
            cells.append({"outputs": outputs, "status": "error", "reason": traceback.format_exc()})
    return {
        "cells": cells,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "pyboost_path": str(Path(py_boost.__file__).parent.name),
        "scope": "weighted scalar/vector GPU fit and JSON reload only",
    }


@app.local_entrypoint()
def pyboost():
    result = pyboost_job.remote()
    result.update(
        source_sha=SOURCE_SHA,
        dirty=DIRTY,
        lock_sha256=LOCK_HASH,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (ROOT / "benchmarks/v1/evidence/pyboost-cuda.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(result["cells"])
    if any(c["status"] != "pass" for c in result["cells"]):
        raise RuntimeError("Py-Boost preflight failed")
