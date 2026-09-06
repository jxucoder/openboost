"""Fixed real-data matrix, run only after the GPU correctness boundary suite."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_baseline_matrix(checks):
    root = Path(__file__).resolve().parent
    from dataset import describe

    source = json.loads((root / "manifest.json").read_text())
    assert describe(root / "cal_housing.tgz") == source["dataset"]
    results = checks["baseline_cells"] = []
    for seed in (0, 1, 2):
        for mode in ("resident", "eval"):
            for backend in ("cpu", "cuda"):
                with tempfile.TemporaryDirectory() as temp:
                    output = Path(temp) / "cell.json"
                    argv = [
                        sys.executable,
                        str(root / "baseline_worker.py"),
                        backend,
                        str(seed),
                        mode,
                        str(root / "cal_housing.tgz"),
                        str(output),
                    ]
                    run = subprocess.run(
                        argv,
                        capture_output=True,
                        text=True,
                        timeout=150,
                        env={**os.environ, "NUMBA_CACHE_DIR": temp},
                    )
                    if run.returncode:
                        checks["baseline_failure"] = {
                            "backend": backend,
                            "seed": seed,
                            "mode": mode,
                            "returncode": run.returncode,
                            "stdout": run.stdout,
                            "stderr": run.stderr,
                        }
                    assert run.returncode == 0, run.stderr
                    result = json.loads(output.read_text())
                    result["worker_stderr"] = run.stderr
                    results.append(result)
    # Predeclared design gates, checked for each seed and mode. No threshold tuning.
    for seed in (0, 1, 2):
        for mode in ("resident", "eval"):
            cpu, gpu = [
                next(c for c in results if (c["seed"], c["mode"], c["backend"]) == (seed, mode, b))
                for b in ("cpu", "cuda")
            ]
            a, b = (c["records"][1]["metrics"] for c in (cpu, gpu))
            assert abs(b["nll"] - a["nll"]) <= 0.01 * max(1, abs(a["nll"]))
            assert b["crps"] <= 1.01 * a["crps"]
            assert abs(b["coverage90"] - a["coverage90"]) <= 0.01
            for key in ("nll", "crps", "coverage90"):
                assert np.isfinite(a[key]) and np.isfinite(b[key])
        for backend in ("cpu", "cuda"):
            cells = [
                next(
                    c for c in results if (c["seed"], c["mode"], c["backend"]) == (seed, m, backend)
                )
                for m in ("resident", "eval")
            ]
            for key in ("nll", "crps", "coverage90"):
                np.testing.assert_allclose(
                    cells[0]["records"][1]["metrics"][key],
                    cells[1]["records"][1]["metrics"][key],
                    rtol=1e-4,
                    atol=1e-5,
                )
