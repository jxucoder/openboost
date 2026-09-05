"""Resident P7 matrix: a negative value result is still complete evidence."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu


def test_value_matrix(checks):
    from dataset import describe
    from value_protocol import STRATEGIES, summarize

    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    assert describe(root / "cal_housing.tgz") == manifest["dataset"]
    frozen = json.loads((root / "p2_baseline.json").read_text())["checks"]["baseline_cells"]
    cells = checks["value_cells"] = []
    checks["frozen_baseline_cells"] = frozen
    checks["unsupported_comparisons"] = ["strict CUDA eval/callbacks"]
    for seed in (0, 1, 2):
        for strategy in STRATEGIES:
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "cell.json"
                argv = [
                    sys.executable,
                    str(root / "value_worker.py"),
                    strategy,
                    str(seed),
                    str(root / "cal_housing.tgz"),
                    str(output),
                ]
                try:
                    run = subprocess.run(
                        argv,
                        capture_output=True,
                        text=True,
                        timeout=180,
                        env={
                            **os.environ,
                            "OPENBOOST_BACKEND": "cpu" if strategy == "legacy_cpu" else "cuda",
                            "NUMBA_CACHE_DIR": temp,
                            "CUPY_CACHE_DIR": temp,
                        },
                    )
                    cell = (
                        json.loads(output.read_text())
                        if output.exists()
                        else dict(strategy=strategy, seed=seed)
                    )
                    cell["worker_stderr"] = run.stderr
                    if run.returncode:
                        cell["error"] = dict(
                            returncode=run.returncode, stdout=run.stdout, stderr=run.stderr
                        )
                except subprocess.TimeoutExpired:
                    cell = dict(
                        strategy=strategy, seed=seed, error="worker timeout after 180 seconds"
                    )
                cells.append(cell)
    # Do not assert quality/speed pass: regressions must survive in the artifact.
    checks["value_summary"] = summarize(cells, frozen)
