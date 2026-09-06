"""Run every new parametric control through the bounded validation worker CLI."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmarks.v1.process_runner import execute

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("fresh output directory required")
    source = Path(__file__).with_name("parametric_worker.py").resolve()
    e, w = np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0])
    x = np.ones((3, 2))
    cells = []
    for app, method in [
        ("A7", "glm"),
        ("A8", "glm"),
        ("A9", "glm"),
        ("A9", "paid_composition"),
        ("A12", "formula_global"),
    ]:
        arrays = dict(x_train=x, x_validation=x, validation_row_ids=np.arange(3, 6), weight_train=w)
        config = dict(alpha=0.0, max_iter=1000)
        if app == "A7":
            arrays.update(y_train=np.array([1, 2, 0]), exposure_train=e, exposure_validation=2 * e)
            expected = 2 * e * 7 / 13
        elif app == "A8":
            arrays.update(
                y_train=np.array([10.0, 20.0, 40.0]), weight_train=np.array([1.0, 3.0, 3.0])
            )
            expected = np.full(3, 190 / 7)
        elif app == "A9":
            arrays.update(
                y_train=np.array([10.0, 60.0, 0.0]), exposure_train=e, exposure_validation=2 * e
            )
            expected = np.full(3, 190 / 13)
            if method == "paid_composition":
                arrays.update(
                    paid_count=np.array([1, 2, 0]),
                    claim_policy=np.array([0, 1, 1]),
                    claim_amount=np.array([10.0, 20.0, 40.0]),
                )
                config = dict(count_alpha=0.0, severity_alpha=0.0, max_iter=1000)
        else:
            age = np.linspace(0.1, 4, 40)
            valid = np.array([0.2, 1.0, 3.0])
            arrays = dict(
                age_train=age,
                y_train=3 * -np.expm1(-0.7 * age),
                age_validation=valid,
                validation_row_ids=np.arange(40, 43),
            )
            config = dict(initial_amplitude_multiplier=1.0, initial_rate=1.0, max_nfev=2000)
            expected = 3 * -np.expm1(-0.7 * valid)
        stem = app + "-" + method
        input_path = root / (stem + ".npz")
        np.savez(input_path, **arrays)
        job = dict(application=app, method=method, config=config, input_npz=str(input_path))
        job_path = root / (stem + ".json")
        job_path.write_text(json.dumps(job))
        out = root / stem
        result = execute([sys.executable, str(source), str(job_path)], out, timeout_s=60, threads=2)
        if result["status"] != "pass":
            raise RuntimeError(f"{stem} failed; inspect {out / 'worker.log'}")
        with np.load(out / "predictions.npz", allow_pickle=False) as pred:
            np.testing.assert_array_equal(pred["row_ids"], arrays["validation_row_ids"])
            np.testing.assert_allclose(pred["prediction"], expected, rtol=1e-4, atol=1e-8)
            cells.append(
                dict(
                    task=app,
                    method=method,
                    status="pass",
                    prediction=pred["prediction"].tolist(),
                    expected=expected.tolist(),
                )
            )
    result = dict(
        scope="synthetic CPU worker contract checks only",
        cells=cells,
        source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        source_hashes={
            n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in ["parametric.py", "parametric_worker.py", "parametric_worker_smoke.py"]
        },
    )
    Path("benchmarks/v1/evidence/parametric-worker-cpu.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(cells)
