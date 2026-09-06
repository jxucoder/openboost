"""Real-data validation plumbing smoke; four rounds, not quality/search evidence."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmarks.v1.process_runner import execute
from benchmarks.v1.worker_data import export


def run(directory):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("fresh smoke directory required")
    repo = Path(__file__).resolve().parents[2]
    result = dict(
        scope="Real-data validation worker plumbing only; four rounds, no test scores or search selection, no quality or performance claim",
        source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        argv=[sys.executable, "-m", "benchmarks.v1.worker_data_smoke", str(directory)],
        python=platform.python_version(),
        os=platform.platform(),
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        device="cpu",
        gpu=None,
        memory_cap=None,
        threads=2,
        source_hashes={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in [
                "worker_data.py",
                "worker_data_smoke.py",
                "baseline_worker.py",
                "process_runner.py",
            ]
        },
        data={},
        cells=[],
    )
    for app in ["A1", "A2", "A3", "A5", "A6", "A11", "A12"]:
        packet = root / app
        manifest = export(app, packet)
        result["data"][app] = manifest
        for fold in manifest["folds"]:
            seed = fold["seed"]
            job = dict(
                application=app,
                library="catboost" if app == "A11" else "xgboost",
                device="cpu",
                threads=2,
                seed=seed,
                **({"classes": 7} if app == "A3" else {}),
                early_stopping_rounds=3,
                config=dict(
                    rounds=4,
                    learning_rate=0.1,
                    **(
                        dict(depth=2, l2_leaf_reg=1)
                        if app == "A11"
                        else dict(max_depth=2, reg_lambda=1)
                    ),
                ),
                input_npz=str(packet / fold["artifacts"]["worker-input"]["path"]),
            )
            job_path = packet / str(seed) / "job.json"
            job_path.write_text(json.dumps(job, indent=2) + "\n")
            output = packet / str(seed) / "fit"
            record = execute(
                [sys.executable, str(repo / "benchmarks/v1/baseline_worker.py"), str(job_path)],
                output,
                timeout_s=90,
                threads=2,
            )
            record.update(application=app, fold=seed, library=job["library"])
            if record["status"] == "pass":
                with (
                    np.load(output / "predictions.npz", allow_pickle=False) as predictions,
                    np.load(
                        packet / fold["artifacts"]["validation"]["path"], allow_pickle=False
                    ) as truth,
                ):
                    expected_shape = (
                        (len(truth["y"]), {"A11": 2, "A3": 7, "A5": 3}[app])
                        if app in ["A11", "A3", "A5"]
                        else truth["y"].shape
                    )
                    assert predictions["prediction"].shape == expected_shape
                    assert np.isfinite(predictions["prediction"]).all()
                    np.testing.assert_array_equal(predictions["row_ids"], truth["row_ids"])
                    if app in ["A2", "A3"]:
                        assert np.all(
                            (predictions["prediction"] >= 0) & (predictions["prediction"] <= 1)
                        )
                    if app == "A3":
                        np.testing.assert_allclose(
                            predictions["prediction"].sum(axis=1), 1, atol=1e-6
                        )
                training = json.loads((output / "training.json").read_text())
                if app == "A6":
                    assert training["target_scale"] == fold["metadata"]["target_scale"]
                record["training"] = training
            result["cells"].append(record)
            (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = run(args.directory)
    print(
        {
            "cells": len(result["cells"]),
            "passed": sum(c["status"] == "pass" for c in result["cells"]),
        }
    )
    if any(c["status"] != "pass" for c in result["cells"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
