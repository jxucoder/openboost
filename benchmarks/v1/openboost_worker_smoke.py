"""Bounded current A1/A2/A3/A5/A6/A7/A8/A11 worker integration on all five frozen folds."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmarks.v1.process_runner import execute
from benchmarks.v1.worker_data import export


def run(directory, applications=("A1", "A6", "A11")):
    if (
        not applications
        or len(set(applications)) != len(applications)
        or set(applications) - {"A1", "A2", "A3", "A5", "A6", "A7", "A8", "A11"}
    ):
        raise ValueError("unique supported applications required")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("fresh output directory required")
    repo = Path(__file__).resolve().parents[2]
    report = dict(
        scope="Current A1/A2/A3/A5/A6/A7/A8/A11 real-data validation plumbing only; four rounds, no test scores, quality or performance claim",
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        argv=[
            sys.executable,
            "-m",
            "benchmarks.v1.openboost_worker_smoke",
            str(root),
            "--applications",
            *applications,
        ],
        python=platform.python_version(),
        os=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
        device="cpu",
        gpu=None,
        threads=1,
        memory_cap=None,
        packages={name: importlib.metadata.version(name) for name in ("numpy", "openboost")},
        sources={
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((repo / "src/openboost").rglob("*.py"))
        },
        data={},
        cells=[],
    )
    for name in (
        "openboost_worker.py",
        "openboost_predict.py",
        "openboost_worker_smoke.py",
        "worker_data.py",
        "preprocessing.py",
        "process_runner.py",
    ):
        p = Path(__file__).with_name(name)
        report["sources"][str(p.relative_to(repo))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for app in applications:
        packet = root / app
        manifest = export(app, packet)
        report["data"][app] = manifest
        for fold in manifest["folds"]:
            seed = fold["seed"]
            job = dict(
                application=app,
                library="openboost",
                device="cpu",
                threads=1,
                seed=seed,
                early_stopping_rounds=3,
                config=dict(rounds=4, learning_rate=0.1, max_depth=2, reg_lambda=1, bins=32),
                input_npz=str(packet / fold["artifacts"]["worker-input"]["path"]),
            )
            if app in {"A2", "A3"}:
                job["classes"] = 2 if app == "A2" else 7
            job_path = packet / str(seed) / "job.json"
            job_path.write_text(json.dumps(job, indent=2) + "\n")
            output = packet / str(seed) / "fit"
            record = execute(
                [sys.executable, str(repo / "benchmarks/v1/openboost_worker.py"), str(job_path)],
                output,
                timeout_s=90,
                threads=1,
            )
            record.update(application=app, fold=seed, job=job)
            if record["status"] == "pass":
                try:
                    with np.load(job["input_npz"], allow_pickle=False) as arrays:
                        np.savez(
                            output / "features.npz",
                            x=arrays["x_validation"],
                            row_ids=arrays["validation_row_ids"],
                            **({"exposure": arrays["exposure_validation"]} if app == "A7" else {}),
                        )
                    command = [
                        sys.executable,
                        str(repo / "benchmarks/v1/openboost_predict.py"),
                        str(output / "model.bin"),
                        str(output / "features.npz"),
                        str(output / "replay.npz"),
                    ]
                    record["replay_command"] = command
                    replay = subprocess.run(
                        command,
                        cwd=output,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env=dict(
                            os.environ,
                            OMP_NUM_THREADS="1",
                            OPENBLAS_NUM_THREADS="1",
                            MKL_NUM_THREADS="1",
                        ),
                    )
                    if replay.returncode:
                        raise RuntimeError(replay.stderr)
                    with (
                        np.load(output / "predictions.npz") as actual,
                        np.load(output / "replay.npz") as restored,
                    ):
                        np.testing.assert_array_equal(actual["row_ids"], restored["row_ids"])
                        np.testing.assert_array_equal(actual["prediction"], restored["prediction"])
                        expected_width = (
                            () if app in {"A1", "A2", "A7", "A8"} else (7,) if app == "A3" else (2,)
                        )
                        if app == "A5":
                            expected_width = (3,)
                        if app == "A6":
                            expected_width = (len(fold["metadata"]["target_scale"]["mean"]),)
                        assert actual["prediction"].shape == (
                            len(actual["row_ids"]),
                            *expected_width,
                        )
                        assert np.isfinite(actual["prediction"]).all()
                        if app in {"A7", "A8"}:
                            assert (actual["prediction"] > 0).all()
                        if app == "A11":
                            assert (actual["prediction"][:, 1] > 0).all()
                        if app in {"A2", "A3"}:
                            assert ((actual["prediction"] >= 0) & (actual["prediction"] <= 1)).all()
                            if app == "A3":
                                np.testing.assert_allclose(actual["prediction"].sum(axis=1), 1.0)
                        record["prediction_shape"] = list(actual["prediction"].shape)
                    record["fresh_process_exact"] = True
                    record["training"] = json.loads((output / "training.json").read_text())
                    if app == "A6":
                        assert (
                            record["training"]["target_scale"] == fold["metadata"]["target_scale"]
                        )
                    record["replay_sha256"] = hashlib.sha256(
                        (output / "replay.npz").read_bytes()
                    ).hexdigest()
                except Exception as error:
                    record.update(status="error", reason=str(error))
            report["cells"].append(record)
            (root / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--applications",
        nargs="+",
        choices=("A1", "A2", "A3", "A5", "A6", "A7", "A8", "A11"),
        default=["A1", "A6", "A11"],
    )
    args = parser.parse_args()
    result = run(args.directory, args.applications)
    passed = sum(c["status"] == "pass" for c in result["cells"])
    print(f"{passed}/{len(result['cells'])} current real-data worker cells passed")
    if passed != len(result["cells"]):
        raise SystemExit(1)
