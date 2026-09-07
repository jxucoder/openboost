"""Rerun current worker jobs from an existing frozen smoke summary, without export."""

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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(previous_path, directory):
    previous_path = Path(previous_path).resolve()
    previous = json.loads(previous_path.read_text())
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("fresh output directory required")
    repo = Path(__file__).resolve().parents[2]
    report = dict(
        scope="Unchanged full-data worker replay; no search, quality or comparative speed claim",
        previous_summary=dict(path=str(previous_path), sha256=digest(previous_path)),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        argv=[
            sys.executable,
            "-m",
            "benchmarks.v1.replay_current_packets",
            str(previous_path),
            str(root),
        ],
        python=platform.python_version(),
        os=platform.platform(),
        cpu_count=os.cpu_count(),
        packages={k: importlib.metadata.version(k) for k in ("openboost", "numpy")},
        threads=1,
        gpu=None,
        memory_cap=None,
        cells=[],
        sources={},
    )
    paths = [
        *sorted((repo / "src/openboost").rglob("*.py")),
        Path(__file__),
        Path(__file__).with_name("openboost_worker.py"),
        Path(__file__).with_name("openboost_predict.py"),
        Path(__file__).with_name("process_runner.py"),
    ]
    report["sources"] = {str(p.relative_to(repo)): digest(p) for p in paths}
    for cell in previous["cells"]:
        job = cell["job"]
        if job["threads"] != 1 or job["device"] != "cpu" or job["library"] != "openboost":
            raise ValueError("single-thread current CPU jobs required")
        app, fold = cell["application"], cell["fold"]
        frozen = next(f for f in previous["data"][app]["folds"] if f["seed"] == fold)
        input_path = Path(job["input_npz"])
        if digest(input_path) != frozen["artifacts"]["worker-input"]["sha256"]:
            raise ValueError("worker packet differs from frozen hash")
        job_path = root / f"{app}-{fold}-job.json"
        job_path.write_text(json.dumps(job, indent=2) + "\n")
        out = root / app / str(fold)
        record = execute(
            [sys.executable, str(repo / "benchmarks/v1/openboost_worker.py"), str(job_path)],
            out,
            timeout_s=90,
            threads=1,
        )
        record.update(application=app, fold=fold, job=job, input_sha256=digest(input_path))
        if record["status"] == "pass":
            try:
                with np.load(input_path, allow_pickle=False) as arrays:
                    np.savez(
                        out / "features.npz",
                        x=arrays["x_validation"],
                        row_ids=arrays["validation_row_ids"],
                    )
                command = [
                    sys.executable,
                    str(repo / "benchmarks/v1/openboost_predict.py"),
                    str(out / "model.bin"),
                    str(out / "features.npz"),
                    str(out / "replay.npz"),
                ]
                record["replay_command"] = command
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    timeout=30,
                    env=dict(
                        os.environ,
                        OMP_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                    ),
                )
                with (
                    np.load(out / "predictions.npz") as a,
                    np.load(out / "replay.npz") as b,
                    np.load(out / "features.npz") as f,
                ):
                    np.testing.assert_array_equal(a["prediction"], b["prediction"])
                    np.testing.assert_array_equal(a["row_ids"], b["row_ids"])
                    np.testing.assert_array_equal(a["row_ids"], f["row_ids"])
                    assert np.isfinite(a["prediction"]).all()
                    if app == "A3":
                        assert a["prediction"].shape == (len(a["row_ids"]), job["classes"])
                        assert ((a["prediction"] >= 0) & (a["prediction"] <= 1)).all()
                        np.testing.assert_allclose(a["prediction"].sum(axis=1), 1.0)
                record.update(
                    fresh_process_exact=True,
                    replay_sha256=digest(out / "replay.npz"),
                    training=json.loads((out / "training.json").read_text()),
                )
            except Exception as error:
                record.update(status="error", reason=str(error))
        report["cells"].append(record)
        (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{app}/{fold}: {record['status']} ({record['wall_s']:.1f}s)", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_summary", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = run(args.previous_summary, args.directory)
    if any(c["status"] != "pass" for c in result["cells"]):
        raise SystemExit(1)
