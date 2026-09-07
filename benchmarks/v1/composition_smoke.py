"""Five hash-bound composition fits and inference replays; no quality gate."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmarks.v1.paid_event_data import digest
from benchmarks.v1.process_runner import execute


def run(manifest, directory):
    manifest, directory = Path(manifest).resolve(), Path(directory).resolve()
    bound = json.loads(manifest.read_text())
    if len(bound["cells"]) != 5 or {c["fold"] for c in bound["cells"]} != set(range(5)):
        raise ValueError("all five bound folds required")
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("fresh output directory required")
    repo = Path(__file__).resolve().parents[2]
    script = Path(__file__).with_name("composition_worker.py")
    sources = [
        *sorted((repo / "src/openboost").glob("*.py")),
        Path(__file__),
        script,
        Path(__file__).with_name("process_runner.py"),
        Path(__file__).with_name("paid_event_data.py"),
    ]
    report = dict(
        scope="Component-selected composition validation only; no joint selection or quality gate",
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        argv=[
            sys.executable,
            "-m",
            "benchmarks.v1.composition_smoke",
            str(manifest),
            str(directory),
        ],
        binding_sha256=digest(manifest),
        python=platform.python_version(),
        os=platform.platform(),
        packages={n: importlib.metadata.version(n) for n in ["numpy", "openboost"]},
        cpu_count=os.cpu_count(),
        threads=1,
        memory_cap=None,
        gpu=None,
        sources={str(p.relative_to(repo)): digest(p) for p in sources},
        cells=[],
    )
    for cell in bound["cells"]:
        packet = manifest.parent / cell["path"]
        if digest(packet) != cell["sha256"]:
            raise ValueError("composition packet hash differs")
        job = dict(
            seed=cell["fold"],
            patience=3,
            config=dict(rounds=4, learning_rate=0.1, bins=32, max_depth=2, reg_lambda=1),
        )
        job_path = directory / f"job-{cell['fold']}.json"
        job_path.write_text(json.dumps(job, indent=2) + "\n")
        output = directory / str(cell["fold"])
        record = execute(
            [sys.executable, str(script), "fit", str(job_path), str(packet)],
            output,
            timeout_s=90,
            threads=1,
        )
        record.update(fold=cell["fold"], job=job, input_sha256=digest(packet))
        if record["status"] == "pass":
            try:
                with np.load(packet) as a:
                    np.savez(
                        output / "features.npz",
                        x=a["x_validation"],
                        row_ids=a["row_ids_validation"],
                        exposure=a["exposure_validation"],
                    )
                command = [
                    sys.executable,
                    str(script),
                    "predict",
                    str(output / "model.bin"),
                    str(output / "features.npz"),
                    str(output / "replay.npz"),
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
                    np.load(output / "predictions.npz") as a,
                    np.load(output / "replay.npz") as b,
                    np.load(output / "features.npz") as f,
                ):
                    assert (
                        set(a.files)
                        == set(b.files)
                        == {
                            "row_ids",
                            "paid_count_rate",
                            "paid_count_mean",
                            "severity_mean",
                            "annualized_mean",
                            "period_mean",
                        }
                    )
                    for k in a.files:
                        np.testing.assert_array_equal(a[k], b[k])
                    np.testing.assert_array_equal(a["row_ids"], f["row_ids"])
                    for k in set(a.files) - {"row_ids"}:
                        assert (
                            a[k].shape == a["row_ids"].shape
                            and np.isfinite(a[k]).all()
                            and (a[k] > 0).all()
                        )
                    np.testing.assert_allclose(
                        a["annualized_mean"], a["paid_count_rate"] * a["severity_mean"], rtol=1e-12
                    )
                    np.testing.assert_allclose(
                        a["period_mean"], a["annualized_mean"] * f["exposure"], rtol=1e-12
                    )
                    np.testing.assert_allclose(
                        a["paid_count_mean"], a["paid_count_rate"] * f["exposure"], rtol=1e-12
                    )
                record.update(
                    fresh_process_exact=True,
                    products_and_units=True,
                    replay_sha256=digest(output / "replay.npz"),
                    training=json.loads((output / "training.json").read_text()),
                )
            except Exception as error:
                record.update(status="error", reason=str(error))
        report["cells"].append(record)
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Fold {cell['fold']}: {record['status']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if any(c["status"] != "pass" for c in run(args.manifest, args.directory)["cells"]):
        raise SystemExit(1)
