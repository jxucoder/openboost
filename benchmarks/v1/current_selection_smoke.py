"""Synthetic current A6 search, scale-bound audit and selected-model replay."""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmarks.v1.preprocessing import fit_target_scale
from benchmarks.v1.process_runner import execute
from benchmarks.v1.selection import audit, digest, release_test, seal


def run(directory, *, protected=False):
    if type(protected) is not bool:
        raise ValueError("protected mode must be boolean")
    if protected and (sys.platform != "linux" or os.geteuid() != 0):
        raise ValueError("protected selection requires a Linux root evaluator")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("fresh output directory required")
    evaluator = root / "evaluator" if protected else root
    candidate = root / "candidate" if protected else root
    if protected:
        root.chmod(0o755)
        evaluator.mkdir(mode=0o700)
        candidate.mkdir(mode=0o755)
    repo = Path(__file__).resolve().parents[2]
    rng = np.random.default_rng(46)
    x = rng.normal(size=(96, 3))
    y = np.column_stack([100 + 20 * x[:, 0], -300 + 0.01 * x[:, 1], np.full(96, 7.0)])
    np.savez(evaluator / "train-rows.npz", row_ids=np.arange(64))
    np.savez(evaluator / "train-targets.npz", row_ids=np.arange(64), y=y[:64])
    np.savez(evaluator / "validation.npz", row_ids=np.arange(64, 80), y=y[64:80])
    np.savez(evaluator / "test-features.npz", row_ids=np.arange(80, 96), x=x[80:])
    np.savez(
        candidate / "worker-input.npz",
        x_train=x[:64],
        y_train=y[:64],
        x_validation=x[64:80],
        y_validation=y[64:80],
        validation_row_ids=np.arange(64, 80),
    )
    if protected:
        (candidate / "worker-input.npz").chmod(0o444)
    scale = fit_target_scale(y[:64])
    (evaluator / "scale.json").write_text(json.dumps(scale, indent=2) + "\n")

    def entry(path):
        return dict(
            path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )

    configs = [
        dict(rounds=4, learning_rate=rate, max_depth=depth, mode=mode, bins=16, reg_lambda=1.0)
        for mode, rate, depth in itertools.product(
            ("shared", "independent"), (0.05, 0.1), (1, 2, 3, 4)
        )
    ]
    sources = {
        str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((repo / "src/openboost").rglob("*.py"))
    }
    for name in (
        "current_selection_smoke.py",
        "openboost_worker.py",
        "openboost_predict.py",
        "selection.py",
        "preprocessing.py",
        "quality.py",
        "quality_report.py",
        "process_runner.py",
    ):
        p = Path(__file__).with_name(name)
        sources[str(p.relative_to(repo))] = hashlib.sha256(p.read_bytes()).hexdigest()
    environment = dict(
        python=platform.python_version(),
        os=platform.platform(),
        threads=1,
        cpu_count=os.cpu_count(),
        gpu=None,
        memory_cap=None,
        address_limit_bytes=8 * 1024**3 if protected else None,
        worker_identity="uid_gid_65534_no_new_privs" if protected else "inherited",
        trial_timeout_s=1800 if protected else 60,
        packages={n: importlib.metadata.version(n) for n in ("openboost", "numpy")},
    )
    protocol = dict(
        schema="openboost-selection-v1",
        application="A6",
        fold=0,
        identity=dict(
            code=digest(sources),
            data=entry(candidate / "worker-input.npz")["sha256"],
            split=digest([64, 80, 96]),
            preprocessing=digest(scale),
            environment=digest(environment),
            search_design=digest(configs),
        ),
        train_rows=entry(evaluator / "train-rows.npz"),
        train_targets=entry(evaluator / "train-targets.npz"),
        target_scale=entry(evaluator / "scale.json"),
        validation=entry(evaluator / "validation.npz"),
        test_features=entry(evaluator / "test-features.npz"),
        selection_weights={f"rmse_{k}": 1 / std for k, std in enumerate(scale["std"])},
        methods={"openboost": configs},
    )
    pinned = digest(protocol)
    (evaluator / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    report = dict(
        scope=(
            "Synthetic A6/A13 selection integration with UID file permissions; "
            "not real quality, cost or complete hostile-code isolation"
            if protected
            else "Synthetic A6/A13 selection integration only; not real quality, cost or OS isolation"
        ),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        argv=[
            sys.executable,
            "-m",
            "benchmarks.v1.current_selection_smoke",
            str(root),
            *(["--protected"] if protected else []),
        ],
        sources=sources,
        environment=environment,
        protocol_sha256=pinned,
        passed=False,
        trials=[],
    )
    records = []
    try:
        for index, config in enumerate(configs):
            job = dict(
                application="A6",
                library="openboost",
                device="cpu",
                threads=1,
                seed=46,
                config=config,
                early_stopping_rounds=2,
                input_npz=str(candidate / "worker-input.npz"),
            )
            job_path = candidate / f"job-{index}.json"
            job_path.write_text(json.dumps(job, indent=2) + "\n")
            if protected:
                job_path.chmod(0o444)
            out = root / f"trial-{index}"
            outcome = execute(
                [sys.executable, str(repo / "benchmarks/v1/openboost_worker.py"), str(job_path)],
                out,
                timeout_s=1800 if protected else 60,
                threads=1,
                address_limit_bytes=8 * 1024**3 if protected else None,
                unprivileged=protected,
            )
            if protected:
                # Reclaim completed output before another worker with the same UID starts.
                os.chown(out, 0, 0)
                out.chmod(0o700)
            report["trials"].append(outcome)
            record = dict(
                id=f"openboost:{index:02}",
                config=config,
                status=outcome["status"],
                exit_code=outcome["exit_code"],
                protocol_sha256=pinned,
                prediction=entry(out / "predictions.npz")
                if (out / "predictions.npz").exists()
                else None,
                model=entry(out / "model.bin") if (out / "model.bin").exists() else None,
                log=entry(out / "execution.json"),
            )
            records.append(record)
            (evaluator / "records.json").write_text(json.dumps(records, indent=2) + "\n")
        receipt = audit(protocol, records, root, pinned)
        receipt_hash = seal(receipt, evaluator / "receipt.json")
        features, selected = release_test(
            protocol, records, evaluator / "receipt.json", root, pinned, receipt_hash
        )
        # Verify the exact selected bytes again before spawning inference.
        model = root / selected["path"]
        assert hashlib.sha256(model.read_bytes()).hexdigest() == selected["sha256"]
        np.savez(evaluator / "released-features.npz", **features)
        command = [
            sys.executable,
            str(repo / "benchmarks/v1/openboost_predict.py"),
            str(model),
            str(evaluator / "released-features.npz"),
            str(evaluator / "selected-predictions.npz"),
        ]
        subprocess.run(
            command,
            cwd=root,
            check=True,
            timeout=30,
            env=dict(
                os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
            ),
        )
        with np.load(evaluator / "selected-predictions.npz") as result:
            assert result["prediction"].shape == (16, 3) and np.isfinite(result["prediction"]).all()
            np.testing.assert_array_equal(result["row_ids"], features["row_ids"])
            np.testing.assert_array_equal(result["prediction"][:, 2], np.full(16, 7.0))
        report.update(
            passed=True,
            selected=receipt["selected"],
            receipt_sha256=receipt_hash,
            inference_command=command,
            selected_predictions=entry(evaluator / "selected-predictions.npz"),
        )
    finally:
        (evaluator / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--protected", action="store_true", help="Linux root evaluator only")
    args = parser.parse_args()
    print(run(args.directory, protected=args.protected)["selected"])
