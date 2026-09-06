"""Execute a synthetic 16-trial CPU search and sealed test release end to end.

Run with the pinned baseline interpreter. Outputs are synthetic harness evidence,
not a preregistered real-data search or an OpenBoost quality/performance result.
"""

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
from benchmarks.v1.quality import metrics
from benchmarks.v1.selection import audit, digest, release_test, seal


def run(directory, patience=None):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("fresh smoke directory required")
    repo = Path(__file__).resolve().parents[2]
    rng = np.random.default_rng(73)
    x = rng.normal(size=(96, 3))
    y = x[:, 0] - 0.3 * x[:, 1]
    np.savez(root / "train-rows.npz", row_ids=np.arange(64))
    np.savez(root / "validation.npz", row_ids=np.arange(64, 80), y=y[64:80])
    np.savez(root / "test-features.npz", row_ids=np.arange(80, 96), x=x[80:])
    np.savez(
        root / "worker-input.npz",
        x_train=x[:64],
        y_train=y[:64],
        x_validation=x[64:80],
        validation_row_ids=np.arange(64, 80),
    )

    def entry(path):
        return {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    if patience is not None:
        with np.load(root / "worker-input.npz", allow_pickle=False) as d:
            arrays = {k: d[k] for k in d.files}
        np.savez(root / "worker-input.npz", **arrays, y_validation=y[64:80])

    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            repo / "benchmarks/v1/baseline_worker.py",
            repo / "benchmarks/v1/selection.py",
            repo / "benchmarks/v1/process_runner.py",
            repo / "benchmarks/v1/quality.py",
        ]
    }
    configs = [dict(rounds=i, learning_rate=0.1, max_depth=2, reg_lambda=1.0) for i in range(1, 17)]
    protocol = dict(
        schema="openboost-selection-v1",
        application="A1",
        fold=0,
        identity=dict(
            code=digest(sources),
            data=digest(dict(seed=73, rows=96, features=3)),
            split=digest([64, 80, 96]),
            preprocessing=digest("identity numeric"),
            environment=digest(versions),
            search_design=digest({"configs": configs, "early_stopping_rounds": patience}),
        ),
        train_rows=entry(root / "train-rows.npz"),
        validation=entry(root / "validation.npz"),
        test_features=entry(root / "test-features.npz"),
        selection_weights={"rmse": 1.0},
        methods={"xgboost": configs},
    )
    pinned_protocol = digest(protocol)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    for i, config in enumerate(configs):
        trial = f"xgboost:{i:02}"
        job = dict(
            application="A1",
            library="xgboost",
            device="cpu",
            seed=73,
            threads=2,
            config=config,
            input_npz=str(root / "worker-input.npz"),
        )
        if patience is not None:
            job["early_stopping_rounds"] = patience
        job_path = root / f"job-{i}.json"
        job_path.write_text(json.dumps(job))
        out = root / f"trial-{i}"
        result = execute(
            [sys.executable, str(repo / "benchmarks/v1/baseline_worker.py"), str(job_path)],
            out,
            timeout_s=60,
            threads=2,
        )
        if result["status"] != "pass":
            raise RuntimeError(f"{trial}: {result['reason']}; see {out / 'worker.log'}")
        records.append(
            dict(
                id=trial,
                config=config,
                protocol_sha256=pinned_protocol,
                status=result["status"],
                exit_code=result["exit_code"],
                prediction=entry(out / "predictions.npz"),
                model=entry(out / "model.bin"),
                log=entry(out / "execution.json"),
            )
        )
    (root / "records.json").write_text(json.dumps(records, indent=2) + "\n")
    receipt = audit(protocol, records, root, pinned_protocol)
    receipt_hash = seal(receipt, root / "receipt.json")
    features, selected_model = release_test(
        protocol, records, root / "receipt.json", root, pinned_protocol, receipt_hash
    )
    # This downstream process is invoked only after release and gets one model.
    # The runner never gives the validation workers the test-feature path.
    code = """import hashlib,pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[1])
from benchmarks.v1.baseline_worker import predict_saved
raw=Path(sys.argv[2]).read_bytes()
if hashlib.sha256(raw).hexdigest()!=sys.argv[3]: raise ValueError("model hash changed")
with np.load(sys.argv[4],allow_pickle=False) as d:
 p=predict_saved(pickle.loads(raw),d["x"])
 np.savez(sys.argv[5],row_ids=d["row_ids"],prediction=p)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(repo),
            str(root / selected_model["path"]),
            selected_model["sha256"],
            str(root / "test-features.npz"),
            str(root / "test-predictions.npz"),
        ],
        check=True,
        timeout=60,
    )
    with np.load(root / "test-predictions.npz", allow_pickle=False) as d:
        np.testing.assert_array_equal(d["row_ids"], features["row_ids"])
        score = metrics("A1", y[80:], d["prediction"])
    result = dict(
        scope="synthetic execution smoke only; no E3 or performance acceptance",
        status="pass",
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo)),
        source_hashes=sources,
        packages=versions,
        python=platform.python_version(),
        os=platform.platform(),
        threads=2,
        seed=73,
        trials=16,
        early_stopping_rounds=patience,
        selected=receipt["selected"],
        protocol_sha256=pinned_protocol,
        receipt_sha256=receipt_hash,
        test_metrics=score,
    )
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--early-stopping-rounds", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.directory, args.early_stopping_rounds), indent=2))
