"""Synthetic baseline worker persistence and external exposure checks."""

import hashlib
import importlib.metadata
import json
import pickle
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.v1.baseline_worker import fit, predict_saved


def run(patience=None):
    rng = np.random.default_rng(41)
    x = rng.normal(size=(96, 5))
    rows = []
    configs = {
        "xgboost": {"max_depth": 2, "reg_lambda": 1.0},
        "lightgbm": {"num_leaves": 4, "lambda_l2": 1.0},
        "catboost": {"depth": 2, "l2_leaf_reg": 1.0},
        "ngboost": {"weak_depth": 2, "minibatch_frac": 1.0},
    }
    for library, config in configs.items():
        for number in range(1, 13):
            task = f"A{number}"
            if task == "A4" or (library == "ngboost" and task != "A11"):
                continue
            if (
                (library == "xgboost" and task == "A11")
                or (library == "lightgbm" and task in ["A10", "A11"])
                or (library == "catboost" and task == "A8")
            ):
                continue
            y = x[:, 0] + 0.1 * rng.normal(size=96)
            if task == "A2":
                y = (y > 0).astype(int)
            elif task == "A3":
                y = np.arange(96) % 3
            elif task == "A6":
                y = np.column_stack([y, x[:, 1]])
            elif task in ["A7", "A8", "A9", "A10"]:
                y = np.exp(y)
                if task == "A7":
                    y = np.round(y)
            arrays = dict(
                x_train=x[:72],
                y_train=y[:72],
                x_validation=x[72:],
                validation_row_ids=np.arange(72, 96),
                weight_train=np.linspace(0.5, 2, 72),
                exposure_train=np.linspace(0.2, 1, 72),
                exposure_validation=np.linspace(0.3, 1, 24),
                event_train=np.arange(72) % 3 != 0,
            )
            if task != "A7":
                arrays.pop("exposure_train")
                arrays.pop("exposure_validation")
            if task != "A10":
                arrays.pop("event_train")
            job = dict(
                application=task,
                library=library,
                seed=41,
                threads=2,
                device="cpu",
                classes=3,
                config=dict(rounds=4, learning_rate=0.1, **config),
            )
            if patience is not None:
                job["early_stopping_rounds"] = patience
                job["config"]["rounds"] = 24
                arrays["y_validation"] = (
                    -y[72:] if task in ["A1", "A12"] or library == "ngboost" else y[72:]
                )
                arrays["weight_validation"] = np.linspace(0.5, 2, 24)
                if task == "A10":
                    arrays["event_validation"] = np.arange(24) % 3 != 0
            prediction, saved = fit(job, arrays)
            reloaded = pickle.loads(pickle.dumps(saved))
            replay = predict_saved(reloaded, x[72:], arrays.get("exposure_validation"))
            np.testing.assert_allclose(prediction, replay, rtol=1e-7, atol=1e-8)
            record = dict(
                library=library,
                application=task,
                status="pass",
                shape=list(prediction.shape),
                reload_max_abs_error=float(np.max(np.abs(prediction - replay))),
            )
            if patience is not None:
                record["stopping"] = saved["stopping"]
                for stop in saved["stopping"]:
                    history = stop["history"]
                    validation = history.get("validation", history.get("val"))
                    values = next(iter(validation.values()))
                    assert stop["selected_rounds"] == int(np.argmin(values)) + 1
                    if task == "A1" or library == "ngboost":
                        assert stop["selected_rounds"] < len(values) < 24
            if patience is not None and (task == "A1" or library == "ngboost"):
                with tempfile.TemporaryDirectory() as temp:
                    folder = Path(temp)
                    (folder / "model.pkl").write_bytes(pickle.dumps(saved))
                    np.save(folder / "x.npy", x[72:])
                    code = "import pickle,sys,numpy as np; from pathlib import Path; from benchmarks.v1.baseline_worker import predict_saved; p=Path(sys.argv[1]); m=pickle.loads((p/'model.pkl').read_bytes()); np.save(p/'prediction.npy',predict_saved(m,np.load(p/'x.npy')))"
                    subprocess.run([sys.executable, "-c", code, temp], check=True, timeout=60)
                    new_process = np.load(folder / "prediction.npy")
                np.testing.assert_allclose(prediction, new_process, rtol=1e-7, atol=1e-8)
                record["new_process_max_abs_error"] = float(
                    np.max(np.abs(prediction - new_process))
                )
            if task == "A7":
                doubled = predict_saved(reloaded, x[72:], 2 * arrays["exposure_validation"])
                np.testing.assert_allclose(doubled, 2 * replay, rtol=1e-6, atol=1e-7)
                record["exposure_doubling_max_abs_error"] = float(
                    np.max(np.abs(doubled - 2 * replay))
                )
            rows.append(record)
    return rows


if __name__ == "__main__":
    result = dict(
        scope="synthetic numeric fixed-round baseline worker checks, not real quality",
        cells=run(),
        seed=41,
        threads=2,
        python=platform.python_version(),
        os=platform.platform(),
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        source_hashes={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ["worker_smoke.py", "baseline_worker.py"]
        },
    )
    Path("benchmarks/v1/evidence/worker-cpu.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["cells"])
