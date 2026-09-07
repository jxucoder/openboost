"""Installed finite-numeric A6 bin parameters, stopping and fresh-process replay."""

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmarks.v1.baseline_worker import fit


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(70)
    x = rng.normal(size=(160, 5))
    y = np.column_stack([10 + 2 * x[:, 0], -30 + x[:, 1], np.full(160, 7)])
    arrays = dict(
        x_train=x[:120],
        y_train=y[:120],
        x_validation=x[120:],
        y_validation=y[120:],
        validation_row_ids=np.arange(120, 160),
    )
    np.savez(output / "input.npz", **arrays)
    configs = dict(
        xgboost=dict(max_depth=2, reg_lambda=1),
        lightgbm=dict(num_leaves=4, lambda_l2=1),
        catboost=dict(depth=2, l2_leaf_reg=1),
    )
    rows = []
    for library, config in configs.items():
        for bins in (7, 255):
            job = dict(
                application="A6",
                library=library,
                device="cpu",
                threads=1,
                seed=70,
                early_stopping_rounds=3,
                config=dict(config, bins=bins, rounds=8, learning_rate=0.1),
            )
            prediction, saved = fit(job, arrays)
            model = saved["model"]
            if library == "xgboost":
                effective = int(
                    json.loads(model.save_config())["learner"]["gradient_booster"][
                        "tree_train_param"
                    ]["max_bin"]
                )
            elif library == "lightgbm":
                effective = [m.params["max_bin"] for m in model]
                assert effective == [bins] * 3
            else:
                effective = model.get_all_params()["border_count"]
            if library != "lightgbm":
                assert effective == (bins - 1 if library == "catboost" else bins)
            assert saved["stopping"]
            stem = f"{library}-{bins}"
            path = output / (stem + ".bin")
            path.write_bytes(pickle.dumps(saved))
            replay = output / (stem + ".npy")
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import pickle,sys,numpy as np; from benchmarks.v1.baseline_worker import predict_saved; "
                    's=pickle.load(open(sys.argv[1],"rb")); a=np.load(sys.argv[2]); '
                    'np.save(sys.argv[3],predict_saved(s,a["x_validation"]))',
                    str(path),
                    str(output / "input.npz"),
                    str(replay),
                ],
                check=True,
                timeout=60,
            )
            np.testing.assert_array_equal(prediction, np.load(replay))
            rows.append(
                dict(
                    library=library,
                    bins=bins,
                    effective=effective,
                    replay="exact",
                    job=job,
                    stopping=saved["stopping"],
                )
            )
    root = Path(__file__).parent
    record = dict(
        scope="Six synthetic CPU fits only; no real quality/resource/GPU gate",
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        argv=sys.argv,
        python=platform.python_version(),
        os=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in (*configs, "numpy")},
        sources={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("baseline_worker.py", "bin_budget_smoke.py", "preprocessing.py")
        },
        artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()},
        cells=rows,
        passed=True,
    )
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output.resolve())
