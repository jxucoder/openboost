"""Weighted query-aware ranking fit, validation stopping and score reload."""

import hashlib
import importlib.metadata
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

from benchmarks.v1.baseline_worker import fit, predict_saved
from benchmarks.v1.quality import metrics

if __name__ == "__main__":
    rng = np.random.default_rng(73)
    x = rng.normal(size=(96, 4))
    y = np.tile(np.arange(4), 24)
    x[:, 0] = y + rng.normal(size=96) * 0.5
    arrays = dict(
        x_train=x[:64],
        y_train=y[:64],
        x_validation=x[64:],
        y_validation=y[64:],
        validation_row_ids=np.arange(64, 96),
        query_train=np.repeat(np.arange(8), 8),
        query_validation=np.repeat(np.arange(8, 12), 8),
        query_weight_train=np.linspace(0.5, 2, 8),
        query_weight_validation=np.linspace(0.5, 2, 4),
    )
    results = []
    for library, cfg in dict(
        xgboost=dict(max_depth=2, reg_lambda=1.0),
        lightgbm=dict(num_leaves=4, lambda_l2=1.0),
        catboost=dict(depth=2, l2_leaf_reg=1.0),
    ).items():
        job = dict(
            application="A4",
            library=library,
            seed=73,
            threads=2,
            device="cpu",
            early_stopping_rounds=3,
            config=dict(rounds=16, learning_rate=0.1, **cfg),
        )
        prediction, saved = fit(job, arrays)
        replay = predict_saved(pickle.loads(pickle.dumps(saved)), x[64:])
        np.testing.assert_allclose(prediction, replay, rtol=1e-7, atol=1e-8)
        stop = saved["stopping"][0]
        native = stop["history"]["validation"]
        metric_name = next(k for k in native if k.lower().startswith("ndcg"))
        vals = native[metric_name]
        assert stop["selected_rounds"] == int(np.argmax(vals)) + 1
        scores = metrics(
            "A4", y[64:], prediction, query=arrays["query_validation"], row_ids=np.arange(64, 96)
        )
        results.append(
            dict(
                library=library,
                status="pass",
                stopping=saved["stopping"],
                metrics=scores,
                reload_max_abs_error=float(np.max(np.abs(prediction - replay))),
            )
        )
    result = dict(
        scope="synthetic CPU ranking adapter only; no real A4 quality or CUDA acceptance",
        cells=results,
        seed=73,
        threads=2,
        source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        source_hashes={
            n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in ["ranking_smoke.py", "ranking.py", "baseline_worker.py", "quality.py"]
        },
    )
    Path("benchmarks/v1/evidence/ranking-cpu.json").write_text(json.dumps(result, indent=2) + "\n")
    print(results)
