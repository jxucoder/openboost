"""One fresh-process baseline cell: first fit and repeated fit, same inputs."""

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

if __package__:
    from .dataset import load_housing, split_indices
else:
    from dataset import load_housing, split_indices

CONFIG = dict(
    n_trees=30,
    max_depth=3,
    learning_rate=0.05,
    n_bins=64,
    min_child_weight=1.0,
    reg_lambda=1.0,
    subsample=1.0,
    colsample_bytree=1.0,
)


def normal_metrics(y, params):
    mu, sigma = (np.asarray(params[k], dtype=np.float64) for k in ("loc", "scale"))
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(sigma > 0)):
        raise ValueError("Invalid Normal parameters")
    z = (y - mu) / sigma
    return {
        "nll": float(np.mean(np.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5 * z**2)),
        "crps": float(
            np.mean(
                sigma
                * (
                    z * (2 * ndtr(z) - 1)
                    + 2 * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
                    - 1 / np.sqrt(np.pi)
                )
            )
        ),
        "coverage90": float(np.mean(np.abs(z) <= ndtri(0.95))),
    }


def run_cell(backend, seed, mode, archive):
    import openboost as ob
    import openboost._trainer as trainer
    from openboost._objectives import DistributionObjective

    X, y = load_housing(archive)
    train, val, test = split_indices(len(y), seed)
    # No scaling or target fitting outside train; fit bins inside timed model.fit.
    X_train, y_train = X[train], y[train]
    X_val, y_val, X_test, y_test = X[val], y[val], X[test], y[test]

    def sync():
        pass

    if backend == "cuda":
        from numba import cuda

        sync = cuda.synchronize
    counts = {}
    original_tree, original_step, original_host = (
        trainer.fit_tree_gpu_native,
        DistributionObjective.step,
        trainer._to_host,
    )

    def native(*args, **kwargs):
        counts["native_tree_calls"] += 1
        return original_tree(*args, **kwargs)

    def step(self, raw, *args, **kwargs):
        output = original_step(self, raw, *args, **kwargs)
        if backend == "cuda":
            assert all(hasattr(a, "__cuda_array_interface__") for a in raw.values())
            assert all(
                hasattr(a, "__cuda_array_interface__") for pair in output.values() for a in pair
            )
        counts["objective_calls"] += 1
        return output

    def to_host(value):
        if hasattr(value, "__cuda_array_interface__"):
            counts["trainer_device_to_host_calls"] += 1
        return original_host(value)

    trainer.fit_tree_gpu_native = native
    DistributionObjective.step = step
    trainer._to_host = to_host
    records = []
    previous_params = None
    try:
        with ob.backend_context(backend):
            for phase in ("first_fit", "repeat_fit"):
                counts = dict(
                    native_tree_calls=0, objective_calls=0, trainer_device_to_host_calls=0
                )
                model = ob.NaturalBoostNormal(**CONFIG, random_state=seed)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    sync()
                    started = time.perf_counter()
                    model.fit(
                        X_train,
                        y_train,
                        **({"eval_set": [(X_val, y_val)]} if mode == "eval" else {}),
                    )
                    sync()
                    fit_s = time.perf_counter() - started
                    fit_counts = counts.copy()
                    started = time.perf_counter()
                    params = model.predict_params(X_test)
                    sync()
                    predict_s = time.perf_counter() - started
                fallback = [str(w.message) for w in caught if "fallback" in str(w.message).lower()]
                record = {
                    "phase": phase,
                    "fit_s": fit_s,
                    "predict_params_s": predict_s,
                    "metrics": normal_metrics(y_test, params),
                    "fit_path": fit_counts,
                    "fallback_warnings": fallback,
                    "eval_history": model.evals_result_,
                    "prediction_sha256": hashlib.sha256(
                        b"".join(
                            np.asarray(params[k], dtype="<f4").tobytes() for k in ("loc", "scale")
                        )
                    ).hexdigest(),
                }
                records.append(record)
                if previous_params is not None:
                    record["repeat_prediction_max_abs_error"] = max(
                        float(np.max(np.abs(params[k] - previous_params[k]))) for k in params
                    )
                    for k in params:
                        if backend == "cpu":
                            np.testing.assert_array_equal(params[k], previous_params[k])
                        else:
                            np.testing.assert_allclose(
                                params[k], previous_params[k], rtol=2e-5, atol=2e-6
                            )
                previous_params = {k: v.copy() for k, v in params.items()}
                assert not fallback, fallback
                assert fit_counts["objective_calls"] == CONFIG["n_trees"]
                assert fit_counts["native_tree_calls"] == (
                    CONFIG["n_trees"] * 2 if backend == "cuda" else 0
                )
    finally:
        trainer.fit_tree_gpu_native = original_tree
        DistributionObjective.step = original_step
        trainer._to_host = original_host
    return {
        "backend": backend,
        "seed": seed,
        "mode": mode,
        "config": CONFIG,
        "split_sizes": [len(train), len(val), len(test)],
        "records": records,
        "timing_scope": "First/repeat fit within fresh Python process, fresh NUMBA_CACHE_DIR; includes binning, gradients, transfers and JIT; excludes imports/data loading/startup. Prediction includes test binning. Path instrumentation included.",
        "transfer_scope": "Counts only trainer._to_host device arguments during fit; excludes compact tree conversion and backend internal copies; not a total transfer count.",
    }


if __name__ == "__main__":
    backend, seed, mode, archive, output = sys.argv[1:]
    result = run_cell(backend, int(seed), mode, archive)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
