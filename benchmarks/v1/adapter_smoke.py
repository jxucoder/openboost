"""Probe baseline category/weight and explicit exposure prediction semantics."""

import hashlib
import importlib.metadata
import json
import pickle
import platform
import subprocess
from pathlib import Path

import numpy as np


def run():
    import catboost as cb
    import lightgbm as lgb
    import pandas as pd
    import xgboost as xgb
    from sklearn.linear_model import PoissonRegressor

    rng = np.random.default_rng(7)
    n = 96
    x = rng.normal(size=(n, 3))
    x[::9, 0] = np.nan
    category = np.array(["a", "b", "missing"] * (n // 3))
    frame = pd.DataFrame(x, columns=["x0", "x1", "x2"])
    frame["category"] = pd.Categorical(category, categories=["a", "b", "missing"])
    y = (np.nan_to_num(x[:, 0]) + rng.normal(size=n) > 0.1).astype(int)
    w = np.where(y, 3.0, 0.5)
    result = []
    for lib in ["xgboost", "lightgbm", "catboost"]:
        predictions = []
        for weight in [w, np.ones(n)]:
            if lib == "xgboost":
                d = xgb.DMatrix(frame, label=y, weight=weight, enable_categorical=True)
                m = xgb.train(
                    {
                        "objective": "binary:logistic",
                        "tree_method": "hist",
                        "max_depth": 2,
                        "nthread": 2,
                        "seed": 7,
                    },
                    d,
                    num_boost_round=4,
                )
                p = m.predict(d)
            elif lib == "lightgbm":
                d = lgb.Dataset(frame, label=y, weight=weight, categorical_feature=["category"])
                m = lgb.train(
                    {
                        "objective": "binary",
                        "num_leaves": 4,
                        "num_threads": 2,
                        "min_data_in_leaf": 2,
                        "verbosity": -1,
                        "seed": 7,
                    },
                    d,
                    num_boost_round=4,
                )
                p = m.predict(frame)
            else:
                d = cb.Pool(frame, label=y, weight=weight, cat_features=["category"])
                m = cb.CatBoostClassifier(
                    iterations=4,
                    depth=2,
                    thread_count=2,
                    verbose=False,
                    random_seed=7,
                    allow_writing_files=False,
                ).fit(d)
                p = m.predict_proba(d)[:, 1]
            if not np.isfinite(p).all() or not np.all((p >= 0) & (p <= 1)):
                raise ValueError("invalid class probabilities")
            predictions.append(p)
        difference = float(np.max(np.abs(predictions[0] - predictions[1])))
        if difference == 0:
            raise ValueError("weights had no observable effect")
        result.append(
            {
                "case": lib + " categories/missing/nonunit weights",
                "status": "pass",
                "weight_prediction_difference": difference,
            }
        )
    # A declared rate model carries exposure outside the saved model at inference.
    clean = np.nan_to_num(x)
    exposure = np.linspace(0.1, 2, n)
    count = rng.poisson(exposure * np.exp(clean[:, 0] / 3))
    m = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
        clean, count / exposure, sample_weight=w * exposure
    )
    loaded = pickle.loads(pickle.dumps(m))
    rate = loaded.predict(clean)
    np.testing.assert_array_equal(m.predict(clean) * exposure, rate * exposure)
    np.testing.assert_array_equal(rate * (2 * exposure), 2 * (rate * exposure))
    result.append({"case": "saved Poisson rate model plus explicit exposure", "status": "pass"})
    # XGBoost base_margin is supplied again after loading; it is not model state.
    d = xgb.DMatrix(clean, label=count, weight=w, base_margin=np.log(exposure))
    m = xgb.train(
        {
            "objective": "count:poisson",
            "tree_method": "hist",
            "max_depth": 2,
            "nthread": 2,
            "seed": 7,
        },
        d,
        num_boost_round=4,
    )
    loaded = xgb.Booster(model_file=m.save_raw())
    p1 = loaded.predict(xgb.DMatrix(clean, base_margin=np.log(exposure)))
    p2 = loaded.predict(xgb.DMatrix(clean, base_margin=np.log(2 * exposure)))
    np.testing.assert_allclose(p2, 2 * p1, rtol=1e-6, atol=1e-7)
    result.append(
        {
            "case": "XGBoost supplied base_margin after reload",
            "status": "pass",
            "double_exposure_max_error": float(np.max(np.abs(p2 - 2 * p1))),
        }
    )
    return {
        "scope": "CPU adapter probes only; full real-task pipeline pending",
        "cells": result,
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
    }


if __name__ == "__main__":
    result = run()
    result.update(
        source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        source_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    Path("benchmarks/v1/evidence/adapter-cpu.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(result["cells"])
