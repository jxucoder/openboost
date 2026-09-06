"""Installed baseline capability probes: tiny weighted fit, prediction, and reload.

These probes are not real-data quality results or performance measurements.
Run separately on CPU and a real CUDA host; failures remain in the returned record.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


def run(device="cpu", library_filter=None, application_filter=None):
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

    rng = np.random.default_rng(41)
    x = rng.normal(size=(96, 5))
    y = 2 + x[:, 0] + 0.1 * rng.normal(size=96)
    weights = np.where(x[:, 1] > 0, 3.0, 0.5)
    binary = (x[:, 0] > 0).astype(int)
    multi = np.arange(96) % 3
    positive = np.exp(y / 2)
    count = rng.poisson(positive)
    rounds = 4
    cells = []
    for library in ["xgboost", "lightgbm", "catboost"]:
        if library_filter is not None and library != library_filter:
            continue
        for task in ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11"]:
            if application_filter is not None and task != application_filter:
                continue
            start = time.perf_counter()
            record = {"library": library, "application": task, "device": device, "status": "error"}
            try:
                if library == "catboost" and task == "A10" and device == "cuda":
                    record.update(
                        status="unsupported",
                        reason="SurvivalAft GPU fit rejected by version 1.2.10; retained initial evidence",
                    )
                    continue
                if (
                    (library == "lightgbm" and task in ["A10", "A11"])
                    or (library == "catboost" and task == "A8")
                    or (library == "xgboost" and task == "A11")
                ):
                    record.update(
                        status="unsupported",
                        reason="no matching builtin; separate outer-loop control required",
                    )
                    continue
                target = {
                    "A2": binary,
                    "A3": multi,
                    "A4": multi,
                    "A6": np.column_stack([y, 2 * y]),
                    "A7": count,
                    "A8": positive,
                    "A9": positive,
                    "A10": positive,
                }.get(task, y)
                with tempfile.TemporaryDirectory() as temp:
                    model_path = Path(temp) / "model.json"
                    if library == "xgboost":
                        objectives = {
                            "A1": "reg:squarederror",
                            "A2": "binary:logistic",
                            "A3": "multi:softprob",
                            "A4": "rank:pairwise",
                            "A5": "reg:quantileerror",
                            "A6": "reg:squarederror",
                            "A7": "count:poisson",
                            "A8": "reg:gamma",
                            "A9": "reg:tweedie",
                            "A10": "survival:aft",
                        }
                        param = {
                            "objective": objectives[task],
                            "tree_method": "hist",
                            "device": device,
                            "max_depth": 2,
                            "eta": 0.1,
                            "nthread": 2,
                            "seed": 41,
                        }
                        d = xgb.DMatrix(x, label=target if task != "A10" else None)
                        if task == "A4":
                            d.set_group([8] * 12)
                            d.set_weight(np.linspace(0.5, 2, 12))
                        else:
                            d.set_weight(weights)
                        if task == "A3":
                            param["num_class"] = 3
                        if task == "A5":
                            param["quantile_alpha"] = 0.5
                        if task == "A6":
                            param["multi_strategy"] = "multi_output_tree"
                        if task == "A9":
                            param["tweedie_variance_power"] = 1.5
                        if task == "A10":
                            d.set_float_info("label_lower_bound", positive)
                            d.set_float_info(
                                "label_upper_bound",
                                np.where(np.arange(96) % 4 == 0, np.inf, positive),
                            )
                            param.update(
                                aft_loss_distribution="normal", aft_loss_distribution_scale=1.0
                            )
                        model = xgb.train(param, d, num_boost_round=rounds)
                        # Actual build/config is recorded; CUDA host additionally validates GPU visibility.
                        actual = json.loads(model.save_config())["learner"]["generic_param"][
                            "device"
                        ]
                        if device == "cuda" and not actual.startswith("cuda"):
                            raise ValueError("silent CPU fallback")
                        before = model.predict(d)
                        model.save_model(model_path)
                        loaded = xgb.Booster()
                        loaded.load_model(model_path)
                        loaded.set_param({"device": device})
                        after = loaded.predict(d)
                        if device == "cuda":
                            loaded.set_param({"device": "cpu"})
                            cpu_prediction = loaded.predict(d)
                            np.testing.assert_allclose(before, cpu_prediction, rtol=1e-4, atol=1e-5)
                            record["cpu_inference_max_abs_error"] = float(
                                np.max(np.abs(before - cpu_prediction))
                            )
                        record["reload_device"] = device
                        record["effective_config"] = json.loads(model.save_config())
                    elif library == "lightgbm":
                        objectives = {
                            "A1": "regression",
                            "A2": "binary",
                            "A3": "multiclass",
                            "A4": "lambdarank",
                            "A5": "quantile",
                            "A6": "regression",
                            "A7": "poisson",
                            "A8": "gamma",
                            "A9": "tweedie",
                        }
                        param = {
                            "objective": objectives[task],
                            "device_type": device,
                            "num_leaves": 4,
                            "learning_rate": 0.1,
                            "num_threads": 2,
                            "min_data_in_leaf": 2,
                            "verbosity": -1,
                            "seed": 41,
                        }
                        if task == "A3":
                            param["num_class"] = 3
                        if task == "A5":
                            param["alpha"] = 0.5
                        if task == "A9":
                            param["tweedie_variance_power"] = 1.5
                        targets = target.T if task == "A6" else [target]
                        predictions = []
                        restored = []
                        for t in targets:
                            d = lgb.Dataset(
                                x,
                                label=t,
                                weight=weights
                                if task != "A4"
                                else np.repeat(np.linspace(0.5, 2, 12), 8),
                                group=[8] * 12 if task == "A4" else None,
                            )
                            model = lgb.train(param, d, num_boost_round=rounds)
                            predictions.append(model.predict(x))
                            model.save_model(str(model_path))
                            restored.append(lgb.Booster(model_file=str(model_path)).predict(x))
                        before = np.column_stack(predictions) if task == "A6" else predictions[0]
                        after = np.column_stack(restored) if task == "A6" else restored[0]
                        record["effective_config"] = param
                    else:
                        losses = {
                            "A1": "RMSE",
                            "A2": "Logloss",
                            "A3": "MultiClass",
                            "A4": "PairLogit",
                            "A5": "Quantile:alpha=0.5",
                            "A6": "MultiRMSE",
                            "A7": "Poisson",
                            "A9": "Tweedie:variance_power=1.5",
                            "A10": "SurvivalAft:dist=Normal;scale=1.0",
                            "A11": "RMSEWithUncertainty",
                        }
                        klass = (
                            cb.CatBoostClassifier
                            if task in ["A2", "A3"]
                            else cb.CatBoostRanker
                            if task == "A4"
                            else cb.CatBoostRegressor
                        )
                        model = klass(
                            loss_function=losses[task],
                            iterations=rounds,
                            depth=2,
                            learning_rate=0.1,
                            thread_count=2,
                            random_seed=41,
                            task_type="GPU" if device == "cuda" else "CPU",
                            verbose=False,
                            allow_writing_files=False,
                        )
                        if task == "A10":
                            target = np.column_stack(
                                [positive, np.where(np.arange(96) % 4 == 0, -1, positive)]
                            )
                        if task == "A4":
                            d = cb.Pool(
                                x,
                                target,
                                group_id=np.repeat(np.arange(12), 8),
                                group_weight=np.repeat(np.linspace(0.5, 2, 12), 8),
                            )
                        else:
                            d = cb.Pool(x, target, weight=weights)
                        model.fit(d)
                        before = (
                            model.predict(d)
                            if task == "A4"
                            else model.predict(d, prediction_type="RawFormulaVal")
                        )
                        model.save_model(str(model_path))
                        loaded = klass()
                        loaded.load_model(str(model_path))
                        after = (
                            loaded.predict(d)
                            if task == "A4"
                            else loaded.predict(d, prediction_type="RawFormulaVal")
                        )
                        record["effective_config"] = model.get_all_params()
                    if not np.isfinite(before).all() or len(before) != 96:
                        raise ValueError("invalid predictions")
                    np.testing.assert_allclose(before, after, rtol=1e-7, atol=1e-8)
                    record.update(
                        status="pass",
                        prediction_shape=list(np.shape(before)),
                        reload_max_abs_error=float(np.max(np.abs(before - after))),
                    )
            except Exception as exc:
                record.update(status="error", reason=f"{type(exc).__name__}: {exc}")
            finally:
                record["diagnostic_wall_s"] = time.perf_counter() - start
                cells.append(record)
    return {
        "schema": "openboost-capability-smoke-v1",
        "scope": "tiny weighted builtin fit/predict/reload only; not full capability or quality gates",
        "device": device,
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "cpu": platform.processor(),
            "cpu_count": os.cpu_count(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
        "cells": cells,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = run(a.device)
    result["source_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"]))
    result["source_file_sha256"] = (
        __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()
    )
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print([(r["library"], r["application"], r["status"]) for r in result["cells"]])
    return int(any(r["status"] == "error" for r in result["cells"]))


if __name__ == "__main__":
    raise SystemExit(main())
