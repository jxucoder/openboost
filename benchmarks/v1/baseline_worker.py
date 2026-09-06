"""One numeric baseline trial producing validation predictions and a saved model.

Input NPZ fields: x_train, y_train, x_validation, validation_row_ids; optional
weight_train, exposure_train/exposure_validation, event_train. Early stopping
requires y_validation and accepts weight_validation/event_validation. Preprocessing and
split identity are supplied by the frozen caller. No test arrays are read.
"""

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np


def fit(job, arrays):
    task, library = job["application"], job["library"]
    if task not in {f"A{i}" for i in range(1, 13)}:
        raise ValueError("unsupported worker task")
    if library not in ["xgboost", "lightgbm", "catboost", "ngboost"]:
        raise ValueError("unknown baseline library")
    allowed_job = {
        "application",
        "library",
        "seed",
        "threads",
        "device",
        "config",
        "classes",
        "early_stopping_rounds",
        "input_npz",
    }
    if set(job) - allowed_job:
        raise ValueError("unsupported job fields")
    allowed_arrays = {"x_train", "y_train", "x_validation", "validation_row_ids", "weight_train"}
    patience = job.get("early_stopping_rounds")
    if patience is not None:
        if type(patience) is not int or patience <= 0:
            raise ValueError("positive integer early stopping patience required")
        allowed_arrays.update({"y_validation", "weight_validation"})
        if task == "A10":
            allowed_arrays.add("event_validation")
    if task == "A4":
        allowed_arrays.update({"query_train", "query_validation", "query_weight_train"})
        if patience is not None:
            allowed_arrays.add("query_weight_validation")
    if task == "A7":
        allowed_arrays.update({"exposure_train", "exposure_validation"})
    if task == "A10":
        allowed_arrays.add("event_train")
    if set(arrays) - allowed_arrays:
        raise ValueError("unsupported input arrays")
    x, y = arrays["x_train"], arrays["y_train"]
    v = arrays["x_validation"]
    if x.ndim != 2 or v.ndim != 2 or x.shape[1] != v.shape[1] or len(y) != len(x):
        raise ValueError("invalid training schema")
    if not np.isfinite(x).all() or not np.isfinite(v).all() or not np.isfinite(y).all():
        raise ValueError("encoded finite inputs required")
    if job["device"] not in {"cpu", "cuda"}:
        raise ValueError("unsupported device")
    if type(job["threads"]) is not int or job["threads"] <= 0:
        raise ValueError("positive thread count required")
    if type(job["seed"]) is not int or job["seed"] < 0:
        raise ValueError("nonnegative integer seed required")
    ids = arrays["validation_row_ids"]
    if ids.ndim != 1 or len(ids) != len(v) or len(np.unique(ids)) != len(ids):
        raise ValueError("invalid validation row IDs")
    if task == "A10":
        event = arrays["event_train"]
        if event.shape != y.shape or not np.isin(event, [0, 1]).all() or np.any(y <= 0):
            raise ValueError("invalid survival targets")
    ranking = None
    if task == "A4":
        # This file is also invoked directly by process_runner.
        if __package__:
            from benchmarks.v1.ranking import validate
        else:
            from ranking import validate
        ranking = validate(arrays, patience)
    w = arrays.get("weight_train", np.ones(len(y)))
    if w.shape != (len(y),) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("invalid sample weights")
    cfg = dict(job["config"])
    rounds = cfg.pop("rounds")
    seed = job["seed"]
    lr = cfg.pop("learning_rate")
    if cfg.pop("seed_from_fold", True) is not True:
        raise ValueError("seed semantics differ")
    vy = vw = None
    if patience is not None:
        if "y_validation" not in arrays:
            raise ValueError("early stopping requires explicit validation targets")
        vy = arrays["y_validation"]
        vw = arrays.get("weight_validation", np.ones(len(v)))
        if vy.shape != (len(v), *y.shape[1:]) or not np.isfinite(vy).all():
            raise ValueError("invalid validation targets")
        if vw.shape != (len(v),) or not np.isfinite(vw).all() or np.any(vw < 0) or vw.sum() <= 0:
            raise ValueError("invalid validation weights")
        if task == "A10":
            ve = arrays.get("event_validation")
            if (
                ve is None
                or ve.shape != vy.shape
                or not np.isin(ve, [0, 1]).all()
                or np.any(vy <= 0)
            ):
                raise ValueError("invalid validation survival targets")
    target_scale = None
    if task == "A6":
        if y.ndim != 2 or not y.shape[1]:
            raise ValueError("nonempty matrix targets required for A6")
        if __package__:
            from benchmarks.v1.preprocessing import fit_target_scale
        else:
            from preprocessing import fit_target_scale
        target_scale = fit_target_scale(y)
        mean, std = np.asarray(target_scale["mean"]), np.asarray(target_scale["std"])
        y = (y - mean) / std
        if vy is not None:
            vy = (vy - mean) / std
    stopping = []
    prediction_rounds = None
    if type(rounds) is not int or rounds <= 0:
        raise ValueError("positive rounds required")
    exposure = validation_exposure = None
    base = 0.0
    if task == "A7":
        exposure = arrays["exposure_train"]
        validation_exposure = arrays["exposure_validation"]
        if (
            exposure.shape != (len(y),)
            or validation_exposure.shape != (len(v),)
            or not np.isfinite(exposure).all()
            or not np.isfinite(validation_exposure).all()
            or np.any(exposure <= 0)
            or np.any(validation_exposure <= 0)
        ):
            raise ValueError("invalid exposure")
        base = math.log(max(float(np.dot(w, y) / np.dot(w, exposure)), 1e-12))
    prediction = None
    if library == "xgboost":
        import xgboost as xgb

        if task == "A11":
            raise ValueError("Normal needs NGBoost/CatBoost or an outer-loop adapter")
        if set(cfg) != {"max_depth", "reg_lambda"}:
            raise ValueError("unsupported XGBoost parameters")
        objectives = {
            "A1": "reg:squarederror",
            "A2": "binary:logistic",
            "A3": "multi:softprob",
            "A4": "rank:ndcg",
            "A5": "reg:quantileerror",
            "A6": "reg:squarederror",
            "A7": "count:poisson",
            "A8": "reg:gamma",
            "A9": "reg:tweedie",
            "A10": "survival:aft",
            "A12": "reg:squarederror",
        }
        params = dict(
            cfg,
            objective=objectives[task],
            eta=lr,
            nthread=job["threads"],
            seed=seed,
            device=job["device"],
            tree_method="hist",
        )
        if task == "A3":
            params["num_class"] = job["classes"]
        if task == "A6":
            params["multi_strategy"] = "multi_output_tree"
            params["base_score"] = 0.0
        if task == "A5":
            params["quantile_alpha"] = [0.1, 0.5, 0.9]
        if task == "A9":
            params["tweedie_variance_power"] = 1.5
        d = xgb.DMatrix(x, label=None if task == "A10" else y, weight=None if task == "A4" else w)
        validation = xgb.DMatrix(v)
        if task == "A4":
            params["eval_metric"] = "ndcg@10"
            d.set_group(ranking[0][1])
            d.set_weight(ranking[0][2])
            validation.set_group(ranking[1][1])
        if task == "A7":
            d.set_base_margin(base + np.log(exposure))
            validation.set_base_margin(base + np.log(validation_exposure))
        if task == "A10":
            event = arrays["event_train"]
            d.set_float_info("label_lower_bound", y)
            d.set_float_info("label_upper_bound", np.where(event, y, np.inf))
            params.update(aft_loss_distribution="normal", aft_loss_distribution_scale=1.0)
        history = {}
        if patience is not None:
            if task == "A10":
                validation.set_float_info("label_lower_bound", vy)
                validation.set_float_info(
                    "label_upper_bound", np.where(arrays["event_validation"], vy, np.inf)
                )
            else:
                validation.set_label(vy)
            validation.set_weight(ranking[1][2] if task == "A4" else vw)
        model = xgb.train(
            params,
            d,
            num_boost_round=rounds,
            evals=[(validation, "validation")] if patience is not None else [],
            early_stopping_rounds=patience,
            evals_result=history,
            verbose_eval=False,
        )
        if patience is not None:
            prediction_rounds = model.best_iteration + 1
            stopping.append(dict(selected_rounds=prediction_rounds, history=history))
        actual = json.loads(model.save_config())["learner"]["generic_param"]["device"]
        if job["device"] == "cuda" and not actual.startswith("cuda"):
            raise ValueError("silent CPU fallback")
        prediction = model.predict(
            validation, output_margin=task == "A10", iteration_range=(0, prediction_rounds or 0)
        )
        if task == "A10":
            prediction = np.column_stack([prediction, np.ones(len(v))])
    elif library == "lightgbm":
        import lightgbm as lgb

        if task in ["A10", "A11"]:
            raise ValueError("no builtin matching task")
        if set(cfg) != {"num_leaves", "lambda_l2"}:
            raise ValueError("unsupported LightGBM parameters")
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
            "A12": "regression",
        }
        params = dict(
            cfg,
            objective=objectives[task],
            learning_rate=lr,
            num_threads=job["threads"],
            seed=seed,
            device_type=job["device"],
            verbosity=-1,
        )
        if task == "A3":
            params["num_class"] = job["classes"]
        if task == "A9":
            params["tweedie_variance_power"] = 1.5
        if task == "A4":
            params.update(metric="ndcg", eval_at=[10])
        if task == "A6":
            params["boost_from_average"] = False
        targets = y.T if task == "A6" else [y] * 3 if task == "A5" else [y]
        model = []
        predictions = []
        for k, target in enumerate(targets):
            if task == "A5":
                params["alpha"] = [0.1, 0.5, 0.9][k]
            d = lgb.Dataset(
                x,
                label=target,
                weight=np.repeat(ranking[0][2], ranking[0][1]) if task == "A4" else w,
                group=ranking[0][1] if task == "A4" else None,
                init_score=base + np.log(exposure) if task == "A7" else None,
            )
            history = {}
            valid = []
            callbacks = []
            if patience is not None:
                vt = vy[:, k] if task == "A6" else vy
                valid = [
                    lgb.Dataset(
                        v,
                        label=vt,
                        weight=np.repeat(ranking[1][2], ranking[1][1]) if task == "A4" else vw,
                        group=ranking[1][1] if task == "A4" else None,
                        reference=d,
                        init_score=base + np.log(validation_exposure) if task == "A7" else None,
                    )
                ]
                callbacks = [
                    lgb.early_stopping(patience, verbose=False),
                    lgb.record_evaluation(history),
                ]
            m = lgb.train(
                params,
                d,
                num_boost_round=rounds,
                valid_sets=valid,
                valid_names=["validation"] if valid else None,
                callbacks=callbacks,
            )
            if patience is not None:
                stopping.append(dict(selected_rounds=m.best_iteration, history=history))
            p = m.predict(v, raw_score=task == "A7")
            if task == "A7":
                p = np.exp(p + base + np.log(validation_exposure))
            model.append(m)
            predictions.append(p)
        prediction = np.column_stack(predictions) if task in ["A5", "A6"] else predictions[0]
    elif library == "catboost":
        import catboost as cb

        if task == "A8" or (task == "A10" and job["device"] == "cuda"):
            raise ValueError("unsupported CatBoost task/device")
        if set(cfg) != {"depth", "l2_leaf_reg"}:
            raise ValueError("unsupported CatBoost parameters")
        losses = {
            "A1": "RMSE",
            "A2": "Logloss",
            "A3": "MultiClass",
            "A4": "PairLogit",
            "A5": "MultiQuantile:alpha=0.1,0.5,0.9",
            "A6": "MultiRMSE",
            "A7": "Poisson",
            "A9": "Tweedie:variance_power=1.5",
            "A10": "SurvivalAft:dist=Normal;scale=1",
            "A11": "RMSEWithUncertainty",
            "A12": "RMSE",
        }
        klass = (
            cb.CatBoostClassifier
            if task in ["A2", "A3"]
            else cb.CatBoostRanker
            if task == "A4"
            else cb.CatBoostRegressor
        )
        target = (
            np.column_stack([y, np.where(arrays["event_train"], y, -1)]) if task == "A10" else y
        )
        data = cb.Pool(
            x,
            label=target,
            weight=None if task == "A4" else w,
            group_id=arrays["query_train"] if task == "A4" else None,
            group_weight=np.repeat(ranking[0][2], ranking[0][1]) if task == "A4" else None,
            baseline=base + np.log(exposure) if task == "A7" else None,
        )
        valid_pool = None
        if patience is not None:
            vt = (
                np.column_stack([vy, np.where(arrays["event_validation"], vy, -1)])
                if task == "A10"
                else vy
            )
            valid_pool = cb.Pool(
                v,
                label=vt,
                weight=None if task == "A4" else vw,
                group_id=arrays["query_validation"] if task == "A4" else None,
                group_weight=np.repeat(ranking[1][2], ranking[1][1]) if task == "A4" else None,
                baseline=base + np.log(validation_exposure) if task == "A7" else None,
            )
        model = klass(
            **cfg,
            iterations=rounds,
            learning_rate=lr,
            loss_function=losses[task],
            eval_metric="NDCG:top=10" if task == "A4" else None,
            random_seed=seed,
            thread_count=job["threads"],
            task_type="GPU" if job["device"] == "cuda" else "CPU",
            verbose=False,
            allow_writing_files=False,
            **({"boost_from_average": False, "allow_const_label": True} if task == "A6" else {}),
        ).fit(
            data,
            eval_set=valid_pool,
            early_stopping_rounds=patience,
            use_best_model=patience is not None,
        )
        if patience is not None:
            stopping.append(
                dict(selected_rounds=model.tree_count_, history=model.get_evals_result())
            )
        if task in ["A2", "A3"]:
            prediction = model.predict_proba(v)
            if task == "A2":
                prediction = prediction[:, 1]
        elif task == "A11":
            prediction = model.predict(v, prediction_type="RMSEWithUncertainty")
            prediction[:, 1] = np.sqrt(prediction[:, 1])
        else:
            prediction = (
                model.predict(v)
                if task == "A4"
                else model.predict(v, prediction_type="RawFormulaVal")
            )
            if task in ["A7", "A9"]:
                prediction = np.exp(
                    prediction + (base + np.log(validation_exposure) if task == "A7" else 0)
                )
            if task == "A10":
                prediction = np.column_stack([prediction, np.ones(len(v))])
    else:
        from ngboost import NGBRegressor
        from sklearn.tree import DecisionTreeRegressor

        if task != "A11" or job["device"] != "cpu" or set(cfg) != {"weak_depth", "minibatch_frac"}:
            raise ValueError("unsupported NGBoost contract")
        model = NGBRegressor(
            n_estimators=rounds,
            learning_rate=lr,
            Base=DecisionTreeRegressor(max_depth=cfg["weak_depth"], random_state=seed),
            minibatch_frac=cfg["minibatch_frac"],
            random_state=seed,
            verbose=False,
        ).fit(
            x,
            y,
            sample_weight=w,
            X_val=v if patience is not None else None,
            Y_val=vy,
            val_sample_weight=vw,
            early_stopping_rounds=patience,
        )
        if patience is not None:
            prediction_rounds = model.best_val_loss_itr + 1
            stopping.append(dict(selected_rounds=prediction_rounds, history=model.evals_result))
        dist = model.pred_dist(v, max_iter=prediction_rounds)
        prediction = np.column_stack([dist.loc, dist.scale])
    if target_scale is not None:
        prediction = prediction * std + mean
    if not np.isfinite(prediction).all() or len(prediction) != len(v):
        raise ValueError("invalid output")
    return prediction, {
        "model": model,
        "application": task,
        "library": library,
        "rate_base": base,
        "target_scale": target_scale,
        "prediction_rounds": prediction_rounds,
        "stopping": stopping,
        "job": job,
    }


def predict_saved(saved, x, exposure=None):
    """Replay a trusted local model bundle with its external prediction state."""
    task, library, model = saved["application"], saved["library"], saved["model"]
    if task == "A7":
        if exposure is None or exposure.shape != (len(x),):
            raise ValueError("prediction exposure required")
        if not np.isfinite(exposure).all() or np.any(exposure <= 0):
            raise ValueError("positive finite prediction exposure required")
        offset = saved["rate_base"] + np.log(exposure)
    if library == "xgboost":
        import xgboost as xgb

        data = xgb.DMatrix(x)
        if task == "A7":
            data.set_base_margin(offset)
        result = model.predict(
            data,
            output_margin=task == "A10",
            iteration_range=(0, saved.get("prediction_rounds") or 0),
        )
    elif library == "lightgbm":
        columns = [m.predict(x, raw_score=task == "A7") for m in model]
        result = np.column_stack(columns) if task in ["A5", "A6"] else columns[0]
        if task == "A7":
            result = np.exp(result + offset)
    elif library == "catboost":
        if task in ["A2", "A3"]:
            result = model.predict_proba(x)
            if task == "A2":
                result = result[:, 1]
        elif task == "A11":
            result = model.predict(x, prediction_type="RMSEWithUncertainty")
            result[:, 1] = np.sqrt(result[:, 1])
        else:
            result = (
                model.predict(x)
                if task == "A4"
                else model.predict(x, prediction_type="RawFormulaVal")
            )
            if task in ["A7", "A9"]:
                result = np.exp(result + (offset if task == "A7" else 0))
    elif library == "ngboost":
        dist = model.pred_dist(x, max_iter=saved.get("prediction_rounds"))
        result = np.column_stack([dist.loc, dist.scale])
    else:
        raise ValueError("unknown saved baseline library")
    if task == "A10":
        result = np.column_stack([result, np.ones(len(x))])
    if task == "A6":
        scale = saved["target_scale"]
        result = result * np.asarray(scale["std"]) + np.asarray(scale["mean"])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job", type=Path)
    args = p.parse_args()
    job = json.loads(args.job.read_text())
    with np.load(job["input_npz"], allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    prediction, model = fit(job, arrays)
    payload = pickle.dumps(model)
    restored = predict_saved(
        pickle.loads(payload), arrays["x_validation"], arrays.get("exposure_validation")
    )
    np.testing.assert_allclose(prediction, restored, rtol=1e-7, atol=1e-8)
    np.savez("predictions.npz", row_ids=arrays["validation_row_ids"], prediction=prediction)
    Path("model.bin").write_bytes(payload)
    Path("training.json").write_text(
        json.dumps(
            {
                "early_stopping_rounds": job.get("early_stopping_rounds"),
                "prediction_rounds": model["prediction_rounds"],
                "stopping": model["stopping"],
                "target_scale": model["target_scale"],
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
