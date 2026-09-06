"""Current CPU A1/A6/A11 trials on frozen encoded train/validation packets.

Explicit validation targets are required even with fixed budgets. The caller
controls process threads and resource limits. Test arrays are always rejected.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from openboost import NumericData, Problem, RunContext
from openboost.multioutput import TargetScale
from openboost.recipes import multi_squared, normal, squared

if __package__:
    from benchmarks.v1.openboost_predict import OUTPUTS, predict_saved
    from benchmarks.v1.preprocessing import fit_target_scale
else:
    from openboost_predict import OUTPUTS, predict_saved
    from preprocessing import fit_target_scale


def fit(job, arrays):
    required = {"application", "library", "device", "seed", "threads", "config"}
    if (
        not required <= set(job)
        or set(job) - required - {"early_stopping_rounds", "input_npz"}
        or job["application"] not in OUTPUTS
        or job["library"] != "openboost"
        or job["device"] != "cpu"
    ):
        raise ValueError("unsupported current OpenBoost job")
    if type(job["threads"]) is not int or job["threads"] != 1:
        raise ValueError("current worker requires one process thread")
    if type(job["seed"]) is not int or job["seed"] < 0:
        raise ValueError("nonnegative integer seed required")
    needed = {"x_train", "y_train", "x_validation", "y_validation", "validation_row_ids"}
    if not needed <= set(arrays) or set(arrays) - needed - {"weight_train", "weight_validation"}:
        raise ValueError("explicit train/validation arrays only; no test arrays")
    cfg = dict(job["config"])
    if cfg.pop("seed_from_fold", True) is not True:
        raise ValueError("seed semantics differ")
    allowed = {"rounds", "learning_rate", "max_depth", "reg_lambda", "bins"}
    if job["application"] == "A11":
        allowed |= {"mode", "damping", "minimum_scale"}
    if job["application"] == "A6":
        allowed |= {"mode"}
    if set(cfg) - allowed or not {"rounds", "learning_rate"} <= set(cfg):
        raise ValueError("unsupported current recipe config")
    if type(cfg["rounds"]) is not int or cfg["rounds"] <= 0:
        raise ValueError("positive round budget required")
    problems = []
    width = 1 if job["application"] == "A1" else 2
    multi = job["application"] == "A6"
    target_scale = None
    scale = None
    names = None
    for part in ("train", "validation"):
        x, y = np.asarray(arrays["x_" + part]), np.asarray(arrays["y_" + part])
        if (
            x.ndim != 2
            or (
                y.ndim != 2 or not y.shape[1] or len(y) != len(x) if multi else y.shape != (len(x),)
            )
            or not len(x)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("finite encoded inputs and aligned task targets required")
        if multi and part == "train":
            width = y.shape[1]
            target_scale = fit_target_scale(y)
            scale = TargetScale(target_scale["mean"], target_scale["std"], target_scale["constant"])
        names = tuple(f"x{i}" for i in range(x.shape[1])) if names is None else names
        ids = np.arange(len(x)) if part == "train" else arrays["validation_row_ids"]
        data = NumericData(x, ids, names)
        problems.append(
            Problem(
                data,
                y if multi else y[:, None],
                data.row_ids,
                weight=arrays.get("weight_" + part),
                raw_width=width,
            )
        )
    if multi:
        problems = [scale.transform(p) for p in problems]
    recipe = multi_squared if multi else (squared if width == 1 else normal)
    result = recipe(
        *problems,
        context=RunContext("evaluation", job["seed"]),
        patience=job.get("early_stopping_rounds"),
        **cfg,
    )
    selection = "final" if job.get("early_stopping_rounds") is None else "best_validation"
    model = result.state.model if selection == "final" else result.state.best_model
    saved = dict(
        format="openboost-evaluation-v1",
        application=job["application"],
        output=OUTPUTS[job["application"]],
        model=model.record(),
    )
    if multi:
        saved["target_scale"] = target_scale
    prediction = predict_saved(saved, arrays["x_validation"])
    training = dict(
        selection=selection,
        stop={**asdict(result.stop), "reason": result.stop.reason},
        accepted_commits=result.state.version,
        selected_model_identity=model.identity,
        best_validation_score=result.state.best_score,
        output=saved["output"],
    )
    if multi:
        training.update(
            target_scale=target_scale,
            scale_convention="unweighted_train_population",
            selection_metric="row_mean_sum_standardized_half_squared_error",
        )
    return prediction, saved, training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    with np.load(job["input_npz"], allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    prediction, saved, training = fit(job, arrays)
    # model.bin is UTF-8 JSON, not pickle; the process runner requires this filename.
    Path("model.bin").write_text(json.dumps(saved, allow_nan=False) + "\n")
    np.savez("predictions.npz", row_ids=arrays["validation_row_ids"], prediction=prediction)
    Path("training.json").write_text(json.dumps(training, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
