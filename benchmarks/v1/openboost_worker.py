"""Current CPU A1/A2/A3/A5/A6/A7/A8/A9/A11 trials on frozen encoded train/validation packets.

Explicit validation targets are required even with fixed budgets. The caller
controls process threads and resource limits. Test arrays are always rejected.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from openboost import ClassSchema, NumericData, Problem, RunContext
from openboost.multioutput import TargetScale
from openboost.recipes import (
    binary,
    gamma,
    multi_squared,
    multiclass,
    normal,
    poisson,
    quantile,
    squared,
    tweedie,
)

if __package__:
    from benchmarks.v1.openboost_predict import OUTPUTS, QUANTILES, predict_saved
    from benchmarks.v1.preprocessing import fit_target_scale
else:
    from openboost_predict import OUTPUTS, QUANTILES, predict_saved
    from preprocessing import fit_target_scale


def fit(job, arrays):
    classification = job.get("application") in {"A2", "A3"}
    required = {"application", "library", "device", "seed", "threads", "config"}
    if (
        not required <= set(job)
        or set(job)
        - required
        - {"early_stopping_rounds", "input_npz"}
        - ({"classes"} if classification else set())
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
    if job["application"] == "A7":
        needed |= {"exposure_train", "exposure_validation"}
    if job["application"] == "A9":
        needed |= {"weight_train", "weight_validation"}
    if not needed <= set(arrays) or set(arrays) - needed - {"weight_train", "weight_validation"}:
        raise ValueError("explicit train/validation arrays only; no test arrays")
    external_ids = np.asarray(arrays["validation_row_ids"])
    if (
        external_ids.ndim != 1
        or external_ids.dtype.kind not in "iuUS"
        or len(external_ids) != len(arrays["x_validation"])
        or len(np.unique(external_ids)) != len(external_ids)
    ):
        raise ValueError("unique aligned external validation row IDs required")
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
    classes = None
    if classification:
        count = job.get("classes")
        if type(count) is not int or (count != 2 if job["application"] == "A2" else count < 3):
            raise ValueError("explicit canonical classification count required")
        classes = ClassSchema(tuple(range(count)))
    problems = []
    width = 1 if job["application"] in {"A1", "A5", "A7", "A8", "A9"} else 2
    if classification:
        width = 1 if job["application"] == "A2" else count
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
        # Packet IDs remain in emitted artifacts; public data uses local integer rows.
        ids = np.arange(len(x))
        data = NumericData(x, ids, names)
        if job["application"] == "A9":
            weight = np.asarray(arrays["weight_" + part])
            if weight.shape != (len(x),) or not np.isfinite(weight).all() or np.any(weight <= 0):
                raise ValueError("positive aligned aggregate exposure weights required")
        structure = None
        if job["application"] == "A7":
            exposure = np.asarray(arrays["exposure_" + part])
            if (
                exposure.shape != (len(x),)
                or not np.isfinite(exposure).all()
                or np.any(exposure <= 0)
            ):
                raise ValueError("aligned positive finite exposure required")
            structure = {"exposure": exposure[:, None]}
        problems.append(
            Problem(
                data,
                y if multi else y[:, None],
                data.row_ids,
                weight=arrays.get("weight_" + part),
                raw_width=width,
                classes=classes,
                structure=structure,
            )
        )
    if multi:
        problems = [scale.transform(p) for p in problems]
    if job["application"] == "A5":
        return fit_quantiles(job, cfg, problems, arrays["x_validation"])
    recipe = {
        "A1": squared,
        "A2": binary,
        "A3": multiclass,
        "A6": multi_squared,
        "A7": poisson,
        "A8": gamma,
        "A9": tweedie,
        "A11": normal,
    }[job["application"]]
    if job["application"] == "A9":
        cfg["power"] = 1.5
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
    if job["application"] == "A9":
        saved["power"] = 1.5
    if multi:
        saved["target_scale"] = target_scale
    prediction = predict_saved(
        saved, arrays["x_validation"], exposure=arrays.get("exposure_validation")
    )
    training = dict(
        selection=selection,
        stop={**asdict(result.stop), "reason": result.stop.reason},
        accepted_commits=result.state.version,
        selected_model_identity=model.identity,
        best_validation_score=result.state.best_score,
        output=saved["output"],
    )
    if job["application"] == "A7":
        from openboost.objectives import Poisson

        training.update(
            selection_metric="weighted_poisson_nll",
            exposure_role="likelihood_once",
            target_units="period_count",
            raw_units="log_rate",
            selected_validation_nll=Poisson().loss(problems[1], model.predict(problems[1].data)),
        )
    if job["application"] == "A8":
        from openboost.objectives import Gamma

        training.update(
            selection_metric="weighted_gamma_objective",
            target_units="positive_claim_amount",
            raw_units="log_mean",
            selected_validation_gamma_objective=Gamma().loss(
                problems[1], model.predict(problems[1].data)
            ),
        )
    if job["application"] == "A9":
        from openboost.objectives import Tweedie

        training.update(
            selection_metric="weighted_tweedie_objective",
            power=1.5,
            target_units="annualized_paid_total",
            weight_role="exposure_once",
            selected_validation_tweedie_objective=Tweedie(1.5).loss(
                problems[1], model.predict(problems[1].data)
            ),
        )
    if classification:
        training.update(class_order=list(classes.values), selection_metric="logloss")
    if multi:
        training.update(
            target_scale=target_scale,
            scale_convention="unweighted_train_population",
            selection_metric="row_mean_sum_standardized_half_squared_error",
        )
    return prediction, saved, training


def fit_quantiles(job, cfg, problems, x_validation):
    """Independent frozen levels with separate validation stopping and selection."""
    from openboost.objectives import Quantile

    models, runs = [], []
    selection = "final" if job.get("early_stopping_rounds") is None else "best_validation"
    for q in QUANTILES:
        result = quantile(
            *problems,
            context=RunContext("evaluation", job["seed"]),
            q=q,
            patience=job.get("early_stopping_rounds"),
            **cfg,
        )
        model = result.state.model if selection == "final" else result.state.best_model
        models.append(model.record())
        runs.append(
            dict(
                q=q,
                stop={**asdict(result.stop), "reason": result.stop.reason},
                accepted_commits=result.state.version,
                selected_model_identity=model.identity,
                best_validation_score=result.state.best_score,
                selected_validation_pinball=Quantile(q).loss(
                    problems[1], model.predict(problems[1].data)
                ),
            )
        )
    saved = dict(
        format="openboost-evaluation-v1",
        application="A5",
        output=OUTPUTS["A5"],
        quantiles=list(QUANTILES),
        models=models,
    )
    prediction = predict_saved(saved, x_validation)
    training = dict(
        selection=selection,
        output=OUTPUTS["A5"],
        quantiles=list(QUANTILES),
        quantile_runs=runs,
        selection_metric="independent_weighted_pinball",
        crossing_rows=int(np.any(np.diff(prediction, axis=1) < 0, axis=1).sum()),
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
