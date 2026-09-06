"""One validation-only GLM, paid-loss composition or global-formula trial."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

if __package__:
    from benchmarks.v1 import parametric as controls
    from benchmarks.v1.judge import read_json
else:
    import parametric as controls
    from judge import read_json


def fit(job, arrays):
    if set(job) != {"application", "method", "config", "input_npz"}:
        raise ValueError("unsupported parametric job fields")
    task, method, config = job["application"], job["method"], job["config"]
    required = {"y_train", "validation_row_ids"}
    optional = {"weight_train"}
    if method == "formula_global" and task == "A12":
        required.update({"age_train", "age_validation"})
        expected = {"initial_amplitude_multiplier", "initial_rate", "max_nfev"}
    elif method in {"glm", "paid_composition"}:
        required.update({"x_train", "x_validation"})
        expected = {"alpha", "max_iter"}
        if task in ["A7", "A9"]:
            required.update({"exposure_train", "exposure_validation"})
        if method == "paid_composition":
            if task != "A9":
                raise ValueError("paid composition requires A9")
            required.update({"paid_count", "claim_policy", "claim_amount"})
            expected = {"count_alpha", "severity_alpha", "max_iter"}
    else:
        raise ValueError("unsupported parametric method/task")
    if set(config) != expected or not required <= set(arrays) or set(arrays) - required - optional:
        raise ValueError("unsupported/missing input or config fields")
    ids = arrays["validation_row_ids"]
    size = len(arrays["age_validation"] if method == "formula_global" else arrays["x_validation"])
    if ids.ndim != 1 or len(ids) != size or len(np.unique(ids)) != size:
        raise ValueError("invalid validation row IDs")
    weight = arrays.get("weight_train")
    if method == "formula_global":
        model = controls.fit_global_formula(
            arrays["age_train"], arrays["y_train"], weight=weight, **config
        )
    elif method == "paid_composition":
        model = controls.fit_paid_composition(
            arrays["x_train"],
            arrays["paid_count"],
            arrays["y_train"],
            arrays["exposure_train"],
            arrays["claim_policy"],
            arrays["claim_amount"],
            weight=weight,
            **config,
        )
    else:
        model = controls.fit_glm(
            task,
            arrays["x_train"],
            arrays["y_train"],
            exposure=arrays.get("exposure_train"),
            weight=weight,
            **config,
        )
    saved = dict(application=task, method=method, model=model)
    prediction = predict(saved, arrays)
    if not np.isfinite(prediction).all() or prediction.shape != (size,):
        raise ValueError("invalid scalar predictions")
    return prediction, saved


def predict(saved, arrays):
    method, task, model = saved["method"], saved["application"], saved["model"]
    if method == "formula_global":
        return controls.predict_global_formula(model, arrays["age_validation"])
    if method == "paid_composition":
        return controls.predict_paid_composition(
            model, arrays["x_validation"], arrays["exposure_validation"]
        )["annualized"]
    result = controls.predict_glm(model, arrays["x_validation"], arrays.get("exposure_validation"))
    return result["mean" if task == "A8" else "period" if task == "A7" else "annualized"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = read_json(args.job.read_bytes())
    with np.load(job["input_npz"], allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    prediction, model = fit(job, arrays)
    raw = pickle.dumps(model)
    np.testing.assert_allclose(prediction, predict(pickle.loads(raw), arrays), rtol=1e-7, atol=1e-8)
    np.savez("predictions.npz", row_ids=arrays["validation_row_ids"], prediction=prediction)
    Path("model.bin").write_bytes(raw)
    Path("training.json").write_text(
        json.dumps(
            {"application": job["application"], "method": job["method"], "config": job["config"]},
            indent=2,
        )
        + "\n"
    )
