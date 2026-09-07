"""Fresh-process inference for current CPU evaluation bundles; no training imports."""

import argparse
import json
from pathlib import Path

import numpy as np

from openboost import NumericData
from openboost.artifacts import Model
from openboost.multioutput import MultiOutputModel, TargetScale
from openboost.objectives import Normal

QUANTILES = (0.1, 0.5, 0.9)

OUTPUTS = {
    "A1": "mean",
    "A5": "quantiles",
    "A2": "positive_class_probability",
    "A3": "class_probabilities",
    "A11": "normal_mean_scale",
    "A6": "multioutput_original_units",
}


def predict_saved(saved, x):
    if isinstance(saved, dict) and saved.get("application") == "A5":
        if (
            set(saved) != {"format", "application", "output", "quantiles", "models"}
            or saved["format"] != "openboost-evaluation-v1"
            or saved["output"] != OUTPUTS["A5"]
            or saved["quantiles"] != list(QUANTILES)
            or not isinstance(saved["models"], list)
            or len(saved["models"]) != len(QUANTILES)
        ):
            raise ValueError("invalid frozen quantile bundle")
        models = [Model.from_record(r) for r in saved["models"]]
        if any(
            m.base.shape != (1,)
            or m.classes is not None
            or m.feature_names != models[0].feature_names
            for m in models
        ):
            raise ValueError("quantile model schema differs")
        # Preserve raw level order, including crossings; no post-hoc sorting.
        return np.column_stack(
            [
                predict_saved(
                    dict(format=saved["format"], application="A1", output="mean", model=r), x
                )
                for r in saved["models"]
            ]
        )
    if (
        not isinstance(saved, dict)
        or set(saved)
        != (
            {"format", "application", "output", "model"}
            | ({"target_scale"} if saved.get("application") == "A6" else set())
        )
        or saved["format"] != "openboost-evaluation-v1"
        or saved["application"] not in OUTPUTS
        or saved["output"] != OUTPUTS[saved["application"]]
    ):
        raise ValueError("unsupported evaluation bundle")
    model = Model.from_record(saved["model"])
    scale = None
    if saved["application"] == "A6":
        record = saved["target_scale"]
        if not isinstance(record, dict) or set(record) != {"mean", "std", "constant"}:
            raise ValueError("invalid evaluation target scale")
        scale = TargetScale(record["mean"], record["std"], record["constant"])
    width = len(scale.mean) if scale is not None else 1 if saved["application"] == "A1" else 2
    classification = saved["application"] in {"A2", "A3"}
    if classification:
        if model.classes is None:
            raise ValueError("classification output requires class schema")
        count = len(model.classes.values)
        if model.classes.values != tuple(range(count)) or any(
            type(v) is not int for v in model.classes.values
        ):
            raise ValueError("canonical encoded class order required")
        if count != 2 if saved["application"] == "A2" else count < 3:
            raise ValueError("class count differs from task")
        width = 1 if saved["application"] == "A2" else count
    if model.base.shape != (width,) or (not classification and model.classes is not None):
        raise ValueError("model differs from declared output semantics")
    x = np.asarray(x)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite encoded prediction matrix required")
    data = NumericData(x, np.arange(len(x)), model.feature_names)
    if classification:
        probability = model.predict_proba(data)
        return probability[:, 1] if saved["application"] == "A2" else probability
    if scale is not None:
        return MultiOutputModel(model, scale).predict(data)
    raw = model.predict(data)
    return raw[:, 0] if width == 1 else Normal.parameters(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.features, allow_pickle=False) as arrays:
        if set(arrays.files) != {"x", "row_ids"}:
            raise ValueError("prediction packet must contain only features and row IDs")
        x, ids = arrays["x"], arrays["row_ids"]
    if ids.ndim != 1 or len(ids) != len(x) or len(np.unique(ids)) != len(ids):
        raise ValueError("unique aligned prediction row IDs required")
    saved = json.loads(args.model.read_text())
    prediction = predict_saved(saved, x)
    np.savez(args.output, row_ids=ids, prediction=prediction)


if __name__ == "__main__":
    main()
