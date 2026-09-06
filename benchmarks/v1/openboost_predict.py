"""Fresh-process inference for current CPU evaluation bundles; no training imports."""

import argparse
import json
from pathlib import Path

import numpy as np

from openboost import NumericData
from openboost.artifacts import Model
from openboost.objectives import Normal

OUTPUTS = {"A1": "mean", "A11": "normal_mean_scale"}


def predict_saved(saved, x):
    if (
        not isinstance(saved, dict)
        or set(saved) != {"format", "application", "output", "model"}
        or saved["format"] != "openboost-evaluation-v1"
        or saved["application"] not in OUTPUTS
        or saved["output"] != OUTPUTS[saved["application"]]
    ):
        raise ValueError("unsupported evaluation bundle")
    model = Model.from_record(saved["model"])
    width = 1 if saved["application"] == "A1" else 2
    if model.base.shape != (width,) or model.classes is not None:
        raise ValueError("model differs from declared output semantics")
    x = np.asarray(x)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite encoded prediction matrix required")
    data = NumericData(x, np.arange(len(x)), model.feature_names)
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
