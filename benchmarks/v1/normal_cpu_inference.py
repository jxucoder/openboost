"""Fresh installed CPU replay; run with -I in an environment without CUDA/plugins."""

import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

import openboost
from openboost.artifacts import Model
from openboost.data import NumericData


def main(directory):
    directory = Path(directory)
    absent = ("ob_cohort_splits", "cupy", "numba")
    assert all(importlib.util.find_spec(name) is None for name in absent)
    installed = Path(openboost.__file__).resolve().parent
    assert "site-packages" in installed.parts
    record = json.loads((directory / "inputs.json").read_text())
    started = time.perf_counter()
    model = Model.load(directory / "model.json")
    data = NumericData(record["values"], record["row_ids"], tuple(record["feature_names"]))
    prediction = model.predict(data)
    seconds = time.perf_counter() - started
    expected = np.array(record["expected_raw"])
    np.testing.assert_allclose(prediction, expected, rtol=2e-4, atol=2e-5)
    decoded = prediction + np.array(record["offset"])
    scale = np.exp(decoded[:, 1])
    assert np.all(np.isfinite(scale)) and np.all(scale > 0)
    result = dict(
        absent=list(absent),
        installed_path=str(installed),
        python=sys.version,
        numpy=importlib.metadata.version("numpy"),
        sources={
            "src/openboost/" + str(p.relative_to(installed)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in installed.rglob("*.py")
        },
        model_sha256=hashlib.sha256((directory / "model.json").read_bytes()).hexdigest(),
        raw=prediction.tolist(),
        mean=decoded[:, 0].tolist(),
        scale=scale.tolist(),
        load_and_predict_seconds=seconds,
    )
    (directory / "cpu-replay.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
