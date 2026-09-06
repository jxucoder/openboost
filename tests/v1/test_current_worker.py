"""Current evaluation worker contracts, separate from baseline implementations."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from benchmarks.v1.openboost_predict import predict_saved
from benchmarks.v1.openboost_worker import fit

from openboost import NumericData, Problem, RunContext
from openboost.objectives import Normal
from openboost.recipes import normal, squared


def test_current_worker_is_available():
    assert callable(importlib.import_module("benchmarks.v1.openboost_worker").fit)


def fixture(app="A1"):
    x = np.arange(12, dtype=float)[:, None]
    arrays = dict(
        x_train=x[:8],
        y_train=np.sin(x[:8, 0]),
        x_validation=x[8:],
        y_validation=np.sin(x[8:, 0]),
        validation_row_ids=np.arange(8, 12),
        weight_train=np.arange(8, dtype=float),
        weight_validation=np.ones(4),
    )
    job = dict(
        application=app,
        library="openboost",
        device="cpu",
        threads=1,
        seed=7,
        config=dict(rounds=3, learning_rate=0.1, bins=8),
    )
    return job, arrays


@pytest.mark.parametrize("app", ["A1", "A11"])
@pytest.mark.parametrize("patience", [None, 1])
def test_direct_recipe_parity_and_fresh_prediction(app, patience, tmp_path):
    job, arrays = fixture(app)
    job["early_stopping_rounds"] = patience
    prediction, saved, training = fit(job, arrays)
    problems = []
    for part in ("train", "validation"):
        x, y = arrays["x_" + part], arrays["y_" + part]
        data = NumericData(x, np.arange(len(x)), ("x0",))
        problems.append(
            Problem(
                data,
                y[:, None],
                data.row_ids,
                weight=arrays["weight_" + part],
                raw_width=1 if app == "A1" else 2,
            )
        )
    direct = (squared if app == "A1" else normal)(
        *problems, context=RunContext("evaluation", 7), patience=patience, **job["config"]
    )
    model = direct.state.model if patience is None else direct.state.best_model
    raw = model.predict(problems[1].data)
    expected = raw[:, 0] if app == "A1" else Normal.parameters(raw)
    np.testing.assert_array_equal(prediction, expected)
    assert training["selected_model_identity"] == model.identity
    assert training["stop"]["completed_rounds"] == direct.stop.completed_rounds
    model_path = tmp_path / "model.bin"
    model_path.write_text(json.dumps(saved))
    np.savez(
        tmp_path / "features.npz", x=arrays["x_validation"], row_ids=arrays["validation_row_ids"]
    )
    script = Path(__file__).resolve().parents[2] / "benchmarks/v1/openboost_predict.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(model_path),
            str(tmp_path / "features.npz"),
            str(tmp_path / "result.npz"),
        ],
        cwd=tmp_path,
        check=True,
    )
    with np.load(tmp_path / "result.npz") as result:
        np.testing.assert_array_equal(result["prediction"], prediction)
        np.testing.assert_array_equal(result["row_ids"], arrays["validation_row_ids"])


@pytest.mark.parametrize(
    "bad", ["test", "validation", "weights", "ids", "width", "config", "device", "threads", "task"]
)
def test_unsupported_or_contaminated_inputs_fail(bad):
    job, arrays = fixture()
    if bad == "test":
        arrays["y_test"] = np.zeros(2)
    elif bad == "validation":
        arrays.pop("y_validation")
    elif bad == "weights":
        arrays["weight_train"][0] = -1
    elif bad == "ids":
        arrays["validation_row_ids"] = np.zeros(4)
    elif bad == "width":
        arrays["x_validation"] = np.ones((4, 2))
    elif bad == "config":
        job["config"]["subsample"] = 0.5
    elif bad == "device":
        job["device"] = "cuda"
    elif bad == "threads":
        job["threads"] = 2
    else:
        job["application"] = "A6"
    with pytest.raises(ValueError):
        fit(job, arrays)


def test_saved_output_semantics_are_validated():
    job, arrays = fixture("A11")
    _, saved, _ = fit(job, arrays)
    saved["output"] = "mean"
    with pytest.raises(ValueError):
        predict_saved(saved, arrays["x_validation"])
