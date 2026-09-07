"""Composition adapters preserve public recipes and named inference roles."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from benchmarks.v1.composition_worker import fit

from openboost import NumericData, RunContext
from openboost.composition import paid_loss_problems
from openboost.recipes import gamma, poisson


def fixture():
    arrays = {}
    for p, ids in [("train", np.arange(6)), ("validation", np.arange(10, 16))]:
        arrays.update(
            {
                f"x_{p}": np.arange(6.0)[:, None],
                f"row_ids_{p}": ids,
                f"paid_count_{p}": np.array([0, 2, 1, 0, 3, 1]),
                f"paid_total_{p}": np.array([0.0, 6, 4, 0, 9, 2]),
                f"exposure_{p}": np.array([0.1, 0.5, 1, 2, 1, 0.2]),
            }
        )
    job = dict(
        seed=7,
        patience=None,
        config=dict(rounds=3, learning_rate=0.1, bins=8, max_depth=2, reg_lambda=1),
    )
    return job, arrays


@pytest.mark.parametrize("patience", [None, 1])
def test_direct_components_and_fresh_replay(patience, tmp_path):
    job, a = fixture()
    job["patience"] = patience
    model, outputs, training = fit(job, a)
    pairs = []
    for p in ["train", "validation"]:
        d = NumericData(a["x_" + p], a["row_ids_" + p], ("x0",))
        pairs.append(
            paid_loss_problems(d, a["paid_count_" + p], a["paid_total_" + p], a["exposure_" + p])
        )
    for i, (name, recipe) in enumerate([("frequency", poisson), ("severity", gamma)]):
        r = recipe(
            pairs[0][i],
            pairs[1][i],
            context=RunContext(name, 7),
            patience=patience,
            **job["config"],
        )
        expected = r.state.model if patience is None else r.state.best_model
        assert getattr(model, name).identity == expected.identity
        assert training["components"][name]["stop"]["completed_rounds"] == r.stop.completed_rounds
    np.testing.assert_allclose(
        outputs["annualized_mean"], outputs["paid_count_rate"] * outputs["severity_mean"]
    )
    np.testing.assert_allclose(
        outputs["period_mean"], outputs["annualized_mean"] * a["exposure_validation"]
    )
    model.save(tmp_path / "model.bin")
    np.savez(
        tmp_path / "features.npz",
        x=a["x_validation"],
        row_ids=a["row_ids_validation"],
        exposure=a["exposure_validation"],
    )
    script = Path(__file__).resolve().parents[2] / "benchmarks/v1/composition_worker.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "predict",
            str(tmp_path / "model.bin"),
            str(tmp_path / "features.npz"),
            str(tmp_path / "replay.npz"),
        ],
        cwd=tmp_path,
        check=True,
    )
    with np.load(tmp_path / "replay.npz") as b:
        for k, v in outputs.items():
            np.testing.assert_array_equal(v, b[k])
        np.testing.assert_array_equal(b["row_ids"], a["row_ids_validation"])


@pytest.mark.parametrize(
    "bad", ["test", "offset", "weight", "overlap", "count", "zero_exposure", "config"]
)
def test_invalid_composition_contract(bad):
    job, a = fixture()
    if bad in ["test", "offset", "weight"]:
        a[bad] = np.ones(6)
    elif bad == "overlap":
        a["row_ids_validation"] = a["row_ids_train"]
    elif bad == "count":
        a["paid_count_train"][0] = 1
    elif bad == "zero_exposure":
        a["exposure_train"][0] = 0
    else:
        job["config"]["sampling"] = 0.5
    with pytest.raises(ValueError):
        fit(job, a)
