"""Quality cannot pass by missing folds, hiding targets, or choosing on test."""

import numpy as np
import pytest
from benchmarks.v1.quality import compare_folds, metrics, select_validation


def test_quality_boundaries_and_missing_folds():
    assert compare_folds([1] * 5, [1] * 5, "loss")["pass"]
    assert not compare_folds([1, 1, 1, 1, 1.16], [1] * 5, "loss")["pass"]
    assert not compare_folds([1] * 4, [1] * 4, "loss")["pass"]
    assert not compare_folds([float("nan")] * 5, [1] * 5, "loss")["pass"]
    assert compare_folds([-2] * 5, [-2.01] * 5, "nll")["pass"]
    assert not compare_folds([0.1] * 5, [0] * 5, "loss")["pass"]


def test_selection_never_reads_test_and_fails_incomplete_search():
    records = [{"id": str(i), "validation": float(i), "status": "pass"} for i in range(16)]
    assert select_validation(records) == "0"
    records[-1]["status"] = "timeout"
    with pytest.raises(ValueError):
        select_validation(records)


def test_multioutput_cannot_average_away_failed_target():
    m = metrics("A6", [[0, 0], [2, 4]], [[0, 0], [2, 0]])
    assert m["rmse_0"] == 0 and m["rmse_1"] == pytest.approx(np.sqrt(8))


def test_proper_scores_and_exposure_weight():
    assert metrics("A7", [0, 2], [1, 2])["poisson_deviance"] == pytest.approx(1)
    assert metrics("A8", [1, 2], [1, 2])["gamma_deviance"] == 0
    assert metrics("A11", [0], [[0, 1]])["nll"] == pytest.approx(0.5 * np.log(2 * np.pi))
    assert metrics("A2", [0, 1], [0.5, 0.5])["logloss"] == pytest.approx(np.log(2))


def test_survival_censor_is_not_an_event():
    event = metrics("A10", [1], [[0, 1]], event=[1])["nll"]
    censored = metrics("A10", [1], [[0, 1]], event=[0])["nll"]
    assert event == pytest.approx(0.5 * np.log(2 * np.pi))
    assert censored == pytest.approx(np.log(2))
    assert np.isfinite(metrics("A10", [np.exp(40)], [[0, 1]], event=[0])["nll"])


@pytest.mark.parametrize(
    "task,y,p",
    [("A2", [1], [2]), ("A3", [0], [[0.2, 0.2]]), ("A8", [0], [1]), ("A11", [0], [[0, -1]])],
)
def test_invalid_predictions_or_targets_fail(task, y, p):
    with pytest.raises(ValueError):
        metrics(task, y, p)
