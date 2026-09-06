"""Reject corrupted evaluation inputs before invoking an external trainer."""

import numpy as np
import pytest
from benchmarks.v1.baseline_worker import fit, predict_saved


def fixture():
    job = dict(
        application="A7",
        library="xgboost",
        device="cpu",
        threads=2,
        seed=0,
        config=dict(rounds=4, learning_rate=0.1, max_depth=2, reg_lambda=1),
    )
    arrays = dict(
        x_train=np.ones((3, 2)),
        y_train=np.ones(3),
        x_validation=np.ones((2, 2)),
        validation_row_ids=np.array([3, 4]),
        exposure_train=np.ones(3),
        exposure_validation=np.ones(2),
    )
    return job, arrays


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("validation_row_ids", np.array([3, 3]), "row IDs"),
        ("exposure_train", np.array([1.0, np.nan, 1.0]), "exposure"),
        ("exposure_validation", np.array([1.0, np.inf]), "exposure"),
        ("weight_train", np.array([0.0, 0.0, 0.0]), "weights"),
    ],
)
def test_invalid_array_rejected(field, value, reason):
    job, arrays = fixture()
    arrays[field] = value
    with pytest.raises(ValueError, match=reason):
        fit(job, arrays)


def test_unsupported_early_stopping_is_explicit():
    job, arrays = fixture()
    job["early_stopping_rounds"] = 50
    with pytest.raises(ValueError, match="early stopping"):
        fit(job, arrays)


def test_censor_indicator_must_not_be_coerced_to_truthiness():
    job, arrays = fixture()
    job["application"] = "A10"
    arrays.pop("exposure_train")
    arrays.pop("exposure_validation")
    arrays["event_train"] = np.array([1, 2, 0])
    with pytest.raises(ValueError, match="survival"):
        fit(job, arrays)


def test_missing_exposure_cannot_silently_change_saved_model():
    saved = dict(application="A7", library="xgboost", model=None)
    with pytest.raises(ValueError, match="exposure required"):
        predict_saved(saved, np.ones((2, 2)))


def test_unknown_job_option_is_not_ignored():
    job, arrays = fixture()
    job["constraints"] = [1, 0]
    with pytest.raises(ValueError, match="unsupported job"):
        fit(job, arrays)


def test_unlocked_test_data_cannot_enter_validation_worker():
    job, arrays = fixture()
    arrays["x_test"] = np.ones((2, 2))
    with pytest.raises(ValueError, match="unsupported input"):
        fit(job, arrays)
