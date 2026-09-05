"""Offline baseline gates and independent data/metric checks."""

import copy

import numpy as np
import pytest
from benchmarks.foundation.baseline_worker import normal_metrics
from benchmarks.foundation.dataset import load_housing, split_indices
from benchmarks.foundation.runner import validate_baseline


def valid_cells():
    record = {
        "fit_s": 1.0,
        "predict_params_s": 0.1,
        "fallback_warnings": [],
        "metrics": {"nll": 1.0, "crps": 0.5, "coverage90": 0.9},
    }
    cells = []
    for seed in (0, 1, 2):
        for mode in ("resident", "eval"):
            for backend in ("cpu", "cuda"):
                records = []
                for phase in ("first_fit", "repeat_fit"):
                    r = copy.deepcopy(record)
                    r.update(
                        phase=phase,
                        fit_path={
                            "objective_calls": 30,
                            "native_tree_calls": 60 if backend == "cuda" else 0,
                        },
                    )
                    records.append(r)
                cells.append(dict(seed=seed, mode=mode, backend=backend, records=records))
    return cells


def test_complete_baseline():
    validate_baseline(valid_cells())


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "phase", "fallback", "device", "time", "metric", "quality"]
)
def test_bad_baseline_rejected(fault):
    cells = valid_cells()
    r = cells[1]["records"][1]
    if fault == "missing":
        cells.pop()
    elif fault == "duplicate":
        cells[1] = cells[0]
    elif fault == "phase":
        r["phase"] = "first_fit"
    elif fault == "fallback":
        r["fallback_warnings"] = ["host"]
    elif fault == "device":
        r["fit_path"]["native_tree_calls"] = 0
    elif fault == "time":
        r["fit_s"] = float("nan")
    elif fault == "metric":
        r["metrics"]["crps"] = float("inf")
    else:
        r["metrics"]["nll"] = 2.0
    with pytest.raises(ValueError):
        validate_baseline(cells)


def test_normal_metrics_at_mean():
    result = normal_metrics(np.zeros(3), {"loc": np.zeros(3), "scale": np.ones(3)})
    assert result["nll"] == pytest.approx(0.5 * np.log(2 * np.pi))
    assert result["crps"] == pytest.approx((np.sqrt(2) - 1) / np.sqrt(np.pi))
    assert result["coverage90"] == 1.0


def test_splits_are_disjoint_complete_seeded():
    a = split_indices(20640, 0)
    b = split_indices(20640, 0)
    assert [len(x) for x in a] == [12384, 4128, 4128]
    np.testing.assert_array_equal(np.sort(np.concatenate(a)), np.arange(20640))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a[0], split_indices(20640, 1)[0])


def test_corrupt_archive_rejected(tmp_path):
    path = tmp_path / "bad.tgz"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_housing(path)
