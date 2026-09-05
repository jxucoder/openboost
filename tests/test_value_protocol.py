import copy

import pytest
from benchmarks.foundation.value_protocol import CONFIG, STRATEGIES, summarize


def fixture():
    metrics = {"nll": 1.0, "crps": 0.4, "coverage90": 0.96}
    cells = [
        {
            "seed": seed,
            "strategy": strategy,
            "records": [
                {
                    "fit_s": 2.0 if strategy == "experimental_cuda" else 1.0,
                    "predict_s": 0.01,
                    "metrics": metrics.copy(),
                    "fallback_warnings": [],
                }
                for _ in range(4)
            ],
        }
        for seed in (0, 1, 2)
        for strategy in STRATEGIES
    ]
    frozen = [
        {
            "seed": s,
            "backend": "cuda",
            "mode": "resident",
            "records": [{}, {"metrics": metrics.copy()}],
        }
        for s in (0, 1, 2)
    ]
    return cells, frozen


def test_regression_remains_a_reported_result():
    cells, frozen = fixture()
    result = summarize(cells, frozen)
    assert result["quality_pass"]
    assert result["profiling_triggered"] and not result["performance_budget_pass"]
    assert result["default_fit_ratio"] == 2
    cells[2]["records"][-1]["metrics"]["crps"] = 0.5
    assert not summarize(cells, frozen)["quality_pass"]


def test_incomplete_or_nonfinite_evidence_rejected():
    cells, frozen = fixture()
    with pytest.raises(ValueError):
        summarize(cells[:-1], frozen)
    bad = copy.deepcopy(cells)
    bad[0]["records"][0]["fit_s"] = float("nan")
    with pytest.raises(ValueError):
        summarize(bad, frozen)


def test_quality_failure_in_first_fit_is_not_hidden_by_warm_repeats():
    cells, frozen = fixture()
    cells[2]["records"][0]["metrics"]["nll"] = 10.0
    assert not summarize(cells, frozen)["quality_pass"]
    with pytest.raises(ValueError, match="frozen"):
        summarize(cells, [])


def test_missing_profile_or_wrong_device_rejected():
    from benchmarks.foundation.value_protocol import validate_profiles

    cells, _ = fixture()
    for c in cells:
        c.update(config=CONFIG.copy(), mode="resident", split_sizes=[12384, 4128, 4128])
        for r, phase in zip(
            c["records"], ("process_first", "warm_1", "warm_2", "warm_3"), strict=True
        ):
            r["phase"] = phase
    with pytest.raises(ValueError, match="profile"):
        validate_profiles(cells)
    cell = cells[1]
    cell["profile"] = {
        "wall_s": 1.0,
        "top_host_functions": [{}],
        "path_functions": [
            dict(file="_tree.py", function="fit_tree_gpu_native", calls=60),
            dict(file="_objectives.py", function="step", calls=30),
        ],
        "memory": {
            "errors": [],
            "samples": 3,
            "initial_used_bytes": 10,
            "sampled_peak_used_bytes": 20,
            "total_bytes": 100,
            "sampled_peak_delta_bytes": 10,
        },
    }
    for record in cell["records"]:
        record["actual_device"] = "cuda"
    validate_profiles([cell])
    cell["records"][0]["actual_device"] = "cpu"
    with pytest.raises(ValueError, match="device"):
        validate_profiles([cell])
    cell["records"][0]["actual_device"] = "cuda"
    cell["profile"]["memory"]["sampled_peak_used_bytes"] = 101
    with pytest.raises(ValueError, match="memory"):
        validate_profiles([cell])
