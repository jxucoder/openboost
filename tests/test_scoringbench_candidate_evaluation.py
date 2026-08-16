"""Tests for the preregistered HistogramBoost acceptance evaluator."""

import pytest
from benchmarks.scoringbench.evaluate_crps_candidate import evaluate_records


def _records(
    candidate_crps=0.99,
    xgb_crps=1.0,
    cat_crps=1.02,
    candidate_coverage=0.9,
    baseline_coverage=0.8,
):
    result = []
    for model, crps, coverage, interval, rmse in (
        ("openboost_histogram_cpu", candidate_crps, candidate_coverage, 2.0, 1.0),
        ("xgboost_quantile", xgb_crps, baseline_coverage, 2.0, 1.0),
        ("catboost_quantile", cat_crps, baseline_coverage, 2.1, 1.02),
    ):
        for fold in range(5):
            result.append(
                {
                    "dataset": "example",
                    "model": model,
                    "fold": fold,
                    "crps": crps,
                    "coverage_90": coverage,
                    "interval_score_90": interval,
                    "rmse": rmse,
                    "error": None,
                }
            )
    return result


def test_development_candidate_passes_every_frozen_guardrail():
    result = evaluate_records(
        _records(),
        candidate="openboost_histogram_cpu",
        baselines=("xgboost_quantile", "catboost_quantile"),
    )

    assert result["selected_strong_baseline"] == "xgboost_quantile"
    assert result["comparisons"]["candidate_fold_wins"] == 5
    assert result["development_pass"] is True
    assert result["accepted"] is True


def test_confirmation_requires_strict_crps_win_and_four_folds():
    result = evaluate_records(
        _records(candidate_crps=1.0),
        candidate="openboost_histogram_cpu",
        baselines=("xgboost_quantile", "catboost_quantile"),
        phase="confirmation",
    )

    assert result["development_pass"] is True
    assert result["confirmation_dataset_win"] is False
    assert result["accepted"] is False


def test_candidate_with_good_crps_but_bad_coverage_fails():
    result = evaluate_records(
        _records(candidate_coverage=0.7),
        candidate="openboost_histogram_cpu",
        baselines=("xgboost_quantile", "catboost_quantile"),
    )

    assert result["guardrails"]["coverage_error_at_most_0_05"] is False
    assert result["guardrails"]["coverage_error_improves_by_0_02"] is False
    assert result["accepted"] is False


def test_incomplete_rows_fail_closed():
    with pytest.raises(ValueError, match="incomplete rows"):
        evaluate_records(
            _records()[:-1],
            candidate="openboost_histogram_cpu",
            baselines=("xgboost_quantile", "catboost_quantile"),
        )
