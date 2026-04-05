"""Composite score computation for autoresearch v2.

Score is a weighted geometric mean in [0, 1] where higher is better.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

SPEED_WEIGHT = 0.60
ACCURACY_WEIGHT = 0.25
COVERAGE_WEIGHT = 0.15

# Speed: cap speedup ratio so one outlier config doesn't dominate
SPEED_RATIO_CAP = 10.0

# Accuracy: hard gate — no dataset parity below this
MIN_PARITY = 0.85

# Speed: hard gate — no config this much slower than XGBoost
MAX_SLOWDOWN = 5.0


# ---------------------------------------------------------------------------
# Sub-score: speed
# ---------------------------------------------------------------------------

def compute_speed_score(config_results: dict[str, dict]) -> tuple[float, bool]:
    """Compute speed sub-score from per-config timing results.

    Each config_results[name] should have:
        ob_median_s: float
        xgb_median_s: float  (or None if XGBoost unavailable)

    Returns (score in [0, 1], gates_passed).
    """
    ratios = []
    for name, r in config_results.items():
        ob_time = r["ob_median_s"]
        xgb_time = r.get("xgb_median_s")

        if xgb_time is None or xgb_time <= 0:
            continue

        ratio = xgb_time / ob_time  # >1 means OB is faster
        if ratio < (1.0 / MAX_SLOWDOWN):
            # Hard gate: OB is >5x slower than XGBoost on this config
            return 0.0, False

        ratios.append(ratio)

    if not ratios:
        # No XGBoost comparison available — use a neutral score
        return 0.5, True

    # Geometric mean of speedup ratios, capped
    capped = [min(r, SPEED_RATIO_CAP) for r in ratios]
    geo_mean = math.exp(sum(math.log(r) for r in capped) / len(capped))

    # Normalize: 1.0 (parity) -> 0.5, SPEED_RATIO_CAP (10x faster) -> 1.0
    score = min(geo_mean, SPEED_RATIO_CAP) / (2.0 * SPEED_RATIO_CAP)
    # Shift so parity=0.5: score = 0.5 * (1 + geo_mean / cap)
    # Simpler: linear map [0, cap] -> [0, 1]
    score = min(geo_mean, SPEED_RATIO_CAP) / SPEED_RATIO_CAP

    return float(score), True


# ---------------------------------------------------------------------------
# Sub-score: accuracy
# ---------------------------------------------------------------------------

def compute_accuracy_score(dataset_results: dict[str, dict]) -> tuple[float, bool]:
    """Compute accuracy sub-score from per-dataset parity ratios.

    Each dataset_results[name] should have:
        parity: float  (>1 means OB is better)

    Parity convention:
        For "higher is better" metrics (R2, accuracy, AUC): ob_metric / xgb_metric
        For "lower is better" metrics (RMSE, NLL): xgb_metric / ob_metric
    So parity > 1.0 always means OpenBoost wins.

    Returns (score in [0, 1], gates_passed).
    """
    parities = []
    for name, r in dataset_results.items():
        parity = r["parity"]
        if parity < MIN_PARITY:
            # Hard gate: OB is >15% worse than XGBoost on this dataset
            return 0.0, False
        parities.append(parity)

    if not parities:
        return 0.5, True

    # Geometric mean of parity ratios
    geo_mean = math.exp(sum(math.log(p) for p in parities) / len(parities))

    # Normalize: 0.85 (min parity) -> ~0.57, 1.0 (parity) -> 0.67, 1.5 -> 1.0
    score = min(geo_mean, 1.5) / 1.5

    return float(score), True


# ---------------------------------------------------------------------------
# Sub-score: coverage
# ---------------------------------------------------------------------------

# Weights for each coverage test
COVERAGE_WEIGHTS = {
    "missing_values": 2.0,
    "categoricals": 2.0,
    "naturalboost": 1.5,
    "dart": 1.0,
    "gam": 1.0,
    "symmetric_growth": 0.5,
    "leafwise_growth": 0.5,
}

HARD_GATE_TESTS = {"missing_values", "categoricals"}


def compute_coverage_score(test_results: dict[str, dict]) -> tuple[float, bool]:
    """Compute coverage sub-score from per-test results.

    Each test_results[name] should have:
        passed: bool
        error: str | None

    Returns (score in [0, 1], gates_passed).
    """
    # Check hard gates
    for name in HARD_GATE_TESTS:
        if name in test_results and not test_results[name]["passed"]:
            return 0.0, False

    # Weighted pass fraction
    total_weight = 0.0
    passed_weight = 0.0
    for name, result in test_results.items():
        weight = COVERAGE_WEIGHTS.get(name, 1.0)
        total_weight += weight
        if result["passed"]:
            passed_weight += weight

    if total_weight == 0:
        return 1.0, True

    score = passed_weight / total_weight
    return float(score), True


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_composite_score(
    speed_result: dict,
    accuracy_result: dict,
    coverage_result: dict,
) -> tuple[float, str]:
    """Compute the composite autoresearch v2 score.

    Returns (composite_score, status).
    Status is one of: PASS, FAIL_SPEED, FAIL_ACCURACY, FAIL_COVERAGE.
    """
    speed_score = speed_result["score"]
    speed_ok = speed_result["gates_passed"]
    accuracy_score = accuracy_result["score"]
    accuracy_ok = accuracy_result["gates_passed"]
    coverage_score = coverage_result["score"]
    coverage_ok = coverage_result["gates_passed"]

    if not speed_ok:
        return float("inf"), "FAIL_SPEED"
    if not accuracy_ok:
        return float("inf"), "FAIL_ACCURACY"
    if not coverage_ok:
        return float("inf"), "FAIL_COVERAGE"

    # Weighted geometric mean
    composite = (
        speed_score ** SPEED_WEIGHT
        * accuracy_score ** ACCURACY_WEIGHT
        * coverage_score ** COVERAGE_WEIGHT
    )

    return float(composite), "PASS"
