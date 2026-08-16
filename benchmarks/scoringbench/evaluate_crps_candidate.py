"""Evaluate a preregistered ScoringBench CRPS candidate from raw Parquet rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

METRICS = ("crps", "coverage_90", "interval_score_90", "rmse")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _finite_number(value) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def evaluate_records(
    records: list[dict],
    *,
    candidate: str,
    baselines: tuple[str, str],
    n_folds: int = 5,
    phase: str = "development",
) -> dict:
    """Apply the frozen CRPS/coverage/interval/RMSE acceptance rules."""
    if phase not in {"development", "confirmation"}:
        raise ValueError("phase must be 'development' or 'confirmation'")
    models = (candidate, *baselines)
    datasets = {str(record.get("dataset")) for record in records}
    if len(datasets) != 1:
        raise ValueError(f"expected exactly one dataset, got {sorted(datasets)}")
    dataset = datasets.pop()

    indexed: dict[tuple[str, int], dict] = {}
    for record in records:
        model = str(record.get("model"))
        if model not in models:
            continue
        try:
            fold = int(record["fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid fold identity: {record.get('fold')!r}") from exc
        key = (model, fold)
        if key in indexed:
            raise ValueError(f"duplicate result row: model={model}, fold={fold}")
        error = record.get("error")
        if error is not None and str(error).strip() and str(error).lower() != "nan":
            raise ValueError(f"captured model error: model={model}, fold={fold}: {error}")
        invalid = [metric for metric in METRICS if not _finite_number(record.get(metric))]
        if invalid:
            raise ValueError(f"non-finite metrics for model={model}, fold={fold}: {invalid}")
        indexed[key] = record

    expected = {(model, fold) for model in models for fold in range(n_folds)}
    missing = sorted(expected - set(indexed))
    unexpected = sorted(set(indexed) - expected)
    if missing or unexpected or len(records) != len(expected):
        raise ValueError(
            f"incomplete rows: expected={len(expected)}, observed={len(records)}, "
            f"missing={missing}, unexpected={unexpected}"
        )

    summaries = {}
    for model in models:
        rows = [indexed[(model, fold)] for fold in range(n_folds)]
        summaries[model] = {
            "mean_crps": _mean([float(row["crps"]) for row in rows]),
            "mean_abs_coverage_90_error": _mean(
                [abs(float(row["coverage_90"]) - 0.9) for row in rows]
            ),
            "mean_interval_score_90": _mean([float(row["interval_score_90"]) for row in rows]),
            "mean_rmse": _mean([float(row["rmse"]) for row in rows]),
        }

    baseline = min(baselines, key=lambda name: summaries[name]["mean_crps"])
    candidate_summary = summaries[candidate]
    baseline_summary = summaries[baseline]
    crps_ratio = candidate_summary["mean_crps"] / baseline_summary["mean_crps"]
    fold_wins = sum(
        float(indexed[(candidate, fold)]["crps"]) <= float(indexed[(baseline, fold)]["crps"])
        for fold in range(n_folds)
    )
    coverage_improvement = (
        baseline_summary["mean_abs_coverage_90_error"]
        - candidate_summary["mean_abs_coverage_90_error"]
    )
    interval_ratio = (
        candidate_summary["mean_interval_score_90"] / baseline_summary["mean_interval_score_90"]
    )
    rmse_ratio = candidate_summary["mean_rmse"] / baseline_summary["mean_rmse"]

    guardrails = {
        "complete_15_rows": len(indexed) == 3 * n_folds,
        "crps_ratio_at_most_1_02": crps_ratio <= 1.02,
        "at_least_3_of_5_fold_wins": fold_wins >= 3,
        "coverage_error_at_most_0_05": (candidate_summary["mean_abs_coverage_90_error"] <= 0.05),
        "coverage_error_improves_by_0_02": coverage_improvement >= 0.02,
        "interval_score_ratio_at_most_1_05": interval_ratio <= 1.05,
        "rmse_ratio_at_most_1_05": rmse_ratio <= 1.05,
    }
    development_pass = all(guardrails.values())
    confirmation_win = development_pass and crps_ratio < 1.0 and fold_wins >= 4

    return {
        "schema_version": 1,
        "dataset": dataset,
        "phase": phase,
        "candidate": candidate,
        "baselines": list(baselines),
        "selected_strong_baseline": baseline,
        "summaries": summaries,
        "comparisons": {
            "crps_ratio": crps_ratio,
            "candidate_fold_wins": fold_wins,
            "coverage_error_improvement": coverage_improvement,
            "interval_score_ratio": interval_ratio,
            "rmse_ratio": rmse_ratio,
        },
        "guardrails": guardrails,
        "development_pass": development_pass,
        "confirmation_dataset_win": confirmation_win,
        "accepted": development_pass if phase == "development" else confirmation_win,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--candidate", default="openboost_histogram_cpu")
    parser.add_argument(
        "--baselines",
        default="xgboost_quantile,catboost_quantile",
        help="Exactly two comma-separated baseline model names",
    )
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    baselines = tuple(part.strip() for part in args.baselines.split(",") if part.strip())
    if len(baselines) != 2:
        raise SystemExit("--baselines must contain exactly two model names")

    import pandas as pd

    parquet_files = sorted((args.result_dir / "raw").glob("*/*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no raw Parquet files found under {args.result_dir / 'raw'}")
    records = []
    for path in parquet_files:
        records.extend(pd.read_parquet(path).to_dict(orient="records"))
    result = evaluate_records(
        records,
        candidate=args.candidate,
        baselines=baselines,
        phase=args.phase,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
