"""Run OpenBoost through an unmodified ScoringBench checkout.

The official suite owns datasets, folds, metrics and Parquet output.  This
launcher only registers OpenBoost (plus selected existing baselines), records
provenance and exposes a small smoke mode for integration testing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _float_csv(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in _csv(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("expected at least one number")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset_registry(path: Path) -> list[dict]:
    """Load and minimally validate a frozen ScoringBench dataset registry."""
    payload = json.loads(path.read_text())
    datasets = payload.get("datasets") if isinstance(payload, dict) else payload
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"dataset registry must contain a non-empty list: {path}")
    if any(not isinstance(dataset, dict) or not dataset.get("name") for dataset in datasets):
        raise ValueError(f"every dataset registry entry must be an object with a name: {path}")
    names = [dataset["name"].casefold() for dataset in datasets]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate case-insensitive dataset names in {path}: {duplicates}")
    return datasets


def _resolve_dataset_registry_path(value: str | None) -> Path | None:
    """Resolve a CLI registry path before changing into the artifact directory."""
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _verify_dataset_files(datasets: list[dict], ensure_cached) -> list[dict]:
    """Materialize and verify dataset files pinned by a frozen registry.

    ScoringBench's processed cache is intentionally fast but cannot prove which
    raw bytes produced an entry. A registry may therefore provide
    ``raw_sha256``. Those entries are downloaded through ScoringBench's own raw
    cache and checked before validation or fold construction.
    """
    verified = []
    for dataset in datasets:
        expected = dataset.get("raw_sha256")
        if expected is None:
            continue
        expected = str(expected).lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError(f"invalid raw_sha256 for dataset {dataset['name']!r}: {expected!r}")
        if dataset.get("source") != "pmlb" or not dataset.get("url"):
            raise ValueError(
                "raw_sha256 verification currently requires a PMLB URL; "
                f"dataset {dataset['name']!r} has source={dataset.get('source')!r}"
            )

        filename = f"{dataset['name']}.tsv.gz"
        path = Path(ensure_cached(dataset["name"], dataset["url"], filename))
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"raw dataset hash mismatch for {dataset['name']!r}: "
                f"expected {expected}, got {actual} at {path}"
            )
        verified.append(
            {
                "name": dataset["name"],
                "url": dataset["url"],
                "sha256": actual,
                "size_bytes": path.stat().st_size,
            }
        )
    return verified


def _enforce_dataset_role_lock(datasets: list[dict], *, allow_confirmation: bool) -> None:
    """Prevent accidental observation of preregistered confirmation data."""
    locked = [
        dataset["name"]
        for dataset in datasets
        if dataset.get("openboost_role") == "untouched_confirmation"
    ]
    if locked and not allow_confirmation:
        raise ValueError(
            "confirmation dataset is still locked; freeze the candidate first, "
            "then rerun with --allow-confirmation: " + ", ".join(locked)
        )


def _git_state(path: Path) -> dict:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(path), *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "changes": status.splitlines() if status else [],
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_info() -> dict | None:
    try:
        from numba import cuda

        if not cuda.is_available():
            return None
        device = cuda.get_current_device()
        name = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
        return {
            "name": name,
            "compute_capability": list(device.compute_capability),
        }
    except Exception as exc:  # provenance should never fail the benchmark
        return {"error": f"{type(exc).__name__}: {exc}"}


def _ci_state() -> dict | None:
    """Return non-secret GitHub Actions identity for artifact provenance."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return None

    names = {
        "event_name": "GITHUB_EVENT_NAME",
        "repository": "GITHUB_REPOSITORY",
        "ref": "GITHUB_REF",
        "tested_sha": "GITHUB_SHA",
        "source_sha": "OPENBOOST_SOURCE_SHA",
        "head_ref": "GITHUB_HEAD_REF",
        "run_id": "GITHUB_RUN_ID",
        "run_attempt": "GITHUB_RUN_ATTEMPT",
    }
    return {
        "provider": "github_actions",
        **{key: os.environ.get(env_name) for key, env_name in names.items()},
    }


@contextmanager
def _working_directory(path: Path):
    """Temporarily direct upstream relative outputs into an artifact directory."""
    original = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


_REQUIRED_DISTRIBUTIONAL_METRICS = (
    "crps",
    "log_score",
    "rmse",
    "coverage_90",
    "interval_score_90",
    "train_time",
)


def _is_present_finite(value) -> bool:
    """Return whether a benchmark value is present and numerically finite."""
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_present_text(value) -> bool:
    return value is not None and bool(str(value).strip()) and str(value).lower() != "nan"


def _audit_records(
    records: list[dict],
    datasets: list[dict],
    model_names: list[str],
    *,
    n_folds: int,
    n_repeats: int,
) -> dict:
    """Audit exact dataset/model/fold coverage after the upstream runner returns.

    ScoringBench intentionally catches dataset and model exceptions so a long
    campaign can continue.  That behavior is useful for throughput, but its
    return code cannot be used as a completeness signal.  This audit turns
    missing, duplicate, error, and non-finite metric rows into explicit data.
    """
    expected_keys = {
        (dataset["name"], model_name, fold)
        for dataset in datasets
        for model_name in model_names
        for fold in range(n_folds * n_repeats)
    }
    rows_by_key: dict[tuple[str, str, int], list[dict]] = {}
    unexpected_rows = []
    for row in records:
        try:
            key = (str(row["dataset"]), str(row["model"]), int(row["fold"]))
        except (KeyError, TypeError, ValueError):
            unexpected_rows.append(
                {
                    "reason": "invalid_identity",
                    "dataset": repr(row.get("dataset")),
                    "model": repr(row.get("model")),
                    "fold": repr(row.get("fold")),
                }
            )
            continue
        if key not in expected_keys:
            unexpected_rows.append(
                {
                    "reason": "unexpected_identity",
                    "dataset": key[0],
                    "model": key[1],
                    "fold": key[2],
                }
            )
            continue
        rows_by_key.setdefault(key, []).append(row)

    missing_rows = [
        {"dataset": dataset, "model": model, "fold": fold}
        for dataset, model, fold in sorted(expected_keys - rows_by_key.keys())
    ]
    duplicate_rows = [
        {
            "dataset": key[0],
            "model": key[1],
            "fold": key[2],
            "count": len(rows),
        }
        for key, rows in sorted(rows_by_key.items())
        if len(rows) != 1
    ]
    error_rows = []
    invalid_metric_rows = []
    valid_keys = set()
    for key, rows in rows_by_key.items():
        if len(rows) != 1:
            continue
        row = rows[0]
        error = row.get("error")
        if _is_present_text(error):
            error_rows.append(
                {
                    "dataset": key[0],
                    "model": key[1],
                    "fold": key[2],
                    "error_type": (
                        str(row["error_type"]) if _is_present_text(row.get("error_type")) else None
                    ),
                    "error": str(error),
                }
            )
            continue
        invalid_metrics = [
            metric
            for metric in _REQUIRED_DISTRIBUTIONAL_METRICS
            if not _is_present_finite(row.get(metric))
        ]
        if invalid_metrics:
            invalid_metric_rows.append(
                {
                    "dataset": key[0],
                    "model": key[1],
                    "fold": key[2],
                    "metrics": invalid_metrics,
                }
            )
            continue
        valid_keys.add(key)

    dataset_outcomes = []
    expected_per_dataset = len(model_names) * n_folds * n_repeats
    for dataset in datasets:
        name = dataset["name"]
        observed = sum(key[0] == name for key in rows_by_key)
        valid = sum(key[0] == name for key in valid_keys)
        dataset_outcomes.append(
            {
                "dataset": name,
                "expected_rows": expected_per_dataset,
                "observed_rows": observed,
                "valid_rows": valid,
                "status": "complete" if valid == expected_per_dataset else "incomplete",
            }
        )

    complete = (
        len(valid_keys) == len(expected_keys)
        and not missing_rows
        and not duplicate_rows
        and not error_rows
        and not invalid_metric_rows
        and not unexpected_rows
    )
    return {
        "schema_version": 1,
        "status": "complete" if complete else "incomplete",
        "expected_rows": len(expected_keys),
        "observed_rows": len(records),
        "valid_rows": len(valid_keys),
        "missing_rows": missing_rows,
        "duplicate_rows": duplicate_rows,
        "error_rows": error_rows,
        "invalid_metric_rows": invalid_metric_rows,
        "unexpected_rows": unexpected_rows,
        "datasets": dataset_outcomes,
    }


def _write_outcome(output_dir: Path, outcome: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "benchmark_outcome.json"
    path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenBoost on the external ScoringBench protocol"
    )
    parser.add_argument(
        "--scoringbench-dir",
        default=os.environ.get("SCORINGBENCH_DIR", ".repos/ScoringBench"),
        help="Path to a ScoringBench git checkout",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results/scoringbench",
        help="ScoringBench Parquet and OpenBoost manifest output",
    )
    parser.add_argument(
        "--models",
        type=_csv,
        default=["openboost_cpu", "ngboost"],
        help=(
            "Comma-separated models: openboost_cpu, openboost_cuda, "
            "openboost_histogram_cpu, openboost_histogram_cpu_v2, ngboost, "
            "xgboost_quantile, xgblss, catboost_quantile"
        ),
    )
    parser.add_argument("--n-trees", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument(
        "--training-objective",
        choices=("nll", "crps"),
        default="nll",
        help=(
            "OpenBoost training objective. CRPS is a development candidate; "
            "use --development-run while evaluating it."
        ),
    )
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--n-quantiles", type=int, default=99)
    parser.add_argument("--histogram-rounds", type=int, default=100)
    parser.add_argument("--histogram-bins", type=int, default=50)
    parser.add_argument("--histogram-learning-rate", type=float, default=0.05)
    parser.add_argument("--histogram-max-depth", type=int, default=6)
    parser.add_argument("--histogram-curvature-scale", type=float, default=1.0)
    parser.add_argument(
        "--histogram-v2-temperature-grid",
        type=_float_csv,
        default=(0.5, 0.7, 0.85, 1.0, 1.2),
    )
    parser.add_argument("--histogram-v2-calibration-fraction", type=float, default=0.2)
    parser.add_argument("--histogram-v2-calibration-seed", type=int, default=42)
    parser.add_argument("--histogram-v2-evaluation-subdivisions", type=int, default=2)
    parser.add_argument(
        "--xgboost-rounds",
        type=int,
        default=100,
        help="Boosting rounds for the ScoringBench XGBoost quantile baseline",
    )
    parser.add_argument(
        "--xgboost-quantiles",
        type=int,
        default=50,
        help="Quantile outputs for the ScoringBench XGBoost quantile baseline",
    )
    parser.add_argument(
        "--xgblss-rounds",
        type=int,
        default=100,
        help="Boosting rounds for the ScoringBench Gaussian XGBoostLSS baseline",
    )
    parser.add_argument(
        "--catboost-rounds",
        type=int,
        default=1000,
        help="Iterations for the ScoringBench CatBoost MultiQuantile baseline",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3000,
        help="Official ScoringBench default is 3000; use 0 only for a scale extension",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--dataset-index",
        type=int,
        action="append",
        help="Run selected index from ScoringBench's validated dataset list (repeatable)",
    )
    selection.add_argument(
        "--dataset-name",
        action="append",
        help="Run exact case-insensitive dataset name from the validated list (repeatable)",
    )
    selection.add_argument(
        "--shard-index",
        type=int,
        help="Run one zero-based strided shard from the dataset registry",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        help="Total number of stable registry shards; requires --shard-index",
    )
    parser.add_argument(
        "--dataset-registry",
        help=(
            "Frozen ScoringBench datasets.json to use instead of rebuilding the "
            "dynamic upstream registry"
        ),
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Use two folds while retaining ScoringBench datasets and metrics",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use sklearn diabetes with 2 folds; validates integration, not leaderboard evidence",
    )
    parser.add_argument(
        "--development-run",
        action="store_true",
        help=(
            "Mark this run as tuning-only evidence that must not be submitted or "
            "reported as a held-out leaderboard result"
        ),
    )
    parser.add_argument(
        "--allow-confirmation",
        action="store_true",
        help=(
            "Unlock a registry entry marked untouched_confirmation. Use only "
            "after the candidate implementation and configuration are committed."
        ),
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print ScoringBench's validated dataset names and exit",
    )
    return parser


def _select_datasets(all_datasets: list[dict], args) -> list[dict]:
    if args.dataset_index:
        invalid = [i for i in args.dataset_index if i < 0 or i >= len(all_datasets)]
        if invalid:
            raise ValueError(
                f"dataset indices out of range: {invalid}; valid range is 0..{len(all_datasets) - 1}"
            )
        return [all_datasets[i] for i in args.dataset_index]

    if args.dataset_name:
        lookup = {dataset["name"].casefold(): dataset for dataset in all_datasets}
        missing = [name for name in args.dataset_name if name.casefold() not in lookup]
        if missing:
            raise ValueError(f"unknown dataset names: {missing}; use --list-datasets")
        return [lookup[name.casefold()] for name in args.dataset_name]

    if args.shard_index is not None:
        if args.shard_count is None or args.shard_count <= 0:
            raise ValueError("--shard-count must be a positive integer with --shard-index")
        if args.shard_index < 0 or args.shard_index >= args.shard_count:
            raise ValueError(
                f"--shard-index must be in 0..{args.shard_count - 1}, got {args.shard_index}"
            )
        selected = [
            dataset
            for position, dataset in enumerate(all_datasets)
            if position % args.shard_count == args.shard_index
        ]
        if not selected:
            raise ValueError(
                f"shard {args.shard_index}/{args.shard_count} selects no datasets "
                f"from a registry of size {len(all_datasets)}"
            )
        return selected

    if args.shard_count is not None:
        raise ValueError("--shard-count requires --shard-index")

    return all_datasets


def _validate_selected_datasets(all_datasets: list[dict], args, validate) -> list[dict]:
    """Validate only named shards; indexed shards retain validated-list semantics."""
    if args.dataset_name or args.shard_index is not None:
        return validate(_select_datasets(all_datasets, args))
    return _select_datasets(validate(all_datasets), args)


def _model_parameters(args) -> dict[str, dict]:
    """Return the exact benchmark constructor parameters for every model."""
    openboost_common = {
        "n_trees": args.n_trees,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "n_quantiles": args.n_quantiles,
        "model_params": {
            "reg_lambda": args.reg_lambda,
            "min_child_weight": args.min_child_weight,
            "training_objective": args.training_objective,
        },
    }
    return {
        "openboost_cpu": {"backend": "cpu", **openboost_common},
        "openboost_cuda": {"backend": "cuda", **openboost_common},
        "openboost_histogram_cpu": {
            "n_distribution_bins": args.histogram_bins,
            "n_trees": args.histogram_rounds,
            "learning_rate": args.histogram_learning_rate,
            "max_depth": args.histogram_max_depth,
            "n_feature_bins": 254,
            "curvature_scale": args.histogram_curvature_scale,
        },
        "openboost_histogram_cpu_v2": {
            "n_distribution_bins": args.histogram_bins,
            "n_trees": args.histogram_rounds,
            "learning_rate": args.histogram_learning_rate,
            "max_depth": args.histogram_max_depth,
            "n_feature_bins": 254,
            "curvature_scale": args.histogram_curvature_scale,
            "temperature_grid": args.histogram_v2_temperature_grid,
            "calibration_fraction": args.histogram_v2_calibration_fraction,
            "calibration_seed": args.histogram_v2_calibration_seed,
            "evaluation_subdivisions": args.histogram_v2_evaluation_subdivisions,
        },
        "ngboost": {
            "dist": "normal",
            "n_estimators": args.n_trees,
            "learning_rate": args.learning_rate,
            "n_quantiles": args.n_quantiles,
            "ngb_params": {"random_state": args.seed},
        },
        "xgboost_quantile": {
            "n_bins": args.xgboost_quantiles,
            "num_boost_round": args.xgboost_rounds,
            "xgb_params": {"device": "cpu", "seed": args.seed, "nthread": 2},
        },
        "xgblss": {
            "n_quantiles": args.n_quantiles,
            "num_boost_round": args.xgblss_rounds,
            "distribution": "Gaussian",
            "xgblss_params": {"device": "cpu", "seed": args.seed, "nthread": 2},
        },
        "catboost_quantile": {
            "n_quantiles": args.n_quantiles,
            "iterations": args.catboost_rounds,
            "catboost_params": {
                "allow_writing_files": False,
                "random_seed": args.seed,
                "thread_count": 2,
            },
        },
    }


def _model_factories(args):
    from benchmarks.scoringbench.openboost_wrapper import (
        OpenBoostHistogramWrapper,
        OpenBoostWrapper,
    )

    parameters = _model_parameters(args)

    factories = {
        "openboost_cpu": lambda: OpenBoostWrapper(**parameters["openboost_cpu"]),
        "openboost_cuda": lambda: OpenBoostWrapper(**parameters["openboost_cuda"]),
        "openboost_histogram_cpu": lambda: OpenBoostHistogramWrapper(
            **parameters["openboost_histogram_cpu"]
        ),
        "openboost_histogram_cpu_v2": lambda: OpenBoostHistogramWrapper(
            **parameters["openboost_histogram_cpu_v2"]
        ),
    }

    if "ngboost" in args.models:
        from scoringbench.wrappers.ngboost_wrapper import NGBoostWrapper

        factories["ngboost"] = lambda: NGBoostWrapper(**parameters["ngboost"])

    if "xgboost_quantile" in args.models:
        from scoringbench.wrappers.xgb_vector import XGBQuantileVectorWrapper

        factories["xgboost_quantile"] = lambda: XGBQuantileVectorWrapper(
            **parameters["xgboost_quantile"]
        )

    if "xgblss" in args.models:
        from scoringbench.wrappers.xgblss_wrapper import XGBLSSWrapper

        factories["xgblss"] = lambda: XGBLSSWrapper(**parameters["xgblss"])

    if "catboost_quantile" in args.models:
        from scoringbench.wrappers.catboost_wrapper import CatBoostQuantileWrapper

        factories["catboost_quantile"] = lambda: CatBoostQuantileWrapper(
            **parameters["catboost_quantile"]
        )

    valid = set(factories)
    unknown = [name for name in args.models if name not in valid]
    if unknown:
        allowed = [
            "openboost_cpu",
            "openboost_cuda",
            "openboost_histogram_cpu",
            "openboost_histogram_cpu_v2",
            "ngboost",
            "xgboost_quantile",
            "xgblss",
            "catboost_quantile",
        ]
        raise ValueError(f"unknown models {unknown}; allowed values: {allowed}")

    return {name: factories[name] for name in args.models}


def _write_provenance(
    output_dir: Path,
    scoringbench_dir: Path,
    args,
    datasets: list[dict],
    result_rows: int,
    outcome: dict,
    verified_dataset_files: list[dict] | None = None,
) -> Path:
    import openboost as ob

    official_shape = (
        not args.smoke and args.sample_size == 3000 and args.n_folds == 5 and args.n_repeats == 1
    )
    official_protocol_compatible = official_shape and not args.development_run
    if args.smoke:
        protocol_mode = "smoke"
    elif args.development_run:
        protocol_mode = "development_tuning"
    elif args.sample_size != 3000:
        protocol_mode = "scoringbench_scale_extension"
    elif official_shape and (
        args.dataset_index or args.dataset_name or args.shard_index is not None
    ):
        protocol_mode = "official_quality_shard"
    elif official_shape:
        protocol_mode = "official_quality"
    else:
        protocol_mode = "scoringbench_protocol_deviation"

    registry_path = output_dir / "datasets.json"
    manifest = {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "ScoringBench",
        "protocol_mode": protocol_mode,
        "official_protocol_compatible": official_protocol_compatible,
        "warning": (
            None
            if official_protocol_compatible
            else (
                "This is a development/tuning run and must not be represented as "
                "held-out leaderboard evidence."
                if args.development_run
                else "This run is not directly comparable to the official 5-fold, sample_size=3000 leaderboard."
            )
        ),
        "openboost_git": _git_state(PROJECT_ROOT),
        "scoringbench_git": _git_state(scoringbench_dir),
        "ci": _ci_state(),
        "arguments": vars(args),
        "model_parameters": {name: _model_parameters(args)[name] for name in args.models},
        "datasets": [
            {
                "name": dataset["name"],
                "source": dataset.get("source", "openml"),
                "id": dataset.get("id", dataset.get("url", dataset.get("loader"))),
            }
            for dataset in datasets
        ],
        "dataset_registry": {
            "mode": "frozen_file" if args.dataset_registry else "scoringbench_dynamic",
            "resolved_sha256": _sha256(registry_path) if registry_path.exists() else None,
            "source_sha256": (
                _sha256(Path(args.dataset_registry).expanduser().resolve())
                if args.dataset_registry
                else None
            ),
            "file": "datasets.json" if registry_path.exists() else None,
        },
        "verified_dataset_files": verified_dataset_files or [],
        "result_rows": result_rows,
        "expected_result_rows": outcome["expected_rows"],
        "outcome": {
            "status": outcome["status"],
            "valid_rows": outcome["valid_rows"],
            "missing_rows": len(outcome["missing_rows"]),
            "duplicate_rows": len(outcome["duplicate_rows"]),
            "error_rows": len(outcome["error_rows"]),
            "invalid_metric_rows": len(outcome["invalid_metric_rows"]),
            "unexpected_rows": len(outcome["unexpected_rows"]),
            "file": "benchmark_outcome.json",
        },
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "gpu": _gpu_info(),
        },
        "versions": {
            "openboost": ob.__version__,
            "numpy": np.__version__,
            **{
                name: _package_version(name)
                for name in (
                    "scipy",
                    "scikit-learn",
                    "pandas",
                    "pyarrow",
                    "torch",
                    "numba",
                    "numba-cuda",
                    "cupy-cuda12x",
                    "ngboost",
                    "xgboost",
                    "xgboostlss",
                    "catboost",
                )
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "openboost_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    args = _build_parser().parse_args()
    dataset_registry_path = _resolve_dataset_registry_path(args.dataset_registry)
    if sys.platform == "darwin" and platform.machine() == "x86_64":
        raise SystemExit(
            "The complete ScoringBench runner is unsupported on Intel macOS: "
            "the available PyTorch wheel uses the NumPy 1.x ABI while "
            "ScoringBench requires NumPy 2.x. Run the benchmark on Linux "
            "(the target environment for published CPU/CUDA results). The "
            "wrapper contract test can still be run separately."
        )
    scoringbench_dir = Path(args.scoringbench_dir).expanduser().resolve()
    if not (scoringbench_dir / "scoringbench" / "runner.py").exists():
        raise SystemExit(
            f"No ScoringBench checkout found at {scoringbench_dir}. "
            "Clone https://github.com/jonaslandsgesell/ScoringBench first."
        )

    sys.path.insert(0, str(SRC_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(scoringbench_dir))

    from scoringbench.datasets import (
        _ensure_cached,
        get_DATASETS_CONFIG,
        validate_datasets,
    )
    from scoringbench.runner import run_benchmark
    from scoringbench.utils import set_seed

    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    verified_dataset_files = []
    if args.smoke:
        datasets = [
            {
                "name": "diabetes_smoke",
                "source": "sklearn",
                "loader": "load_diabetes",
                "abbr": "DBS",
                "sample_size": min(args.sample_size or 442, 442),
            }
        ]
        args.n_folds = 2
    else:
        # ScoringBench exports its resolved dataset registry to Path.cwd(). Keep
        # that reproducibility artifact with the benchmark instead of dirtying
        # the OpenBoost checkout.
        with _working_directory(output_dir):
            if args.dataset_registry:
                all_datasets = _load_dataset_registry(dataset_registry_path)
                Path("datasets.json").write_text(
                    json.dumps(all_datasets, indent=2, ensure_ascii=False) + "\n"
                )
            else:
                all_datasets = get_DATASETS_CONFIG()
            if args.list_datasets:
                datasets = validate_datasets(all_datasets)
                for index, dataset in enumerate(datasets):
                    print(f"{index:3d}  {dataset['name']}")
                return 0

            def validate_with_raw_verification(selected):
                nonlocal verified_dataset_files
                _enforce_dataset_role_lock(
                    selected,
                    allow_confirmation=args.allow_confirmation,
                )
                verified_dataset_files = _verify_dataset_files(
                    selected,
                    _ensure_cached,
                )
                if verified_dataset_files:
                    # Force the pinned raw file through preprocessing instead
                    # of accepting an opaque processed cache entry.
                    os.environ["SCORINGBENCH_NO_CACHE"] = "1"
                return validate_datasets(selected)

            datasets = _validate_selected_datasets(
                all_datasets,
                args,
                validate_with_raw_verification,
            )

    if args.lite:
        args.n_folds = 2

    model_factories = _model_factories(args)
    result = run_benchmark(
        datasets_config=datasets,
        model_factories=model_factories,
        output_dir=output_dir,
        n_folds=args.n_folds,
        n_repeats_cv=args.n_repeats,
        seed=args.seed,
        sample_size=args.sample_size,
    )
    outcome = _audit_records(
        result.to_dict(orient="records"),
        datasets,
        list(model_factories),
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
    )
    outcome_path = _write_outcome(output_dir, outcome)
    manifest = _write_provenance(
        output_dir,
        scoringbench_dir,
        args,
        datasets,
        result_rows=len(result),
        outcome=outcome,
        verified_dataset_files=verified_dataset_files,
    )
    print(f"OpenBoost outcome: {outcome_path} ({outcome['status']})")
    print(f"OpenBoost provenance: {manifest}")
    return 0 if outcome["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
