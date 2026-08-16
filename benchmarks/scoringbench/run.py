"""Run OpenBoost through an unmodified ScoringBench checkout.

The official suite owns datasets, folds, metrics and Parquet output.  This
launcher only registers OpenBoost (plus selected existing baselines), records
provenance and exposes a small smoke mode for integration testing.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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
            "Comma-separated models: openboost_cpu, openboost_cuda, ngboost, "
            "xgblss, catboost_quantile"
        ),
    )
    parser.add_argument("--n-trees", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--n-quantiles", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3000,
        help="Official ScoringBench default is 3000; use 0 only for a scale extension",
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        action="append",
        help="Run selected index from ScoringBench's validated dataset list (repeatable)",
    )
    parser.add_argument(
        "--dataset-name",
        action="append",
        help="Run exact case-insensitive dataset name from the validated list (repeatable)",
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

    return all_datasets


def _model_factories(args):
    from benchmarks.scoringbench.openboost_wrapper import OpenBoostWrapper

    common = {
        "n_trees": args.n_trees,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "n_quantiles": args.n_quantiles,
    }

    def openboost(backend: str):
        return lambda: OpenBoostWrapper(backend=backend, **common)

    factories = {
        "openboost_cpu": openboost("cpu"),
        "openboost_cuda": openboost("cuda"),
    }

    if "ngboost" in args.models:
        from scoringbench.wrappers.ngboost_wrapper import NGBoostWrapper

        factories["ngboost"] = lambda: NGBoostWrapper(
            dist="normal",
            n_estimators=args.n_trees,
            learning_rate=args.learning_rate,
            n_quantiles=args.n_quantiles,
            ngb_params={"random_state": args.seed},
        )

    if "xgblss" in args.models:
        from scoringbench.wrappers.xgblss_wrapper import XGBLSSWrapper

        factories["xgblss"] = lambda: XGBLSSWrapper(
            n_quantiles=args.n_quantiles,
            num_boost_round=args.n_trees,
            distribution="Gaussian",
            xgblss_params={"max_depth": args.max_depth, "eta": args.learning_rate},
        )

    if "catboost_quantile" in args.models:
        from scoringbench.wrappers.catboost_wrapper import CatBoostQuantileWrapper

        factories["catboost_quantile"] = lambda: CatBoostQuantileWrapper(
            n_quantiles=args.n_quantiles,
            iterations=args.n_trees,
            catboost_params={
                "depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "random_seed": args.seed,
            },
        )

    valid = set(factories)
    unknown = [name for name in args.models if name not in valid]
    if unknown:
        allowed = [
            "openboost_cpu",
            "openboost_cuda",
            "ngboost",
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
) -> Path:
    import openboost as ob

    official_shape = (
        not args.smoke
        and args.sample_size == 3000
        and args.n_folds == 5
        and args.n_repeats == 1
    )
    if args.smoke:
        protocol_mode = "smoke"
    elif args.sample_size != 3000:
        protocol_mode = "scoringbench_scale_extension"
    elif official_shape and (args.dataset_index or args.dataset_name):
        protocol_mode = "official_quality_shard"
    elif official_shape:
        protocol_mode = "official_quality"
    else:
        protocol_mode = "scoringbench_protocol_deviation"

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "ScoringBench",
        "protocol_mode": protocol_mode,
        "official_protocol_compatible": official_shape,
        "warning": (
            None
            if official_shape
            else "This run is not directly comparable to the official 5-fold, sample_size=3000 leaderboard."
        ),
        "openboost_git": _git_state(PROJECT_ROOT),
        "scoringbench_git": _git_state(scoringbench_dir),
        "arguments": vars(args),
        "datasets": [
            {
                "name": dataset["name"],
                "source": dataset.get("source", "openml"),
                "id": dataset.get("id", dataset.get("loader")),
            }
            for dataset in datasets
        ],
        "result_rows": result_rows,
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

    from scoringbench.datasets import get_DATASETS_CONFIG, validate_datasets
    from scoringbench.runner import run_benchmark
    from scoringbench.utils import set_seed

    set_seed(args.seed)
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
        datasets = validate_datasets(get_DATASETS_CONFIG())
        if args.list_datasets:
            for index, dataset in enumerate(datasets):
                print(f"{index:3d}  {dataset['name']}")
            return 0
        datasets = _select_datasets(datasets, args)

    if args.lite:
        args.n_folds = 2

    model_factories = _model_factories(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    result = run_benchmark(
        datasets_config=datasets,
        model_factories=model_factories,
        output_dir=output_dir,
        n_folds=args.n_folds,
        n_repeats_cv=args.n_repeats,
        seed=args.seed,
        sample_size=args.sample_size,
    )
    manifest = _write_provenance(
        output_dir,
        scoringbench_dir,
        args,
        datasets,
        result_rows=len(result),
    )
    print(f"OpenBoost provenance: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
