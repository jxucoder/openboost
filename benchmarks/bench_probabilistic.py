"""Probabilistic GBDT benchmark: the yardstick for the parametric-boosting pivot.

Two suites (see tasks/todo.md and planning/unified-engine-design.md):

  quality  Paired-split comparison vs NGBoost on real datasets.
           Metrics: NLL, CRPS, RMSE, 90% interval coverage, pinball loss.
           Paired Wilcoxon test on per-split NLL. CPU by default (NGBoost is
           CPU-only; set OPENBOOST_BACKEND to control OpenBoost's side).

  speed    Wall-clock fit time + time-to-same-NLL vs NGBoost (and PGBM if
           installed) on large synthetic heteroscedastic data. Meant for
           Modal A100; also runs locally at reduced sizes.

Usage:
    # Local quality suite (CPU, honest vs NGBoost)
    uv run --with ngboost python benchmarks/bench_probabilistic.py --suite quality
    uv run --with ngboost python benchmarks/bench_probabilistic.py --suite quality --quick

    # Local speed suite (small sizes, sanity only)
    uv run --with ngboost python benchmarks/bench_probabilistic.py --suite speed --quick

    # Modal A100 (speed is the ≥10x gate; quality is CPU-honest vs NGBoost)
    uv run modal run benchmarks/bench_probabilistic.py --suite speed
    uv run modal run benchmarks/bench_probabilistic.py --suite all

Output: benchmarks/results/probabilistic_<suite>_<YYYYMMDD_HHMMSS>.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"

SEED = 42
# Shared budget, deliberately close to NGBoost defaults (its native regime).
N_TREES = 500
LEARNING_RATE = 0.03
MAX_DEPTH = 3

# quality suite: (n_splits_small, n_splits_large) — large datasets get fewer
# paired splits because each fit is expensive.
N_SPLITS_SMALL = 20
N_SPLITS_LARGE = 5
LARGE_THRESHOLD = 50_000
# NGBoost-paper-style protocol: hold out a validation set from the training
# split; both libraries early-stop on the SAME validation set.
VAL_FRACTION = 0.2
ES_PATIENCE = 50

# =============================================================================
# Dataset registry (quality suite)
# =============================================================================
# NGBoost-paper-style UCI datasets. OpenML's name->id resolution endpoint is
# flaky (frequent 503s), so each row carries the numeric data_id as well; the
# loader fetches by id first (skips the flaky name endpoint) and falls back to
# name, with retries. A fetch failure skips the dataset with a note instead of
# crashing the run. data_ids verified against OpenML for version 1.

OPENML_DATASETS = [
    # (short name, openml name, version, data_id)
    ("boston", "boston", 1, 531),
    ("concrete", "Concrete_Compressive_Strength", 1, 4353),
    ("energy", "energy-efficiency", 1, 1472),
    ("kin8nm", "kin8nm", 1, 189),
    # naval by-id 44898 is a deactivated version; use name resolution instead.
    ("naval", "naval_propulsion_plant", 1, None),
    ("power", "combined_cycle_power_plant", 1, None),
    ("protein", "physicochemical-protein", 1, 42903),
    ("wine", "wine_quality", 1, 287),
    ("yacht", "yacht_hydrodynamics", 1, 42370),
]


def _fetch_openml_robust(name, version, data_id=None, as_frame=True, retries=3):
    """Fetch from OpenML by id first (avoids the flaky name endpoint), then by
    name, retrying with backoff. Raises the last error if all attempts fail."""
    import time

    from sklearn.datasets import fetch_openml

    attempts = []
    if data_id is not None:
        attempts.append({"data_id": data_id})
    attempts.append({"name": name, "version": version})

    last = None
    for r in range(retries):
        for kw in attempts:
            try:
                return fetch_openml(as_frame=as_frame, parser="auto", **kw)
            except Exception as exc:  # noqa: BLE001 - retry every failure mode
                last = exc
        time.sleep(2 * (r + 1))
    raise last


def load_quality_datasets(quick: bool = False) -> tuple[list[dict], list[str]]:
    import numpy as np

    datasets: list[dict] = []
    notes: list[str] = []

    from sklearn.datasets import fetch_california_housing

    wanted = OPENML_DATASETS[:1] if quick else OPENML_DATASETS

    for short, name, version, data_id in wanted:
        try:
            bunch = _fetch_openml_robust(name, version, data_id)
            frame = bunch.frame.select_dtypes(include=[np.number]).dropna()
            target_col = bunch.target_names[0] if bunch.target_names else \
                frame.columns[-1]
            if target_col not in frame.columns:
                target_col = frame.columns[-1]
            y = frame[target_col].to_numpy(dtype=np.float64)
            X = frame.drop(columns=[target_col]).to_numpy(dtype=np.float64)
            datasets.append({"name": short, "X": X, "y": y,
                             "source": f"openml:{name}/v{version}"})
        except Exception as exc:
            notes.append(f"{short} skipped ({type(exc).__name__}: {exc})")
            print(f"WARNING: {notes[-1]}")

    try:
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        if quick:
            X, y = X[:2_000], y[:2_000]
        datasets.append({"name": "california", "X": X.astype("float64"),
                         "y": y.astype("float64"),
                         "source": "sklearn:california_housing"})
    except Exception as exc:
        notes.append(f"california skipped ({type(exc).__name__}: {exc})")
        print(f"WARNING: {notes[-1]}")

    if not quick:
        try:
            # "active" version: name+version=1 does not resolve on OpenML.
            msd = _fetch_openml_robust("YearPredictionMSD", "active",
                                       data_id=None, as_frame=False)
            X = msd.data.astype("float64")
            y = msd.target.astype("float64")
            # Subsample for the quality suite: NGBoost's exact-split trees at
            # this budget take hours on the full 515K. Full size is covered
            # by the speed suite.
            rng = np.random.RandomState(SEED)
            idx = rng.choice(len(y), 100_000, replace=False)
            datasets.append({"name": "year_msd_100k", "X": X[idx], "y": y[idx],
                             "source": "openml:YearPredictionMSD/v1 (100K subsample)"})
        except Exception as exc:
            notes.append(f"year_msd skipped ({type(exc).__name__}: {exc})")
            print(f"WARNING: {notes[-1]}")

    return datasets, notes


def make_heteroscedastic(n_samples: int, n_features: int = 80, seed: int = SEED):
    """Speed-suite synthetic: Friedman#1 signal + feature-dependent noise."""
    import numpy as np

    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n_samples, n_features)).astype(np.float32)
    f = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )
    sigma = 1.0 + 2.0 * X[:, 5]
    y = (f + sigma * rng.randn(n_samples)).astype(np.float64)
    return X.astype(np.float64), y


# =============================================================================
# Models
# =============================================================================

def make_openboost():
    import openboost as ob

    return ob.NaturalBoostNormal(
        n_trees=N_TREES, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
        n_bins=254,
    )


def make_ngboost():
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from sklearn.tree import DecisionTreeRegressor

    return NGBRegressor(
        Dist=Normal,
        n_estimators=N_TREES,
        learning_rate=LEARNING_RATE,
        Base=DecisionTreeRegressor(criterion="friedman_mse", max_depth=MAX_DEPTH),
        natural_gradient=True,
        verbose=False,
        random_state=SEED,
    )


def make_pgbm():
    """PGBM (torch, GPU-capable probabilistic GBDT). Optional."""
    from pgbm.sklearn import HistGradientBoostingRegressor as PGBMRegressor

    return PGBMRegressor(max_iter=N_TREES, learning_rate=LEARNING_RATE,
                         max_depth=MAX_DEPTH, random_state=SEED)


def predict_normal_params(model, X):
    import numpy as np

    if hasattr(model, "predict_distribution"):  # OpenBoost
        # early_stopping_rounds restores the best iteration internally
        params = model.predict_distribution(X).params
        return (np.asarray(params["loc"], dtype=np.float64),
                np.asarray(params["scale"], dtype=np.float64))
    if hasattr(model, "pred_dist"):  # NGBoost
        best_iter = getattr(model, "best_val_loss_itr", None)
        dist = (model.pred_dist(X, max_iter=best_iter)
                if best_iter is not None else model.pred_dist(X))
        params = dist.params
        return (np.asarray(params["loc"], dtype=np.float64),
                np.asarray(params["scale"], dtype=np.float64))
    # PGBM sklearn wrapper
    mean, std = model.predict(X, return_std=True)
    return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)


# =============================================================================
# Metrics (shared closed forms — identical for every library)
# =============================================================================

def gaussian_nll(y, mean, std):
    import numpy as np

    std = np.clip(std, 1e-12, None)
    return float(np.mean(0.5 * np.log(2 * np.pi * std**2)
                         + (y - mean) ** 2 / (2 * std**2)))


def coverage_90(y, mean, std):
    import numpy as np
    from scipy.stats import norm

    lo = mean + norm.ppf(0.05) * std
    hi = mean + norm.ppf(0.95) * std
    return float(np.mean((y >= lo) & (y <= hi)))


def pinball(y, mean, std, quantiles=(0.05, 0.5, 0.95)):
    import numpy as np
    from scipy.stats import norm

    losses = {}
    for q in quantiles:
        pred_q = mean + norm.ppf(q) * std
        diff = y - pred_q
        losses[str(q)] = float(np.mean(np.maximum(q * diff, (q - 1) * diff)))
    return losses


def score_model(model, X_te, y_te):
    import numpy as np

    import openboost as ob

    mean, std = predict_normal_params(model, X_te)
    return {
        "nll": gaussian_nll(y_te, mean, std),
        "crps": float(ob.crps_gaussian(y_te, mean, std)),
        "rmse": float(np.sqrt(np.mean((y_te - mean) ** 2))),
        "coverage_90": coverage_90(y_te, mean, std),
        "pinball": pinball(y_te, mean, std),
    }


# =============================================================================
# Quality suite
# =============================================================================

def run_quality(quick: bool = False) -> dict:
    import numpy as np
    from scipy.stats import wilcoxon
    from sklearn.model_selection import train_test_split

    import openboost as ob

    datasets, notes = load_quality_datasets(quick=quick)

    # Untimed warmup (JIT compile both sides)
    Xw, yw = make_heteroscedastic(512, n_features=10, seed=0)
    make_openboost().fit(Xw[:256], yw[:256])
    ngb_w = make_ngboost()
    ngb_w.set_params(n_estimators=5)
    ngb_w.fit(Xw[:256], yw[:256])

    results = []
    for ds in datasets:
        X, y = ds["X"], ds["y"]
        n_splits = (2 if quick else
                    N_SPLITS_LARGE if len(y) > LARGE_THRESHOLD else
                    N_SPLITS_SMALL)
        per_split = {"openboost": [], "ngboost": []}
        t_ds = time.perf_counter()
        for split_seed in range(n_splits):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=split_seed)
            # Both libraries early-stop on the SAME validation set.
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=VAL_FRACTION, random_state=split_seed)
            for lib, factory in (("openboost", make_openboost),
                                 ("ngboost", make_ngboost)):
                model = factory()
                t0 = time.perf_counter()
                if lib == "openboost":
                    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                              early_stopping_rounds=ES_PATIENCE)
                else:
                    model.fit(X_fit, y_fit, X_val=X_val, Y_val=y_val,
                              early_stopping_rounds=ES_PATIENCE)
                fit_time = time.perf_counter() - t0
                metrics = score_model(model, X_te, y_te)
                metrics["fit_time_s"] = round(fit_time, 3)
                per_split[lib].append(metrics)

        def agg(lib, key, per_split=per_split):
            vals = [m[key] for m in per_split[lib]]
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        ob_nll = np.array([m["nll"] for m in per_split["openboost"]])
        ngb_nll = np.array([m["nll"] for m in per_split["ngboost"]])
        if n_splits >= 5 and not np.allclose(ob_nll, ngb_nll):
            stat = wilcoxon(ob_nll, ngb_nll)
            p_value = float(stat.pvalue)
        else:
            p_value = None

        row = {
            "dataset": ds["name"],
            "source": ds["source"],
            "n": int(len(y)),
            "n_features": int(X.shape[1]),
            "n_splits": n_splits,
            "openboost": {k: agg("openboost", k)
                          for k in ("nll", "crps", "rmse", "coverage_90",
                                    "fit_time_s")},
            "ngboost": {k: agg("ngboost", k)
                        for k in ("nll", "crps", "rmse", "coverage_90",
                                  "fit_time_s")},
            "nll_delta_mean": float(np.mean(ob_nll - ngb_nll)),
            "nll_paired_wilcoxon_p": p_value,
            "per_split": per_split,
        }
        results.append(row)
        print(f"{ds['name']:<12} splits={n_splits}  "
              f"OB NLL {row['openboost']['nll']['mean']:.4f}  "
              f"NGB NLL {row['ngboost']['nll']['mean']:.4f}  "
              f"delta {row['nll_delta_mean']:+.4f}  p={p_value}  "
              f"OB cov90 {row['openboost']['coverage_90']['mean']:.3f}  "
              f"({time.perf_counter() - t_ds:.0f}s)")

    return {"suite": "quality", "results": results, "skipped": notes,
            "backend": ob.get_backend()}


# =============================================================================
# Speed suite
# =============================================================================

def _time_to_nll(model_name, nll_curve_fn, target_nll):
    """Placeholder hook: time-to-same-NLL requires per-round eval; wired in
    once the unified trainer exposes cheap incremental eval on GPU."""
    return None


def run_speed(quick: bool = False, use_gpu: bool = False) -> dict:
    import openboost as ob

    # 1M hits the ≥1M acceptance gate. NGBoost exact-split trees took
    # ~24 min at 45K; skip it past 100K. OpenBoost still times 500K/1M.
    sizes = [50_000] if quick else [100_000, 500_000, 1_000_000]
    sync = use_gpu

    # Warmup (few trees — JIT only)
    Xw, yw = make_heteroscedastic(512, n_features=10, seed=0)
    for _ in range(2):
        ob.NaturalBoostNormal(
            n_trees=5, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
            n_bins=254,
        ).fit(Xw, yw)
    if sync:
        from numba import cuda
        cuda.synchronize()

    have_ngboost = True
    try:
        import ngboost  # noqa: F401
    except ImportError:
        have_ngboost = False
    have_pgbm = True
    try:
        import pgbm  # noqa: F401
    except ImportError:
        have_pgbm = False

    results = []
    for n in sizes:
        X, y = make_heteroscedastic(n)
        split = int(0.9 * n)
        X_tr, y_tr = X[:split], y[:split]
        X_te, y_te = X[split:], y[split:]
        row = {"n_train": split, "n_features": int(X.shape[1])}
        print(f"size n={n:,} train={split:,} backend={ob.get_backend()}",
              flush=True)

        # OpenBoost NaturalBoost
        times = []
        n_trials = 1 if n >= 500_000 else 3
        for trial in range(n_trials):
            model = make_openboost()
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            if sync:
                from numba import cuda
                cuda.synchronize()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"  openboost trial {trial + 1}/{n_trials}: {elapsed:.1f}s",
                  flush=True)
        times.sort()
        row["openboost"] = {"fit_time_s": round(times[len(times) // 2], 2),
                            **score_model(model, X_te, y_te)}

        # NGBoost (CPU only) — skip past 100K: 45K already took ~24 min.
        if have_ngboost and (quick or n <= 100_000):
            print(f"  ngboost starting (n={split:,})...", flush=True)
            model = make_ngboost()
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            row["ngboost"] = {"fit_time_s": round(time.perf_counter() - t0, 2),
                              **score_model(model, X_te, y_te)}
            row["speedup_vs_ngboost"] = round(
                row["ngboost"]["fit_time_s"] / row["openboost"]["fit_time_s"], 2)

        # PGBM (torch; GPU-capable) — optional
        if have_pgbm:
            try:
                model = make_pgbm()
                t0 = time.perf_counter()
                model.fit(X_tr, y_tr)
                row["pgbm"] = {"fit_time_s": round(time.perf_counter() - t0, 2),
                               **score_model(model, X_te, y_te)}
                row["speedup_vs_pgbm"] = round(
                    row["pgbm"]["fit_time_s"] / row["openboost"]["fit_time_s"], 2)
            except Exception as exc:
                row["pgbm"] = {"error": f"{type(exc).__name__}: {exc}"}

        results.append(row)
        parts = [f"n={split:,}",
                 f"OB {row['openboost']['fit_time_s']}s "
                 f"NLL {row['openboost']['nll']:.4f}"]
        if "ngboost" in row:
            parts.append(f"NGB {row['ngboost']['fit_time_s']}s "
                         f"({row['speedup_vs_ngboost']}x)")
        if isinstance(row.get("pgbm"), dict) and "fit_time_s" in row["pgbm"]:
            parts.append(f"PGBM {row['pgbm']['fit_time_s']}s "
                         f"({row['speedup_vs_pgbm']}x)")
        print("  ".join(parts), flush=True)

    return {"suite": "speed", "results": results,
            "backend": ob.get_backend(),
            "ngboost_available": have_ngboost,
            "pgbm_available": have_pgbm,
            "ngboost_max_n": 100_000}


# =============================================================================
# Runner / report
# =============================================================================

def run_suites(suite: str, quick: bool, use_gpu: bool) -> dict:
    import numpy as np

    import openboost as ob

    report = {
        "benchmark": "bench_probabilistic",
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "quick_mode": quick,
        "use_gpu": use_gpu,
        "budget": {"n_trees": N_TREES, "learning_rate": LEARNING_RATE,
                   "max_depth": MAX_DEPTH},
        "platform": {
            "python": platform.python_version(),
            "system": f"{platform.system()} {platform.machine()}",
            "cpu_count": os.cpu_count(),
        },
        "versions": {"openboost": ob.__version__, "numpy": np.__version__},
        "suites": {},
    }
    try:
        import ngboost
        report["versions"]["ngboost"] = ngboost.__version__
    except ImportError:
        pass

    if suite in ("quality", "all"):
        report["suites"]["quality"] = run_quality(quick=quick)
    if suite in ("speed", "all"):
        report["suites"]["speed"] = run_speed(quick=quick, use_gpu=use_gpu)
    return report


def save_report(report: dict, suite: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"probabilistic_{suite}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out}")
    return out


# =============================================================================
# Modal entry points
# =============================================================================

try:
    import modal

    app = modal.App("openboost-probabilistic-bench")
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
        .pip_install(
            "numpy>=1.24,<2.5", "numba>=0.60", "numba-cuda>=0.23",
            "scipy>=1.10", "scikit-learn>=1.0", "ngboost>=0.5",
            "pandas>=2.0", "joblib>=1.2",
        )
        .add_local_dir(
            str(PROJECT_ROOT / "src" / "openboost"),
            remote_path="/root/openboost",
            copy=True,
        )
        .env({"PYTHONPATH": "/root", "OPENBOOST_BACKEND": "cuda"})
    )
except ImportError:
    modal = None
    app = None
    image = None

if modal is not None and app is not None:

    @app.function(gpu="A100", image=image, timeout=4 * 3600)
    def _run_remote(suite: str = "all", quick: bool = False):
        sys.path.insert(0, "/root")
        import openboost as ob

        ob.set_backend("cuda")
        print(f"backend={ob.get_backend()} suite={suite} quick={quick}",
              flush=True)
        return run_suites(suite=suite, quick=quick, use_gpu=True)

    @app.function(image=image, timeout=4 * 3600)
    def _run_quality_remote(quick: bool = False):
        # Quality suite is CPU-only (NGBoost is CPU); run it off-GPU. Modal's
        # datacenter network to OpenML is reliable, unlike some local networks.
        sys.path.insert(0, "/root")
        import openboost as ob

        ob.set_backend("cpu")
        print(f"backend={ob.get_backend()} suite=quality quick={quick}",
              flush=True)
        return run_suites(suite="quality", quick=quick, use_gpu=False)

    @app.local_entrypoint()
    def main(suite: str = "all", quick: bool = False):
        report = _run_remote.remote(suite=suite, quick=quick)
        save_report(report, suite)

    @app.local_entrypoint()
    def quality(quick: bool = False):
        report = _run_quality_remote.remote(quick=quick)
        save_report(report, "quality")
        q = report["suites"].get("quality", {})
        print(f"\nquality: {len(q.get('results', []))} datasets, "
              f"{len(q.get('skipped', []))} skipped")


# =============================================================================
# Local execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["quality", "speed", "all"],
                        default="quality")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke-test sizes/splits")
    args = parser.parse_args()

    os.environ.setdefault("OPENBOOST_BACKEND", "cpu")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    report = run_suites(suite=args.suite, quick=args.quick, use_gpu=False)
    save_report(report, args.suite)
