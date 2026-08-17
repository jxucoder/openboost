"""Survival capability benchmark: Weibull AFT with covariate-dependent shape.

The second capability proof for "Boost every parameter". Data is Weibull AFT
where BOTH the scale lambda(z) AND the shape k(z) depend on covariates:

    T ~ Weibull(scale=lambda(z), shape=k(z)),  right-censored at ~35%.

Baselines:

  global    constant (lambda, k) MLE (no covariate dependence).
  xgb-aft   XGBoost's built-in `survival:aft`. It learns only the location;
            the distribution scale `aft_loss_distribution_scale` is a single
            GLOBAL hyperparameter, so the Weibull shape k = 1/sigma is the SAME
            for every row. It structurally cannot vary shape with covariates.
            We set its global sigma to the MLE 1/k_global (its best setting).

OpenBoost WeibullAFT boosts both lambda(z) and k(z) from the censored NLL, so
it is the only model that recovers the true shape surface. NGBoost has no
censored likelihood at all, so it cannot enter this benchmark.

Metrics (identical closed forms for both libraries via the implied Weibull):
  C-index (Harrell), censored NLL, 80% interval coverage, shape recovery
  (corr of predicted vs true log k).

Usage:
    uv run --with xgboost python benchmarks/bench_survival.py
    uv run --with xgboost python benchmarks/bench_survival.py --quick
    uv run modal run benchmarks/bench_survival.py            # A100

Output: benchmarks/results/survival_<YYYYMMDD_HHMMSS>.json
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

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"

SEED = 42
N_FEATURES = 6
ROUNDS = 300
LR = 0.1
DEPTH = 3
DAMP = 1.0
CENSOR_Q = 0.65          # exponential censoring quantile -> ~35% censored
CINDEX_CAP = 4000        # subsample cap for the O(n^2) concordance

# Gates
GATE_CINDEX_SLACK = 0.005   # OpenBoost C-index >= xgb - this
GATE_SHAPE_CORR = 0.5       # OpenBoost must recover the shape surface


# =============================================================================
# Data generating process: varying scale AND shape
# =============================================================================

def make_data(n, rng):
    Z = rng.uniform(0, 1, (n, N_FEATURES))
    lam = np.exp(0.5 + 1.0 * Z[:, 0] - 0.8 * Z[:, 1] + 0.4 * Z[:, 4])
    k = np.exp(0.2 + 1.2 * Z[:, 2] - 0.6 * Z[:, 3])   # shape varies with z
    u = rng.uniform(1e-9, 1.0, n)
    T = lam * (-np.log(u)) ** (1.0 / k)
    C = rng.exponential(np.quantile(T, CENSOR_Q), n)
    t = np.minimum(T, C)
    event = (T <= C).astype(np.float64)
    return Z, t, event, lam, k, T


# =============================================================================
# Metrics (shared closed forms via the implied Weibull(lambda, k))
# =============================================================================

def weibull_nll(t, event, lam, k):
    lam = np.clip(lam, 1e-9, None)
    k = np.clip(k, 1e-6, None)
    m = np.log(t) - np.log(lam)
    z = np.exp(np.clip(k * m, -60, 60))
    nll = -event * (np.log(k) - np.log(lam) + (k - 1.0) * m) + z
    return float(np.mean(nll))


def weibull_quantile(lam, k, q):
    return lam * (-np.log(1.0 - q)) ** (1.0 / k)


def interval_coverage(true_T, lam, k, lo=0.1, hi=0.9):
    """Coverage of the true (uncensored) event time by the [lo, hi] interval."""
    q_lo = weibull_quantile(lam, k, lo)
    q_hi = weibull_quantile(lam, k, hi)
    return float(np.mean((true_T >= q_lo) & (true_T <= q_hi)))


def c_index(time, event, risk, seed=SEED, cap=CINDEX_CAP):
    """Harrell's concordance. Higher risk should mean earlier event."""
    n = len(time)
    if n > cap:
        idx = np.random.default_rng(seed).choice(n, cap, replace=False)
        time, event, risk = time[idx], event[idx], risk[idx]
    ti = time[:, None]
    tj = time[None, :]
    ei = event[:, None]
    # comparable: i had an event and died strictly before j
    comparable = (ti < tj) & (ei == 1)
    ri = risk[:, None]
    rj = risk[None, :]
    concordant = (ri > rj) & comparable
    tied = (ri == rj) & comparable
    denom = comparable.sum()
    if denom == 0:
        return float("nan")
    return float((concordant.sum() + 0.5 * tied.sum()) / denom)


# =============================================================================
# Models
# =============================================================================

def run_openboost(Ztr, ttr, evtr, sets, rounds):
    import openboost as ob

    m = ob.WeibullAFT(n_trees=rounds, max_depth=DEPTH, learning_rate=LR, damp=DAMP)
    t0 = time.perf_counter()
    m.fit(Ztr, ttr, event=evtr)
    fit_time = time.perf_counter() - t0

    out = {"fit_time_s": round(fit_time, 2)}
    for name, (Z, t, ev, true_T, _lam_true, k_true) in sets.items():
        p = m.predict_params(Z)
        lam, k = p["scale"], p["shape"]
        med = weibull_quantile(lam, k, 0.5)
        out[name] = {
            "c_index": c_index(t, ev, -med),
            "nll": weibull_nll(t, ev, lam, k),
            "coverage_80": interval_coverage(true_T, lam, k),
            "shape_corr": float(np.corrcoef(np.log(k), np.log(k_true))[0, 1]),
        }
    return out


def run_global(Ztr, ttr, evtr, sets):
    from openboost._objectives import WeibullAFTObjective

    obj = WeibullAFTObjective()
    init = obj.init_raw(ttr, None, {"event": evtr})
    lam_g, k_g = float(np.exp(init["scale"])), float(np.exp(init["shape"]))
    out = {"lambda": lam_g, "shape": k_g}
    for name, (_Z, t, ev, true_T, _lam, _k) in sets.items():
        lam = np.full(len(t), lam_g)
        k = np.full(len(t), k_g)
        med = weibull_quantile(lam, k, 0.5)
        out[name] = {
            "c_index": c_index(t, ev, -med),
            "nll": weibull_nll(t, ev, lam, k),
            "coverage_80": interval_coverage(true_T, lam, k),
            "shape_corr": None,  # constant shape: not defined
        }
    return out, k_g


def run_xgb_aft(Ztr, ttr, evtr, sets, rounds, k_global):
    """XGBoost survival:aft with a single global scale sigma = 1/k_global."""
    try:
        import xgboost as xgb
    except ImportError as exc:
        return {"error": f"xgboost unavailable: {exc}"}

    try:
        sigma = float(1.0 / max(k_global, 1e-6))
        dtrain = xgb.DMatrix(np.ascontiguousarray(Ztr))
        upper = np.where(evtr == 1, ttr, np.inf)
        dtrain.set_float_info("label_lower_bound", ttr)
        dtrain.set_float_info("label_upper_bound", upper)
        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            # 'extreme' value log-time distribution == Weibull survival time.
            "aft_loss_distribution": "extreme",
            "aft_loss_distribution_scale": sigma,   # GLOBAL shape, not per-row
            "tree_method": "hist",
            "max_depth": DEPTH,
            "eta": LR,
            "lambda": DAMP,
            "seed": SEED,
        }
        t0 = time.perf_counter()
        bst = xgb.train(params, dtrain, num_boost_round=rounds)
        fit_time = time.perf_counter() - t0

        k_const = 1.0 / sigma
        out = {"fit_time_s": round(fit_time, 2), "global_shape": k_const,
               "note": "single global scale; shape identical for all rows"}
        for name, (Z, t, ev, true_T, _lam, _k_true) in sets.items():
            lam = np.asarray(bst.predict(xgb.DMatrix(np.ascontiguousarray(Z))),
                             dtype=np.float64)   # AFT location = Weibull scale
            k = np.full(len(t), k_const)
            med = weibull_quantile(lam, k, 0.5)
            out[name] = {
                "c_index": c_index(t, ev, -med),
                "nll": weibull_nll(t, ev, lam, k),
                "coverage_80": interval_coverage(true_T, lam, k),
                "shape_corr": None,  # constant shape by construction
            }
        return out
    except Exception as exc:  # noqa: BLE001 - baseline must not crash the suite
        import traceback
        return {"error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


# =============================================================================
# Gates
# =============================================================================

def compute_gates(res):
    ob = res["openboost"]["test"]
    glob = res["global"]["test"]
    xgb = res.get("xgb", {})
    gates = {
        "openboost_shape_corr": round(ob["shape_corr"], 3),
        "recovers_shape_pass": ob["shape_corr"] >= GATE_SHAPE_CORR,
        "beats_global_cindex_pass": ob["c_index"] > glob["c_index"],
        "beats_global_nll_pass": ob["nll"] < glob["nll"],
    }
    if xgb and "error" not in xgb:
        xt = xgb["test"]
        gates["cindex_vs_xgb_delta"] = round(ob["c_index"] - xt["c_index"], 4)
        gates["nll_vs_xgb_delta"] = round(ob["nll"] - xt["nll"], 4)
        gates["cindex_not_worse_than_xgb_pass"] = (
            ob["c_index"] >= xt["c_index"] - GATE_CINDEX_SLACK)
        gates["nll_better_than_xgb_pass"] = ob["nll"] < xt["nll"]
    else:
        gates["cindex_not_worse_than_xgb_pass"] = None
        gates["nll_better_than_xgb_pass"] = None

    hard = [
        gates["recovers_shape_pass"],
        gates["beats_global_cindex_pass"],
        gates["beats_global_nll_pass"],
    ]
    if gates["cindex_not_worse_than_xgb_pass"] is not None:
        hard.append(gates["cindex_not_worse_than_xgb_pass"])
        hard.append(gates["nll_better_than_xgb_pass"])
    gates["passed"] = all(hard)
    return gates


def _pf(v):
    return "skip" if v is None else ("PASS" if v else "FAIL")


# =============================================================================
# Runner
# =============================================================================

def run_size(n_train, n_test, rounds):
    import openboost as ob

    rng = np.random.default_rng(SEED)
    Ztr, ttr, evtr, *_ = make_data(n_train, rng)
    Zte, tte, evte, lam_te, k_te, T_te = make_data(n_test, rng)
    sets = {"test": (Zte, tte, evte, T_te, lam_te, k_te)}

    print(f"\n=== n_train={n_train:,}  rounds={rounds}  "
          f"backend={ob.get_backend()}  censor={1 - evtr.mean():.2f} ===",
          flush=True)

    res = {"n_train": n_train, "n_test": n_test, "rounds": rounds}
    res["openboost"] = run_openboost(Ztr, ttr, evtr, sets, rounds)
    o = res["openboost"]["test"]
    print(f"  OpenBoost WeibullAFT  C-index {o['c_index']:.4f}  "
          f"NLL {o['nll']:.4f}  cov80 {o['coverage_80']:.3f}  "
          f"shape_corr {o['shape_corr']:.3f}  ({res['openboost']['fit_time_s']}s)",
          flush=True)

    res["global"], k_global = run_global(Ztr, ttr, evtr, sets)
    g = res["global"]["test"]
    print(f"  global                C-index {g['c_index']:.4f}  "
          f"NLL {g['nll']:.4f}  cov80 {g['coverage_80']:.3f}  shape_corr n/a",
          flush=True)

    res["xgb"] = run_xgb_aft(Ztr, ttr, evtr, sets, rounds, k_global)
    if "error" in res["xgb"]:
        print(f"  xgb survival:aft      SKIPPED ({res['xgb']['error']})", flush=True)
    else:
        x = res["xgb"]["test"]
        print(f"  xgb survival:aft      C-index {x['c_index']:.4f}  "
              f"NLL {x['nll']:.4f}  cov80 {x['coverage_80']:.3f}  "
              f"shape_corr n/a (global k={res['xgb']['global_shape']:.2f})  "
              f"({res['xgb']['fit_time_s']}s)", flush=True)

    res["gates"] = compute_gates(res)
    gt = res["gates"]
    print(f"  GATES  recovers_shape({gt['openboost_shape_corr']})="
          f"{_pf(gt['recovers_shape_pass'])}  "
          f"NLL<xgb={_pf(gt['nll_better_than_xgb_pass'])}  "
          f"C-index>=xgb={_pf(gt['cindex_not_worse_than_xgb_pass'])}  "
          f"beats_global={_pf(gt['beats_global_nll_pass'])}  "
          f"=> {'PASS' if gt['passed'] else 'FAIL'}", flush=True)
    return res


def run_suite(quick=False):
    import openboost as ob

    sizes = ([(8_000, 4_000, 80)] if quick
             else [(40_000, 10_000, ROUNDS), (200_000, 20_000, ROUNDS)])

    # Warmup (JIT the tree kernels off the clock).
    rng = np.random.default_rng(0)
    Zw, tw, evw, *_ = make_data(512, rng)
    ob.WeibullAFT(n_trees=5, max_depth=DEPTH).fit(Zw, tw, event=evw)

    report = {
        "benchmark": "bench_survival",
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "quick_mode": quick,
        "model": "Weibull AFT, scale(z) and shape(z) both boosted",
        "budget": {"rounds": ROUNDS, "lr": LR, "depth": DEPTH, "damp": DAMP,
                   "censor_q": CENSOR_Q},
        "platform": {
            "python": platform.python_version(),
            "system": f"{platform.system()} {platform.machine()}",
        },
        "backend": ob.get_backend(),
        "versions": {"openboost": ob.__version__, "numpy": np.__version__},
        "sizes": [],
    }
    try:
        import xgboost
        report["versions"]["xgboost"] = xgboost.__version__
    except ImportError:
        report["versions"]["xgboost"] = None

    for n_train, n_test, rounds in sizes:
        report["sizes"].append(run_size(n_train, n_test, rounds))

    report["passed"] = all(s["gates"]["passed"] for s in report["sizes"])
    return report


def save_report(report):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"survival_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out}")
    return out


# =============================================================================
# Modal entry point
# =============================================================================

try:
    import modal

    app = modal.App("openboost-survival-bench")
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
        .pip_install(
            "numpy>=1.24,<2.5", "numba>=0.60", "numba-cuda>=0.23",
            "scipy>=1.10", "scikit-learn>=1.0", "xgboost>=2.0",
            "joblib>=1.2",
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

    @app.function(gpu="A100", image=image, timeout=2 * 3600)
    def _run_remote(quick: bool = False):
        sys.path.insert(0, "/root")
        import openboost as ob

        ob.set_backend("cuda")
        print(f"backend={ob.get_backend()} quick={quick}", flush=True)
        return run_suite(quick=quick)

    @app.local_entrypoint()
    def main(quick: bool = False):
        report = _run_remote.remote(quick=quick)
        save_report(report)
        print(f"\nOverall: {'PASS' if report['passed'] else 'FAIL'}")


# =============================================================================
# Local execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke-test sizes/rounds")
    args = parser.parse_args()

    os.environ.setdefault("OPENBOOST_BACKEND", "cpu")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    report = run_suite(quick=args.quick)
    save_report(report)
    print(f"\nOverall: {'PASS' if report['passed'] else 'FAIL'}")
