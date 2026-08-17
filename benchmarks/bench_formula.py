"""Formula capability benchmark: the yardstick for "Boost every parameter".

FormulaBoost fits a user formula ``y ~ f(theta(z), x)`` where each parameter
surface ``theta_k(z)`` is its own boosting ensemble and ``x`` enters only
through the formula. This suite proves the claim NGBoost and black-box GBDT
cannot express, on the sales/saturation curve from the GGN spike:

    sales = a(z) * x ** sigmoid(b(z) * x)

Baselines (what a practitioner would actually reach for):

  global    one (a, b) fit to all rows (formula, but no theta(z)).
  blackbox  GradientBoosting on [z, x] -> y (no formula; trees flatten
            outside the training x-range, so extrapolation collapses).
  xgb       hand-rolled XGBoost multi-output custom objective with the SAME
            FD Jacobian as FormulaBoost, but a DIAGONAL Hessian. XGBoost's
            custom-objective API only accepts a diagonal Hessian ("the Hessian
            for each row should be diagonal", XGBoost docs), so the off-diagonal
            GGN term is inexpressible. This baseline is exactly FormulaBoost's
            ``precond='diag'`` on xgb trees.

FormulaBoost is run in all three preconditioning modes (plain / diag / full);
``full`` is the only one that uses the off-diagonal GGN coupling.

Metrics: in-range test RMSE, extrapolation RMSE (x beyond training range, vs
the true noiseless curve), parameter recovery (corr + RMSE of a_hat, b_hat),
fit time.

Usage:
    # Local CPU
    uv run --with xgboost python benchmarks/bench_formula.py
    uv run --with xgboost python benchmarks/bench_formula.py --quick

    # Modal A100 (proves the formula path runs on GPU-built trees)
    uv run modal run benchmarks/bench_formula.py
    uv run modal run benchmarks/bench_formula.py --quick

Output: benchmarks/results/formula_<YYYYMMDD_HHMMSS>.json
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
BLACKBOX_DEPTH = 6
X_TRAIN_RANGE = (0.25, 2.5)
X_EXTRAP_RANGE = (3.0, 5.0)
NOISE_FRAC = 0.05

# Acceptance gates (see planning/formula_capability_next plan).
GATE_EXTRAP_VS_BLACKBOX = 5.0   # full must beat black-box extrap by >= this
GATE_XGB_EXTRAP_SLACK = 1.15    # full extrap must be within this factor of xgb
GATE_XGB_CORRB_SLACK = 0.02     # full b-recovery must be >= xgb - this


# =============================================================================
# Data generating process (ported from development/paramboost/spike_ggn.py)
# =============================================================================

def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-np.clip(t, -30.0, 30.0)))


def sales_curve(theta, x):
    """formula(theta, x) = a * x ** sigmoid(b*x), stable in log space."""
    a, b = theta
    return np.exp(np.log(np.clip(a, 1e-12, None)) + sigmoid(b * x) * np.log(x))


def true_params(Z):
    u = 0.5 + 1.0 * Z[:, 0] - 0.8 * Z[:, 1] + 0.5 * np.sin(2 * np.pi * Z[:, 2])
    a = np.exp(u)
    b = 0.5 + 2.0 * Z[:, 3] + Z[:, 4] * Z[:, 0]
    return a, b


def make_data(n, rng, x_range):
    Z = rng.uniform(0, 1, (n, N_FEATURES))
    x = rng.uniform(*x_range, n)
    a, b = true_params(Z)
    f = sales_curve((a, b), x)
    y = f + NOISE_FRAC * f.std() * rng.standard_normal(n)
    return Z, x, y, a, b, f


# =============================================================================
# Metrics
# =============================================================================

def rmse(pred, target):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(target)) ** 2)))


def _corr(pred, target):
    """Pearson corr, or None when either side is constant (e.g. a global fit)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.std() < 1e-12 or target.std() < 1e-12:
        return None
    return float(np.corrcoef(pred, target)[0, 1])


def param_recovery(a_hat, b_hat, a_true, b_true):
    return {
        "rmse_a": rmse(a_hat, a_true),
        "rmse_b": rmse(b_hat, b_true),
        "corr_a": _corr(a_hat, a_true),
        "corr_b": _corr(b_hat, b_true),
    }


# =============================================================================
# Shared formula math (identical FD Jacobian to FormulaObjective, so the only
# difference between the xgb baseline and FormulaBoost is diagonal-vs-full
# Hessian and xgb-vs-openboost trees).
# =============================================================================

def _raw_to_theta(raw):
    """raw (n,2) -> (a, b): log link on a, identity on b."""
    return np.exp(np.clip(raw[:, 0], -30.0, 30.0)), raw[:, 1]


def _fd_jac_raw(raw, x, eps=1e-5):
    """Forward-difference Jacobian df/d(raw), shape (n,2)."""
    a, b = _raw_to_theta(raw)
    f0 = sales_curve((a, b), x)
    # bump raw_a
    a1, _ = _raw_to_theta(np.column_stack([raw[:, 0] + eps, raw[:, 1]]))
    f_a = sales_curve((a1, b), x)
    # bump raw_b
    _, b1 = _raw_to_theta(np.column_stack([raw[:, 0], raw[:, 1] + eps]))
    f_b = sales_curve((a, b1), x)
    jac = np.column_stack([(f_a - f0) / eps, (f_b - f0) / eps])
    return f0, jac


def fit_global_raw(x, y):
    """Single global raw (u, b) via L-BFGS-B (matches FormulaObjective init)."""
    from scipy.optimize import minimize

    v0 = np.zeros(2, dtype=np.float64)
    if np.all(y > 0):
        v0[0] = float(np.log(np.mean(y)))

    def loss(v):
        a = np.exp(np.clip(v[0], -30.0, 30.0))
        pred = sales_curve((np.full_like(x, a), np.full_like(x, v[1])), x)
        if not np.all(np.isfinite(pred)):
            return 1e12
        return float(np.mean(0.5 * (pred - y) ** 2))

    res = minimize(loss, v0, method="L-BFGS-B")
    return res.x if res.success else v0


# =============================================================================
# Models
# =============================================================================

def run_formulaboost(precond, Z_tr, x_tr, y_tr, sets, rounds):
    import openboost as ob

    model = ob.FormulaBoost(
        formula=sales_curve,
        n_params=2,
        links=("log", "identity"),
        param_names=("a", "b"),
        precond=precond,
        damp=DAMP,
        n_trees=rounds,
        max_depth=DEPTH,
        learning_rate=LR,
    )
    t0 = time.perf_counter()
    model.fit(Z_tr, y_tr, model_input=x_tr)
    fit_time = time.perf_counter() - t0

    out = {"fit_time_s": round(fit_time, 2)}
    for name, (Z, x, target, a_true, b_true) in sets.items():
        params = model.predict_params(Z)
        pred = model.predict(Z, model_input=x)
        entry = {"rmse": rmse(pred, target)}
        entry.update(param_recovery(params["a"], params["b"], a_true, b_true))
        out[name] = entry
    return out


def run_global(Z_tr, x_tr, y_tr, sets):
    v = fit_global_raw(x_tr, y_tr)
    a_g, b_g = float(np.exp(v[0])), float(v[1])
    out = {"a": a_g, "b": b_g}
    for name, (_Z, x, target, a_true, b_true) in sets.items():
        pred = sales_curve((np.full_like(x, a_g), np.full_like(x, b_g)), x)
        entry = {"rmse": rmse(pred, target)}
        # a global constant "recovers" nothing; report corr for completeness.
        entry.update(param_recovery(
            np.full_like(a_true, a_g), np.full_like(b_true, b_g),
            a_true, b_true))
        out[name] = entry
    return out


def run_blackbox(Z_tr, x_tr, y_tr, sets, rounds):
    import openboost as ob

    model = ob.GradientBoosting(
        n_trees=rounds, max_depth=BLACKBOX_DEPTH, learning_rate=LR,
        random_state=SEED,
    )
    t0 = time.perf_counter()
    model.fit(np.column_stack([Z_tr, x_tr]), y_tr)
    fit_time = time.perf_counter() - t0

    out = {"fit_time_s": round(fit_time, 2)}
    for name, (Z, x, target, _a, _b) in sets.items():
        pred = model.predict(np.column_stack([Z, x]))
        out[name] = {"rmse": rmse(pred, target)}  # no theta(z) to recover
    return out


def run_xgb_custom(Z_tr, x_tr, y_tr, sets, rounds):
    """Hand-rolled XGBoost multi-output custom objective (diagonal Hessian).

    Same FD Jacobian and global init as FormulaBoost; the only difference is
    that XGBoost's custom-objective API accepts a diagonal Hessian only, so
    this is precond='diag' on xgb trees. Returns {"error": ...} if the
    installed xgboost lacks multi-output custom objectives.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        return {"error": f"xgboost unavailable: {exc}"}

    try:
        v0 = fit_global_raw(x_tr, y_tr)
        n = len(y_tr)
        base_tr = np.tile(v0.astype(np.float64), (n, 1))

        def obj(pred, dtrain):
            pred = np.asarray(pred, dtype=np.float64).reshape(n, 2)
            f0, jac = _fd_jac_raw(pred, x_tr)
            residual = f0 - y_tr
            grad = residual[:, None] * jac
            hess = np.maximum(jac * jac, 1e-6)  # diagonal only
            return grad.astype(np.float32), hess.astype(np.float32)

        dtrain = xgb.DMatrix(
            np.ascontiguousarray(Z_tr),
            label=np.zeros((n, 2)),          # unused: obj closes over y, x
            base_margin=base_tr,
        )
        params = {
            "tree_method": "hist",
            "num_target": 2,
            "base_score": 0.0,
            "disable_default_eval_metric": True,
            "max_depth": DEPTH,
            "eta": LR,
            "lambda": DAMP,
            "seed": SEED,
        }
        t0 = time.perf_counter()
        booster = xgb.train(params, dtrain, num_boost_round=rounds, obj=obj)
        fit_time = time.perf_counter() - t0

        out = {"fit_time_s": round(fit_time, 2), "note": "diagonal Hessian only"}
        for name, (Z, x, target, a_true, b_true) in sets.items():
            m = xgb.DMatrix(
                np.ascontiguousarray(Z),
                base_margin=np.tile(v0.astype(np.float64), (len(x), 1)),
            )
            raw = np.asarray(booster.predict(m), dtype=np.float64).reshape(-1, 2)
            a_hat, b_hat = _raw_to_theta(raw)
            pred = sales_curve((a_hat, b_hat), x)
            entry = {"rmse": rmse(pred, target)}
            entry.update(param_recovery(a_hat, b_hat, a_true, b_true))
            out[name] = entry
        return out
    except Exception as exc:  # noqa: BLE001 - baseline must never crash the suite
        import traceback
        return {"error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


# =============================================================================
# Gates
# =============================================================================

def compute_gates(res):
    """Boolean acceptance gates from one size's results."""
    full = res["formulaboost"]["full"]
    plain = res["formulaboost"]["plain"]
    blackbox = res["blackbox"]
    glob = res["global"]
    xgb = res.get("xgb", {})

    gates = {}

    # 1. formula constrains the x-shape: extrapolation crushes black-box.
    ratio_bb = blackbox["extrap"]["rmse"] / max(full["extrap"]["rmse"], 1e-12)
    gates["extrap_vs_blackbox_ratio"] = round(ratio_bb, 2)
    gates["extrap_vs_blackbox_pass"] = ratio_bb >= GATE_EXTRAP_VS_BLACKBOX

    # 2. off-diagonal GGN pays: full beats plain and recovers b better.
    gates["full_vs_plain_test_ratio"] = round(
        plain["test"]["rmse"] / max(full["test"]["rmse"], 1e-12), 2)
    gates["full_corr_b"] = round(full["test"]["corr_b"], 3)
    gates["plain_corr_b"] = round(plain["test"]["corr_b"], 3)
    gates["full_beats_plain_pass"] = (
        full["test"]["rmse"] < plain["test"]["rmse"]
        and full["test"]["corr_b"] > plain["test"]["corr_b"]
    )

    # 3. theta(z) is necessary: full beats the global constant curve.
    gates["full_beats_global_pass"] = (
        full["test"]["rmse"] < glob["test"]["rmse"]
        and full["extrap"]["rmse"] < glob["extrap"]["rmse"]
    )

    # 4. not worse than the hand-rolled XGBoost diagonal baseline.
    if xgb and "error" not in xgb:
        extrap_ok = (full["extrap"]["rmse"]
                     <= xgb["extrap"]["rmse"] * GATE_XGB_EXTRAP_SLACK)
        corrb_ok = (full["test"]["corr_b"]
                    >= xgb["test"]["corr_b"] - GATE_XGB_CORRB_SLACK)
        gates["full_vs_xgb_extrap_ratio"] = round(
            xgb["extrap"]["rmse"] / max(full["extrap"]["rmse"], 1e-12), 2)
        gates["full_corr_b_minus_xgb"] = round(
            full["test"]["corr_b"] - xgb["test"]["corr_b"], 3)
        gates["full_not_worse_than_xgb_pass"] = bool(extrap_ok and corrb_ok)
    else:
        gates["full_not_worse_than_xgb_pass"] = None  # skipped (xgb missing)

    hard = [
        gates["extrap_vs_blackbox_pass"],
        gates["full_beats_plain_pass"],
        gates["full_beats_global_pass"],
    ]
    gates["passed"] = all(hard)
    return gates


# =============================================================================
# Runner
# =============================================================================

def run_size(n_train, n_test, rounds):
    import openboost as ob

    rng = np.random.default_rng(SEED)
    Z_tr, x_tr, y_tr, *_ = make_data(n_train, rng, X_TRAIN_RANGE)
    Z_te, x_te, y_te, a_te, b_te, _ = make_data(n_test, rng, X_TRAIN_RANGE)
    # Extrapolation: same z, x beyond the training range; target is the TRUE
    # noiseless curve (did the model learn the functional form, not the noise?).
    Z_ex, x_ex, _, a_ex, b_ex, f_ex = make_data(n_test, rng, X_EXTRAP_RANGE)

    sets = {
        "test": (Z_te, x_te, y_te, a_te, b_te),
        "extrap": (Z_ex, x_ex, f_ex, a_ex, b_ex),
    }

    print(f"\n=== n_train={n_train:,}  rounds={rounds}  "
          f"backend={ob.get_backend()} ===", flush=True)

    res = {"n_train": n_train, "n_test": n_test, "rounds": rounds,
           "noise_floor_test_rmse": None, "formulaboost": {}}

    _, _, _, _, _, f_te = make_data(n_test, np.random.default_rng(SEED + 1),
                                    X_TRAIN_RANGE)
    res["noise_floor_test_rmse"] = round(float(NOISE_FRAC * f_te.std()), 4)

    for precond in ("plain", "diag", "full"):
        out = run_formulaboost(precond, Z_tr, x_tr, y_tr, sets, rounds)
        res["formulaboost"][precond] = out
        print(f"  FormulaBoost[{precond:>5}]  test {out['test']['rmse']:.4f}  "
              f"extrap {out['extrap']['rmse']:.4f}  "
              f"corr_b {out['test']['corr_b']:.3f}  ({out['fit_time_s']}s)",
              flush=True)

    res["global"] = run_global(Z_tr, x_tr, y_tr, sets)
    print(f"  global            test {res['global']['test']['rmse']:.4f}  "
          f"extrap {res['global']['extrap']['rmse']:.4f}", flush=True)

    res["blackbox"] = run_blackbox(Z_tr, x_tr, y_tr, sets, rounds)
    print(f"  blackbox GBDT     test {res['blackbox']['test']['rmse']:.4f}  "
          f"extrap {res['blackbox']['extrap']['rmse']:.4f}  "
          f"({res['blackbox']['fit_time_s']}s)", flush=True)

    res["xgb"] = run_xgb_custom(Z_tr, x_tr, y_tr, sets, rounds)
    if "error" in res["xgb"]:
        print(f"  xgb custom        SKIPPED ({res['xgb']['error']})", flush=True)
    else:
        print(f"  xgb custom(diag)  test {res['xgb']['test']['rmse']:.4f}  "
              f"extrap {res['xgb']['extrap']['rmse']:.4f}  "
              f"corr_b {res['xgb']['test']['corr_b']:.3f}  "
              f"({res['xgb']['fit_time_s']}s)", flush=True)

    res["gates"] = compute_gates(res)
    g = res["gates"]
    print(f"  GATES  extrap_vs_blackbox={g['extrap_vs_blackbox_ratio']}x "
          f"({_pf(g['extrap_vs_blackbox_pass'])})  "
          f"full>plain={_pf(g['full_beats_plain_pass'])}  "
          f"full>global={_pf(g['full_beats_global_pass'])}  "
          f"full>=xgb={_pf(g['full_not_worse_than_xgb_pass'])}  "
          f"=> {'PASS' if g['passed'] else 'FAIL'}", flush=True)
    return res


def _pf(v):
    if v is None:
        return "skip"
    return "PASS" if v else "FAIL"


def run_suite(quick=False):
    import openboost as ob

    sizes = ([(8_000, 3_000, 80)] if quick
             else [(40_000, 10_000, ROUNDS), (200_000, 20_000, ROUNDS)])

    # Warmup (JIT the tree kernels off the clock).
    rng = np.random.default_rng(0)
    Zw, xw, yw, *_ = make_data(512, rng, X_TRAIN_RANGE)
    run_formulaboost("full", Zw, xw, yw,
                     {"test": (Zw, xw, yw, *true_params(Zw))}, rounds=5)

    report = {
        "benchmark": "bench_formula",
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "quick_mode": quick,
        "formula": "sales = a(z) * x ** sigmoid(b(z) * x)",
        "budget": {"rounds": ROUNDS, "lr": LR, "depth": DEPTH, "damp": DAMP,
                   "blackbox_depth": BLACKBOX_DEPTH},
        "ranges": {"x_train": X_TRAIN_RANGE, "x_extrap": X_EXTRAP_RANGE,
                   "noise_frac": NOISE_FRAC},
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
    out = RESULTS_DIR / f"formula_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out}")
    return out


# =============================================================================
# Modal entry point
# =============================================================================

try:
    import modal

    app = modal.App("openboost-formula-bench")
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
