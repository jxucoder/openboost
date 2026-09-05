"""Spike: generalized Gauss-Newton (GGN) parametric boosting on a custom formula.

Model:  sales y = a(z) * x ** sigmoid(b(z) * x) + noise
        a = exp(F_a(z))  (log link, positive ceiling parameter)
        b = F_b(z)       (identity link, saturation-speed parameter)

Each parameter channel F_k(z) is a boosting ensemble over feature vector z;
x (spend/price) enters only through the user formula.

Question this spike answers: is GGN preconditioning + tree pooling stable and
necessary?  We compare three per-sample update rules for d = "step direction
the trees are fit to":

    plain : d = g                      (raw gradient, single learning rate)
    diag  : d = g / (diag(G) + lam)    (per-channel GGN scaling)
    full  : d = (G + lam*I)^{-1} g     (full 2x2 GGN solve, off-diagonal)

where g = dL/dF and G = J^T J (MSE loss => GGN middle matrix is identity).

Baselines:
    global : one (a, b) fit to all samples by Adam (no per-sample params)
    blackbox: ob.GradientBoosting on [z, x] -> y (no formula)

Metrics:
    - in-range test RMSE      (new z, x within training range)
    - extrapolation RMSE      (new z, x beyond training range: [3, 5])
    - parameter recovery      (RMSE of a_hat vs a_true, b_hat vs b_true)

Derivatives (analytic; JAX unavailable on this machine, same math):
    s  = sigmoid(b*x)
    f  = exp(u + s*ln x)          with u = F_a, b = F_b
    df/du = f
    df/db = f * ln x * s*(1-s) * x

Usage:  uv run python development/paramboost/spike_ggn.py [--quick]
Output: development/paramboost/spike_ggn_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import openboost as ob

SEED = 42
N_TRAIN = 40_000
N_TEST = 10_000
N_FEATURES = 6
ROUNDS = 300
LR = 0.1
DEPTH = 3
DAMP = 1.0
X_TRAIN_RANGE = (0.25, 2.5)
X_EXTRAP_RANGE = (3.0, 5.0)
NOISE_FRAC = 0.05  # additive noise sd as fraction of signal sd


# =============================================================================
# Data generating process
# =============================================================================

def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def true_params(Z):
    """Ground-truth parameter surfaces a(z), b(z)."""
    u = 0.5 + 1.0 * Z[:, 0] - 0.8 * Z[:, 1] + 0.5 * np.sin(2 * np.pi * Z[:, 2])
    a = np.exp(u)
    b = 0.5 + 2.0 * Z[:, 3] + Z[:, 4] * Z[:, 0]
    return a, b


def formula(a, b, x):
    """sales = a * x ** sigmoid(b*x), computed stably in log space."""
    s = sigmoid(b * x)
    return np.exp(np.log(a) + s * np.log(x))


def make_data(n, rng, x_range):
    Z = rng.uniform(0, 1, (n, N_FEATURES))
    x = rng.uniform(*x_range, n)
    a, b = true_params(Z)
    f = formula(a, b, x)
    y = f + NOISE_FRAC * f.std() * rng.standard_normal(n)
    return Z, x, y, a, b, f


# =============================================================================
# Per-sample gradients / GGN for the formula under MSE loss
# =============================================================================

def grads_and_ggn(u, b, x, y):
    """Return g (n,2), G entries (n,3): [G_uu, G_bb, G_ub]."""
    lnx = np.log(x)
    s = sigmoid(b * x)
    f = np.exp(np.clip(u + s * lnx, -30.0, 30.0))
    fu = f
    fb = f * lnx * s * (1.0 - s) * x
    r = f - y  # dL/df for L = 0.5*(f-y)^2
    g = np.stack([r * fu, r * fb], axis=1)
    G_uu = fu * fu
    G_bb = fb * fb
    G_ub = fu * fb
    return g, G_uu, G_bb, G_ub, f


def step_direction(mode, g, G_uu, G_bb, G_ub, damp):
    """Compute d per sample for the chosen preconditioning mode."""
    if mode == "plain":
        return g
    if mode == "diag":
        return g / np.stack([G_uu + damp, G_bb + damp], axis=1)
    # full 2x2 solve, analytic inverse with damping
    det = (G_uu + damp) * (G_bb + damp) - G_ub * G_ub
    d_u = ((G_bb + damp) * g[:, 0] - G_ub * g[:, 1]) / det
    d_b = ((G_uu + damp) * g[:, 1] - G_ub * g[:, 0]) / det
    return np.stack([d_u, d_b], axis=1)


# =============================================================================
# Global curve fit (baseline + boosting init)
# =============================================================================

def fit_global(x, y, iters=2000, lr=0.05):
    """Adam on scalar (u, b) minimizing mean 0.5*(f-y)^2."""
    u, b = float(np.log(np.mean(y))), 1.0
    m = np.zeros(2)
    v = np.zeros(2)
    for t in range(1, iters + 1):
        g, *_ = grads_and_ggn(np.full_like(x, u), np.full_like(x, b), x, y)
        gm = g.mean(axis=0)
        m = 0.9 * m + 0.1 * gm
        v = 0.999 * v + 0.001 * gm * gm
        mh = m / (1 - 0.9**t)
        vh = v / (1 - 0.999**t)
        u -= lr * mh[0] / (np.sqrt(vh[0]) + 1e-8)
        b -= lr * mh[1] / (np.sqrt(vh[1]) + 1e-8)
    return u, b


# =============================================================================
# Parametric booster
# =============================================================================

def fit_paramboost(mode, Z_tr, x_tr, y_tr, eval_sets, rounds=ROUNDS,
                   lr=LR, depth=DEPTH, damp=DAMP):
    """Boost F_u, F_b on binned Z. Returns dict with curves and final F fns.

    eval_sets: {name: (Z_binned_data, x, y)} evaluated every 10 rounds.
    """
    ba = ob.array(Z_tr)
    Zb = np.ascontiguousarray(ba.data)

    u0, b0 = fit_global(x_tr, y_tr)
    F_u = np.full(len(y_tr), u0, dtype=np.float64)
    F_b = np.full(len(y_tr), b0, dtype=np.float64)

    # Per-eval-set running ensemble predictions (base + sum of tree outputs)
    eval_state = {
        name: {
            "Zb": np.ascontiguousarray(ba.transform(Z).data),
            "F_u": np.full(len(y), u0, dtype=np.float64),
            "F_b": np.full(len(y), b0, dtype=np.float64),
            "x": x, "y": y,
        }
        for name, (Z, x, y) in eval_sets.items()
    }

    curves = {name: {} for name in eval_sets}
    diverged = False
    t0 = time.perf_counter()
    for r in range(rounds):
        g, G_uu, G_bb, G_ub, f = grads_and_ggn(F_u, F_b, x_tr, y_tr)
        if not np.isfinite(g).all() or f.max() > 1e12:
            diverged = True
            break
        d = step_direction(mode, g, G_uu, G_bb, G_ub, damp)

        ones = np.ones(len(y_tr), dtype=np.float64)
        for ch, F in ((0, F_u), (1, F_b)):
            # fit_tree leaf value = -sum(grad)/(sum(hess)+reg); grad=d, hess=1
            # gives leaf = -mean(d) so F += lr*tree is a descent step.
            tree = ob.fit_tree(ba, np.ascontiguousarray(d[:, ch]), ones,
                               max_depth=depth)
            F += lr * tree(Zb)
            for st in eval_state.values():
                if ch == 0:
                    st["F_u"] += lr * tree(st["Zb"])
                else:
                    st["F_b"] += lr * tree(st["Zb"])

        if (r + 1) % 10 == 0 or r == 0:
            for name, st in eval_state.items():
                pred = formula(np.exp(st["F_u"]), st["F_b"], st["x"])
                curves[name][r + 1] = float(
                    np.sqrt(np.mean((pred - st["y"]) ** 2)))

    fit_time = time.perf_counter() - t0
    return {
        "curves": curves,
        "eval_state": eval_state,
        "diverged": diverged,
        "rounds_done": r + 1 if not diverged else r,
        "fit_time_s": fit_time,
        "init": (u0, b0),
    }


# =============================================================================
# Runner
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    global N_TRAIN, N_TEST, ROUNDS
    if args.quick:
        N_TRAIN, N_TEST, ROUNDS = 5_000, 2_000, 60

    rng = np.random.default_rng(SEED)
    Z_tr, x_tr, y_tr, _, _, _ = make_data(N_TRAIN, rng, X_TRAIN_RANGE)
    Z_te, x_te, y_te, a_te, b_te, f_te = make_data(N_TEST, rng, X_TRAIN_RANGE)
    # Extrapolation: same z distribution, x beyond training range, NOISELESS
    # target (we compare against the true curve).
    Z_ex, x_ex, _, a_ex, b_ex, f_ex = make_data(N_TEST, rng, X_EXTRAP_RANGE)

    eval_sets = {"test": (Z_te, x_te, y_te), "extrap": (Z_ex, x_ex, f_ex)}
    results = {}

    # --- Parametric boosting variants ---
    for mode in ["plain", "diag", "full"]:
        out = fit_paramboost(mode, Z_tr, x_tr, y_tr, eval_sets, rounds=ROUNDS)
        st_te = out["eval_state"]["test"]
        a_hat = np.exp(st_te["F_u"])
        b_hat = st_te["F_b"]
        test_curve = out["curves"]["test"]
        best_r = min(test_curve, key=test_curve.get) if test_curve else None
        results[mode] = {
            "diverged": out["diverged"],
            "rounds_done": out["rounds_done"],
            "fit_time_s": round(out["fit_time_s"], 2),
            "test_rmse_final": test_curve.get(ROUNDS),
            "test_rmse_best": test_curve[best_r] if best_r else None,
            "best_round": best_r,
            "extrap_rmse_final": out["curves"]["extrap"].get(ROUNDS),
            "param_rmse_a": float(np.sqrt(np.mean((a_hat - a_te) ** 2))),
            "param_rmse_b": float(np.sqrt(np.mean((b_hat - b_te) ** 2))),
            "param_corr_a": float(np.corrcoef(a_hat, a_te)[0, 1]),
            "param_corr_b": float(np.corrcoef(b_hat, b_te)[0, 1]),
        }
        r = results[mode]
        print(f"{mode:>6}: diverged={r['diverged']}  "
              f"test RMSE best {r['test_rmse_best']}@{r['best_round']} "
              f"final {r['test_rmse_final']}  extrap {r['extrap_rmse_final']}  "
              f"a: rmse {r['param_rmse_a']:.3f} corr {r['param_corr_a']:.3f}  "
              f"b: rmse {r['param_rmse_b']:.3f} corr {r['param_corr_b']:.3f}  "
              f"({r['fit_time_s']}s)")

    # --- Baseline: global curve fit ---
    u_g, b_g = fit_global(x_tr, y_tr)
    pred_te = formula(np.exp(u_g), b_g, x_te)
    pred_ex = formula(np.exp(u_g), b_g, x_ex)
    a_true_mean, b_true_mean = true_params(Z_te)
    results["global"] = {
        "test_rmse": float(np.sqrt(np.mean((pred_te - y_te) ** 2))),
        "extrap_rmse": float(np.sqrt(np.mean((pred_ex - f_ex) ** 2))),
        "param_rmse_a": float(np.sqrt(np.mean((np.exp(u_g) - a_true_mean) ** 2))),
        "param_rmse_b": float(np.sqrt(np.mean((b_g - b_true_mean) ** 2))),
    }
    print(f"global: test RMSE {results['global']['test_rmse']:.4f}  "
          f"extrap {results['global']['extrap_rmse']:.4f}")

    # --- Baseline: black-box GBDT on [z, x] ---
    X_tr = np.column_stack([Z_tr, x_tr])
    gb = ob.GradientBoosting(n_trees=ROUNDS, max_depth=6, learning_rate=0.1,
                             random_state=SEED)
    t0 = time.perf_counter()
    gb.fit(X_tr, y_tr)
    gb_time = time.perf_counter() - t0
    pred_te = gb.predict(np.column_stack([Z_te, x_te]))
    pred_ex = gb.predict(np.column_stack([Z_ex, x_ex]))
    results["blackbox"] = {
        "fit_time_s": round(gb_time, 2),
        "test_rmse": float(np.sqrt(np.mean((pred_te - y_te) ** 2))),
        "extrap_rmse": float(np.sqrt(np.mean((pred_ex - f_ex) ** 2))),
    }
    print(f"blackbox: test RMSE {results['blackbox']['test_rmse']:.4f}  "
          f"extrap {results['blackbox']['extrap_rmse']:.4f}")

    # Reference: irreducible noise floor on the test set
    results["noise_floor_test_rmse"] = float(NOISE_FRAC * f_te.std())
    results["config"] = {
        "n_train": N_TRAIN, "n_test": N_TEST, "rounds": ROUNDS, "lr": LR,
        "depth": DEPTH, "damp": DAMP, "seed": SEED,
        "x_train_range": X_TRAIN_RANGE, "x_extrap_range": X_EXTRAP_RANGE,
        "noise_frac": NOISE_FRAC,
    }
    print(f"noise floor (test RMSE lower bound): "
          f"{results['noise_floor_test_rmse']:.4f}")

    out_path = Path(__file__).parent / "spike_ggn_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
