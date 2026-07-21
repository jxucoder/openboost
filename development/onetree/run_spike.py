"""Feasibility benchmark: single vector-leaf tree vs per-parameter trees.

Compares OneTreeNormalBooster (1 tree/round, joint splits) against the
library's NaturalBoostNormal (one tree per parameter per round) on:

  E1  coupled synthetic     — mean and scale depend on the SAME features
  E2  decoupled synthetic   — mean and scale depend on DISJOINT features
                              (adversarial for shared tree structure)
  E3  California housing    — real data

Both models get identical binning, depth, learning rate, and lambda. Results
are reported at matched ROUNDS and at matched TOTAL TREE COUNT (the one-tree
model spends half the trees per round, so at equal tree budget it gets twice
the rounds).

Run:  OPENBOOST_BACKEND=cpu uv run python development/onetree/run_spike.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vectorleaf_normal import OneTreeNormalBooster  # noqa: E402

import openboost as ob  # noqa: E402

SEED = 42
DEPTH = 4
LR = 0.1
LAM = 1.0
ROUNDS = 300


def make_coupled(n, rng):
    X = rng.uniform(-2, 2, size=(n, 10)).astype(np.float32)
    mu = 2 * np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + X[:, 2]
    sigma = 0.5 + np.abs(X[:, 0]) + 0.5 * np.abs(X[:, 1])
    y = mu + sigma * rng.standard_normal(n)
    return X, y.astype(np.float32), mu, sigma


def make_decoupled(n, rng):
    X = rng.uniform(-2, 2, size=(n, 10)).astype(np.float32)
    mu = 2 * np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + X[:, 2]
    sigma = 0.5 + np.abs(X[:, 5]) + 0.5 * np.abs(X[:, 6])
    y = mu + sigma * rng.standard_normal(n)
    return X, y.astype(np.float32), mu, sigma


def load_california():
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing()
    return d.data.astype(np.float32), d.target.astype(np.float32)


def run_experiment(name, X_tr, y_tr, X_te, y_te):
    print(f"\n=== {name} (n_train={len(y_tr)}, n_test={len(y_te)}) ===")
    tr_binned_arr = ob.array(X_tr)
    # transform test data with the training bin edges
    te_binned_arr = tr_binned_arr.transform(X_te)
    tr_binned = np.ascontiguousarray(tr_binned_arr.data)
    te_binned = np.ascontiguousarray(te_binned_arr.data)

    # One-tree runs 2x rounds so both models can spend the SAME total tree
    # budget (library uses k=2 trees per round).
    one_rounds = ROUNDS * 2
    curve_one = {}

    def hook(r, model):
        if (r + 1) % 10 == 0 or r == 0:
            curve_one[r + 1] = model.nll(te_binned, y_te)

    t0 = time.time()
    one = OneTreeNormalBooster(
        n_rounds=one_rounds, learning_rate=LR, max_depth=DEPTH, reg_lambda=LAM
    ).fit(tr_binned, y_tr, eval_hook=hook)
    t_one = time.time() - t0

    # Library NaturalBoost (one tree per parameter per round), with a per-round
    # validation NLL curve so both models can be scored best-on-validation.
    t0 = time.time()
    nb = ob.NaturalBoostNormal(
        n_trees=ROUNDS, learning_rate=LR, max_depth=DEPTH, reg_lambda=LAM
    )
    nb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
    t_nb = time.time() - t0
    nb_curve = nb.evals_result_["eval_0"]["nll"]
    nb_final = nb.nll(X_te, y_te)
    nb_best_round = int(np.argmin(nb_curve)) + 1
    nb_best = float(np.min(nb_curve))
    nb_trees = sum(len(v) for v in nb.trees_.values())

    one_best_round = min(curve_one, key=curve_one.get)
    one_best = curve_one[one_best_round]
    # equal-tree-budget checkpoint: one-tree at 2*ROUNDS rounds == nb tree count
    one_equal_budget = curve_one[max(r for r in curve_one if r <= one_rounds)]

    print(f"one-tree : best NLL {one_best:.4f} @ {one_best_round}r ({one_best_round} trees)   "
          f"final@{one_rounds}r(={one_rounds} trees, equal budget) {one_equal_budget:.4f}   [{t_one:.1f}s numpy-prototype]")
    print(f"library  : best NLL {nb_best:.4f} @ {nb_best_round}r ({2*nb_best_round} trees)   "
          f"final@{ROUNDS}r(={nb_trees} trees) {nb_final:.4f}   [{t_nb:.1f}s numba]")
    print(f"best-vs-best delta (one-tree minus library): {one_best - nb_best:+.4f} nats; "
          f"tree budget at best: {one_best_round} vs {2*nb_best_round}")
    return {
        "one_best": one_best, "one_best_round": one_best_round,
        "one_equal_budget": one_equal_budget,
        "nb_best": nb_best, "nb_best_round": nb_best_round,
        "nb_final": nb_final, "nb_tree_count": nb_trees,
        "one_curve": curve_one, "nb_curve": [round(v, 5) for v in nb_curve],
    }


def main():
    rng = np.random.default_rng(SEED)
    results = {}

    X, y, _, _ = make_coupled(25_000, rng)
    results["coupled"] = run_experiment("E1 coupled heteroscedastic", X[:20_000], y[:20_000], X[20_000:], y[20_000:])

    X, y, _, _ = make_decoupled(25_000, rng)
    results["decoupled"] = run_experiment("E2 decoupled (adversarial)", X[:20_000], y[:20_000], X[20_000:], y[20_000:])

    X, y = load_california()
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    n_tr = int(0.8 * len(y))
    results["california"] = run_experiment("E3 california housing", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:])

    out = Path(__file__).parent / "spike_results.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
