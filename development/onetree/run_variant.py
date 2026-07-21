
"""Variant: slower scale channel (channel_lr = (1.0, 0.5)) on E1 + E3."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_spike import DEPTH, LAM, LR, SEED, load_california, make_coupled
from vectorleaf_normal import OneTreeNormalBooster

import openboost as ob

rng = np.random.default_rng(SEED)

def run(name, X_tr, y_tr, X_te, y_te, rounds=600):
    ba = ob.array(X_tr)
    tr = np.ascontiguousarray(ba.data)
    te = np.ascontiguousarray(ba.transform(X_te).data)
    curve = {}
    def hook(r, m):
        if (r + 1) % 10 == 0 or r == 0:
            curve[r + 1] = m.nll(te, y_te)
    m = OneTreeNormalBooster(n_rounds=rounds, learning_rate=LR, max_depth=DEPTH,
                             reg_lambda=LAM, channel_lr=(1.0, 0.5)).fit(tr, y_tr, eval_hook=hook)
    br = min(curve, key=curve.get)
    print(f"{name}: best NLL {curve[br]:.4f} @ {br}r   final@{rounds}r {curve[rounds]:.4f}")

X, y, _, _ = make_coupled(25_000, rng)
run("E1 slow-scale", X[:20_000], y[:20_000], X[20_000:], y[20_000:])
X, y = load_california()
perm = rng.permutation(len(y)); X, y = X[perm], y[perm]
n = int(0.8 * len(y))
run("E3 slow-scale", X[:n], y[:n], X[n:], y[n:])
