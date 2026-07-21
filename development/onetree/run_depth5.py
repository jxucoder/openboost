"""Equal per-round leaf budget: one-tree depth-5 (32 leaves/round) vs the
depth-4 library baseline (2 trees x 16 leaves/round)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from vectorleaf_normal import OneTreeNormalBooster
import openboost as ob
from run_spike import make_coupled, load_california, SEED, LR, LAM

rng = np.random.default_rng(SEED)

def run(name, X_tr, y_tr, X_te, y_te, depth, rounds=600):
    ba = ob.array(X_tr)
    tr = np.ascontiguousarray(ba.data)
    te = np.ascontiguousarray(ba.transform(X_te).data)
    curve = {}
    def hook(r, m):
        if (r + 1) % 10 == 0 or r == 0:
            curve[r + 1] = m.nll(te, y_te)
    OneTreeNormalBooster(n_rounds=rounds, learning_rate=LR, max_depth=depth,
                         reg_lambda=LAM).fit(tr, y_tr, eval_hook=hook)
    br = min(curve, key=curve.get)
    print(f"{name} depth={depth}: best NLL {curve[br]:.4f} @ {br}r ({br} trees)")

X, y = load_california()
perm = rng.permutation(len(y)); X, y = X[perm], y[perm]
n = int(0.8 * len(y))
for d in (5, 6):
    run("E3", X[:n], y[:n], X[n:], y[n:], d)

Xc, yc, _, _ = make_coupled(25_000, rng)
run("E1", Xc[:20_000], yc[:20_000], Xc[20_000:], yc[20_000:], 5)
