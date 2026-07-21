"""Prototype: single-tree vector-leaf boosting for multi-parameter distributions.

Instead of fitting one tree per distribution parameter per round (the current
NaturalBoost strategy), each round fits ONE tree whose leaves hold a vector
value — one component per distribution parameter. Split gain and leaf values
generalize the scalar second-order formulas:

    leaf value:  v_L = -(sum_{i in L} F_i + lam*I)^{-1} sum_{i in L} g_i
    split gain:  ||sum g||^2 in the (sum F + lam*I)^{-1} metric,
                 summed left/right minus parent

where g_i is the per-sample gradient of the NLL w.r.t. the raw (link-space)
parameters and F_i is the per-sample Fisher information (expected Hessian).
For k=1 both formulas reduce exactly to the classic XGBoost gain/leaf value.

This file implements the Normal(mu, sigma) case with raw params (mu, log sigma),
whose Fisher is diagonal: diag(1/sigma^2, 2). The solver below therefore uses
per-parameter channels; families with non-diagonal Fisher need a k x k solve
per candidate split, which is deliberately out of scope for this prototype.

Pure NumPy, CPU, level-wise growth on uint8-binned features (reuses
openboost's BinnedArray). Optimized for clarity, not speed — wall-clock
comparisons against the Numba-optimized library are not meaningful; tree
counts and NLL-per-round are.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_BINS = 256  # uint8 bin space (255 reserved for missing; data here is clean)


def normal_grad_fisher(y: np.ndarray, mu: np.ndarray, s: np.ndarray):
    """Gradients and diagonal Fisher of the Normal NLL w.r.t. raw params (mu, s=log sigma).

    NLL_i = s_i + 0.5 * ((y_i - mu_i) / exp(s_i))^2 + const
    d/dmu = (mu - y) / sigma^2
    d/ds  = 1 - ((y - mu) / sigma)^2
    Fisher (expected Hessian): diag(1 / sigma^2, 2)
    """
    sigma2 = np.exp(2.0 * s)
    z2 = (y - mu) ** 2 / sigma2
    g = np.stack([(mu - y) / sigma2, 1.0 - z2], axis=1)
    f = np.stack([1.0 / sigma2, np.full_like(y, 2.0)], axis=1)
    return g.astype(np.float64), f.astype(np.float64)


def normal_nll(y: np.ndarray, mu: np.ndarray, s: np.ndarray) -> float:
    sigma2 = np.exp(2.0 * s)
    return float(np.mean(s + 0.5 * (y - mu) ** 2 / sigma2 + 0.5 * np.log(2 * np.pi)))


@dataclass
class VectorLeafTree:
    """Level-wise binary tree over binned features with k-vector leaf values."""

    feature: np.ndarray   # (n_nodes,) int32, -1 = leaf
    threshold: np.ndarray  # (n_nodes,) uint8, go left if bin <= threshold
    value: np.ndarray      # (n_nodes, k) float64, defined on leaves
    max_depth: int

    def leaf_index(self, binned: np.ndarray) -> np.ndarray:
        n = binned.shape[1]
        pos = np.zeros(n, dtype=np.int64)
        for _ in range(self.max_depth):
            feat = self.feature[pos]
            internal = feat >= 0
            if not internal.any():
                break
            idx = np.where(internal)[0]
            bins = binned[feat[idx], idx]
            go_left = bins <= self.threshold[pos[idx]]
            pos[idx] = np.where(go_left, 2 * pos[idx] + 1, 2 * pos[idx] + 2)
        return pos

    def predict(self, binned: np.ndarray) -> np.ndarray:
        return self.value[self.leaf_index(binned)]


def fit_vectorleaf_tree(
    binned: np.ndarray,
    grad: np.ndarray,
    fisher: np.ndarray,
    max_depth: int = 4,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    min_gain: float = 1e-12,
) -> VectorLeafTree:
    """Fit one tree on k gradient/Fisher channels jointly.

    binned: (n_features, n) uint8, grad/fisher: (n, k) float64.
    """
    n_feat, n = binned.shape
    k = grad.shape[1]
    max_nodes = 2 ** (max_depth + 1) - 1
    feature = np.full(max_nodes, -1, dtype=np.int32)
    threshold = np.zeros(max_nodes, dtype=np.uint8)
    value = np.zeros((max_nodes, k), dtype=np.float64)

    pos = np.zeros(n, dtype=np.int64)

    for depth in range(max_depth):
        first = 2**depth - 1
        n_lvl = 2**depth
        rel = pos - first
        # samples sitting at this level in a node that is still open
        at_level = (rel >= 0) & (rel < n_lvl)
        if not at_level.any():
            break
        idx = np.where(at_level)[0]
        rel_idx = rel[idx]

        hist_g = np.zeros((n_lvl, n_feat, N_BINS, k))
        hist_f = np.zeros((n_lvl, n_feat, N_BINS, k))
        for f in range(n_feat):
            np.add.at(hist_g[:, f], (rel_idx, binned[f, idx]), grad[idx])
            np.add.at(hist_f[:, f], (rel_idx, binned[f, idx]), fisher[idx])

        # cumulative left sums over bins; totals per node
        gl = np.cumsum(hist_g, axis=2)          # (n_lvl, n_feat, bins, k)
        fl = np.cumsum(hist_f, axis=2)
        gt = gl[:, :, -1:, :]
        ft = fl[:, :, -1:, :]
        gr = gt - gl
        fr = ft - fl

        def score(g, f):
            return np.sum(g**2 / (f + reg_lambda), axis=-1)

        gain = score(gl, fl) + score(gr, fr) - score(gt, ft)  # (n_lvl, n_feat, bins)
        # feasibility: both children need enough Fisher mass (trace)
        wl = fl.sum(axis=-1)
        wr = fr.sum(axis=-1)
        gain[(wl < min_child_weight) | (wr < min_child_weight)] = -np.inf
        gain[:, :, -1] = -np.inf  # splitting at the last bin sends everything left

        flat = gain.reshape(n_lvl, -1)
        best = flat.argmax(axis=1)
        best_gain = flat[np.arange(n_lvl), best]
        best_feat, best_bin = np.unravel_index(best, (n_feat, N_BINS))

        for node_rel in range(n_lvl):
            node = first + node_rel
            members = idx[rel_idx == node_rel]
            if members.size == 0:
                continue
            if best_gain[node_rel] > min_gain and depth < max_depth:
                feature[node] = best_feat[node_rel]
                threshold[node] = best_bin[node_rel]
                bins = binned[best_feat[node_rel], members]
                left = bins <= best_bin[node_rel]
                pos[members[left]] = 2 * node + 1
                pos[members[~left]] = 2 * node + 2

    # leaf values: aggregate G, F by final node
    g_leaf = np.zeros((max_nodes, k))
    f_leaf = np.zeros((max_nodes, k))
    np.add.at(g_leaf, pos, grad)
    np.add.at(f_leaf, pos, fisher)
    occupied = f_leaf.sum(axis=1) > 0
    value[occupied] = -g_leaf[occupied] / (f_leaf[occupied] + reg_lambda)
    return VectorLeafTree(feature, threshold, value, max_depth)


@dataclass
class OneTreeNormalBooster:
    """Boosted Normal(mu, sigma) with one vector-leaf tree per round."""

    n_rounds: int = 300
    learning_rate: float = 0.1
    max_depth: int = 4
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    # log-scale box around the init value. The Fisher for s = log sigma is the
    # constant 2, so once training residuals shrink the s-gradient keeps
    # pushing sigma toward 0 with no curvature counterweight; an s-box is the
    # minimal safeguard against that late-training collapse.
    s_box: float = 4.0
    line_search: bool = False
    # Per-channel learning-rate multipliers (mu, s). The joint split gain
    # actively selects splits that isolate low-residual regions, which
    # overfits the scale channel much faster than the mean channel; a slower
    # scale channel is the minimal counterweight.
    channel_lr: tuple = (1.0, 1.0)
    trees_: list = field(default_factory=list)
    init_: tuple = (0.0, 0.0)

    def fit(self, binned: np.ndarray, y: np.ndarray, eval_hook=None):
        y = np.asarray(y, dtype=np.float64)
        mu0 = float(np.mean(y))
        s0 = float(np.log(np.std(y) + 1e-12))
        self.init_ = (mu0, s0)
        raw = np.tile(np.array([mu0, s0]), (len(y), 1))
        for r in range(self.n_rounds):
            g, f = normal_grad_fisher(y, raw[:, 0], raw[:, 1])
            tree = fit_vectorleaf_tree(
                binned, g, f,
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                min_child_weight=self.min_child_weight,
            )
            step = tree.predict(binned) * np.asarray(self.channel_lr)
            if self.line_search:
                # Backtracking on the train NLL: the Newton step is locally
                # optimal under the quadratic model, but the scale channel's
                # constant Fisher underestimates curvature once residuals are
                # small; a cheap step-size search restores monotone descent.
                base = normal_nll(y, raw[:, 0], raw[:, 1])
                best_scale, best_nll = 0.0, base
                for scale in (1.0, 0.5, 0.25, 0.125):
                    cand = raw + self.learning_rate * scale * step
                    nll = normal_nll(y, cand[:, 0], cand[:, 1])
                    if nll < best_nll:
                        best_scale, best_nll = scale, nll
                tree.value *= best_scale  # bake the scale into the tree
                raw += self.learning_rate * best_scale * tree.predict(binned)
            else:
                raw += self.learning_rate * step
            np.clip(raw[:, 1], s0 - self.s_box, s0 + self.s_box, out=raw[:, 1])
            self.trees_.append(tree)
            if eval_hook is not None:
                eval_hook(r, self)
        return self

    def predict_raw(self, binned: np.ndarray, n_rounds: int | None = None) -> np.ndarray:
        raw = np.tile(np.array(self.init_), (binned.shape[1], 1))
        s0 = self.init_[1]
        trees = self.trees_ if n_rounds is None else self.trees_[:n_rounds]
        mult = np.asarray(self.channel_lr)
        for tree in trees:
            raw += self.learning_rate * tree.predict(binned) * mult
            np.clip(raw[:, 1], s0 - self.s_box, s0 + self.s_box, out=raw[:, 1])
        return raw

    def nll(self, binned: np.ndarray, y: np.ndarray) -> float:
        raw = self.predict_raw(binned)
        return normal_nll(np.asarray(y, dtype=np.float64), raw[:, 0], raw[:, 1])

    @property
    def n_trees_total(self) -> int:
        return len(self.trees_)
