"""Finite reference compositions and immutable state; not a public model/runtime API."""

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from .positive import gamma, gamma_base, poisson, poisson_base, poisson_predict
from .quantile import fit_quantile_tree, weighted_quantile
from .runs import derive_seed
from .tree import fit_tree, numeric_bins


@dataclass(frozen=True)
class Ensemble:
    base: float
    terms: tuple
    n_features: int

    def raw(self, values):
        x = numeric_bins(values)
        if x.shape[1] != self.n_features:
            raise ValueError("feature schema mismatch")
        raw = np.full(len(x), self.base)
        for tree, coefficient in self.terms:
            raw += coefficient * tree.predict(x)
        if not np.all(np.isfinite(raw)):
            raise ValueError("non-finite ensemble prediction")
        return raw


def fit_positive(bins, target, *, kind, exposure=None, weight=None):
    x = numeric_bins(bins)
    if kind == "poisson":
        base = poisson_base(target, exposure, weight=weight, minimum_rate=1e-8)

        def objective(raw):
            return poisson(raw, target, exposure, weight=weight)
    elif kind == "gamma":
        if exposure is not None:
            raise ValueError("Gamma severity does not consume exposure")
        base = gamma_base(target, weight=weight)

        def objective(raw):
            return gamma(raw, target, weight=weight)
    else:
        raise ValueError("unknown positive recipe")
    raw = np.full(len(x), base)
    terms, trace = [], []
    for _ in range(2):
        _, g, h = objective(raw)
        tree = fit_tree(x, g, h, weight=weight, max_depth=2)
        updated = raw + 0.1 * tree.predict(x)
        objective(updated)  # finite/domain checks before committing
        trace.append((tuple(raw), tuple(g), tuple(h), tuple(updated)))
        terms.append((tree, 0.1))
        raw = updated
    return Ensemble(base, tuple(terms), x.shape[1]), tuple(trace)


@dataclass(frozen=True)
class TwoStage:
    frequency: Ensemble
    severity: Ensemble

    @classmethod
    def fit(cls, policy_bins, paid_count, exposure, claim_bins, positive_payments):
        frequency, _ = fit_positive(policy_bins, paid_count, kind="poisson", exposure=exposure)
        severity, _ = fit_positive(claim_bins, positive_payments, kind="gamma")
        return cls(frequency, severity)

    def predict(self, bins, exposure):
        frequency = poisson_predict(self.frequency.raw(bins), exposure)
        with np.errstate(over="raise", invalid="raise"):
            severity = np.exp(self.severity.raw(bins))
            annualized = frequency["rate"] * severity
            amount = frequency["count_mean"] * severity
        if (
            not np.all(np.isfinite(amount))
            or not np.all(np.isfinite(annualized))
            or np.any(severity <= 0)
        ):
            raise ValueError("invalid two-stage prediction")
        return {"annualized": annualized, "amount": amount}


def fit_quantiles(bins, target, *, weight=None):
    x = numeric_bins(bins)
    models = {}
    for q in (0.1, 0.5, 0.9):
        base = weighted_quantile(target, q, weight)
        raw = np.full(len(x), base)
        terms = []
        for _ in range(2):
            tree = fit_quantile_tree(x, raw, target, q, weight=weight, max_depth=2)
            raw += 0.1 * tree.predict(x)
            terms.append((tree, 0.1))
        models[q] = Ensemble(base, tuple(terms), x.shape[1])
    return models


def _snapshot(raw):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or min(raw.shape) == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("raw must be nonempty finite [N,K]")
    return tuple(tuple(row) for row in raw)


@dataclass(frozen=True)
class Snapshot:
    version: int
    step_id: tuple
    train_raw: tuple
    valid_raw: tuple
    terms: tuple


@dataclass(frozen=True)
class Track:
    run_id: str
    seed: int
    current: Snapshot
    best: Snapshot
    best_score: float

    @classmethod
    def initialize(cls, run_id, seed, train_raw, valid_raw, *, score):
        derive_seed(seed, run_id, 0, "state", "initialize")
        train, valid = _snapshot(train_raw), _snapshot(valid_raw)
        if len(train[0]) != len(valid[0]):
            raise ValueError("output schema mismatch")
        metric = float(score(np.array(valid)))
        if not np.isfinite(metric):
            raise ValueError("non-finite initial metric")
        snapshot = Snapshot(0, (0, -1), train, valid, ())
        return cls(run_id, seed, snapshot, snapshot, metric)


@dataclass(frozen=True)
class Proposal:
    run_id: str
    parent_version: int
    step_id: tuple
    update: object


def advance(track, proposal, train_bins, valid_bins, *, score):
    current, update = track.current, proposal.update
    if proposal.run_id != track.run_id:
        raise ValueError("proposal belongs to another run")
    if proposal.parent_version != current.version:
        raise ValueError("stale parent version")
    if update.raw_before != current.train_raw:
        raise ValueError("proposal raw does not match accepted parent")
    if not update.accepted:
        if update.terms or update.raw_after != current.train_raw:
            raise ValueError("rejected proposal contains a state change")
        return track
    step_id = proposal.step_id
    if (
        len(step_id) != 2
        or any(not isinstance(i, Integral) or isinstance(i, bool) for i in step_id)
        or step_id <= current.step_id
        or step_id[0] < 1
        or step_id[1] not in (0, 1)
    ):
        raise ValueError("invalid logical step order")
    train, valid = np.array(current.train_raw), np.array(current.valid_raw)
    train_bins, valid_bins = numeric_bins(train_bins), numeric_bins(valid_bins)
    if len(train_bins) != len(train) or len(valid_bins) != len(valid):
        raise ValueError("proposal data rows do not match state caches")
    if np.asarray(update.raw_after).shape != train.shape:
        raise ValueError("proposal raw shape differs from state")
    for channel, tree, coefficient in update.terms:
        if channel not in range(train.shape[1]) or not np.isfinite(coefficient):
            raise ValueError("invalid term mapping or coefficient")
        train[:, channel] += coefficient * tree.predict(train_bins)
        valid[:, channel] += coefficient * tree.predict(valid_bins)
    train_snapshot, valid_snapshot = _snapshot(train), _snapshot(valid)
    if not np.allclose(train, np.asarray(update.raw_after), rtol=1e-12, atol=1e-12):
        raise ValueError("proposal prediction does not match terms")
    metric = float(score(valid.copy()))
    if not np.isfinite(metric):
        raise ValueError("non-finite validation metric")
    accepted = Snapshot(
        current.version + 1, step_id, train_snapshot, valid_snapshot, current.terms + update.terms
    )
    if metric < track.best_score:
        return replace(track, current=accepted, best=accepted, best_score=metric)
    return replace(track, current=accepted)


def restore_best(track):
    # Restore the immutable matched payload/cache/step bundle but give it a fresh
    # version so a proposal from the abandoned future cannot be replayed.
    restored = replace(track.best, version=track.current.version + 1)
    return replace(track, current=restored)
