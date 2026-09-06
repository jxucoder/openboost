"""Explicit CPU run identity and immutable B03 proposal transactions."""

from dataclasses import dataclass, field, replace

import numpy as np

from .artifacts import ConstantTerm, Model, TreeTerm
from .data import Problem, _identity, _owned


@dataclass(frozen=True)
class RunContext:
    run_id: str
    seed: int
    device: str = "cpu"

    def __post_init__(self):
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("nonempty run ID required")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("nonnegative integer seed required")
        if self.device != "cpu":
            raise ValueError("RunContext currently supports CPU only")

    def rng(self, round_index, component, purpose):
        """Fresh generator for a logical key; retries/rejection cannot consume it.

        Reusing a key intentionally reproduces the same stream. Use distinct
        component/purpose labels for independent draws, never a global RNG.
        """
        if type(round_index) is not int or round_index < 0:
            raise ValueError("nonnegative logical round required")
        if any(not isinstance(v, str) or not v for v in (component, purpose)):
            raise ValueError("nonempty RNG component and purpose required")
        key = _identity("run-rng-v1", self.seed, self.run_id, round_index, component, purpose)
        return np.random.default_rng(int(key[:32], 16))


@dataclass(frozen=True, eq=False)
class AcceptedState:
    context: RunContext
    train: Problem
    validation: Problem
    model: Model
    best_model: Model
    best_score: float
    version: int = 0
    train_raw: np.ndarray = field(init=False)
    validation_raw: np.ndarray = field(init=False)
    identity: str = field(init=False)

    def __post_init__(self):
        if (
            not isinstance(self.context, RunContext)
            or not isinstance(self.train, Problem)
            or not isinstance(self.validation, Problem)
        ):
            raise ValueError("explicit context and problems required")
        if type(self.version) is not int or self.version < 0 or not np.isfinite(self.best_score):
            raise ValueError("valid version and finite best score required")
        if not isinstance(self.model, Model) or not isinstance(self.best_model, Model):
            raise ValueError("state requires ensemble artifacts")
        for model in (self.model, self.best_model):
            if (
                model.feature_names != self.train.data.feature_names
                or model.feature_names != self.validation.data.feature_names
            ):
                raise ValueError("train/validation/model feature schema differs")
            if (
                len(model.base) != self.train.target.shape[1]
                or len(model.base) != self.validation.target.shape[1]
            ):
                raise ValueError("problem and model output widths differ")
        object.__setattr__(self, "train_raw", _owned(self.model.predict(self.train.data), ndim=2))
        object.__setattr__(
            self, "validation_raw", _owned(self.model.predict(self.validation.data), ndim=2)
        )
        object.__setattr__(
            self,
            "identity",
            _identity(
                "accepted-v1",
                self.context.run_id,
                self.context.seed,
                self.train.identity,
                self.validation.identity,
                self.model.identity,
                self.best_model.identity,
                float(self.best_score),
                self.version,
            ),
        )


@dataclass(frozen=True)
class Proposal:
    parent_identity: str
    terms: tuple[ConstantTerm | TreeTerm, ...]

    def __post_init__(self):
        terms = tuple(self.terms)
        if (
            not isinstance(self.parent_identity, str)
            or not self.parent_identity
            or not terms
            or any(not isinstance(t, (ConstantTerm, TreeTerm)) for t in terms)
        ):
            raise ValueError("valid parent and nonempty supported proposal terms required")
        object.__setattr__(self, "terms", terms)


def initialize(context, train, validation, base, *, score):
    """Initialize with a finite validation score; smaller scores are better.

    score(problem, raw) owns objective weighting/offset semantics. Runtime raw
    caches exclude input offsets. Terms and best snapshots are immutable.
    """
    model = Model(train.data.feature_names, base)
    initial = AcceptedState(context, train, validation, model, model, 0.0)
    value = float(score(validation, initial.validation_raw))
    if not np.isfinite(value):
        raise ValueError("finite initial validation score required")
    return replace(initial, best_score=value)


def propose(state, value, *, coefficient=1.0):
    return propose_terms(state, (ConstantTerm(value, coefficient),))


def propose_terms(state, terms):
    """Propose an atomic tuple of mapped learner/constant updates."""
    proposal = Proposal(state.identity, tuple(terms))
    candidate = preview(state, proposal)
    _owned(candidate.predict(state.train.data), ndim=2)
    _owned(candidate.predict(state.validation.data), ndim=2)
    return proposal


def preview(state, proposal):
    """Build a candidate model without changing any accepted or best state."""
    if not isinstance(proposal, Proposal) or proposal.parent_identity != state.identity:
        raise ValueError("stale or foreign proposal parent")
    return Model(state.model.feature_names, state.model.base, (*state.model.terms, *proposal.terms))


def resolve(state, proposal, *, accept, score):
    """Commit atomically or return the identical state on rejection.

    Acceptance is decided by caller algorithm code. It need not mean validation
    improvement. Validation chooses an immutable best snapshot independently.
    """
    if type(accept) is not bool:
        raise ValueError("explicit boolean acceptance required")
    candidate = preview(state, proposal)
    if not accept:
        return state
    raw = _owned(candidate.predict(state.validation.data), ndim=2)
    value = float(score(state.validation, raw))
    if not np.isfinite(value):
        raise ValueError("finite candidate validation score required")
    improved = value < state.best_score
    return AcceptedState(
        state.context,
        state.train,
        state.validation,
        candidate,
        candidate if improved else state.best_model,
        value if improved else state.best_score,
        state.version + 1,
    )
