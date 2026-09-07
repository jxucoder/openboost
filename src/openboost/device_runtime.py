"""Experimental resident scalar transactions with privately owned accepted state."""

from dataclasses import dataclass

import numpy as np

from . import device_objectives as objective
from . import device_tree as trees
from .artifacts import Model, TreeTerm
from .binning import Binning
from .data import _identity
from .device import DeviceOperations, _atomic, _workspace
from .objectives import Squared
from .runtime import RunContext


def _coefficient(value):
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(
        value, (bool, np.bool_)
    ):
        raise ValueError("finite real float32 coefficient required")
    with np.errstate(over="raise", invalid="raise"):
        try:
            value = np.float32(value)
        except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
            raise ValueError("finite real float32 coefficient required") from error
    if not np.isfinite(value):
        raise ValueError("finite real float32 coefficient required")
    return value


@_atomic
def add_raw(ops, raw, delta, coefficient):
    """New resident raw + coefficient * delta; neither input is modified."""
    coefficient = _coefficient(coefficient)
    source = ops.execution._array(raw)
    if source.ndim != 2 or source.shape[1] != 1 or not source.shape[0]:
        raise ValueError("nonempty scalar raw matrix required")
    source = ops._float(raw, source.shape)
    update = ops._float(delta, source.shape)
    ops._validate(source)
    ops._validate(update)
    output = ops.execution._empty(source.shape, np.float32)
    ops._launch(
        "scalar_add_raw", source.shape[0], source, update, coefficient, ops.execution._array(output)
    )
    ops._validate(ops.execution._array(output))
    return output


@dataclass(frozen=True, eq=False)
class DeviceState:
    """Read-only diagnostics; raw arrays and model terms remain private to the run."""

    identity: str
    run_id: str
    version: int
    loss: float
    validation_score: float
    best_score: float
    n_terms: int
    best_n_terms: int


@dataclass(frozen=True, eq=False)
class DeviceProposal:
    identity: str
    parent_identity: str
    coefficient: float
    loss: float
    validation_score: float


@dataclass(eq=False)
class _Term:
    tree: trees.DeviceTree
    coefficient: float
    references: int = 1


@dataclass(frozen=True)
class _StateStorage:
    raw: tuple
    terms: tuple[_Term, ...]


@dataclass(frozen=True)
class _ProposalStorage:
    parent: DeviceState
    raw: tuple
    term: _Term


class DeviceRun:
    """Numeric scalar squared run, explicit preparation and separate model export.

    Retained states own raw snapshots; release them explicitly. Immutable tree
    terms are reference-counted across states/proposals. Public raw() always copies.
    ExecutionContext remains caller-owned. This is not a CPU RunContext backend.
    """

    def __init__(self, ops, train, validation, *, run_id, seed, binning=None, bins=254):
        Squared.validate(train)
        Squared.validate(validation)
        if not isinstance(ops, DeviceOperations):
            raise ValueError("DeviceOperations required")
        self._keys = RunContext(run_id, seed)  # Identity/RNG metadata only; no CPU training.
        if train.data.feature_names != validation.data.feature_names:
            raise ValueError("training/validation feature schemas differ")
        if binning is not None and bins != 254:
            raise ValueError("explicit binning owns its bin configuration")
        if binning is None:
            binning = Binning.fit(train.data, bins=bins)
        if not isinstance(binning, Binning):
            raise ValueError("fitted Binning required")
        self.ops, self.execution, self.binning = ops, ops.execution, binning
        self._closed, self._serial = False, 0
        self._states, self._proposals = {}, {}
        with _workspace(ops) as retained:
            train_data = ops.prepare(binning.transform(train.data), train)
            validation_data = ops.prepare(binning.transform(validation.data), validation)
            self._train = objective.prepare(ops, train_data, train)
            self._validation = objective.prepare(ops, validation_data, validation)
            self._base = objective.base(ops, self._train)
            self._prepared = (self._train, self._validation, train_data, validation_data)
            retained.update((*self._prepared, self._base))

    @property
    def run_id(self):
        return self._keys.run_id

    @property
    def seed(self):
        return self._keys.seed

    @property
    def data(self):
        self._check()
        return self._train.data

    @property
    def validation_data(self):
        self._check()
        return self._validation.data

    def _check(self):
        self.execution._check()
        if self._closed:
            raise RuntimeError("device run is closed")
        self.ops._get(self._train, objective.DeviceProblem)
        self.ops._get(self._validation, objective.DeviceProblem)
        self.execution._array(self._base)

    def _get(self, record, kind=None):
        self._check()
        if isinstance(record, DeviceState) and kind in (None, DeviceState):
            storage = self._states.get(record)
            terms = storage.terms if storage else ()
        elif isinstance(record, DeviceProposal) and kind in (None, DeviceProposal):
            storage = self._proposals.get(record)
            terms = (storage.term,) if storage else ()
        else:
            storage, terms = None, ()
        if storage is None:
            raise ValueError("foreign, forged or released run record")
        for handle in storage.raw:
            self.execution._array(handle)
        for term in terms:
            self.ops._get(term.tree, trees.DeviceTree)
        return storage

    def _identity(self, kind):
        # Not an RNG draw. Failed device work never advances the successful-record counter.
        identity = _identity(
            "device-run-record-v1",
            self.run_id,
            self.seed,
            self._train.data.problem_identity,
            self._validation.data.problem_identity,
            kind,
            self._serial,
        )
        self._serial += 1
        return identity

    def rng(self, round_index, component, purpose):
        """Fresh keyed generator with the same logical identity semantics as CPU runs."""
        self._check()
        return self._keys.rng(round_index, component, purpose)

    def initialize(self):
        self._check()
        with _workspace(self.ops) as retained:
            raw = tuple(
                objective.broadcast(self.ops, self._base, p.data.n_rows)
                for p in (self._train, self._validation)
            )
            loss, score = (
                objective.loss(self.ops, p, r)
                for p, r in zip((self._train, self._validation), raw, strict=True)
            )
            retained.update(raw)
        state = DeviceState(self._identity("state"), self.run_id, 0, loss, score, score, 0, 0)
        self._states[state] = _StateStorage(raw, ())
        return state

    def raw(self, record, *, validation=False):
        """Independent resident snapshot; releasing it cannot affect the run record."""
        if type(validation) is not bool:
            raise ValueError("explicit boolean validation selector required")
        return self.execution.copy(self._get(record).raw[int(validation)])

    def gradient(self, state):
        return objective.gradient(self.ops, self._train, self._get(state, DeviceState).raw[0])

    def fields(self, state):
        return objective.fields(self.ops, self._train, self._get(state, DeviceState).raw[0])

    def propose(self, state, tree, *, coefficient=1.0):
        """Snapshot the learner and evaluate just its new contribution on both sets."""
        prior = self._get(state, DeviceState)
        coefficient = _coefficient(coefficient)
        with _workspace(self.ops) as retained:
            snapshot = trees.copy(self.ops, tree)
            raw = tuple(
                add_raw(self.ops, previous, trees.predict(self.ops, snapshot, p.data), coefficient)
                for p, previous in zip((self._train, self._validation), prior.raw, strict=True)
            )
            loss, score = (
                objective.loss(self.ops, p, r)
                for p, r in zip((self._train, self._validation), raw, strict=True)
            )
            retained.update((*raw, snapshot))
        proposal = DeviceProposal(
            self._identity("proposal"), state.identity, float(coefficient), loss, score
        )
        self._proposals[proposal] = _ProposalStorage(
            state, raw, _Term(snapshot, float(coefficient))
        )
        return proposal

    def resolve(self, state, proposal, *, accept):
        """Caller acceptance; strict validation improvement selects best independently."""
        if type(accept) is not bool:
            raise ValueError("explicit boolean acceptance required")
        prior = self._get(state, DeviceState)
        candidate = self._get(proposal, DeviceProposal)
        if candidate.parent is not state:
            raise ValueError("stale or foreign proposal parent")
        if not accept:
            return state
        with _workspace(self.ops) as retained:
            raw = tuple(self.execution.copy(handle) for handle in candidate.raw)
            retained.update(raw)
        terms = (*prior.terms, candidate.term)
        improved = proposal.validation_score < state.best_score
        updated = DeviceState(
            self._identity("state"),
            self.run_id,
            state.version + 1,
            proposal.loss,
            proposal.validation_score,
            proposal.validation_score if improved else state.best_score,
            len(terms),
            len(terms) if improved else state.best_n_terms,
        )
        # All fallible CUDA work has finished before shared ownership changes.
        for term in terms:
            term.references += 1
        self._states[updated] = _StateStorage(raw, terms)
        return updated

    def export(self, state, *, best=False):
        """Explicit CPU Model export; prediction thereafter needs no CUDA or plugins."""
        if type(best) is not bool:
            raise ValueError("explicit boolean best selector required")
        storage = self._get(state, DeviceState)
        n_terms = state.best_n_terms if best else state.n_terms
        terms = tuple(
            TreeTerm(trees.export(self.ops, term.tree), [[1]], term.coefficient)
            for term in storage.terms[:n_terms]
        )
        return Model(self.binning.feature_names, self.execution.export(self._base), terms)

    def release(self, record):
        """Release one state's/proposal's storage without invalidating other records."""
        storage = self._get(record)
        for handle in storage.raw:
            self.execution.release(handle)
        terms = storage.terms if isinstance(record, DeviceState) else (storage.term,)
        for term in terms:
            term.references -= 1
            if term.references == 0:
                self.ops.release(term.tree)
        (self._states if isinstance(record, DeviceState) else self._proposals).pop(record)

    def close(self):
        """Release this run's resources, leaving its caller-owned execution context open."""
        if self._closed:
            return
        self.execution._check()
        # Release is also possible after a caller explicitly invalidates prepared data.
        self.execution.synchronize()
        handles, records = {self._base}, set(self._prepared)
        for storage in (*self._states.values(), *self._proposals.values()):
            handles.update(storage.raw)
            terms = storage.terms if isinstance(storage, _StateStorage) else (storage.term,)
            records.update(term.tree for term in terms)
        for record in records:
            if record in self.ops._records:
                handles.update(self.ops._records.pop(record)[1])
        for handle in handles & self.execution._buffers.keys():
            del self.execution._buffers[handle]
            self.execution._counts["live_bytes"] -= handle.nbytes
        self._states.clear()
        self._proposals.clear()
        self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        self.close()
