"""Experimental mapped resident transactions with privately owned accepted state."""

from dataclasses import dataclass

import numpy as np

from . import device_objectives as objectives
from . import device_tree as trees
from .artifacts import Model, TreeTerm
from .binning import Binning
from .data import _identity
from .device import DeviceOperations, _atomic, _workspace
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
    if source.ndim != 2 or not all(source.shape):
        raise ValueError("nonempty raw matrix required")
    source = ops._float(raw, source.shape)
    update = ops._float(delta, source.shape)
    ops._validate(source)
    ops._validate(update)
    output = ops.execution._empty(source.shape, np.float32)
    ops._launch(
        "scalar_add_raw" if source.shape[1] == 1 else "matrix_add_raw",
        source.shape[0],
        source,
        update,
        coefficient,
        ops.execution._array(output),
    )
    ops._validate(ops.execution._array(output))
    return output


def _mapping(value):
    try:
        with np.errstate(over="raise", invalid="raise"):
            values = np.asarray(value, dtype=np.float32)
    except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
        raise ValueError("finite float32 [1,K] mapping required") from error
    if (
        values.ndim != 2
        or values.shape[0] != 1
        or not values.shape[1]
        or not np.isfinite(values).all()
    ):
        raise ValueError("finite float32 [1,K] mapping required")
    return np.frombuffer(values.tobytes(), dtype=np.float32).reshape(values.shape)


@dataclass(frozen=True, eq=False)
class DeviceTerm:
    """Scalar tree and owned immutable [1,K] output map; run validates tree ownership."""

    tree: trees.DeviceTree
    mapping: object

    def __post_init__(self):
        if not isinstance(self.tree, trees.DeviceTree):
            raise ValueError("DeviceTree required")
        object.__setattr__(self, "mapping", _mapping(self.mapping))


@_atomic
def map_update(ops, raw, prediction, mapping, coefficient=1.0):
    """Map a resident scalar prediction into K raw columns with independent output.

    Mapping is small host metadata passed as kernel scalar arguments, not a bulk
    array upload. Stored term order and separate product rounding define replay.
    """
    mapping, coefficient = _mapping(mapping), _coefficient(coefficient)
    source = ops.execution._array(raw)
    if source.ndim != 2 or not source.shape[0] or source.shape[1] != mapping.shape[1]:
        raise ValueError("raw width and mapping differ")
    source = ops._float(raw, source.shape)
    scalar = ops._float(prediction, (source.shape[0], 1))
    if source.shape[1] == 1 and mapping[0, 0] == 1:
        return add_raw(ops, raw, prediction, coefficient)
    ops._validate(source)
    ops._validate(scalar)
    output = ops.execution._empty(source.shape, np.float32)
    ops._launch(
        "mapped_add_raw",
        source.shape[0],
        source,
        scalar,
        tuple(mapping[0]),
        coefficient,
        ops.execution._array(output),
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
    mapping: object
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
    terms: tuple[_Term, ...]


class DeviceRun:
    """Numeric run with explicit objective dependencies and mapped scalar trees.

    Retained states own raw snapshots; release them explicitly. Immutable tree
    terms are reference-counted across states/proposals. Public raw() always copies.
    ExecutionContext remains caller-owned. This is not a CPU RunContext backend.
    """

    def __init__(
        self,
        ops,
        train,
        validation,
        *,
        run_id,
        seed,
        binning=None,
        bins=254,
        objective=objectives.SQUARED,
    ):
        if not isinstance(objective, objectives.ObjectiveOperations):
            raise ValueError("explicit ObjectiveOperations required")
        objective.validate(train)
        objective.validate(validation)
        if train.raw_width != validation.raw_width:
            raise ValueError("training/validation raw widths differ")
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
        self.objective = objective
        self._closed, self._serial = False, 0
        self._states, self._proposals = {}, {}
        with _workspace(ops) as retained:
            train_data = ops.prepare(binning.transform(train.data), train)
            validation_data = ops.prepare(binning.transform(validation.data), validation)
            self._train = objective.prepare(ops, train_data, train)
            self._validation = objective.prepare(ops, validation_data, validation)
            for p, expected in ((self._train, train_data), (self._validation, validation_data)):
                ops._get(p, objectives.DeviceProblem)
                if p.data is not expected or p.raw_width != train.raw_width:
                    raise ValueError("objective preparation returned foreign data or raw width")
            self._base = objective.base(ops, self._train)
            ops._validate(ops._float(self._base, (train.raw_width,)).reshape(1, -1))
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

    @property
    def problem(self):
        """Prepared training record for public geometry operations; raw remains private."""
        self._check()
        return self._train

    @property
    def raw_width(self):
        return self.problem.raw_width

    def _check(self):
        self.execution._check()
        if self._closed:
            raise RuntimeError("device run is closed")
        self.ops._get(self._train, objectives.DeviceProblem)
        self.ops._get(self._validation, objectives.DeviceProblem)
        self.execution._array(self._base)

    def _get(self, record, kind=None):
        self._check()
        if isinstance(record, DeviceState) and kind in (None, DeviceState):
            storage = self._states.get(record)
            terms = storage.terms if storage else ()
        elif isinstance(record, DeviceProposal) and kind in (None, DeviceProposal):
            storage = self._proposals.get(record)
            terms = storage.terms if storage else ()
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
                objectives.broadcast(self.ops, self._base, p.data.n_rows)
                for p in (self._train, self._validation)
            )
            loss, score = (
                self._loss(p, r) for p, r in zip((self._train, self._validation), raw, strict=True)
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
        if self.objective.gradient is None:
            raise ValueError("objective has no scalar gradient convenience; use public geometry")
        return self.objective.gradient(self.ops, self._train, self._get(state, DeviceState).raw[0])

    def fields(self, state):
        if self.objective.fields is None:
            raise ValueError("objective has no scalar fields convenience; compose public fields")
        return self.objective.fields(self.ops, self._train, self._get(state, DeviceState).raw[0])

    def _loss(self, problem, raw):
        value = float(self.objective.loss(self.ops, problem, raw))
        if not np.isfinite(value):
            raise ValueError("finite objective loss required")
        return value

    def validate_terms(self, terms):
        """Structural validation outside numerical search; no candidate arrays allocated."""
        self._check()
        terms = tuple(terms)
        if not terms or any(not isinstance(term, DeviceTerm) for term in terms):
            raise ValueError("nonempty DeviceTerm tuple required")
        for term in terms:
            self.ops._get(term.tree, trees.DeviceTree)
            if term.mapping.shape != (1, self.raw_width):
                raise ValueError("term mapping width differs from run")
            if term.tree.binning.identity != self.binning.identity:
                raise ValueError("term fitted binning identity differs from run")
        return terms

    def propose(self, state, tree, *, coefficient=1.0):
        """Scalar convenience, using the same mapped transaction as other algorithms."""
        return self.propose_terms(state, (DeviceTerm(tree, [[1]]),), coefficient=coefficient)

    def propose_terms(self, state, terms, *, coefficient=1.0):
        """Snapshot all terms; atomically evaluate their ordered contribution on both sets."""
        prior = self._get(state, DeviceState)
        coefficient = _coefficient(coefficient)
        terms = self.validate_terms(terms)
        with _workspace(self.ops) as retained:
            snapshots = tuple(
                _Term(trees.copy(self.ops, t.tree), t.mapping, float(coefficient)) for t in terms
            )
            raw = []
            for p, previous in zip((self._train, self._validation), prior.raw, strict=True):
                candidate = previous
                for term in snapshots:
                    prediction = trees.predict(self.ops, term.tree, p.data)
                    candidate = map_update(
                        self.ops, candidate, prediction, term.mapping, coefficient
                    )
                raw.append(candidate)
            raw = tuple(raw)
            loss, score = (
                self._loss(p, r) for p, r in zip((self._train, self._validation), raw, strict=True)
            )
            retained.update((*raw, *(t.tree for t in snapshots)))
        proposal = DeviceProposal(
            self._identity("proposal"), state.identity, float(coefficient), loss, score
        )
        self._proposals[proposal] = _ProposalStorage(state, raw, snapshots)
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
        terms = (*prior.terms, *candidate.terms)
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
            TreeTerm(trees.export(self.ops, term.tree), term.mapping, term.coefficient)
            for term in storage.terms[:n_terms]
        )
        return Model(self.binning.feature_names, self.execution.export(self._base), terms)

    def release(self, record):
        """Release one state's/proposal's storage without invalidating other records."""
        storage = self._get(record)
        for handle in storage.raw:
            self.execution.release(handle)
        terms = storage.terms
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
            terms = storage.terms
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
