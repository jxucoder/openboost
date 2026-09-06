"""Typed content identity and sequential run oracle; no production scheduler/cache."""

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .tree import fit_tree, numeric_bins


def _canonical(value):
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", value.hex() if not math.isnan(value) else "nan"]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (list, tuple)):
        return ["sequence", [_canonical(v) for v in value]]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return ["mapping", [[k, _canonical(value[k])] for k in sorted(value)]]
    raise ValueError("identity only accepts explicitly typed primitive data")


def _digest(value):
    payload = json.dumps(_canonical(value), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _ids(values):
    ids = tuple(values)
    if (
        not ids
        or any(not isinstance(i, Integral) or isinstance(i, (bool, np.bool_)) for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("row IDs must be nonempty unique integers")
    return ids


@dataclass(frozen=True)
class DataIdentity:
    digest: str
    row_ids: tuple


def data_identity(row_ids, values, schema, transformer):
    ids = _ids(row_ids)
    matrix = np.asarray(values, dtype=object)
    schema = tuple(schema)
    if matrix.ndim != 2 or matrix.shape != (len(ids), len(schema)) or not schema:
        raise ValueError("data shape must match row IDs and schema")
    if any(not isinstance(s, str) for s in schema) or len(set(schema)) != len(schema):
        raise ValueError("feature names must be unique strings")
    return DataIdentity(_digest(("data-v1-reference", ids, matrix, schema, transformer)), ids)


def bind_identity(prepared, row_ids, **fields):
    ids = _ids(row_ids)
    if not isinstance(prepared, DataIdentity) or ids != prepared.row_ids or "target" not in fields:
        raise ValueError("binding requires prepared row IDs and a target role")
    payload = {}
    for name, (field_ids, values) in fields.items():
        if _ids(field_ids) != ids or len(values) != len(ids):
            raise ValueError(f"{name} row IDs or length do not match prepared order")
        payload[name] = values
    # This hashes role contents; objective-specific support validation is separate.
    return _digest(("problem-v1-reference", prepared.digest, ids, payload))


def derive_seed(seed, run_id, round_index, component, purpose):
    if any(
        not isinstance(v, Integral) or isinstance(v, bool) or v < 0 for v in (seed, round_index)
    ):
        raise ValueError("seed and round must be nonnegative integers")
    if any(not isinstance(v, str) or not v for v in (run_id, component, purpose)):
        raise ValueError("RNG identifiers must be nonempty strings")
    return int(
        _digest(("rng-v1-reference", int(seed), run_id, int(round_index), component, purpose))[:16],
        16,
    )


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    seed: int
    budget: int
    learning_rate: float
    patience: int
    sample_count: int
    fail_round: int | None = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: str
    rounds: int
    best_round: int
    best_loss: float
    base: tuple
    best_raw: tuple
    best_terms: tuple
    samples: tuple
    learning_rate: float
    problem_id: str
    error: str | None


def _target(values, length):
    y = np.asarray(values, dtype=float)
    if y.ndim != 2 or y.shape[0] != length or y.shape[1] < 1 or not np.all(np.isfinite(y)):
        raise ValueError("target must be finite [N,K]")
    return y.copy()


def _snapshot(values):
    return tuple(tuple(row) for row in values)


def _one(spec, train, validation, problems):
    rounds, best_round, best_loss = 0, 0, float("inf")
    base, best_raw, best_terms, samples = (), (), (), []
    problem_id = ""
    status, error = "completed", None
    try:
        for name, value, minimum in (
            ("budget", spec.budget, 0),
            ("patience", spec.patience, 1),
            ("sample_count", spec.sample_count, 1),
        ):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"invalid {name}")
        if spec.fail_round is not None and (
            not isinstance(spec.fail_round, Integral)
            or isinstance(spec.fail_round, bool)
            or not 1 <= spec.fail_round <= spec.budget
        ):
            raise ValueError("fail_round must identify a budgeted round")
        if (
            spec.sample_count > len(train)
            or not np.isfinite(spec.learning_rate)
            or spec.learning_rate < 0
        ):
            raise ValueError("invalid sampling or learning rate")
        derive_seed(spec.seed, spec.run_id, 0, "tree", "rows")
        if spec.run_id not in problems:
            raise ValueError("missing run problem")
        target, valid_target = problems[spec.run_id]
        target, valid_target = _target(target, len(train)), _target(valid_target, len(validation))
        if target.shape[1] != valid_target.shape[1]:
            raise ValueError("validation output schema mismatch")
        identities = []
        for features, labels in ((train, target), (validation, valid_target)):
            rows = tuple(range(len(features)))
            prepared = data_identity(
                rows,
                features,
                [f"f{i}" for i in range(features.shape[1])],
                {"encoding": "fixed-bins-reference-v1"},
            )
            identities.append(bind_identity(prepared, rows, target=(rows, labels)))
        problem_id = _digest(("squared-run-problem", identities))
        base = tuple(np.mean(target, axis=0))
        raw = np.tile(base, (len(train), 1))
        valid_raw = np.tile(base, (len(validation), 1))
        best_loss = float(np.mean((valid_raw - valid_target) ** 2) / 2)
        if not np.isfinite(best_loss):
            raise ValueError("non-finite initial validation metric")
        best_raw = _snapshot(valid_raw)
        terms, stale = [], 0
        for iteration in range(1, spec.budget + 1):
            if iteration == spec.fail_round:
                raise RuntimeError(f"injected failure at round {iteration}")
            seed = derive_seed(spec.seed, spec.run_id, iteration, "tree", "rows")
            rows = tuple(
                sorted(
                    int(i)
                    for i in np.random.default_rng(seed).choice(
                        len(train), spec.sample_count, replace=False
                    )
                )
            )
            chosen = list(rows)
            gradient = raw - target
            trees = tuple(
                fit_tree(train[chosen], gradient[chosen, k], np.ones(len(chosen)), max_depth=1)
                for k in range(target.shape[1])
            )
            updated = raw + spec.learning_rate * np.column_stack([t.predict(train) for t in trees])
            valid_updated = valid_raw + spec.learning_rate * np.column_stack(
                [t.predict(validation) for t in trees]
            )
            score = float(np.mean((valid_updated - valid_target) ** 2) / 2)
            if not np.all(np.isfinite(updated)) or not np.isfinite(score):
                raise ValueError("non-finite candidate state")
            # All output channels commit together after candidate validation.
            raw, valid_raw = updated, valid_updated
            rounds = iteration
            samples.append(rows)
            terms.append(trees)
            if score < best_loss:
                best_loss, best_round, best_raw, best_terms = (
                    score,
                    iteration,
                    _snapshot(valid_raw),
                    tuple(terms),
                )
                stale = 0
            else:
                stale += 1
            if stale >= spec.patience:
                status = "early_stopped"
                break
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as failure:
        status, error = "failed", f"{type(failure).__name__}: {failure}"
    return RunResult(
        spec.run_id,
        status,
        rounds,
        best_round,
        best_loss,
        base,
        best_raw,
        best_terms,
        tuple(samples),
        spec.learning_rate,
        problem_id,
        error,
    )


def run_many(specs, train_bins, validation_bins, problems):
    specs = tuple(specs)
    ids = [s.run_id for s in specs]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("run IDs must be unique nonempty strings")
    train, validation = numeric_bins(train_bins), numeric_bins(validation_bins)
    if train.shape[1] != validation.shape[1]:
        raise ValueError("train/validation feature schema mismatch")
    return {spec.run_id: _one(spec, train, validation, problems) for spec in specs}


def select_best(records):
    candidates = [
        r
        for r in records.values()
        if r.status in ("completed", "early_stopped") and np.isfinite(r.best_loss)
    ]
    if not candidates:
        raise ValueError("no successful run can be selected")
    if len({r.problem_id for r in candidates}) != 1:
        raise ValueError("model selection requires the same problem and validation metric")
    return min(candidates, key=lambda r: (r.best_loss, r.run_id))
