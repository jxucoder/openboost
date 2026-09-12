"""Explicit resident feature encodings with independent per-problem bindings."""

from dataclasses import dataclass

import numpy as np

from .binning import BinnedData, Binning
from .data import Problem
from .device import DeviceData, _atomic
from .execution import DeviceBuffer


@dataclass(frozen=True, eq=False)
class DeviceFeatures:
    """Registered codes/missingness; no targets, offsets, weights or mutable run state.

    Release with ops.release(features) after all borrowers finish. Explicit
    release invalidates borrowed bindings, just like releasing a device buffer.
    The exact operations instance owns registration; copying metadata cannot
    forge another usable record. Binning is immutable host inference metadata.
    """

    data_identity: str
    prepared_identity: str
    binning: Binning
    n_rows: int
    codes: DeviceBuffer
    missing: DeviceBuffer


@_atomic
def prepare(ops, binned):
    """Upload one numeric feature encoding; later binds may change targets/weights."""
    if not isinstance(binned, BinnedData):
        raise ValueError("explicit BinnedData required")
    if any(k != "numeric" for k in binned.binning.feature_kinds):
        raise ValueError("resident feature sharing supports numeric features only")
    rows = len(binned.data.row_ids)
    if rows > np.iinfo(np.int32).max:
        raise ValueError("int32 row capacity exceeded")
    context = ops.execution
    codes, missing = (context.upload(v) for v in (binned.codes, binned.missing))
    record = ops._record(
        DeviceFeatures(binned.data.identity, binned.identity, binned.binning, rows, codes, missing),
        (codes, missing), (codes, missing),
    )
    context._counts["feature_prepare_calls"] = context._counts.get("feature_prepare_calls", 0) + 1
    context._counts["feature_upload_bytes"] = context._counts.get("feature_upload_bytes", 0) + codes.nbytes + missing.nbytes
    return record


def _match(features, problem):
    if not isinstance(features, DeviceFeatures) or not isinstance(problem, Problem):
        raise ValueError("DeviceFeatures and explicit Problem required")
    if features.data_identity != problem.data.identity:
        raise ValueError("prepared features differ in content, schema or row identity")


def check_pair(prepared, train, validation, *, binning=None, bins=254):
    """Host metadata checks only; bind separately verifies live device registration."""
    if not isinstance(prepared, tuple) or len(prepared) != 2:
        raise ValueError("explicit training/validation DeviceFeatures pair required")
    for features, problem in zip(prepared, (train, validation), strict=True):
        _match(features, problem)
    first, second = (f.binning for f in prepared)
    if not isinstance(first, Binning) or not isinstance(second, Binning) or first.identity != second.identity:
        raise ValueError("prepared training/validation binning differs")
    if bins != 254:
        raise ValueError("prepared features own their bin configuration")
    if binning is not None and (not isinstance(binning, Binning) or binning.identity != first.identity):
        raise ValueError("explicit binning differs from prepared features")
    return first


@_atomic
def bind(ops, features, problem):
    """Upload this problem's weights, borrowing codes/missingness without copies.

    Targets and offsets remain the objective's separate preparation boundary.
    Closing/releasing the returned DeviceData releases only its owned weights.
    """
    _match(features, problem)
    ops._get(features, DeviceFeatures)
    with np.errstate(over="raise", invalid="raise"):
        weight = problem.weight.astype(np.float32)
    if not np.isfinite(weight).all() or not np.any(weight > 0):
        raise ValueError("weights must retain finite positive mass in float32")
    weights = ops.execution.upload(weight)
    record = DeviceData(
        features.data_identity, features.prepared_identity, problem.identity,
        features.binning.identity, features.n_rows, features.binning.feature_names,
        features.binning.bin_counts, features.codes, features.missing, weights,
    )
    ops._record(record, (features.codes, features.missing, weights), (weights,))
    counts = ops.execution._counts
    counts["feature_bind_calls"] = counts.get("feature_bind_calls", 0) + 1
    counts["feature_bind_upload_bytes"] = counts.get("feature_bind_upload_bytes", 0) + weights.nbytes
    return record
