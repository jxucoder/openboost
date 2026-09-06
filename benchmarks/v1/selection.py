"""Independent validation selection and sealed test-feature release.

Protocol and receipt digests must be held by the trusted orchestrator, separately
from producer artifacts. This API orders access; it is not an OS sandbox.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from benchmarks.v1.judge import read_json
from benchmarks.v1.preprocessing import fit_target_scale
from benchmarks.v1.quality import metrics
from benchmarks.v1.quality_report import PRIMARY, load


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _fields(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise ValueError("invalid object fields")


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("invalid hash")


def _bytes(root, entry):
    _fields(entry, "path sha256")
    _hash(entry["sha256"])
    rel = Path(entry["path"])
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError("unsafe artifact path")
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError("artifact escapes root")
    raw = path.read_bytes()
    if not raw or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("artifact hash mismatch or empty artifact")
    return raw


def _ids(arrays):
    ids = arrays["row_ids"]
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("invalid row IDs")
    if ids.dtype.kind not in "iuUS":
        raise ValueError("integer or string row IDs required")
    return ids


def _disjoint(a, b):
    if a.dtype.kind != b.dtype.kind:
        raise ValueError("inconsistent row ID types")
    if np.intersect1d(a, b).size:
        raise ValueError("partition row overlap")


def audit(protocol, records, directory, pinned_protocol_sha256):
    """Recompute all validation scores; never open the test-feature artifact."""
    _hash(pinned_protocol_sha256)
    if digest(protocol) != pinned_protocol_sha256:
        raise ValueError("changed protocol")
    _fields(
        protocol,
        "schema application fold identity train_rows validation test_features selection_weights methods"
        + (" train_targets target_scale" if protocol.get("application") == "A6" else ""),
    )
    if protocol["schema"] != "openboost-selection-v1":
        raise ValueError("unknown selection protocol")
    app = protocol["application"]
    if (
        app not in {*PRIMARY, "A6"}
        or type(protocol["fold"]) is not int
        or protocol["fold"] not in range(5)
    ):
        raise ValueError("invalid application/fold")
    _fields(protocol["identity"], "code data split preprocessing environment search_design")
    for h in protocol["identity"].values():
        _hash(h)
    # Validate its descriptor without reading its contents before selection.
    _fields(protocol["test_features"], "path sha256")
    _hash(protocol["test_features"]["sha256"])
    root = Path(directory).resolve()
    training = load(root, protocol["train_rows"])
    _fields(training, "row_ids")
    train_ids = _ids(training)
    truth = load(root, protocol["validation"])
    if not {"row_ids", "y"} <= set(truth) or set(truth) - {
        "row_ids",
        "y",
        "weight",
        "event",
        "query",
    }:
        raise ValueError("invalid validation truth")
    ids = _ids(truth)
    if len(ids) != len(truth["y"]):
        raise ValueError("validation row mismatch")
    _disjoint(train_ids, ids)
    if app == "A6":
        if truth["y"].ndim != 2 or not truth["y"].shape[1]:
            raise ValueError("vector targets required")
        primary = [f"rmse_{k}" for k in range(truth["y"].shape[1])]
    else:
        primary = PRIMARY[app]
    weights = protocol["selection_weights"]
    if not isinstance(weights, dict) or set(weights) != set(primary):
        raise ValueError("all primary metrics required for selection")
    if any(type(w) not in (int, float) or not np.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("positive finite selection weights required")
    if app == "A6":
        targets = load(root, protocol["train_targets"])
        _fields(targets, "row_ids y")
        if not np.array_equal(_ids(targets), train_ids):
            raise ValueError("training target row mismatch")
        if targets["y"].shape != (len(train_ids), len(primary)):
            raise ValueError("training target shape mismatch")
        frozen_scale = read_json(_bytes(root, protocol["target_scale"]))
        expected_scale = fit_target_scale(targets["y"])
        if digest(frozen_scale) != digest(expected_scale):
            raise ValueError("target scale differs from training population")
        expected_weights = {
            key: 1.0 / std for key, std in zip(primary, expected_scale["std"], strict=True)
        }
        if weights != expected_weights:
            raise ValueError("selection weights differ from training scale")
    methods = protocol["methods"]
    if not isinstance(methods, dict) or not methods:
        raise ValueError("missing methods")
    expected = {}
    for method, configs in methods.items():
        if not isinstance(method, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", method):
            raise ValueError("invalid method ID")
        if (
            not isinstance(configs, list)
            or len(configs) != 16
            or any(not isinstance(c, dict) or not c for c in configs)
        ):
            raise ValueError("exactly 16 configurations per method required")
        if len({digest(c) for c in configs}) != 16:
            raise ValueError("duplicate configurations")
        expected.update({f"{method}:{i:02}": config for i, config in enumerate(configs)})
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("missing search trials")
    scores, artifacts = {}, {}
    for record in records:
        _fields(record, "id config status exit_code protocol_sha256 prediction model log")
        trial = record["id"]
        if trial not in expected or trial in scores:
            raise ValueError("unknown or duplicate trial")
        if (
            digest(record["config"]) != digest(expected[trial])
            or record["protocol_sha256"] != pinned_protocol_sha256
        ):
            raise ValueError("changed trial configuration/protocol")
        if (
            record["status"] != "pass"
            or type(record["exit_code"]) is not int
            or record["exit_code"] != 0
        ):
            raise ValueError("failed search trial")
        _bytes(root, record["log"])
        _bytes(root, record["model"])
        prediction = load(root, record["prediction"])
        _fields(prediction, "row_ids prediction")
        if not np.array_equal(ids, prediction["row_ids"]):
            raise ValueError("validation prediction row mismatch")
        measured = metrics(
            app,
            truth["y"],
            prediction["prediction"],
            row_ids=ids,
            **{k: v for k, v in truth.items() if k not in ["row_ids", "y"]},
        )
        # A6 is a mean of standardized errors, not normalized inverse-scale weights.
        denominator = len(primary) if app == "A6" else sum(weights.values())
        value = sum(weights[k] * measured[k] for k in primary) / denominator
        if not np.isfinite(value):
            raise ValueError("invalid selection score")
        scores[trial] = {"metrics": measured, "selection": value}
        artifacts[trial] = {k: record[k] for k in ["model", "prediction", "log"]}
    direction = -1 if app == "A4" else 1
    selected = min(scores, key=lambda t: (direction * scores[t]["selection"], t))
    return dict(
        schema="openboost-selection-receipt-v1",
        protocol_sha256=pinned_protocol_sha256,
        records_sha256=digest(sorted(records, key=lambda r: r["id"])),
        selected=selected,
        scores=scores,
        artifacts=artifacts,
    )


def seal(receipt, path):
    """Write once; the orchestrator retains the returned digest independently."""
    raw = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with Path(path).open("xb") as f:
        f.write(raw)
        f.flush()
        __import__("os").fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()


def release_test(
    protocol, records, receipt_path, directory, pinned_protocol_sha256, pinned_receipt_sha256
):
    """Return test features and the selected model descriptor only after re-audit.

    Does not deserialize models or read test labels. External workers must verify
    the returned model hash when loading and enforce process access restrictions.
    """
    _hash(pinned_receipt_sha256)
    raw = Path(receipt_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pinned_receipt_sha256:
        raise ValueError("changed receipt")
    receipt = read_json(raw)
    fresh = audit(protocol, records, directory, pinned_protocol_sha256)
    if digest(receipt) != digest(fresh):
        raise ValueError("receipt does not match independent selection")
    root = Path(directory).resolve()
    features = load(root, protocol["test_features"])
    if not {"row_ids", "x"} <= set(features) or set(features) - {
        "row_ids",
        "x",
        "exposure",
        "query",
    }:
        raise ValueError("test features must exclude targets")
    ids = _ids(features)
    if (
        features["x"].ndim != 2
        or len(features["x"]) != len(ids)
        or not np.isfinite(features["x"]).all()
    ):
        raise ValueError("finite encoded test inputs required")
    for field in ["exposure", "query"]:
        if field in features and features[field].shape != ids.shape:
            raise ValueError("unaligned test metadata")
    if "exposure" in features and (
        not np.isfinite(features["exposure"]).all() or np.any(features["exposure"] <= 0)
    ):
        raise ValueError("invalid test exposure")
    for partition in ["train_rows", "validation"]:
        _disjoint(ids, _ids(load(root, protocol[partition])))
    return features, fresh["artifacts"][fresh["selected"]]["model"]
