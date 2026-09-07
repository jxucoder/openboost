"""Fail-closed evidence integrity checks; no statistical or E-gate evaluator."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "openboost-integrity-v0"
STATUSES = {"not_run", "pass", "fail", "unsupported", "error", "timeout"}
APPLICATIONS = {f"A{i}" for i in range(1, 14)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(value, fields):
    _require(isinstance(value, dict) and set(value) == set(fields.split()), "invalid object fields")


def _text(value):
    _require(isinstance(value, str) and bool(value.strip()), "expected nonempty text")


def _hash(value, length=64):
    _require(
        isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value), "invalid hash"
    )


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def cache_key(manifest, cell):
    """Bind the full declared provenance, protocol, expected matrix and cell inputs."""
    return hashlib.sha256(_canonical({"manifest": manifest, "cell": cell})).hexdigest()


def _manifest(manifest):
    _object(manifest, "schema protocol_sha256 provenance expected")
    _require(manifest["schema"] == SCHEMA, "unknown schema")
    _hash(manifest["protocol_sha256"])
    provenance = manifest["provenance"]
    _object(provenance, "code_sha dirty environment")
    _hash(provenance["code_sha"], 40)
    _require(type(provenance["dirty"]) is bool, "dirty must be boolean")
    # Dirty trees need a patch digest before their cache identity is trustworthy.
    _require(not provenance["dirty"], "dirty code unsupported by integrity-v0")
    _require(
        isinstance(provenance["environment"], dict) and provenance["environment"],
        "missing environment",
    )
    cells = manifest["expected"]
    _require(isinstance(cells, list) and cells, "empty expected matrix")
    seen, applications = set(), set()
    for cell in cells:
        _object(
            cell,
            "id application required backend dataset_sha256 split_sha256 preprocessing_sha256 config seed model fold",
        )
        for field in ("id", "model", "fold"):
            _text(cell[field])
        _require(cell["id"] not in seen, "duplicate expected case")
        seen.add(cell["id"])
        _require(cell["application"] in APPLICATIONS, "unknown application")
        _require(type(cell["required"]) is bool, "required must be boolean")
        _require(cell["backend"] in {"cpu", "cuda"}, "unknown backend")
        if cell["required"] and cell["backend"] == "cpu":
            applications.add(cell["application"])
        for field in ("dataset_sha256", "split_sha256", "preprocessing_sha256"):
            _hash(cell[field])
        _require(type(cell["seed"]) is int and cell["seed"] >= 0, "invalid seed")
        _require(isinstance(cell["config"], dict), "config must be an object")
    _require(applications == APPLICATIONS, "each A1-A13 needs a required CPU cell")
    _canonical(manifest)  # disallow non-finite nested config/provenance values
    return {cell["id"]: cell for cell in cells}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def read_json(data):
    value = json.loads(data, object_pairs_hook=_pairs)
    _canonical(value)  # also rejects JSON NaN/Infinity and overflowing numeric literals
    return value


def _artifact(root, entry):
    _object(entry, "path sha256")
    _text(entry["path"])
    _hash(entry["sha256"])
    relative = Path(entry["path"])
    _require(not relative.is_absolute() and ".." not in relative.parts, "unsafe artifact path")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root), "artifact escapes run directory")
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == entry["sha256"], "artifact hash mismatch")
    return data


def _predictions(data):
    values = read_json(data)
    _require(isinstance(values, list) and values, "predictions must be a nonempty JSON array")

    def shape(value):
        if isinstance(value, list):
            _require(bool(value), "empty prediction dimension")
            shapes = [shape(v) for v in value]
            _require(all(s == shapes[0] for s in shapes), "ragged predictions")
            return (len(value), *shapes[0])
        _require(type(value) in (int, float) and math.isfinite(value), "invalid prediction value")
        return ()

    shape(values)


def _case(record, cell, manifest, root):
    _object(record, "id status cache_key backend fallback exit_code artifacts metrics reason")
    _require(record["status"] in STATUSES, "unknown status")
    _require(record["cache_key"] == cache_key(manifest, cell), "cache identity mismatch")
    _require(record["backend"] in {"cpu", "cuda", "none"}, "unknown actual backend")
    _require(type(record["fallback"]) is bool, "fallback must be boolean")
    _require(type(record["exit_code"]) is int or record["exit_code"] is None, "invalid exit code")
    _require(isinstance(record["reason"], str), "invalid reason")
    _require(isinstance(record["metrics"], dict), "invalid metrics")
    for name, value in record["metrics"].items():
        _text(name)
        _require(type(value) in (int, float) and math.isfinite(value), "non-finite/invalid metric")
    artifacts = record["artifacts"]
    _require(isinstance(artifacts, dict), "invalid artifacts")
    _require(
        set(artifacts) <= {"predictions", "model", "log"} and "log" in artifacts,
        "missing log or unknown artifact role",
    )
    payloads = {role: _artifact(root, entry) for role, entry in artifacts.items()}
    if "predictions" in payloads:
        _predictions(payloads["predictions"])
    if record["status"] == "pass":
        _require(record["exit_code"] == 0, "worker did not exit successfully")
        _require(
            record["backend"] == cell["backend"] and not record["fallback"],
            "backend mismatch or fallback",
        )
        _require(set(artifacts) == {"predictions", "model", "log"}, "missing raw predictions/model")
        _require(bool(record["metrics"]), "missing metrics")
    else:
        _text(record["reason"])
        _require(not cell["required"], f"required status is {record['status']}")


def judge(manifest, records, directory, *, frozen_manifest=None):
    """Check integrity against declared cells and, optionally, an evaluator-owned freeze.

    The caller must obtain frozen_manifest independently of producer output. An
    exact match binds provenance and the entire expected matrix, not gate validity.
    """
    report = {
        "schema": SCHEMA,
        "integrity_pass": False,
        "errors": [],
        "statuses": {},
        "gate_results": {},
        "frozen_manifest_match": None,
        "frozen_manifest_sha256": None,
        "scope": "artifact integrity only; quality and E0-E7 not evaluated",
    }
    errors = report["errors"]
    try:
        if frozen_manifest is not None:
            report["frozen_manifest_match"] = False
            _manifest(frozen_manifest)
            frozen_bytes = _canonical(frozen_manifest)
            report["frozen_manifest_sha256"] = hashlib.sha256(frozen_bytes).hexdigest()
            _require(
                _canonical(manifest) == frozen_bytes,
                "producer manifest differs from evaluator frozen manifest",
            )
            report["frozen_manifest_match"] = True
        expected = _manifest(manifest)
        _require(isinstance(records, list), "cases must be a list")
        root = Path(directory).resolve()
        seen = set()
        for record in records:
            try:
                _require(isinstance(record, dict), "case must be an object")
                case_id = record.get("id")
                _text(case_id)
                _require(case_id in expected, f"unknown case: {case_id}")
                _require(case_id not in seen, f"duplicate case: {case_id}")
                seen.add(case_id)
                report["statuses"][case_id] = record.get("status")
                _case(record, expected[case_id], manifest, root)
            except (ValueError, TypeError, OSError, OverflowError) as exc:
                errors.append(
                    f"case {record.get('id') if isinstance(record, dict) else '?'}: {exc}"
                )
        errors.extend(f"missing case: {case_id}" for case_id in sorted(expected.keys() - seen))
    except (ValueError, TypeError, OSError, OverflowError) as exc:
        errors.append(f"manifest: {exc}")
    report["integrity_pass"] = not errors
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        help="Evaluator-owned execution manifest, outside producer run directory",
    )
    parser.add_argument("--frozen-sha256", help="Evaluator-pinned SHA256 of frozen file bytes")
    args = parser.parse_args()
    try:
        frozen = None
        _require(
            bool(args.frozen_manifest) == bool(args.frozen_sha256),
            "frozen manifest and pinned SHA256 must be supplied together",
        )
        if args.frozen_manifest is not None:
            _hash(args.frozen_sha256)
            _require(
                not args.frozen_manifest.resolve().is_relative_to(args.directory.resolve()),
                "frozen manifest must be outside producer run directory",
            )
            payload = args.frozen_manifest.read_bytes()
            _require(
                hashlib.sha256(payload).hexdigest() == args.frozen_sha256,
                "frozen manifest file hash mismatch",
            )
            frozen = read_json(payload)
        manifest = read_json((args.directory / "manifest.json").read_bytes())
        records = [
            read_json(line) for line in (args.directory / "cases.jsonl").read_bytes().splitlines()
        ]
        report = judge(manifest, records, args.directory, frozen_manifest=frozen)
    except (ValueError, TypeError, OSError, OverflowError) as exc:
        report = {
            "schema": SCHEMA,
            "integrity_pass": False,
            "errors": [str(exc)],
            "gate_results": {},
        }
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["integrity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
