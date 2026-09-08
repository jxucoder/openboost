"""Standalone trusted D1/D2 development judge; never imports author extensions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    if path.is_symlink() or not path.is_file():
        raise ValueError("JSON input must be a regular file")
    return json.loads(path.read_text(), parse_constant=invalid, object_pairs_hook=pairs)


def compare(actual, expected, path="result"):
    """Require exact structure and finite values; never broadcast missing rows."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{path}: fields differ")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path}: length differs")
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            compare(a, e, f"{path}[{index}]")
    elif expected is None:
        if actual is not None:
            raise ValueError(f"{path}: expected a leaf")
    elif isinstance(expected, bool):
        if actual is not expected:
            raise ValueError(f"{path}: rejection observation differs")
    elif (
        isinstance(actual, bool)
        or not isinstance(actual, (float, int))
        or not np.isfinite(actual)
        or not np.isclose(actual, expected, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError(f"{path}: numerical mismatch")


def verify_bundle(bundle):
    manifest = read_json(bundle / "manifest.json")
    if manifest["schema"] != "openboost-d1-d2-development-verifier-v2":
        raise ValueError("unknown verifier schema")
    files = manifest["runtime_files"]
    if set(files) != {"judge.py", "inputs.json", "expected.json"}:
        raise ValueError("incomplete verifier closure")
    for name, digest in files.items():
        path = bundle / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"verifier input changed: {name}")


def judge(bundle, observations, task):
    from openboost import NumericData
    from openboost.artifacts import Model

    bundle, observations = Path(bundle).resolve(), Path(observations).resolve()
    verify_bundle(bundle)
    cases = [c for c in read_json(bundle / "inputs.json") if c["task"] == task]
    if not cases:
        raise ValueError("unknown or empty task")
    expected = read_json(bundle / "expected.json")
    actual = read_json(observations / f"{task}.json")
    compare(actual, {c["id"]: expected[c["id"]] for c in cases})
    models = {}
    for case in cases:
        path = observations / f"{case['id']}.model.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("model must be a regular file")
        data = NumericData(
            np.asarray(case["values"], dtype=float), np.arange(len(case["values"])), ("x",)
        )
        model = Model.load(path)
        prediction = model.predict(data)
        if prediction.shape != (len(case["values"]), 1):
            raise ValueError(f"{case['id']}.saved_model: output shape differs")
        compare(
            prediction[:, 0].tolist(),
            expected[case["id"]]["raw"],
            f"{case['id']}.saved_model",
        )
        models[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(
        passed=True,
        scope="development numerical and saved-model checks only",
        task=task,
        cases=len(cases),
        models=models,
        observations_sha256=hashlib.sha256(
            (observations / f"{task}.json").read_bytes()
        ).hexdigest(),
        verifier_manifest_sha256=hashlib.sha256(
            (bundle / "manifest.json").read_bytes()
        ).hexdigest(),
        dispatch_ready=False,
        attempts=[],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("task", choices=("D1", "D2"))
    args = parser.parse_args()
    print(json.dumps(judge(args.bundle, args.observations, args.task), indent=2))
