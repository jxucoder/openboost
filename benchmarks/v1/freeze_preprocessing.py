"""Freeze train-only numeric encoding and support metadata for available real tasks."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmarks.v1 import adult, bike, housing, real_data
from benchmarks.v1.preprocessing import censoring_support, fit_encoder, fit_target_scale, transform


def prepare(name, data, folds, categories=None):
    categories = categories or {}
    records = []
    for seed, parts in enumerate(folds):
        train = parts["train"]
        enc = fit_encoder(data["x"][train], {k: v[train] for k, v in categories.items()})
        record = {"seed": seed, "encoder": enc, "partitions": {}}
        if name == "parkinsons":
            record["target_scale"] = fit_target_scale(data["y"][train])
        if name == "veteran":
            record["censoring_support"] = censoring_support(data["y"][train], data["event"][train])
        for part, rows in parts.items():
            result = transform(enc, data["x"][rows], {k: v[rows] for k, v in categories.items()})
            record["partitions"][part] = {
                "shape": list(result.shape),
                "x_sha256": real_data.array_hash(result),
                "rows_sha256": real_data.array_hash(rows),
            }
        records.append(record)
    return records


def describe(root, adult_path, bike_path, housing_path):
    result = {}
    for name in ["covertype", "parkinsons", "concrete", "veteran", "insurance"]:
        if name == "insurance":
            data, _ = real_data.insurance(root)
            rule = "group"
        else:
            data, rule = real_data.load(root, name)
        categories = {k[9:]: v for k, v in data.items() if k.startswith("category_")}
        if name == "veteran":
            # Restore source-declared categories before one-hot encoding; no ordinal assumption.
            for col, key, labels in [
                (0, "Treatment", ["standard", "test"]),
                (1, "Celltype", ["adeno", "large", "smallcell", "squamous"]),
                (5, "Prior_therapy", ["no", "yes"]),
            ]:
                categories[key] = np.asarray(labels)[data["x"][:, col].astype(int)]
            data["x"] = data["x"][:, 2:5]
        folds = []
        for seed in range(5):
            split = (
                real_data.group_splits(data["group"], seed)
                if rule == "group"
                else real_data.stratified_splits(
                    data["event"] if rule == "event" else data["y"], seed
                )
            )
            folds.append(dict(zip(["train", "validation", "test"], split, strict=True)))
        result[name] = prepare(name, data, folds, categories)
        if name == "insurance":
            for suffix, original_rows in [
                ("severity", data["severity_policy_row"]),
                ("aggregate", np.flatnonzero(data["aggregate_eligible"])),
            ]:
                filtered = {"x": data["x"][original_rows]}
                selected_folds = [
                    {
                        part: np.flatnonzero(np.isin(original_rows, rows))
                        for part, rows in fold.items()
                    }
                    for fold in folds
                ]
                selected_categories = {k: v[original_rows] for k, v in categories.items()}
                result["insurance_" + suffix] = prepare(
                    suffix, filtered, selected_folds, selected_categories
                )

    first, last = adult.load_archive(adult_path)
    n = len(first["y"])
    raw = np.asarray(first["x"] + last["x"], dtype=object)
    numeric = [i for i, c in enumerate(adult.FEATURES) if c in adult.NUMERIC]
    cats = {c: raw[:, i] for i, c in enumerate(adult.FEATURES) if c not in adult.NUMERIC}
    data = {"x": raw[:, numeric].astype(float)}
    folds = []
    for seed in range(5):
        train, val = adult.stratified_split(first["y"], seed)
        folds.append(
            {"train": train, "validation": val, "test": np.arange(n, len(raw), dtype="<i8")}
        )
    result["adult"] = prepare("adult", data, folds, cats)
    b = bike.load_archive(bike_path)
    result["bike"] = prepare("bike", {"x": b.x}, bike.rolling_splits(b.dates))
    x, _, _ = housing.load_archive(housing_path)
    result["housing"] = prepare(
        "housing",
        {"x": x},
        [
            dict(
                zip(["train", "validation", "test"], housing.split_indices(len(x), s), strict=True)
            )
            for s in range(5)
        ],
    )
    return {
        "schema": "openboost-preprocessing-freeze-v1",
        "datasets": result,
        "scope": "train-fitted dense controls, including separate A8 claim and A9 eligible-policy encoders; no model quality evaluation",
        "source_files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("preprocessing.py"),
                Path(__file__).with_name("real_data.py"),
                Path(__file__).with_name("adult.py"),
                Path(__file__).with_name("bike.py"),
                Path(__file__).with_name("housing.py"),
            ]
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("build/v1-data"))
    p.add_argument("--adult", type=Path, default=Path("/tmp/openboost-v1-adult.zip"))
    p.add_argument("--bike", type=Path, default=Path("/tmp/openboost-v1-bike.zip"))
    p.add_argument("--housing", type=Path, default=Path("build/foundation_data/cal_housing.tgz"))
    p.add_argument("--verify", type=Path)
    a = p.parse_args()
    r = describe(a.root, a.adult, a.bike, a.housing)
    if a.verify:
        expected = json.loads(a.verify.read_text())
        expected.pop("provenance")
        if r != expected:
            raise ValueError("preprocessing freeze mismatch")
    r["provenance"] = {
        "argv": [sys.executable, "-m", "benchmarks.v1.freeze_preprocessing", *sys.argv[1:]],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
    }
    print(json.dumps(r, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
