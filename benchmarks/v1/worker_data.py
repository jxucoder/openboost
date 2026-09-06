"""Bind frozen Housing, Parkinsons and Concrete folds to numeric worker packets.

Run as an evaluation-side preparer. Separate files are not an OS access boundary.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.v1 import housing, real_data
from benchmarks.v1.preprocessing import fit_encoder, fit_target_scale, transform

NAMES = {"A1": "housing", "A11": "housing", "A6": "parkinsons", "A12": "concrete"}
PARTS = ("train", "validation", "test")


def bind(application, data, parts, frozen):
    """Check source-aligned partitions and training metadata before assembling arrays."""
    if application not in NAMES or set(parts) != set(PARTS):
        raise ValueError("unsupported application or partitions")
    x, y = data["x"], data["y"]
    if len(x) != len(y) or not np.isfinite(y).all():
        raise ValueError("invalid targets")
    for rows in parts.values():
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows):
            raise ValueError("nonempty integer row indices required")
    joined = np.concatenate(list(parts.values()))
    if not np.array_equal(np.sort(joined), np.arange(len(x))):
        raise ValueError("partitions must cover each source row exactly once")
    train = parts["train"]
    if "group" in data:
        groups = [set(data["group"][parts[p]].tolist()) for p in PARTS]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("group crosses partitions")
    if fit_encoder(x[train]) != frozen["encoder"]:
        raise ValueError("training encoder differs from freeze")
    metadata = {}
    if application == "A6":
        metadata["target_scale"] = fit_target_scale(y[train])
        if metadata["target_scale"] != frozen["target_scale"]:
            raise ValueError("training target scale differs from freeze")
    encoded = {}
    for part, rows in parts.items():
        result = transform(frozen["encoder"], x[rows])
        expected = frozen["partitions"][part]
        if (
            real_data.array_hash(rows) != expected["rows_sha256"]
            or real_data.array_hash(result) != expected["x_sha256"]
            or list(result.shape) != expected["shape"]
        ):
            raise ValueError("partition differs from preprocessing freeze")
        encoded[part] = result
    if application == "A12":
        age = data["structure"]
        if age.shape != (len(x),) or not np.isfinite(age).all() or np.any(age <= 0):
            raise ValueError("positive finite structure age required")
        # Ordinary GBDT receives age as a feature; formula learners have a separate packet.
        encoded = {p: np.column_stack([v, age[parts[p]]]) for p, v in encoded.items()}
        metadata["age_train_range"] = [float(age[train].min()), float(age[train].max())]
    worker = dict(
        x_train=encoded["train"],
        y_train=y[train],
        x_validation=encoded["validation"],
        y_validation=y[parts["validation"]],
        validation_row_ids=parts["validation"],
    )
    packets = {
        "worker-input": worker,
        "train-rows": {"row_ids": train},
        "validation": {"row_ids": parts["validation"], "y": y[parts["validation"]]},
        "test-features": {"row_ids": parts["test"], "x": encoded["test"]},
        "test-truth": {"row_ids": parts["test"], "y": y[parts["test"]]},
    }
    if application == "A12":
        for part in ("validation", "test"):
            packets[part + "-structure"] = dict(
                row_ids=parts[part],
                age=age[parts[part]],
                train_min=np.asarray(metadata["age_train_range"][0]),
                train_max=np.asarray(metadata["age_train_range"][1]),
            )
    return packets, metadata


def export(
    application,
    directory,
    *,
    root=Path("build/v1-data"),
    housing_path=Path("build/foundation_data/cal_housing.tgz"),
):
    if application not in NAMES:
        raise ValueError("unsupported application")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("fresh output directory required")
    source_dir = Path(__file__).parent / "datasets"
    name = NAMES[application]
    prep_path = source_dir / "preprocessing.json"
    prep = json.loads(prep_path.read_text())
    # Refuse to silently reinterpret a freeze with changed reader/preprocessing code.
    for filename, digest in prep["source_files"].items():
        if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen source implementation changed")
    source_path = source_dir / (name + ".json")
    source = json.loads(source_path.read_text())
    if name == "housing":
        x, y, _ = housing.load_archive(housing_path)
        data = dict(x=x, y=y)
        splits = [housing.split_indices(len(y), s) for s in range(5)]
        if housing.digest(x.tobytes() + y.tobytes()) != source["arrays_sha256"]:
            raise ValueError("source targets/features differ from freeze")
    else:
        data, _ = real_data.load(root, name)
        for field, expected in source["arrays"].items():
            if real_data.array_hash(data[field]) != expected["sha256"]:
                raise ValueError("source arrays differ from freeze")
        splits = [real_data.group_splits(data["group"], s) for s in range(5)]
    records = []
    # Validate every fold before writing any worker input.
    prepared = [
        bind(application, data, dict(zip(PARTS, split, strict=True)), prep["datasets"][name][s])
        for s, split in enumerate(splits)
    ]
    for seed, (packets, metadata) in enumerate(prepared):
        folder = directory / str(seed)
        folder.mkdir()
        artifacts = {}
        for key, arrays in packets.items():
            path = folder / (key + ".npz")
            np.savez(path, **arrays)
            artifacts[key] = dict(
                path=str(path.relative_to(directory)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        records.append(dict(seed=seed, artifacts=artifacts, metadata=metadata))
    result = dict(
        application=application,
        dataset=name,
        folds=records,
        source_freeze_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        preprocessing_freeze_sha256=hashlib.sha256(prep_path.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Five frozen folds; validation worker input includes early-stopping labels. Test files require evaluation-side custody, not enforced by this exporter.",
    )
    (directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("application", choices=NAMES)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    export(args.application, args.directory)


if __name__ == "__main__":
    main()
