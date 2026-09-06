"""Pinned A1/A11 Housing inputs; five seeded random partitions, no model training."""

import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

SOURCE = "https://ndownloader.figshare.com/files/5976036"
ARCHIVE_SHA256 = "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681"
MEMBER = "CaliforniaHousing/cal_housing.data"
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def transform(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 9 or not len(raw) or not np.all(np.isfinite(raw)):
        raise ValueError("raw must be nonempty finite [N,9]")
    if np.any(raw[:, 6] <= 0) or np.any(raw[:, 2:] < 0):
        raise ValueError("households must be positive; counts/income/value nonnegative")
    households = raw[:, 6]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        x = np.column_stack(
            (
                raw[:, 7],
                raw[:, 2],
                raw[:, 3] / households,
                raw[:, 4] / households,
                raw[:, 5],
                raw[:, 5] / households,
                raw[:, 1],
                raw[:, 0],
            )
        ).astype("<f4")
        y = (raw[:, 8] / 100000.0).astype("<f4")
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def load_archive(path):
    content = Path(path).read_bytes()
    if digest(content) != ARCHIVE_SHA256:
        raise ValueError("archive hash mismatch")
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
        member = archive.extractfile(MEMBER).read()
    raw = np.loadtxt(io.BytesIO(member), delimiter=",")
    x, y = transform(raw)
    return x, y, digest(member)


def split_indices(n_samples, seed):
    if type(n_samples) is not int or n_samples < 5:
        raise ValueError("n_samples must be an integer >= 5")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    order = np.random.default_rng(seed).permutation(n_samples).astype("<i8")
    return np.split(order, [n_samples * 3 // 5, n_samples * 4 // 5])


def describe(path):
    x, y, member_hash = load_archive(path)
    return {
        "schema": "openboost-housing-data-v1",
        "applications": ["A1", "A11"],
        "name": "California Housing",
        "source_url": SOURCE,
        "archive_sha256": ARCHIVE_SHA256,
        "member": MEMBER,
        "member_sha256": member_hash,
        "features": FEATURES,
        "shape": list(x.shape),
        "target_unit": "100,000 USD",
        "dtype": "little-endian float32",
        "arrays_sha256": digest(x.tobytes() + y.tobytes()),
        "row_ids": "zero-based source row positions; never model features",
        "array_hash_format": "SHA256(X C-order <f4 bytes + y C-order <f4 bytes); legacy-compatible",
        "split": "numpy.default_rng(seed), permutation, 60/20/20; integer floor endpoints",
        "split_hash_format": "SHA256(C-order <i8 source row positions); legacy-compatible",
        "split_sizes": [len(p) for p in split_indices(len(y), 0)],
        "split_sha256": {
            str(seed): [digest(p.tobytes()) for p in split_indices(len(y), seed)]
            for seed in range(5)
        },
        "license_status": "unresolved: archive has no license file; original source page unavailable during verification",
        "source_documentation": "https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset",
        "status": "data hashes and splits prepared; license, budgets and model quality not verified",
    }


def check_legacy(record, legacy):
    for field in ("archive_sha256", "features", "shape", "target_unit", "dtype", "arrays_sha256"):
        if record[field] != legacy[field]:
            raise ValueError(f"legacy mismatch: {field}")
    for seed in ("0", "1", "2"):
        if record["split_sha256"][seed] != legacy["split_sha256"][seed]:
            raise ValueError(f"legacy split mismatch: {seed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = describe(args.archive)
    root = Path(__file__).resolve().parents[2]
    check_legacy(result, json.loads((root / "benchmarks/foundation/housing.json").read_text()))
    if args.verify:
        frozen = json.loads(args.verify.read_text())
        provenance = frozen.pop("provenance")
        if frozen != result or provenance["adapter_sha256"] != digest(Path(__file__).read_bytes()):
            raise ValueError("frozen data/splits/source mismatch")
    result["provenance"] = {
        "adapter_sha256": digest(Path(__file__).read_bytes()),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.platform(),
        "argv": ["python", "-m", "benchmarks.v1.housing", *sys.argv[1:]],
    }
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
