"""Pinned Adult data and official-test-preserving stratified evaluation splits."""

import argparse
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

SOURCE = "https://archive.ics.uci.edu/static/public/2/adult.zip"
ARCHIVE_SHA256 = "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"
MEMBERS = {
    "adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
    "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
}
COLUMNS = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
)
NUMERIC = {"age", "education-num", "capital-gain", "capital-loss", "hours-per-week"}
FEATURES = tuple(c for c in COLUMNS if c != "fnlwgt")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def parse_source(content, source):
    if source not in MEMBERS:
        raise ValueError("unknown official source")
    result = {"x": [], "y": [], "row_ids": []}
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if source == "adult.test" and line_number == 1 and line == "|1x3 Cross validator":
            continue
        values = [v.strip() for v in next(csv.reader([line]))]
        if len(values) != 15 or any(not v for v in values):
            raise ValueError("invalid Adult row schema")
        label = values[-1]
        if source == "adult.test":
            if not label.endswith("."):
                raise ValueError("official test label needs one trailing dot")
            label = label[:-1]
        if label not in ("<=50K", ">50K"):
            raise ValueError("invalid Adult label")
        row = []
        for name, value in zip(COLUMNS, values[:-1], strict=True):
            if name == "fnlwgt":
                continue  # never infer sample weight from a source column
            if value == "?":
                if name in NUMERIC:
                    raise ValueError("numeric missing value outside source contract")
                row.append(None)
            elif name in NUMERIC:
                number = int(value)
                if number < 0:
                    raise ValueError("negative numeric feature")
                row.append(number)
            else:
                row.append(value)
        result["x"].append(row)
        result["y"].append(int(label == ">50K"))
        result["row_ids"].append(f"{source}:{line_number}")
    if not result["y"]:
        raise ValueError("empty Adult source")
    return result


def load_archive(path):
    content = Path(path).read_bytes()
    if digest(content) != ARCHIVE_SHA256:
        raise ValueError("archive hash mismatch")
    result = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member, expected in MEMBERS.items():
            data = archive.read(member)
            if digest(data) != expected:
                raise ValueError("member hash mismatch")
            result[member] = parse_source(data, member)
    return result["adult.data"], result["adult.test"]


def stratified_split(labels, seed):
    labels = np.asarray(labels)
    if (
        labels.ndim != 1
        or labels.dtype.kind not in "iu"
        or set(labels.tolist()) != {0, 1}
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("binary integer labels and nonnegative integer seed required")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in (0, 1):
        rows = rng.permutation(np.flatnonzero(labels == label))
        end = len(rows) * 4 // 5
        if not 0 < end < len(rows):
            raise ValueError("each class needs train and validation rows")
        train.extend(rows[:end])
        validation.extend(rows[end:])
    return np.sort(np.array(train, dtype="<i8")), np.sort(np.array(validation, dtype="<i8"))


def summarize(data, rows):
    ids = [data["row_ids"][i] for i in rows]
    labels = [data["y"][i] for i in rows]
    return {
        "rows": len(ids),
        "row_ids_sha256": digest(canonical(ids)),
        "class_counts": [labels.count(0), labels.count(1)],
    }


def describe(path):
    train, test = load_archive(path)
    folds = []
    official_test = summarize(test, range(len(test["y"])))
    for seed in range(5):
        fit, validation = stratified_split(train["y"], seed)
        folds.append(
            {
                "seed": seed,
                "train": summarize(train, fit),
                "validation": summarize(train, validation),
                "test": official_test,
            }
        )
    # Equal predictors are not proof of equal entities; preserve official records.
    train_x = {canonical(row) for row in train["x"]}
    return {
        "schema": "openboost-adult-data-v1",
        "application": "A2",
        "source_url": SOURCE,
        "archive_sha256": ARCHIVE_SHA256,
        "member_sha256": MEMBERS,
        "citation": "Becker, B. & Kohavi, R. (1996). Adult. UCI. DOI:10.24432/C5XW20",
        "license": "CC-BY-4.0",
        "license_source": "https://archive.ics.uci.edu/dataset/2/adult",
        "features": list(FEATURES),
        "numeric": [c for c in FEATURES if c in NUMERIC],
        "categorical": [c for c in FEATURES if c not in NUMERIC],
        "labels": ["<=50K", ">50K"],
        "weight": "unit; fnlwgt excluded",
        "hash_format": "SHA256(canonical JSON with sorted keys and compact separators; ensure_ascii=True)",
        "source_records": {
            name: {
                "rows": len(data["y"]),
                "data_sha256": digest(canonical(data)),
                "missing_by_feature": [
                    sum(row[j] is None for row in data["x"]) for j in range(len(FEATURES))
                ],
            }
            for name, data in (("adult.data", train), ("adult.test", test))
        },
        "test_rows_with_predictors_seen_in_official_train": sum(
            canonical(row) in train_x for row in test["x"]
        ),
        "identity_note": "source:physical-line IDs; repeated predictor rows retained; no person IDs available",
        "split_rule": "official test unchanged; per seed PCG64 default_rng, class 0 then 1 permutation; floor(0.8*n_class) train, rest validation; sort source indices",
        "folds": folds,
        "status": "raw data/splits only; numeric encoding, budgets and quality pending",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    record = describe(args.archive)
    if args.verify:
        frozen = json.loads(args.verify.read_text())
        provenance = frozen.pop("provenance")
        if frozen != record or provenance["adapter_sha256"] != digest(Path(__file__).read_bytes()):
            raise ValueError("frozen data/splits/source mismatch")
    root = Path(__file__).resolve().parents[2]
    record["provenance"] = {
        "adapter_sha256": digest(Path(__file__).read_bytes()),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.platform(),
        "argv": ["python", "-m", "benchmarks.v1.adult", *sys.argv[1:]],
    }
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
