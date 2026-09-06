"""Verified real-data parsing and deterministic partitions; no model fitting."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

SOURCES = Path(__file__).parent / "datasets/sources.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def array_hash(value):
    a = np.asarray(value)
    if a.dtype.kind == "O":
        raise ValueError("object arrays have no portable byte identity")
    if a.dtype.kind in "f" and not np.isfinite(a).all():
        raise ValueError("nonfinite data")
    a = np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<")))
    return sha(
        json.dumps([a.dtype.str, list(a.shape)], separators=(",", ":")).encode()
        + b"\n"
        + a.tobytes()
    )


def group_splits(groups, seed):
    a = np.asarray(groups)
    if a.ndim != 1 or not len(a) or a.dtype.kind not in "iuUSf":
        raise ValueError("one-dimensional group identifiers required")
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        raise ValueError("nonfinite group")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer seed required")
    ids, inverse = np.unique(a, return_inverse=True)
    if len(ids) < 5:
        raise ValueError("at least five groups required")
    order = np.random.default_rng(seed).permutation(len(ids))
    cuts = np.split(order, [len(ids) * 3 // 5, len(ids) * 4 // 5])
    return tuple(np.flatnonzero(np.isin(inverse, part)) for part in cuts)


def stratified_splits(labels, seed):
    a = np.asarray(labels)
    if a.ndim != 1 or a.dtype.kind not in "iu" or type(seed) is not int or seed < 0:
        raise ValueError("integer labels and nonnegative seed required")
    classes = np.unique(a)
    if len(classes) < 2:
        raise ValueError("at least two classes required")
    rng = np.random.default_rng(seed)
    splits = [[], [], []]
    for cls in classes:
        rows = rng.permutation(np.flatnonzero(a == cls))
        if len(rows) < 5:
            raise ValueError("at least five rows per class required")
        for out, part in zip(
            splits, np.split(rows, [len(rows) * 3 // 5, len(rows) * 4 // 5]), strict=True
        ):
            out.extend(part)
    return tuple(np.sort(np.asarray(s, dtype="<i8")) for s in splits)


def source(root, name):
    spec = json.loads(SOURCES.read_text())[name]
    data = (Path(root) / spec["file"]).read_bytes()
    if sha(data) != spec["sha256"]:
        raise ValueError(f"{name}: source hash mismatch")
    if "members" not in spec:
        return data
    archive = zipfile.ZipFile(io.BytesIO(data))
    result = {k: archive.read(k) for k in spec["members"]}
    if any(sha(v) != spec["members"][k] for k, v in result.items()):
        raise ValueError("member hash mismatch")
    return result


def numeric(data, columns):
    a = np.asarray(data, dtype="<f8")
    if a.ndim != 2 or a.shape[1] != columns or not len(a) or not np.isfinite(a).all():
        raise ValueError("invalid numeric table")
    return a


def arff_rows(data):
    lines = data.decode().splitlines()
    start = next(i for i, line in enumerate(lines) if line.lower() == "@data")
    return list(csv.reader(lines[start + 1 :], quotechar="'"))


def load(root, name):
    raw = source(root, name)
    if name == "covertype":
        a = numeric(
            np.loadtxt(io.BytesIO(gzip.decompress(raw["covtype.data.gz"])), delimiter=",", ndmin=2),
            55,
        )
        if not np.isin(a[:, -1], np.arange(1, 8)).all() or not np.isin(a[:, 10:54], [0, 1]).all():
            raise ValueError("invalid Covertype indicators or labels")
        if not np.all(a[:, 10:14].sum(axis=1) == 1) or not np.all(a[:, 14:54].sum(axis=1) == 1):
            raise ValueError("invalid one-hot source indicators")
        return {"x": a[:, :54], "y": a[:, -1].astype("<i8") - 1}, "stratified"
    if name == "parkinsons":
        a = numeric(
            np.loadtxt(
                io.BytesIO(raw["parkinsons_updrs.data"]), delimiter=",", skiprows=1, ndmin=2
            ),
            22,
        )
        return {
            "x": a[:, [1, 2, 3, *range(6, 22)]],
            "y": a[:, 4:6],
            "group": a[:, 0].astype("<i8"),
        }, "group"
    if name == "concrete":
        import xlrd

        sheet = xlrd.open_workbook(file_contents=raw["Concrete_Data.xls"]).sheet_by_index(0)
        a = numeric([sheet.row_values(i) for i in range(1, sheet.nrows)], 9)
        if np.any(a[:, 7:] <= 0):
            raise ValueError("positive age and strength required")
        _, groups = np.unique(a[:, :7], axis=0, return_inverse=True)
        return {
            "x": a[:, :7],
            "structure": a[:, 7] / 28,
            "y": a[:, 8],
            "group": groups.astype("<i8"),
        }, "group"
    if name == "veteran":
        rows = arff_rows(raw)
        maps = {
            0: ["standard", "test"],
            1: ["adeno", "large", "smallcell", "squamous"],
            7: ["no", "yes"],
        }
        x = [
            [maps[j].index(r[j]) if j in maps else float(r[j]) for j in [0, 1, 4, 5, 6, 7]]
            for r in rows
        ]
        if any(r[3] not in ["dead", "censored"] for r in rows):
            raise ValueError("invalid event label")
        time = np.array([float(r[2]) for r in rows])
        if np.any(time <= 0):
            raise ValueError("positive survival time required")
        return {
            "x": numeric(x, 6),
            "y": time,
            "event": np.array([r[3] == "dead" for r in rows], dtype="<i8"),
        }, "event"
    raise ValueError("unknown dataset")


def insurance(root):
    freq, sev = arff_rows(source(root, "freq")), arff_rows(source(root, "sev"))
    f = numeric([[r[i] for i in [0, 1, 2, 4, 5, 6, 7, 10]] for r in freq], 8)
    s = numeric(sev, 2)
    ids = f[:, 0].astype("<i8")
    if (
        len(np.unique(ids)) != len(ids)
        or np.any(f[:, 0] != ids)
        or np.any(s[:, 0] != s[:, 0].astype("<i8"))
    ):
        raise ValueError("invalid or duplicate policy identity")
    if np.any(f[:, 2] <= 0) or np.any(f[:, 1] < 0) or np.any(f[:, 1] != np.floor(f[:, 1])):
        raise ValueError("invalid exposure/count; never clip silently")
    lookup = {int(v): i for i, v in enumerate(ids)}
    joined = np.array([lookup.get(int(v), -1) for v in s[:, 0]])
    positive = s[:, 1] > 0
    valid = (joined >= 0) & positive
    counts = np.bincount(joined[valid], minlength=len(f))
    totals = np.bincount(joined[valid], weights=s[valid, 1], minlength=len(f))
    contradictory = ((f[:, 1] == 0) & (counts > 0)) | ((f[:, 1] > 0) & (counts == 0))
    data = {
        "x": f[:, 3:],
        "group": ids,
        "y": f[:, 1],
        "exposure": f[:, 2],
        "paid_count": counts,
        "paid_total": totals,
        "aggregate_eligible": ~contradictory,
    }
    for label, j in [("Area", 3), ("VehBrand", 8), ("VehGas", 9), ("Region", 11)]:
        data["category_" + label] = np.array([r[j] for r in freq])
    # Policy-row mapping gives claims the exact same entity partitions as frequency.
    data["severity_policy_row"] = joined[valid]
    data["severity_y"] = s[valid, 1]
    audit = {
        "frequency_rows": len(f),
        "severity_rows": len(s),
        "nonpositive_claims": int((~positive).sum()),
        "orphan_claims": int((joined < 0).sum()),
        "retained_claims": int(valid.sum()),
        "positive_count_without_payment": int(((f[:, 1] > 0) & (counts == 0)).sum()),
        "zero_count_with_payment": int(((f[:, 1] == 0) & (counts > 0)).sum()),
        "aggregate_retained": int((~contradictory).sum()),
    }
    return data, audit


def summarize(data, split, name):
    rows = np.asarray(split, dtype="<i8")
    d = {"rows": len(rows), "row_indices_sha256": array_hash(rows)}
    if "group" in data:
        d["groups"] = len(np.unique(data["group"][rows]))
    if name in ["covertype", "veteran"]:
        labels = data["y"] if name == "covertype" else data["event"]
        d["class_counts"] = np.bincount(labels[rows]).tolist()
    if name == "concrete":
        d["age_days_min_max"] = [
            float(v)
            for v in [data["structure"][rows].min() * 28, data["structure"][rows].max() * 28]
        ]
    return d


def describe(root, name):
    if name == "insurance":
        data, audit = insurance(root)
        rule = "group"
    else:
        data, rule = load(root, name)
        audit = {}
    for a in data.values():
        array_hash(a)
    folds = []
    for seed in range(5):
        split = (
            group_splits(data["group"], seed)
            if rule == "group"
            else stratified_splits(data["event"] if rule == "event" else data["y"], seed)
        )
        fold = {"seed": seed}
        for part, rows in zip(["train", "validation", "test"], split, strict=True):
            fold[part] = summarize(data, rows, name)
            if name == "insurance":
                eligible = rows[data["aggregate_eligible"][rows]]
                claims = np.flatnonzero(np.isin(data["severity_policy_row"], rows))
                fold[part]["aggregate"] = summarize(data, eligible, name)
                fold[part]["severity"] = {
                    "rows": len(claims),
                    "row_indices_sha256": array_hash(claims),
                }
        if name == "concrete":
            lo, hi = data["structure"][split[0]].min(), data["structure"][split[0]].max()
            fold["test_age_outside_train_support"] = int(
                ((data["structure"][split[2]] < lo) | (data["structure"][split[2]] > hi)).sum()
            )
        folds.append(fold)
    return {
        "schema": "openboost-real-data-v1",
        "dataset": name,
        "split_rule": rule
        + "; sorted unique groups/classes; PCG64(seed); floor 60/80 percent boundaries; sorted source rows",
        "arrays": {k: {"shape": list(v.shape), "sha256": array_hash(v)} for k, v in data.items()},
        "audit": audit,
        "folds": folds,
        "sources_sha256": sha(SOURCES.read_bytes()),
        "adapter_sha256": sha(Path(__file__).read_bytes()),
        "hash_format": "SHA256(JSON [little-endian dtype,shape] compact + newline + contiguous C-order bytes)",
        "status": "data preparation only; no model quality claim",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", choices=["covertype", "parkinsons", "concrete", "veteran", "insurance"])
    p.add_argument("directory", type=Path)
    p.add_argument("--verify", type=Path)
    args = p.parse_args()
    record = describe(args.directory, args.name)
    if args.verify:
        frozen = json.loads(args.verify.read_text())
        frozen.pop("provenance")
        if record != frozen:
            raise ValueError("frozen data mismatch")
    record["provenance"] = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
        "argv": [sys.executable, "-m", "benchmarks.v1.real_data", *sys.argv[1:]],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.platform(),
    }
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
