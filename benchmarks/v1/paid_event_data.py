"""Bind matched positive-payment aggregates to existing frozen A9 worker rows."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmarks.v1 import real_data


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_paid_events(raw):
    """Independent claim iteration, not the source reader's bincount aggregation."""
    size = len(raw["group"])
    count = np.zeros(size, dtype=np.int64)
    total = np.zeros(size)
    rows, amounts = raw["severity_policy_row"], raw["severity_y"]
    if rows.shape != amounts.shape or rows.ndim != 1 or rows.dtype.kind not in "iu":
        raise ValueError("aligned integer claim-policy mapping required")
    for row, amount in zip(rows, amounts, strict=True):
        if row < 0 or row >= size or not np.isfinite(amount) or amount <= 0:
            raise ValueError("invalid retained positive payment")
        count[row] += 1
        total[row] += amount
    if not np.array_equal(count, raw["paid_count"]) or not np.array_equal(total, raw["paid_total"]):
        raise ValueError("paid aggregates differ from retained claim records")
    return count, total


def bind(raw, arrays, train_ids):
    """Require exact source units/rows; only emit train/validation composition roles."""
    count, total = verify_paid_events(raw)
    required = {
        "x_train",
        "y_train",
        "weight_train",
        "x_validation",
        "y_validation",
        "weight_validation",
        "validation_row_ids",
    }
    if set(arrays) != required:
        raise ValueError("exact annualized train/validation packet required")
    ids = np.asarray(raw["group"])
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("unique source policies required")
    lookup = {v: i for i, v in enumerate(ids.tolist())}
    output = {}
    used = set()
    for part, labels in (
        ("train", np.asarray(train_ids)),
        ("validation", arrays["validation_row_ids"]),
    ):
        if labels.ndim != 1 or len(np.unique(labels)) != len(labels) or not len(labels):
            raise ValueError("unique nonempty partition IDs required")
        if any(v not in lookup or v in used for v in labels.tolist()):
            raise ValueError("foreign or overlapping policies")
        used.update(labels.tolist())
        rows = np.array([lookup[v] for v in labels.tolist()])
        exposure = raw["exposure"][rows]
        if not np.all(raw["aggregate_eligible"][rows]):
            raise ValueError("ineligible aggregate policy")
        if not np.array_equal(arrays["weight_" + part], exposure):
            raise ValueError("exposure weights differ from source")
        if not np.array_equal(arrays["y_" + part], total[rows] / exposure):
            raise ValueError("annualized targets differ from matched payments")
        x = arrays["x_" + part]
        if x.ndim != 2 or len(x) != len(rows) or not np.isfinite(x).all():
            raise ValueError("aligned finite encoded features required")
        output.update(
            {
                "x_" + part: x,
                "row_ids_" + part: labels,
                "paid_count_" + part: count[rows],
                "paid_total_" + part: total[rows],
                "exposure_" + part: exposure,
            }
        )
    return output


def run(previous_summary, directory, source_root=Path("build/v1-data")):
    previous_summary, directory = Path(previous_summary).resolve(), Path(directory).resolve()
    previous = json.loads(previous_summary.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("fresh output directory required")
    repo = Path(__file__).resolve().parents[2]
    source_path = repo / "benchmarks/v1/datasets/insurance.json"
    freeze = json.loads(source_path.read_text())
    raw, audit = real_data.insurance(source_root)
    for name, record in freeze["arrays"].items():
        if real_data.array_hash(raw[name]) != record["sha256"]:
            raise ValueError("source array differs from freeze")
    report = dict(
        scope="Matched paid-event binding only; no composition fit or test scores",
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        argv=[
            sys.executable,
            "-m",
            "benchmarks.v1.paid_event_data",
            str(previous_summary),
            str(directory),
        ],
        python=platform.python_version(),
        os=platform.platform(),
        numpy=np.__version__,
        previous_summary_sha256=digest(previous_summary),
        source_manifest_sha256=digest(source_path),
        sources={
            str(p.relative_to(repo)): digest(p) for p in [Path(__file__), Path(real_data.__file__)]
        },
        source_audit=audit,
        cells=[],
    )
    folds = previous["data"]["A9"]["folds"]
    cells = previous["cells"]
    if {c["fold"] for c in cells} != set(range(5)) or len(cells) != 5:
        raise ValueError("all five frozen folds required")
    for cell in cells:
        if cell["application"] != "A9":
            raise ValueError("A9 summary required")
        fold = next(f for f in folds if f["seed"] == cell["fold"])
        packet = Path(cell["job"]["input_npz"])
        train_path = packet.parent / "train-rows.npz"
        for path, role in [(packet, "worker-input"), (train_path, "train-rows")]:
            if digest(path) != fold["artifacts"][role]["sha256"]:
                raise ValueError("frozen packet hash differs")
        with np.load(packet, allow_pickle=False) as a, np.load(train_path, allow_pickle=False) as t:
            bound = bind(raw, dict(a), t["row_ids"])
        dest = directory / f"fold-{cell['fold']}.npz"
        np.savez(dest, **bound)
        report["cells"].append(
            dict(
                fold=cell["fold"],
                status="pass",
                path=dest.name,
                sha256=digest(dest),
                input_sha256=digest(packet),
                train_ids_sha256=digest(train_path),
                rows={p: len(bound["row_ids_" + p]) for p in ("train", "validation")},
                paid_events={
                    p: int(bound["paid_count_" + p].sum()) for p in ("train", "validation")
                },
            )
        )
    (directory / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_summary", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(
        f"{len(run(args.previous_summary, args.directory)['cells'])}/5 paid-event bindings passed"
    )
