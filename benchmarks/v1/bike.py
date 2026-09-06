"""A5 calendar-only Bike Sharing adapter and frozen full-date rolling splits."""

import argparse
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

SOURCE = "https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip"
ARCHIVE_SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
MEMBER_SHA256 = "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"
FEATURES = ("season", "yr", "mnth", "hr", "holiday", "weekday", "workingday")
COLUMNS = (
    "instant",
    "dteday",
    *FEATURES[:3],
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
)


@dataclass(frozen=True)
class BikeData:
    x: np.ndarray
    y: np.ndarray
    row_ids: np.ndarray
    dates: np.ndarray


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def array_hash(values, dtype):
    array = np.asarray(values, dtype=dtype, order="C")
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(header + b"\n" + array.tobytes(order="C"))


def parse_hour(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise ValueError("unexpected hour.csv schema")
    features, target, ids, days, timestamps = [], [], [], [], []
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("malformed CSV row")
        day = date.fromisoformat(row["dteday"])
        values = [int(row[name]) for name in FEATURES]
        season, year, month, hour, holiday, weekday, working = values
        count, row_id = int(row["cnt"]), int(row["instant"])
        if (
            season not in range(1, 5)
            or year != day.year - 2011
            or year not in (0, 1)
            or month != day.month
            or hour not in range(24)
            or holiday not in (0, 1)
            or weekday != (day.weekday() + 1) % 7
            or working != int(weekday not in (0, 6) and not holiday)
            or count < 0
            or row_id <= 0
        ):
            raise ValueError("invalid calendar/count/row ID")
        features.append(values)
        target.append(count)
        ids.append(row_id)
        days.append(day.isoformat())
        timestamps.append((day, hour))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty data or duplicate row ID")
    if any(a >= b for a, b in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("timestamps must be unique and chronological")
    arrays = (
        np.array(features, dtype="<f8"),
        np.array(target, dtype="<f8"),
        np.array(ids, dtype="<i8"),
        np.array(days),
    )
    for array in arrays:
        array.flags.writeable = False
    return BikeData(*arrays)


def load_archive(path):
    archive = Path(path).read_bytes()
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("archive hash mismatch")
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        member = zipped.read("hour.csv")
    if sha256(member) != MEMBER_SHA256:
        raise ValueError("hour.csv hash mismatch")
    return parse_hour(member)


def rolling_splits(dates):
    dates = np.asarray(dates)
    if dates.ndim != 1 or not len(dates):
        raise ValueError("dates must be a nonempty vector")
    if any(not isinstance(d, str) or date.fromisoformat(d).isoformat() != d for d in dates):
        raise ValueError("dates must be canonical ISO dates")
    if np.any(dates[1:] < dates[:-1]):
        raise ValueError("dates must be chronological")
    unique = np.unique(dates)
    day_indices = np.searchsorted(unique, dates)
    result = []
    for percent in (50, 55, 60, 65, 70):
        ends = [len(unique) * p // 100 for p in (percent, percent + 10, percent + 20)]
        if not 0 < ends[0] < ends[1] < ends[2] <= len(unique):
            raise ValueError("too few dates for nonempty rolling windows")
        result.append(
            {
                key: np.flatnonzero((day_indices >= start) & (day_indices < end))
                for key, start, end in zip(
                    ("train", "validation", "test"), (0, *ends[:2]), ends, strict=True
                )
            }
        )
    return result


def describe(data):
    folds = []
    for i, split in enumerate(rolling_splits(data.dates)):
        parts = {}
        for name, rows in split.items():
            parts[name] = {
                "rows": len(rows),
                "days": len(np.unique(data.dates[rows])),
                "first_date": str(data.dates[rows[0]]),
                "last_date": str(data.dates[rows[-1]]),
                "row_ids_sha256": array_hash(data.row_ids[rows], "<i8"),
            }
        folds.append({"origin": i, "training_percent": 50 + 5 * i, "splits": parts})
    return {
        "schema": "openboost-bike-data-v1",
        "application": "A5",
        "source_url": SOURCE,
        "citation": "Fanaee-T, H. (2013). Bike Sharing. UCI. DOI:10.24432/C5W894",
        "license": "CC-BY-4.0",
        "license_source": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
        "archive_sha256": ARCHIVE_SHA256,
        "member": "hour.csv",
        "member_sha256": MEMBER_SHA256,
        "rows": len(data.y),
        "days": len(np.unique(data.dates)),
        "features": list(FEATURES),
        "target": "cnt",
        "target_unit": "hourly rental count",
        "weight": "unit",
        "x_sha256": array_hash(data.x, "<f8"),
        "y_sha256": array_hash(data.y, "<f8"),
        "row_ids_sha256": array_hash(data.row_ids, "<i8"),
        "array_hash_format": "SHA256(canonical JSON dtype/shape + newline + C-order little-endian bytes)",
        "split_rule": "sorted unique dates; floor(D*p/100) endpoints; expanding train, next 10% validation, next 10% test",
        "folds": folds,
        "status": "data prepared only; no quality or budget gate evaluated",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--verify", type=Path, help="compare data fields with a frozen record")
    args = parser.parse_args()
    result = describe(load_archive(args.archive))
    if args.verify is not None:
        frozen = json.loads(args.verify.read_text())
        recorded_provenance = frozen.pop("provenance")
        if recorded_provenance["adapter_sha256"] != sha256(Path(__file__).read_bytes()):
            raise ValueError("adapter source hash differs from frozen record")
        if frozen != result:
            raise ValueError("prepared data/splits differ from frozen record")
    repository = Path(__file__).resolve().parents[2]
    result["provenance"] = {
        "adapter_sha256": sha256(Path(__file__).read_bytes()),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.platform(),
        "argv": ["python", "-m", "benchmarks.v1.bike", *sys.argv[1:]],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
