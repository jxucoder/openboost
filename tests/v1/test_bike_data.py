"""Calendar-only data and full-date rolling splits for A5."""

import csv
import io
from datetime import date, timedelta

import numpy as np
import pytest
from benchmarks.v1.bike import FEATURES, parse_hour, rolling_splits


def fixture_csv():
    names = [
        "instant",
        "dteday",
        "season",
        "yr",
        "mnth",
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
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=names)
    writer.writeheader()
    for i in range(40):
        day = date(2011, 1, 1) + timedelta(days=i // 2)
        row = dict.fromkeys(names, "0")
        row.update(
            instant=i + 1,
            dteday=day.isoformat(),
            season=1,
            yr=0,
            mnth=day.month,
            hr=i % 2,
            weekday=(day.weekday() + 1) % 7,
            workingday=int(day.weekday() < 5),
            cnt=i,
        )
        writer.writerow(row)
    return stream.getvalue().encode()


def test_full_dates_and_calendar_only():
    data = parse_hour(fixture_csv())
    assert data.x.shape == (40, 7)
    assert FEATURES == ("season", "yr", "mnth", "hr", "holiday", "weekday", "workingday")
    for fold in rolling_splits(data.dates):
        parts = [set(data.dates[fold[k]]) for k in ("train", "validation", "test")]
        assert not parts[0] & parts[1] and not parts[1] & parts[2]
        assert max(parts[0]) < min(parts[1]) < min(parts[2])
    first = rolling_splits(data.dates)[0]
    np.testing.assert_array_equal(first["train"], np.arange(20))
    np.testing.assert_array_equal(first["validation"], np.arange(20, 24))
    np.testing.assert_array_equal(first["test"], np.arange(24, 28))


def rewrite(data, field, value, row_index=0):
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    rows[row_index][field] = value
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def test_excluded_observations_cannot_change_features_or_split():
    original = fixture_csv()
    modified = original
    for field in ("weathersit", "temp", "atemp", "hum", "windspeed", "casual", "registered"):
        modified = rewrite(modified, field, "unavailable-at-prediction-time")
    a, b = parse_hour(original), parse_hour(modified)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.y, b.y)
    for left, right in zip(rolling_splits(a.dates), rolling_splits(b.dates), strict=True):
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
    changed_target = parse_hour(rewrite(original, "cnt", "9999"))
    np.testing.assert_array_equal(a.x, changed_target.x)
    assert a.y[0] != changed_target.y[0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("instant", "2"),
        ("instant", "0"),
        ("cnt", "-1"),
        ("cnt", "NaN"),
        ("hr", "24"),
        ("hr", "1"),
        ("dteday", "2011-02-30"),
        ("yr", "1"),
        ("weekday", "1"),
        ("workingday", "1"),
        ("season", "0"),
        ("holiday", "2"),
        ("mnth", "2"),
    ],
)
def test_invalid_calendar_counts_and_ids_fail(field, value):
    with pytest.raises(ValueError):
        parse_hour(rewrite(fixture_csv(), field, value))


def test_empty_and_wrong_schema_fail():
    with pytest.raises(ValueError, match="schema"):
        parse_hour(b"date,count\n2011-01-01,1\n")
    with pytest.raises(ValueError, match="empty"):
        parse_hour(fixture_csv().splitlines()[0] + b"\n")


def test_date_splits_are_not_row_percentage_and_retain_full_days():
    dates = np.array(["2011-01-01"] * 20 + [f"2011-01-{i:02}" for i in range(2, 21)])
    folds = rolling_splits(dates)
    # Ten of twenty dates, but 29 of 39 rows. Missing hours are not synthesized.
    assert len(folds[0]["train"]) == 29
    assert dates[folds[0]["validation"][0]] == "2011-01-11"
    assert dates[folds[4]["test"][-1]] == "2011-01-18"


@pytest.mark.parametrize("dates", [[], ["2011-01-01"], ["2011-01-02", "2011-01-01"], ["bad"]])
def test_invalid_split_dates_fail(dates):
    with pytest.raises(ValueError):
        rolling_splits(dates)


def test_archive_corruption_rejected_before_parsing(tmp_path):
    from benchmarks.v1.bike import load_archive

    archive = tmp_path / "wrong.zip"
    archive.write_bytes(b"not the pinned archive")
    with pytest.raises(ValueError, match="archive hash"):
        load_archive(archive)


def test_array_hash_binds_dtype_shape_and_order():
    from benchmarks.v1.bike import array_hash

    assert array_hash([1, 2], "<i8") != array_hash([2, 1], "<i8")
    assert array_hash([1, 2], "<i8") != array_hash([[1, 2]], "<i8")
    assert array_hash([1, 2], "<i8") != array_hash([1, 2], "<f8")
    assert array_hash(np.array([1, 2], dtype=">i8"), "<i8") == array_hash([1, 2], "<i8")


def test_committed_freeze_binds_adapter_source():
    import hashlib
    import json
    from pathlib import Path

    from benchmarks.v1 import bike

    module = Path(bike.__file__)
    frozen = json.loads((module.parent / "datasets/bike.json").read_text())
    assert frozen["provenance"]["adapter_sha256"] == hashlib.sha256(module.read_bytes()).hexdigest()
    assert frozen["rows"] == 17379 and frozen["days"] == 731
    assert len(frozen["folds"]) == 5
