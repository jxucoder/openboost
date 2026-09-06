"""Independent hand calculations and seeded partition checks for Housing."""

import numpy as np
import pytest
from benchmarks.v1.housing import split_indices, transform


def test_raw_columns_ratios_and_target_units():
    raw = [[-120, 35, 10, 100, 20, 60, 10, 4, 250000]]
    x, y = transform(raw)
    np.testing.assert_array_equal(x, [[4, 10, 10, 2, 60, 6, 35, -120]])
    np.testing.assert_array_equal(y, [2.5])
    assert x.dtype == y.dtype == np.dtype("<f4")


@pytest.mark.parametrize(
    "column,value", [(6, 0), (6, -1), (8, -1), (3, float("nan")), (4, float("inf")), (3, 1e100)]
)
def test_invalid_or_overflowing_raw_rejected(column, value):
    raw = np.array([[-120, 35, 10, 100, 20, 60, 10, 4, 250000]], dtype=float)
    raw[0, column] = value
    with pytest.raises((ValueError, FloatingPointError)):
        transform(raw)


@pytest.mark.parametrize("raw", [[], [[1, 2]], [1] * 9])
def test_shape_rejected(raw):
    with pytest.raises(ValueError):
        transform(raw)


def test_transformation_does_not_learn_from_other_rows():
    row = [-120, 35, 10, 100, 20, 60, 10, 4, 250000]
    x, y = transform([row])
    extra = [-110, 30, 50, 1e6, 200, 300, 100, 20, 500000]
    xx, yy = transform([row, extra])
    np.testing.assert_array_equal(x[0], xx[0])
    np.testing.assert_array_equal(y[0], yy[0])


def test_five_partitions_complete_disjoint_and_rng_isolated():
    before = np.random.get_state()
    for seed in range(5):
        parts = split_indices(101, seed)
        assert [len(p) for p in parts] == [60, 20, 21]
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(101))
        for a, b in zip(parts, split_indices(101, seed), strict=True):
            np.testing.assert_array_equal(a, b)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert not np.array_equal(split_indices(101, 0)[0], split_indices(101, 1)[0])


@pytest.mark.parametrize("n,seed", [(0, 0), (4, 0), (10, -1), (10, True), (10, 1.5), (10.5, 0)])
def test_invalid_split_arguments(n, seed):
    with pytest.raises(ValueError):
        split_indices(n, seed)


def test_corrupt_archive_fails(tmp_path):
    from benchmarks.v1.housing import load_archive

    path = tmp_path / "bad.tgz"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash"):
        load_archive(path)


def test_freeze_matches_legacy_and_source():
    import json
    from pathlib import Path

    from benchmarks.v1 import housing

    module = Path(housing.__file__)
    frozen = json.loads((module.parent / "datasets/housing.json").read_text())
    legacy = json.loads((module.parent.parent / "foundation/housing.json").read_text())
    housing.check_legacy(frozen, legacy)
    assert set(frozen["split_sha256"]) == {"0", "1", "2", "3", "4"}
    assert frozen["provenance"]["adapter_sha256"] == housing.digest(module.read_bytes())
    frozen["split_sha256"]["1"][0] = "0" * 64
    with pytest.raises(ValueError, match="legacy split"):
        housing.check_legacy(frozen, legacy)
