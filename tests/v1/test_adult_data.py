"""Official source separation, categorical missingness and stratified Adult splits."""

import numpy as np
import pytest
from benchmarks.v1.adult import parse_source, stratified_split

ROW = "39, Private, 77516, Bachelors, 13, Never-married, ?, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K"


def test_missing_and_excluded_weight_and_test_label():
    train = parse_source((ROW + "\n").encode(), "adult.data")
    test = parse_source(("|1x3 Cross validator\n" + ROW + ".\n").encode(), "adult.test")
    assert len(train["x"][0]) == 13
    assert train["x"][0][5] is None
    assert train["x"] == test["x"] and train["y"] == test["y"] == [0]
    assert train["row_ids"] != test["row_ids"]
    altered = parse_source((ROW.replace("77516", "99999") + "\n").encode(), "adult.data")
    assert altered["x"] == train["x"]


@pytest.mark.parametrize("label", ["<=50K.", ">50K..", "unknown"])
def test_invalid_training_labels(label):
    with pytest.raises(ValueError):
        parse_source(ROW.replace("<=50K", label).encode(), "adult.data")


@pytest.mark.parametrize("line", [ROW, ROW + "..", ROW.replace("39,", "NaN,") + "."])
def test_invalid_test_rows(line):
    with pytest.raises(ValueError):
        parse_source(line.encode(), "adult.test")


def test_stratification_is_complete_seeded_and_isolated():
    labels = np.array([0] * 11 + [1] * 9)
    before = np.random.get_state()
    for seed in range(5):
        train, val = stratified_split(labels, seed)
        assert np.bincount(labels[train]).tolist() == [8, 7]
        assert np.bincount(labels[val]).tolist() == [3, 2]
        np.testing.assert_array_equal(np.sort(np.concatenate([train, val])), np.arange(20))
        for a, b in zip((train, val), stratified_split(labels, seed), strict=True):
            np.testing.assert_array_equal(a, b)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize(
    "labels,seed",
    [
        ([0, 0], 0),
        ([0, 1], 0),
        ([False, True], 0),
        ([0, 1, 2], 0),
        ([0, 0, 1, 1], -1),
        ([0, 0, 1, 1], True),
    ],
)
def test_invalid_split(labels, seed):
    with pytest.raises(ValueError):
        stratified_split(labels, seed)


def test_test_categories_do_not_change_training_records():
    train = parse_source(ROW.encode(), "adult.data")
    test = parse_source((ROW.replace("Private", "Test-only") + ".").encode(), "adult.test")
    assert test["x"][0][1] == "Test-only" and train["x"][0][1] == "Private"


def test_archive_and_frozen_source(tmp_path):
    import json
    from pathlib import Path

    from benchmarks.v1 import adult

    path = tmp_path / "bad.zip"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash"):
        adult.load_archive(path)
    module = Path(adult.__file__)
    frozen = json.loads((module.parent / "datasets/adult.json").read_text())
    assert frozen["provenance"]["adapter_sha256"] == adult.digest(module.read_bytes())
    assert frozen["source_records"]["adult.data"]["rows"] == 32561
    assert frozen["source_records"]["adult.test"]["rows"] == 16281
    assert len({f["test"]["row_ids_sha256"] for f in frozen["folds"]}) == 1
