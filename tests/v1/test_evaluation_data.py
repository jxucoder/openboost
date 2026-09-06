"""Independent partition and task-field counterexamples for evaluation data."""

import numpy as np
import pytest
from benchmarks.v1.real_data import group_splits, stratified_splits


def test_groups_never_cross_partitions_and_global_rng_is_untouched():
    groups = np.repeat(np.arange(10), np.arange(1, 11))
    before = np.random.get_state()
    for seed in range(5):
        split = group_splits(groups, seed)
        assert [len(set(groups[r])) for r in split] == [6, 2, 2]
        assert set(groups[split[0]]).isdisjoint(groups[split[1]])
        assert set(groups[split[0]]).isdisjoint(groups[split[2]])
        assert set(groups[split[1]]).isdisjoint(groups[split[2]])
        np.testing.assert_array_equal(np.sort(np.concatenate(split)), np.arange(len(groups)))
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_stratification_preserves_every_class_and_row():
    labels = np.repeat(np.arange(3), 10)
    split = stratified_splits(labels, 0)
    for rows, expected in zip(split, [6, 2, 2], strict=True):
        assert np.bincount(labels[rows]).tolist() == [expected] * 3
    assert len(set(np.concatenate(split))) == 30


@pytest.mark.parametrize("groups", [[], [1, 1], [float("nan"), 1, 2]])
def test_invalid_group_splits(groups):
    with pytest.raises(ValueError):
        group_splits(groups, 0)


def test_insurance_join_retains_only_declared_targets(monkeypatch):
    from benchmarks.v1 import real_data as rd

    freq = b"@data\n1,0,0.5,'A',4,1,30,50,'B1','Regular',10,'R11'\n2,1,1,'A',4,1,30,50,'B1','Regular',10,'R11'\n3,1,2,'A',4,1,30,50,'B1','Regular',10,'R11'\n4,0,1,'A',4,1,30,50,'B1','Regular',10,'R11'\n"
    sev = b"@data\n2,10\n2,20\n2,-3\n4,5\n999,10\n"
    monkeypatch.setattr(rd, "source", lambda root, name: freq if name == "freq" else sev)
    data, audit = rd.insurance(".")
    np.testing.assert_array_equal(data["paid_total"], [0, 30, 0, 5])
    np.testing.assert_array_equal(data["paid_count"], [0, 2, 0, 1])
    np.testing.assert_array_equal(data["aggregate_eligible"], [True, True, False, False])
    np.testing.assert_array_equal(data["severity_policy_row"], [1, 1, 3])
    assert audit["orphan_claims"] == 1 and audit["nonpositive_claims"] == 1
    assert data["x"].shape == (4, 5)  # No ID, count, exposure, or payment target in X.


def test_parkinsons_excludes_subject_and_both_targets(monkeypatch):
    from benchmarks.v1 import real_data as rd

    row = ",".join(str(i) for i in range(22))
    monkeypatch.setattr(
        rd, "source", lambda *args: {"parkinsons_updrs.data": ("header\n" + row).encode()}
    )
    data, rule = rd.load(".", "parkinsons")
    assert rule == "group"
    np.testing.assert_array_equal(data["y"], [[4, 5]])
    np.testing.assert_array_equal(data["x"][0], [1, 2, 3, *range(6, 22)])


def test_survival_event_is_not_an_input_or_death_imputation(monkeypatch):
    from benchmarks.v1 import real_data as rd

    raw = b"@data\nstandard,adeno,20,censored,50,2,60,no\ntest,large,30,dead,40,3,61,yes\n"
    monkeypatch.setattr(rd, "source", lambda *args: raw)
    data, rule = rd.load(".", "veteran")
    assert rule == "event"
    np.testing.assert_array_equal(data["event"], [0, 1])
    np.testing.assert_array_equal(data["y"], [20, 30])
    assert data["x"].shape == (2, 6)


def test_source_corruption_rejected(tmp_path):
    from benchmarks.v1.real_data import source

    (tmp_path / "covertype.zip").write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash"):
        source(tmp_path, "covertype")
