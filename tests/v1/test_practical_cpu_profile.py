"""Diagnostic input isolation, executable workers and independent replay checks."""

import json

import numpy as np
import pytest
from benchmarks.v1 import practical_cpu_profile as coordinator
from benchmarks.v1.cpu_profile_worker import rejection_fixtures, run


def test_freeze_preserves_prefixes_and_excludes_test_rows(tmp_path, monkeypatch):
    x = np.arange(20640 * 8, dtype="<f4").reshape(-1, 8)
    y = np.arange(20640, dtype="<f4")
    root = tmp_path / "repo"
    freeze_path = root / "benchmarks/v1/datasets/housing.json"
    freeze_path.parent.mkdir(parents=True)
    partitions = coordinator.housing.split_indices(len(y), 0)
    freeze_path.write_text(
        json.dumps(
            dict(
                arrays_sha256=coordinator.housing.digest(x.tobytes() + y.tobytes()),
                split_sha256={"0": [coordinator.housing.digest(i.tobytes()) for i in partitions]},
            )
        )
    )
    archive = tmp_path / "archive"
    archive.write_bytes(b"synthetic freeze fixture")
    monkeypatch.setattr(coordinator, "ROOT", root)
    monkeypatch.setattr(coordinator.housing, "load_archive", lambda p: (x, y, "member"))
    destination = tmp_path / "freeze"
    protocol = coordinator.freeze(archive, destination)
    with np.load(destination / "input.npz", allow_pickle=False) as data:
        assert not any("test" in name for name in data.files)
        np.testing.assert_array_equal(data["train_ids"], partitions[0][:8192])
        np.testing.assert_array_equal(data["validation_ids"], partitions[1][:1024])
        np.testing.assert_array_equal(data["x_train"], x[partitions[0][:8192]])
        assert not np.intersect1d(data["train_ids"], partitions[2]).size
    assert protocol["budgets"]["retries"] == 0
    assert len(protocol["cases"]) == 8
    assert len({c["id"] for c in protocol["cases"]}) == 8
    with pytest.raises(FileExistsError):
        coordinator.freeze(archive, destination)
    frozen = json.loads(freeze_path.read_text())
    frozen["arrays_sha256"] = "corrupt"
    freeze_path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="arrays differ"):
        coordinator.freeze(archive, tmp_path / "corrupt")


def packet(tmp_path, **changes):
    values = dict(
        x_train=np.arange(8)[:, None],
        y_train=np.arange(8),
        train_ids=np.arange(8),
        x_validation=np.linspace(0, 7, 1024)[:, None],
        y_validation=np.linspace(0, 7, 1024),
        validation_ids=np.arange(100, 1124),
    )
    values.update(changes)
    path = tmp_path / "input.npz"
    np.savez(path, **values)
    return dict(
        input_path=str(path),
        input_sha256=coordinator.digest(path),
        recipe="squared",
        train_rows=8,
        rounds=2,
        id="fixture",
        features=["x"],
        instrumented=False,
    )


@pytest.mark.parametrize("recipe", ["squared", "normal"])
def test_worker_records_full_replay_and_independent_metric(tmp_path, recipe, monkeypatch):
    job = packet(tmp_path)
    job["recipe"] = recipe
    # Source mathematics tests do not certify this host's BLAS configuration.
    # Actual nonempty two-thread inspection is required by the real Modal worker.
    monkeypatch.setattr(
        "benchmarks.v1.cpu_profile_worker.threadpool_info",
        lambda: [dict(num_threads=2, scope="source-test inspection stub")],
    )
    result = run(job, tmp_path)
    with np.load(tmp_path / "predictions.npz") as data:
        raw = data["raw"]
    y = np.linspace(0, 7, 1024)
    if recipe == "squared":
        expected = np.dot(raw[:, 0] - y, raw[:, 0] - y) / (2 * len(y))
    else:
        from math import log
        from statistics import NormalDist

        expected = -sum(
            log(NormalDist(float(mu), float(sigma)).pdf(float(value)))
            for value, mu, sigma in zip(y, raw[:, 0], np.exp(raw[:, 1]), strict=True)
        ) / len(y)
    assert result["validation_metric"] == pytest.approx(expected)
    assert result["final_raw_exact"] and result["trace_array_bytes"] > 0
    assert result["tree_predict_calls"] is None


@pytest.mark.parametrize("corruption", ["hash", "test_labels", "overlap", "short_prefix"])
def test_worker_rejects_invalid_inputs(tmp_path, corruption):
    additions = {"y_test": np.ones(1)} if corruption == "test_labels" else {}
    if corruption == "overlap":
        additions["validation_ids"] = np.arange(1024)
    job = packet(tmp_path, **additions)
    if corruption == "hash":
        job["input_sha256"] = "invalid"
    if corruption == "short_prefix":
        job["train_rows"] = 8192
    with pytest.raises(ValueError):
        run(job, tmp_path)
    assert not (tmp_path / "result.json").exists()


def test_forced_rejection_fixtures():
    result = rejection_fixtures()
    assert result["rejection_state_unchanged"]
    assert result["backtracking_coefficients"] == [16, 8, 4, 2]
