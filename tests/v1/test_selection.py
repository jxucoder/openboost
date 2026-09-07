import copy
import hashlib

import numpy as np
import pytest
from benchmarks.v1.selection import audit, digest, release_test, seal


def entry(path):
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def example(root):
    np.savez(root / "train.npz", row_ids=[0, 1])
    np.savez(root / "valid.npz", row_ids=[2, 3], y=[0.0, 1.0])
    np.savez(root / "test.npz", row_ids=[4, 5], x=[[2.0], [3.0]])
    protocol = dict(
        schema="openboost-selection-v1",
        application="A1",
        fold=0,
        identity={
            k: "a" * 64
            for k in ["code", "data", "split", "preprocessing", "environment", "search_design"]
        },
        train_rows=entry(root / "train.npz"),
        validation=entry(root / "valid.npz"),
        test_features=entry(root / "test.npz"),
        selection_weights={"rmse": 1.0},
        methods={
            "first": [{"depth": i + 1} for i in range(16)],
            "second": [{"leaves": i + 1} for i in range(16)],
        },
    )
    records = []
    for method, configs in protocol["methods"].items():
        for i, config in enumerate(configs):
            trial_id = f"{method}:{i:02}"
            np.savez(
                root / f"{trial_id}.npz",
                row_ids=[2, 3],
                prediction=np.array([0.0, 1.0]) + (16 - i if method == "first" else 0.5 + i),
            )
            (root / f"{trial_id}.model").write_bytes(trial_id.encode())
            (root / f"{trial_id}.log").write_text("synthetic fixture\n")
            records.append(
                dict(
                    id=trial_id,
                    config=config,
                    status="pass",
                    exit_code=0,
                    protocol_sha256=digest(protocol),
                    prediction=entry(root / f"{trial_id}.npz"),
                    model=entry(root / f"{trial_id}.model"),
                    log=entry(root / f"{trial_id}.log"),
                )
            )
    return protocol, records


def test_selects_method_and_trial_from_recomputed_validation(tmp_path):
    p, records = example(tmp_path)
    # Test features are not read by the selection audit.
    features = (tmp_path / "test.npz").read_bytes()
    (tmp_path / "test.npz").unlink()
    receipt = audit(p, records, tmp_path, digest(p))
    assert receipt["selected"] == "second:00"
    assert receipt["scores"]["first:15"]["selection"] == 1.0
    saved = tmp_path / "receipt.json"
    receipt_hash = seal(receipt, saved)
    with pytest.raises(FileExistsError):
        seal(receipt, saved)
    (tmp_path / "test.npz").write_bytes(features)
    arrays, model = release_test(p, records, saved, tmp_path, digest(p), receipt_hash)
    assert arrays["row_ids"].tolist() == [4, 5]
    assert model == records[16]["model"]


@pytest.mark.parametrize("mutation", ["missing", "failed", "config", "score", "duplicate"])
def test_incomplete_or_manipulated_search_fails(tmp_path, mutation):
    p, records = example(tmp_path)
    if mutation == "missing":
        records.pop()
    elif mutation == "failed":
        records[0]["status"] = "timeout"
    elif mutation == "config":
        records[0]["config"] = {"depth": 99}
    elif mutation == "score":
        records[0]["validation_score"] = -1000.0
    else:
        records[-1] = copy.deepcopy(records[0])
    with pytest.raises(ValueError):
        audit(p, records, tmp_path, digest(p))


def test_protocol_cannot_shrink_expected_search(tmp_path):
    p, records = example(tmp_path)
    pinned = digest(p)
    p["methods"].pop("second")
    with pytest.raises(ValueError, match="protocol"):
        audit(p, records[:16], tmp_path, pinned)


def test_changed_model_and_forged_receipt_cannot_release_test(tmp_path):
    p, records = example(tmp_path)
    receipt = audit(p, records, tmp_path, digest(p))
    saved = tmp_path / "receipt.json"
    pinned = seal(receipt, saved)
    (tmp_path / records[16]["model"]["path"]).write_bytes(b"replacement")
    with pytest.raises(ValueError, match="hash"):
        release_test(p, records, saved, tmp_path, digest(p), pinned)
    saved.write_text("{}")
    with pytest.raises(ValueError, match="receipt"):
        release_test(p, records, saved, tmp_path, digest(p), pinned)


def test_overlap_fails_even_with_rehashed_features(tmp_path):
    p, records = example(tmp_path)
    np.savez(tmp_path / "test.npz", row_ids=[1, 4], x=[[2.0], [3.0]])
    p["test_features"] = entry(tmp_path / "test.npz")
    for r in records:
        r["protocol_sha256"] = digest(p)
    receipt = audit(p, records, tmp_path, digest(p))
    saved = tmp_path / "receipt.json"
    pinned = seal(receipt, saved)
    with pytest.raises(ValueError, match="overlap"):
        release_test(p, records, saved, tmp_path, digest(p), pinned)


def test_forged_winner_is_rejected_even_if_receipt_hash_is_supplied(tmp_path):
    p, records = example(tmp_path)
    receipt = audit(p, records, tmp_path, digest(p))
    receipt["selected"] = "first:00"
    saved = tmp_path / "receipt.json"
    forged_hash = seal(receipt, saved)
    with pytest.raises(ValueError, match="independent selection"):
        release_test(p, records, saved, tmp_path, digest(p), forged_hash)


def test_test_targets_cannot_be_released_as_features(tmp_path):
    p, records = example(tmp_path)
    np.savez(tmp_path / "test.npz", row_ids=[4, 5], x=[[2.0], [3.0]], y=[1.0, 2.0])
    p["test_features"] = entry(tmp_path / "test.npz")
    for record in records:
        record["protocol_sha256"] = digest(p)
    receipt = audit(p, records, tmp_path, digest(p))
    saved = tmp_path / "receipt.json"
    pinned = seal(receipt, saved)
    with pytest.raises(ValueError, match="exclude targets"):
        release_test(p, records, saved, tmp_path, digest(p), pinned)


def test_validation_overlap_fails_before_scoring(tmp_path):
    p, records = example(tmp_path)
    np.savez(tmp_path / "train.npz", row_ids=[0, 2])
    p["train_rows"] = entry(tmp_path / "train.npz")
    with pytest.raises(ValueError, match="overlap"):
        audit(p, records, tmp_path, digest(p))


def multi_example(root):
    p, records = example(root)
    p["application"] = "A6"
    p["selection_weights"] = {"rmse_0": 1.0, "rmse_1": 0.01}
    np.savez(root / "valid.npz", row_ids=[2, 3], y=[[0.0, 0.0], [0.0, 0.0]])
    p["validation"] = entry(root / "valid.npz")
    for i, record in enumerate(records):
        path = root / record["prediction"]["path"]
        np.savez(path, row_ids=[2, 3], prediction=np.tile([1 + i, 100.0], (2, 1)))
        record["prediction"] = entry(path)
        record["protocol_sha256"] = digest(p)
    return p, records


def test_a6_requires_scale_binding(tmp_path):
    p, records = multi_example(tmp_path)
    with pytest.raises(ValueError):
        audit(p, records, tmp_path, digest(p))


def bound_multi_example(root):
    import json

    p, records = multi_example(root)
    np.savez(root / "targets.npz", row_ids=[0, 1], y=[[-1.0, -100.0], [1.0, 100.0]])
    (root / "scale.json").write_text(
        json.dumps(dict(mean=[0.0, 0.0], std=[1.0, 100.0], constant=[False, False]))
    )
    p.update(train_targets=entry(root / "targets.npz"), target_scale=entry(root / "scale.json"))
    for record in records:
        record["protocol_sha256"] = digest(p)
    return p, records


def test_a6_mean_standardized_rmse_and_no_test_access(tmp_path):
    p, records = bound_multi_example(tmp_path)
    (tmp_path / "test.npz").unlink()
    receipt = audit(p, records, tmp_path, digest(p))
    assert receipt["selected"] == "first:00"
    assert receipt["scores"]["first:00"]["selection"] == 1.0
    assert receipt["scores"]["first:01"]["selection"] == 1.5


@pytest.mark.parametrize("mutation", ["weights", "scale", "rows", "targets", "width"])
def test_a6_forged_binding_rejected(tmp_path, mutation):
    import json

    p, records = bound_multi_example(tmp_path)
    if mutation == "weights":
        p["selection_weights"]["rmse_1"] = 1.0
    elif mutation == "scale":
        (tmp_path / "scale.json").write_text(
            json.dumps(dict(mean=[0.0, 0.0], std=[1.0, 1.0], constant=[False, False]))
        )
        p["target_scale"] = entry(tmp_path / "scale.json")
    else:
        rows = [1, 0] if mutation == "rows" else [0, 1]
        y = [[-1.0, -100.0], [1.0, 100.0]]
        if mutation == "targets":
            y[1][0] = 50.0
        if mutation == "width":
            y = [[-1.0], [1.0]]
        np.savez(tmp_path / "targets.npz", row_ids=rows, y=y)
        p["train_targets"] = entry(tmp_path / "targets.npz")
    for record in records:
        record["protocol_sha256"] = digest(p)
    with pytest.raises(ValueError):
        audit(p, records, tmp_path, digest(p))


def test_a6_scale_changes_winner_as_declared(tmp_path):
    p, records = bound_multi_example(tmp_path)
    for record, values in zip(records[:2], ([0.0, 150.0], [2.0, 0.0]), strict=True):
        path = tmp_path / record["prediction"]["path"]
        np.savez(path, row_ids=[2, 3], prediction=np.tile(values, (2, 1)))
        record["prediction"] = entry(path)
    receipt = audit(p, records, tmp_path, digest(p))
    assert receipt["selected"] == "first:00"
    assert receipt["scores"]["first:00"]["selection"] == 0.75
    assert receipt["scores"]["first:01"]["selection"] == 1.0


@pytest.mark.parametrize("steps,accepted", [(1, True), (8, True), (9, False)])
def test_receipt_score_replay_has_bounded_roundoff(tmp_path, steps, accepted):
    p, records = example(tmp_path)
    receipt = audit(p, records, tmp_path, digest(p))
    value = receipt["scores"]["first:00"]["metrics"]["rmse"]
    # Move toward zero to stay in the same binade for the ULP boundary check.
    for _ in range(steps):
        value = np.nextafter(value, 0.0)
    receipt["scores"]["first:00"]["metrics"]["rmse"] = float(value)
    saved = tmp_path / "receipt.json"
    pin = seal(receipt, saved)
    if accepted:
        release_test(p, records, saved, tmp_path, digest(p), pin)
    else:
        with pytest.raises(ValueError, match="independent selection"):
            release_test(p, records, saved, tmp_path, digest(p), pin)


def test_committed_linux_receipt_releases_unchanged_on_current_host():
    import json
    from pathlib import Path

    root = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/v1/evidence/protected-selection-070/selection"
    )
    evaluator = root / "evaluator"
    protocol = json.loads((evaluator / "protocol.json").read_text())
    records = json.loads((evaluator / "records.json").read_text())
    summary = json.loads((evaluator / "summary.json").read_text())
    features, model = release_test(
        protocol,
        records,
        evaluator / "receipt.json",
        root,
        digest(protocol),
        summary["receipt_sha256"],
    )
    assert features["x"].shape == (16, 3)
    assert model == records[15]["model"]


def test_roundoff_allowance_cannot_reverse_a_near_tie(tmp_path):
    p, records = example(tmp_path)
    competitor = records[15]
    np.savez(
        tmp_path / competitor["prediction"]["path"],
        row_ids=[2, 3],
        prediction=[0.5000000000000002, 1.5000000000000002],
    )
    competitor["prediction"] = entry(tmp_path / competitor["prediction"]["path"])
    receipt = audit(p, records, tmp_path, digest(p))
    assert receipt["selected"] == "second:00"
    receipt["scores"]["first:15"]["selection"] = float(np.nextafter(0.5, 0.0))
    saved = tmp_path / "receipt.json"
    pin = seal(receipt, saved)
    with pytest.raises(ValueError, match="independent selection"):
        release_test(p, records, saved, tmp_path, digest(p), pin)


@pytest.mark.parametrize("bad", [True, "16", None, {"extra": 16}, 16.00001])
def test_roundoff_allowance_rejects_invalid_or_material_scores(tmp_path, bad):
    p, records = example(tmp_path)
    receipt = audit(p, records, tmp_path, digest(p))
    receipt["scores"]["first:00"]["metrics"]["rmse"] = bad
    saved = tmp_path / "receipt.json"
    pin = seal(receipt, saved)
    with pytest.raises(ValueError, match="independent selection"):
        release_test(p, records, saved, tmp_path, digest(p), pin)
