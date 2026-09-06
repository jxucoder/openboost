import hashlib

import numpy as np
from benchmarks.v1.quality_report import report


def test_recomputes_scores_and_rejects_row_permutation(tmp_path):
    cells = []
    for fold in range(5):
        truth = tmp_path / f"truth{fold}.npz"
        pred = tmp_path / f"pred{fold}.npz"
        np.savez(truth, row_ids=[1, 2], y=[0.0, 1.0])
        np.savez(pred, row_ids=[1, 2], prediction=[0.0, 1.0])

        def entry(p):
            return {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

        cells.append(
            {
                "application": "A1",
                "fold": fold,
                "kind": "loss",
                "primary": ["rmse"],
                "truth": entry(truth),
                "candidate": entry(pred),
                "baseline": entry(pred),
            }
        )
    manifest = {"schema": "openboost-quality-pairs-v1", "cells": cells}
    r = report(manifest, tmp_path)
    assert r["comparisons"]["A1"]["pass"]
    assert not r["E3_pass"]  # A2–A13 are absent.
    np.savez(tmp_path / "pred0.npz", row_ids=[2, 1], prediction=[0.0, 1.0])
    # Even with an updated byte hash, row misalignment must fail.
    cells[0]["candidate"]["sha256"] = hashlib.sha256(
        (tmp_path / "pred0.npz").read_bytes()
    ).hexdigest()
    assert report(manifest, tmp_path)["errors"]


def test_no_declared_scores_can_replace_arrays(tmp_path):
    r = report({"schema": "openboost-quality-pairs-v1", "cells": []}, tmp_path)
    assert not r["E3_pass"]


def test_survival_auxiliary_is_hashed_reported_and_support_checked(tmp_path):
    import json

    from benchmarks.v1.preprocessing import censoring_support

    def entry(path):
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    truth = tmp_path / "survival.npz"
    pred = tmp_path / "survival-pred.npz"
    support_path = tmp_path / "censoring.json"
    np.savez(truth, row_ids=[1, 2, 3], y=[1.0, 2.0, 4.0], event=[1, 0, 1])
    np.savez(
        pred, row_ids=[1, 2, 3], prediction=np.column_stack([np.full(3, np.log(2)), np.ones(3)])
    )
    support = censoring_support([1, 2, 3, 4], [1, 0, 1, 1])
    support["grid"] = [2.0]
    support_path.write_text(json.dumps(support))
    cells = [
        dict(
            application="A10",
            fold=f,
            kind="nll",
            primary=["nll"],
            truth=entry(truth),
            candidate=entry(pred),
            baseline=entry(pred),
        )
        for f in range(5)
    ]
    manifest = dict(schema="openboost-quality-pairs-v1", cells=cells)
    missing = report(manifest, tmp_path)
    assert len(missing["auxiliary_missing"]) == 5
    for cell in cells:
        cell["auxiliary"] = {"censoring": entry(support_path)}
    result = report(manifest, tmp_path)
    assert not result["errors"] and not result["auxiliary_missing"]
    assert result["comparisons"]["A10"]["fold_metrics"]["0"]["candidate"]["contributing_rows"] == [
        2
    ]
    assert result["comparisons"]["A10"]["metrics"]["nll"]["paired_summary"]["percentile95"] == [
        0.0,
        0.0,
    ]
    assert not result["E3_pass"]
    support["grid"] = [4.0]
    support_path.write_text(json.dumps(support))
    for cell in cells:
        cell["auxiliary"]["censoring"] = entry(support_path)
    assert report(manifest, tmp_path)["errors"]


def test_structural_auxiliary_preserves_row_identity(tmp_path):
    def entry(path):
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    truth, pred, support = [tmp_path / n for n in ["y.npz", "p.npz", "age.npz"]]
    np.savez(truth, row_ids=[1, 2], y=[0.0, 0.0])
    np.savez(pred, row_ids=[1, 2], prediction=[1.0, 1.0])
    np.savez(support, row_ids=[1, 2], age=[1.0, 3.0], train_min=1.0, train_max=2.0)
    cells = [
        dict(
            application="A12",
            fold=f,
            kind="loss",
            primary=["rmse"],
            truth=entry(truth),
            candidate=entry(pred),
            baseline=entry(pred),
            auxiliary={"structure": entry(support)},
        )
        for f in range(5)
    ]
    manifest = dict(schema="openboost-quality-pairs-v1", cells=cells)
    result = report(manifest, tmp_path)
    assert not result["errors"] and not result["auxiliary_missing"]
    assert result["comparisons"]["A12"]["fold_metrics"]["0"]["candidate"]["structure_errors"][
        "above"
    ] == {"rows": 1, "rmse": 1.0}
    np.savez(support, row_ids=[2, 1], age=[1.0, 3.0], train_min=1.0, train_max=2.0)
    for cell in cells:
        cell["auxiliary"]["structure"] = entry(support)
    assert report(manifest, tmp_path)["errors"]


def entry(path):
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def multioutput_fixture(root):
    import json

    from benchmarks.v1.preprocessing import fit_target_scale

    cells = []
    for fold in range(5):

        def save(name, fold=fold, **values):
            path = root / f"{fold}-{name}.npz"
            np.savez(path, **values)
            return entry(path)

        target = np.array([[-1.0, -100.0, 7.0], [1.0, 100.0, 7.0]])
        scale_path = root / f"{fold}-scale.json"
        scale_path.write_text(json.dumps(fit_target_scale(target)))
        cells.append(
            dict(
                application="A6",
                fold=fold,
                kind="loss",
                primary=["rmse_0", "rmse_1", "rmse_2", "standardized_rmse"],
                truth=save("truth", row_ids=[2, 3], y=np.zeros((2, 3))),
                candidate=save(
                    "candidate", row_ids=[2, 3], prediction=np.tile([1.0, 100.0, 0.0], (2, 1))
                ),
                baseline=save(
                    "baseline", row_ids=[2, 3], prediction=np.tile([1.0, 100.0, 0.0], (2, 1))
                ),
                auxiliary=dict(
                    train_rows=save("rows", row_ids=[0, 1]),
                    train_targets=save("targets", row_ids=[0, 1], y=target),
                    target_scale=entry(scale_path),
                ),
            )
        )
    return dict(schema="openboost-quality-pairs-v1", cells=cells)


def test_a6_reports_standardized_average(tmp_path):
    manifest = multioutput_fixture(tmp_path)
    result = report(manifest, tmp_path)
    assert not result["errors"]
    actual = result["comparisons"]["A6"]
    assert actual["pass"]
    assert actual["fold_metrics"]["0"]["candidate"]["standardized_rmse"] == 2 / 3
    assert set(actual["metrics"]) == {"rmse_0", "rmse_1", "rmse_2", "standardized_rmse"}
    assert not result["E3_pass"]


def test_a6_average_cannot_hide_target_regression(tmp_path):
    manifest = multioutput_fixture(tmp_path)
    for cell in manifest["cells"]:
        path = tmp_path / cell["candidate"]["path"]
        np.savez(path, row_ids=[2, 3], prediction=np.tile([1.2, 0.0, 0.0], (2, 1)))
        cell["candidate"] = entry(path)
    result = report(manifest, tmp_path)
    assert not result["errors"]
    comparison = result["comparisons"]["A6"]
    assert comparison["metrics"]["standardized_rmse"]["pass"]
    assert not comparison["metrics"]["rmse_0"]["pass"]
    assert not comparison["pass"]


def test_a6_bad_scale_support_is_rejected(tmp_path):
    import copy
    import json

    manifest = multioutput_fixture(tmp_path)
    original = copy.deepcopy(manifest)
    manifest["cells"][0].pop("auxiliary")
    assert report(manifest, tmp_path)["errors"]
    manifest = copy.deepcopy(original)
    manifest["cells"][0]["primary"].remove("rmse_0")
    assert report(manifest, tmp_path)["errors"]
    manifest = copy.deepcopy(original)
    path = tmp_path / manifest["cells"][0]["auxiliary"]["target_scale"]["path"]
    record = json.loads(path.read_text())
    record["std"][1] = 1.0
    path.write_text(json.dumps(record))
    manifest["cells"][0]["auxiliary"]["target_scale"] = entry(path)
    assert report(manifest, tmp_path)["errors"]


def test_a6_training_overlap_and_target_permutation_fail(tmp_path):
    manifest = multioutput_fixture(tmp_path)
    support = manifest["cells"][0]["auxiliary"]
    path = tmp_path / support["train_targets"]["path"]
    np.savez(path, row_ids=[1, 0], y=[[-1.0, -100.0, 7.0], [1.0, 100.0, 7.0]])
    support["train_targets"] = entry(path)
    assert report(manifest, tmp_path)["errors"]
    row_path = tmp_path / support["train_rows"]["path"]
    np.savez(row_path, row_ids=[1, 2])
    np.savez(path, row_ids=[1, 2], y=[[-1.0, -100.0, 7.0], [1.0, 100.0, 7.0]])
    support.update(train_rows=entry(row_path), train_targets=entry(path))
    assert report(manifest, tmp_path)["errors"]


def test_a6_scale_cannot_use_evaluation_targets(tmp_path):
    import json

    from benchmarks.v1.preprocessing import fit_target_scale

    manifest = multioutput_fixture(tmp_path)
    cell = manifest["cells"][0]
    path = tmp_path / cell["auxiliary"]["target_scale"]["path"]
    path.write_text(json.dumps(fit_target_scale(np.zeros((2, 3)))))
    cell["auxiliary"]["target_scale"] = entry(path)
    assert report(manifest, tmp_path)["errors"]
