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
