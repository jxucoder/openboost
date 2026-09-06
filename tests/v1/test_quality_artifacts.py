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
