"""Recompute paired quality comparisons from hashed, row-aligned NPZ artifacts.

This comparison layer does not certify validation-only model selection or the
full required recipe/device matrix, and therefore never declares E3 complete.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from benchmarks.v1.judge import read_json
from benchmarks.v1.quality import compare_folds, metrics

PRIMARY = {
    "A1": ["rmse"],
    "A2": ["logloss"],
    "A3": ["logloss"],
    "A4": ["ndcg10"],
    "A5": ["pinball_0.1", "pinball_0.5", "pinball_0.9"],
    "A7": ["poisson_deviance"],
    "A8": ["gamma_deviance"],
    "A9": ["tweedie_deviance"],
    "A10": ["nll"],
    "A11": ["nll"],
    "A12": ["rmse"],
}


def load(root, entry):
    if set(entry) != {"path", "sha256"}:
        raise ValueError("invalid artifact entry")
    rel = Path(entry["path"])
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("unsafe artifact path")
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError("artifact escapes root")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("artifact hash mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return arrays


def report(manifest, directory):
    result = {
        "schema": "openboost-quality-pairs-v1",
        "E3_pass": False,
        "comparisons": {},
        "errors": [],
        "scope": "paired metrics only; selection provenance and complete recipe/device coverage not certified",
    }
    root = Path(directory).resolve()
    try:
        if (
            set(manifest) != {"schema", "cells"}
            or manifest["schema"] != result["schema"]
            or not manifest["cells"]
        ):
            raise ValueError("invalid or empty quality matrix")
        groups = {}
        seen = set()
        for cell in manifest["cells"]:
            if set(cell) != {
                "application",
                "fold",
                "kind",
                "primary",
                "truth",
                "candidate",
                "baseline",
            }:
                raise ValueError("invalid comparison cell")
            app, fold = cell["application"], cell["fold"]
            if (
                app not in {*PRIMARY, "A6"}
                or type(fold) is not int
                or fold not in range(5)
                or (app, fold) in seen
            ):
                raise ValueError("unknown/duplicate application or fold")
            seen.add((app, fold))
            truth = load(root, cell["truth"])
            if (
                "row_ids" not in truth
                or "y" not in truth
                or set(truth) - {"row_ids", "y", "weight", "event", "query"}
            ):
                raise ValueError("invalid target schema")
            ids = truth["row_ids"]
            if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or len(ids) != len(truth["y"]):
                raise ValueError("invalid target row IDs")
            primary = (
                [f"rmse_{k}" for k in range(truth["y"].shape[1])] if app == "A6" else PRIMARY[app]
            )
            kind = "nll" if app in ["A10", "A11"] else "ndcg" if app == "A4" else "loss"
            if cell["primary"] != primary or cell["kind"] != kind:
                raise ValueError("cannot remove primary metrics or change comparison kind")
            scores = []
            for role in ["candidate", "baseline"]:
                pred = load(root, cell[role])
                if set(pred) != {"row_ids", "prediction"} or not np.array_equal(
                    ids, pred["row_ids"]
                ):
                    raise ValueError("prediction row identity mismatch")
                scores.append(
                    metrics(
                        app,
                        truth["y"],
                        pred["prediction"],
                        row_ids=ids,
                        **{k: v for k, v in truth.items() if k not in ["y", "row_ids"]},
                    )
                )
            groups.setdefault(app, {})[fold] = (scores, kind, primary)
        for app, folds in groups.items():
            if set(folds) != set(range(5)):
                raise ValueError(f"{app}: missing fold")
            primary = folds[0][2]
            kind = folds[0][1]
            if any(f[2] != primary or f[1] != kind for f in folds.values()):
                raise ValueError("inconsistent fold output schema")
            comparisons = {
                name: compare_folds(
                    [folds[f][0][0][name] for f in range(5)],
                    [folds[f][0][1][name] for f in range(5)],
                    kind,
                )
                for name in primary
            }
            result["comparisons"][app] = {
                "pass": all(c["pass"] for c in comparisons.values()),
                "metrics": comparisons,
            }
    except (
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        OSError,
        OverflowError,
        FloatingPointError,
    ) as exc:
        result["errors"].append(str(exc))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    r = report(read_json((a.directory / "quality-manifest.json").read_bytes()), a.directory)
    print(json.dumps(r, indent=2, sort_keys=True, allow_nan=False))
    return int(
        bool(r["errors"])
        or not r["comparisons"]
        or any(not c["pass"] for c in r["comparisons"].values())
    )


if __name__ == "__main__":
    raise SystemExit(main())
