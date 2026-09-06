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

from benchmarks.v1.auxiliary import (
    classification,
    normal_pit,
    paired_interval,
    structure_errors,
    survival,
)
from benchmarks.v1.judge import read_json
from benchmarks.v1.preprocessing import fit_target_scale
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


def load_bytes(root, entry):
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
    return raw


def load(root, entry):
    with np.load(io.BytesIO(load_bytes(root, entry)), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return arrays


def report(manifest, directory):
    result = {
        "schema": "openboost-quality-pairs-v1",
        "E3_pass": False,
        "comparisons": {},
        "errors": [],
        "auxiliary_missing": [],
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
            if set(cell) - {"auxiliary"} != {
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
            if app == "A6" and (truth["y"].ndim != 2 or not truth["y"].shape[1]):
                raise ValueError("nonempty vector targets required")
            primary = (
                [f"rmse_{k}" for k in range(truth["y"].shape[1])] + ["standardized_rmse"]
                if app == "A6"
                else PRIMARY[app]
            )
            kind = "nll" if app in ["A10", "A11"] else "ndcg" if app == "A4" else "loss"
            if cell["primary"] != primary or cell["kind"] != kind:
                raise ValueError("cannot remove primary metrics or change comparison kind")
            auxiliary = cell.get("auxiliary", {})
            if not isinstance(auxiliary, dict):
                raise ValueError("auxiliary must be an object")
            if app == "A6":
                if set(auxiliary) != {"train_rows", "train_targets", "target_scale"}:
                    raise ValueError("A6 requires training rows, targets and scale")
                rows = load(root, auxiliary["train_rows"])
                targets = load(root, auxiliary["train_targets"])
                if set(rows) != {"row_ids"} or set(targets) != {"row_ids", "y"}:
                    raise ValueError("invalid A6 training schema")
                train_ids = rows["row_ids"]
                if (
                    train_ids.ndim != 1
                    or not len(train_ids)
                    or len(np.unique(train_ids)) != len(train_ids)
                    or train_ids.dtype.kind not in "iuUS"
                    or ids.dtype.kind != train_ids.dtype.kind
                    or not np.array_equal(train_ids, targets["row_ids"])
                    or np.intersect1d(train_ids, ids).size
                ):
                    raise ValueError("A6 training row mismatch or evaluation overlap")
                if targets["y"].shape != (len(train_ids), truth["y"].shape[1]):
                    raise ValueError("A6 training target shape mismatch")
                scale = read_json(load_bytes(root, auxiliary["target_scale"]))
                expected_scale = fit_target_scale(targets["y"])
                if json.dumps(scale, sort_keys=True) != json.dumps(expected_scale, sort_keys=True):
                    raise ValueError("A6 scale differs from training population")
            elif app == "A10" and auxiliary:
                if set(auxiliary) != {"censoring"}:
                    raise ValueError("invalid survival auxiliary entry")
                support = read_json(load_bytes(root, auxiliary["censoring"]))
            elif app == "A12" and auxiliary:
                if set(auxiliary) != {"structure"}:
                    raise ValueError("invalid structure auxiliary entry")
                support = load(root, auxiliary["structure"])
                if (
                    set(support) != {"age", "row_ids", "train_min", "train_max"}
                    or not np.array_equal(support["row_ids"], ids)
                    or support["train_min"].shape != ()
                    or support["train_max"].shape != ()
                ):
                    raise ValueError("invalid structural support arrays")
            elif auxiliary:
                raise ValueError("unexpected auxiliary entry")
            if app in ["A10", "A12"] and not auxiliary:
                result["auxiliary_missing"].append({"application": app, "fold": fold})
            scores = []
            for role in ["candidate", "baseline"]:
                pred = load(root, cell[role])
                if set(pred) != {"row_ids", "prediction"} or not np.array_equal(
                    ids, pred["row_ids"]
                ):
                    raise ValueError("prediction row identity mismatch")
                kwargs = {k: v for k, v in truth.items() if k not in ["y", "row_ids"]}
                measured = metrics(app, truth["y"], pred["prediction"], row_ids=ids, **kwargs)
                if app == "A6":
                    measured["standardized_rmse"] = float(
                        np.mean([measured[f"rmse_{k}"] / std for k, std in enumerate(scale["std"])])
                    )
                elif app in ["A2", "A3"]:
                    measured = classification(
                        app, truth["y"], pred["prediction"], truth.get("weight")
                    )
                elif app == "A11":
                    measured = normal_pit(truth["y"], pred["prediction"], truth.get("weight"))
                elif app == "A10" and auxiliary:
                    measured = survival(
                        truth["y"], truth["event"], pred["prediction"], support, truth.get("weight")
                    )
                elif app == "A12" and auxiliary:
                    measured["structure_errors"] = structure_errors(
                        support["age"],
                        truth["y"],
                        pred["prediction"],
                        float(support["train_min"]),
                        float(support["train_max"]),
                        truth.get("weight"),
                    )
                scores.append(measured)
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
            for name in primary:
                comparisons[name]["paired_summary"] = paired_interval(
                    [folds[f][0][0][name] for f in range(5)],
                    [folds[f][0][1][name] for f in range(5)],
                )
            result["comparisons"][app] = {
                "pass": all(c["pass"] for c in comparisons.values()),
                "metrics": comparisons,
                "fold_metrics": {
                    str(f): dict(candidate=folds[f][0][0], baseline=folds[f][0][1])
                    for f in range(5)
                },
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
