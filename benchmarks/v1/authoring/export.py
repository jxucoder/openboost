"""Export public D1/D2 development fixtures and a standalone numerical judge."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from tests.v1.reference.author import expectile, expectile_base
from tests.v1.reference.tree import fit_tree

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_MODULES = ("__init__", "author", "coupled", "positive", "quantile", "scalar", "tree")


def fixtures():
    """Public development cases only; never load sealed author tasks."""
    cases = []
    for tau in (0.5, 0.8):
        cases.append(
            dict(
                id=f"d1-{tau}",
                task="D1",
                tau=tau,
                values=[[0], [1], [2], [3], [4], [None]],
                target=[-3, 0, 2, 2, 90, 7],
                weight=[2, 1, 3, 1, 0, 2],
                offset=[1, -1, 0, 2, 1, -2],
                geometry_raw=[-4, 1, 2, 0, 90, 9],
                rounds=2,
                learning_rate=0.1,
                bins=6,
                policy="depthwise",
            )
        )
    for policy in ("depthwise", "best_first", "symmetric"):
        for variant in ("ordinary", "zero-weight", "impossible"):
            grouped = variant == "impossible"
            cases.append(
                dict(
                    id=f"d2-{policy}-{variant}",
                    task="D2",
                    policy=policy,
                    values=[[i // 3 if grouped else i] for i in range(6)],
                    target=[6, -1, -1, -1, -1, -2],
                    offset=[0] * 6,
                    weight=[0 if variant == "zero-weight" else 1, 1, 1, 1, 1, 1],
                    information=np.eye(2)[
                        [0, 0, 0, 1, 1, 1] if grouped else [0, 1, 0, 1, 0, 1]
                    ].tolist(),
                    probe_gradient=[-6, 1, 1, 1, 1, 2],
                    rounds=2,
                    learning_rate=0.1,
                    bins=6,
                )
            )
    return cases


def expected(case):
    x = np.asarray(case["values"], dtype=float)
    y, w, offset = (np.array(case[k], dtype=float) for k in ("target", "weight", "offset"))
    d1 = case["task"] == "D1"
    info = {} if d1 else dict(information=case["information"], min_information=1)
    base = (
        expectile_base(y - offset, weight=w, tau=case["tau"])
        if d1
        else float(np.dot(w / w.sum(), y - offset))
    )

    def geometry(raw):
        if d1:
            return expectile(raw + offset, y, weight=w, tau=case["tau"])
        residual = raw + offset - y
        return float(np.dot(w / w.sum(), residual**2) / 2), residual, np.ones(len(y))

    raw = np.full(len(y), base)
    trace = []
    for _ in range(case["rounds"]):
        loss, g, h = geometry(raw)
        tree = fit_tree(x, g, h, weight=w, policy=case["policy"], max_depth=2, **info)
        before = raw.copy()
        raw = raw + case["learning_rate"] * tree.predict(x)
        trace.append(
            dict(
                loss=loss,
                gradient=g.tolist(),
                curvature=h.tolist(),
                before=before.tolist(),
                raw=raw.tolist(),
            )
        )
    result = dict(base=base, trace=trace, raw=raw.tolist())
    rejection_names = (
        ("tau_zero", "tau_one", "tau_nan", "tau_boolean", "rate_zero", "rate_infinite")
        if d1
        else (
            "rows",
            "rank",
            "empty",
            "negative",
            "nan",
            "infinite",
            "foreign_problem",
            "foreign_rows",
        )
    )
    result["rejections"] = dict.fromkeys(rejection_names, True)
    if d1:
        loss, g, h = geometry(np.array(case["geometry_raw"]))
        result["geometry"] = dict(loss=loss, gradient=g.tolist(), curvature=h.tolist())
    else:
        tree = fit_tree(
            x,
            case["probe_gradient"],
            np.ones(len(y)),
            weight=w,
            policy=case["policy"],
            max_depth=1,
            **info,
        )
        result["probe"] = dict(
            root_cut=None if tree.nodes[0].condition is None else tree.nodes[0].condition.threshold,
            prediction=tree.predict(x).tolist(),
            information=case["information"],
        )
    return result


def export(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    cases = fixtures()
    payloads = {"inputs.json": cases, "expected.json": {c["id"]: expected(c) for c in cases}}
    for name, payload in payloads.items():
        (destination / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    shutil.copyfile(Path(__file__).with_name("judge.py"), destination / "judge.py")
    # Explicit public closure; unrelated modules loaded by a caller cannot widen it.
    sources = [Path(__file__)] + [
        ROOT / f"tests/v1/reference/{name}.py" for name in REFERENCE_MODULES
    ]
    manifest = dict(
        schema="openboost-d1-d2-development-verifier-v2",
        dispatch_ready=False,
        attempts=[],
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        runtime_files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.iterdir())
        },
        oracle_sources={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(sources))
        },
        dependencies=["Python", "numpy", "installed openboost wheel"],
        limitation="development numerical checks; OS isolation and budget accounting unverified",
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    export(parser.parse_args().destination)
