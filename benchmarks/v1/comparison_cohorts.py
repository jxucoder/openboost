"""Local 092 requirement bindings and reference trajectories; never dispatches CUDA."""

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path

from tests.v1.reference import compared_normal, device_normal

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = "v1-sprints/092-historical-case-mapping.json"
BINDINGS = "v1-sprints/092-collected-case-bindings.json"
REFERENCE = "v1-sprints/092-reference-trajectories.json"


def bindings():
    historical = json.loads((ROOT / HISTORICAL).read_text())
    cases = []
    for item in historical["cases"]:
        node = item["historical_node"]
        revised = node
        if item["disposition"] != "unchanged-operations":
            revised = node.replace("test_device_normal_", "test_compared_normal_")
        cases.append(
            dict(
                historical_node=node,
                planned_requirement=item["planned_requirement"],
                revised_node=revised,
                disposition=item["disposition"],
                historical_outcome=item["historical_outcome"],
                revised_status="collected_not_run",
            )
        )
    return dict(
        schema="openboost-092-collected-bindings-v1",
        historical_mapping_sha256=hashlib.sha256((ROOT / HISTORICAL).read_bytes()).hexdigest(),
        original_tolerances=historical["original_tolerances"],
        counts=dict(Counter(c["disposition"] for c in cases)),
        cases=cases,
        device_execution=False,
    )


def summary(steps):
    return [
        dict(
            round=s["round"],
            channels=list(s["channels"]),
            accepted=s["accepted"],
            coefficients=[a[0] for a in s["attempts"]],
            version=s["version"],
            n_terms=s["nterms"],
            best_terms=s["best_terms"],
            loss=s["loss"],
            validation_score=s["validation_score"],
            best_score=s["best_score"],
        )
        for s in steps
    ]


def trajectories():
    cases = []
    settings = product(
        [
            ("weighted", 1, None),
            ("d2", 1, None),
            ("d2", 2, 1),
            ("conflict", 0, None),
            ("conflict", 2, None),
        ],
        [("ordinary", 0), ("natural", 0), ("natural", 0.25)],
        ["joint", "forward", "reverse"],
        [(True, 0.1), (False, 8.0)],
    )
    for (case, depth, minimum), (mode, damping), update, (fixed, rate) in settings:
        options = dict(
            depth=depth,
            minimum=minimum,
            mode=mode,
            damping=damping,
            update=update,
            fixed=fixed,
            rate=rate,
        )
        initial, old = device_normal.rounds(case, **options)
        _, new = compared_normal.rounds(case, **options)
        before, after = summary(old), summary(new)
        cases.append(
            dict(
                case=case,
                options=options,
                initial=initial.tolist(),
                historical=before,
                revised=after,
                changed=before != after,
            )
        )
    return dict(
        schema="openboost-092-reference-trajectories-v1",
        device_execution=False,
        interpretation="Independent float64 original-row trajectories; CUDA results remain unrun.",
        sources={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "tests/v1/reference").glob("*.py")
            if p.name
            in (
                "compared_normal.py",
                "device_normal.py",
                "device_rounds.py",
                "device_splits.py",
                "normal_comparison.py",
            )
        },
        changed_cases=sum(c["changed"] for c in cases),
        cases=cases,
    )


if __name__ == "__main__":
    for path, value in ((BINDINGS, bindings()), (REFERENCE, trajectories())):
        (ROOT / path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
