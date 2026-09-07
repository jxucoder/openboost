"""Freeze the full A6 CPU method matrix without launching or certifying a search."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.v1.a6_preflight_plan import compile_plan as openboost_plan
from benchmarks.v1.selection import digest

COMPARATORS = {
    "xgboost": "shared multi_output_tree",
    "lightgbm": "independent target models",
    "catboost": "shared MultiRMSE",
}


def compile_plan(design):
    partial = openboost_plan(design)
    jobs = partial["jobs"]
    for library in COMPARATORS:
        configs = design["families"][library]
        if (
            len(configs) != 16
            or len({digest(c) for c in configs}) != 16
            or {c.get("rounds") for c in configs} != {300, 1000}
            or any(c.get("seed_from_fold") is not True for c in configs)
        ):
            raise ValueError("complete frozen comparator family required")
        jobs.extend(
            dict(
                id=f"{library}:{fold}:{index:02}",
                fold=fold,
                application="A6",
                library=library,
                device="cpu",
                threads=1,
                seed=fold,
                early_stopping_rounds=design["shared"]["early_stopping_rounds"],
                config={**cfg, "bins": design["shared"]["bin_budget"]},
            )
            for fold in range(5)
            for index, cfg in enumerate(configs)
        )
    return dict(
        schema="openboost-a6-cpu-search-plan-v1",
        scope="A6 CPU configuration search only; not complete R/C/A/E coverage or execution evidence",
        search_design_sha256=digest(design),
        methods={
            "openboost-shared": "shared vector topology",
            "openboost-independent": "independent target models",
            **COMPARATORS,
        },
        jobs=jobs,
        job_count=len(jobs),
        trials_per_fold=80,
        maximum_sequential_worker_seconds=len(jobs) * partial["policy"]["timeout_s"],
        maximum_reserved_cpu_seconds=len(jobs) * partial["policy"]["timeout_s"] * 2,
        policy=partial["policy"],
        dispatch_ready=False,
        blockers=[
            "Translate explicit bins=255 in all three comparator adapters; current worker rejects this field",
            "Verify comparator installed fit, stopping, saved-model replay and exact resource policy",
            "Bind all five verified train/validation packets and evaluator-owned selection protocols",
            "Qualify deeper/1000-round resource cases, retaining failures without retries or budget shrink",
            "Bind the complete protocol-derived R/C/A/E ledger before declaring full search readiness",
        ],
        accounting_scope="Worker timeout ceilings only; excludes image/startup, replay, selection, M=1/8/32 and GPU",
        outcome="not_run",
    )


def validate_plan(plan, design):
    """The design must come from the evaluator's pinned input, not the producer."""
    if digest(plan) != digest(compile_plan(design)):
        raise ValueError("A6 CPU plan differs from evaluator-owned design")


def main(output):
    root = Path(__file__).resolve().parent
    plan = compile_plan(json.loads((root / "search-design.json").read_text()))
    record = dict(
        plan=plan,
        input_files={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "search-design.json",
                "datasets/parkinsons.json",
                "datasets/preprocessing.json",
                "baseline_worker.py",
                "requirements-cpu.txt",
                "a6_search_plan.py",
                "a6_preflight_plan.py",
            )
        },
    )
    with Path(output).open("x") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
