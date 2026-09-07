"""Compile the OpenBoost portion of A6 resource preflight; never launch jobs."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.v1.selection import digest


def compile_plan(design):
    if design["schema"] != "openboost-search-design-v1":
        raise ValueError("unknown search design")
    budgets = design["budgets"]
    if any(
        budgets[k] != v
        for k, v in dict(trial_wall_s=1800, ram_mib=8192, cpu_threads=2, trial_retries=0).items()
    ):
        raise ValueError("resource policy differs from frozen preflight")
    if design["selection"]["folds"] != 5 or design["selection"]["trials_per_method"] != 16:
        raise ValueError("five folds and sixteen trials required")
    configs = design["families"]["xgboost"]
    fields = {"rounds", "learning_rate", "max_depth", "reg_lambda", "seed_from_fold"}
    if (
        len(configs) != 16
        or any(set(c) != fields for c in configs)
        or len({digest(c) for c in configs}) != 16
        or {c["rounds"] for c in configs} != {300, 1000}
        or any(c["seed_from_fold"] is not True for c in configs)
    ):
        raise ValueError("complete supported frozen numeric-tree configuration family required")
    jobs = [
        dict(
            id=f"openboost-{mode}:{fold}:{index:02}",
            fold=fold,
            application="A6",
            library="openboost",
            device="cpu",
            threads=1,
            seed=fold,
            early_stopping_rounds=design["shared"]["early_stopping_rounds"],
            config={**config, "mode": mode, "bins": design["shared"]["bin_budget"]},
        )
        for mode in ("shared", "independent")
        for fold in range(5)
        for index, config in enumerate(configs)
    ]
    return dict(
        schema="openboost-a6-resource-plan-v1",
        search_design_sha256=digest(design),
        scope="OpenBoost-only resource planning; not complete comparator coverage or launch authorization",
        jobs=jobs,
        job_count=len(jobs),
        trials_per_fold=32,
        maximum_sequential_worker_seconds=len(jobs) * budgets["trial_wall_s"],
        maximum_reserved_cpu_seconds=len(jobs) * budgets["trial_wall_s"] * budgets["cpu_threads"],
        policy=dict(
            timeout_s=budgets["trial_wall_s"],
            address_limit_bytes=8 * 1024**3,
            requested_container_memory_mib=budgets["ram_mib"],
            reserved_cpus=2,
            worker_threads=1,
            retries=0,
            retention="summary",
        ),
        first_probe_ids=["openboost-shared:0:00", "openboost-independent:0:00"],
        stop_on_failure=True,
        open_requirements=[
            "Bind verified real train/validation packets without test material",
            "Run exact full-round workers under protected resource policy",
            "Complete all required comparator methods and coverage ledger",
            "Qualify selected-model release and per-target real quality",
        ],
    )


def main(output):
    design_path = Path(__file__).with_name("search-design.json")
    plan = compile_plan(json.loads(design_path.read_text()))
    plan["input_files"] = {
        str(path.relative_to(design_path.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            design_path,
            design_path.parent / "datasets/preprocessing.json",
            design_path.parent / "datasets/parkinsons.json",
        )
    }
    with Path(output).open("x") as stream:
        stream.write(json.dumps(plan, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
