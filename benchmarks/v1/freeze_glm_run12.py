"""Freeze and collect the bounded GLM packet locally; never dispatch hardware."""

import argparse
import hashlib
import json
from pathlib import Path

from tests.v1.reference.glm_comparison import CASES
from tests.v1.reference.glm_recipe import SETTINGS

from benchmarks.v1.cuda_aggregation_preflight import snapshot_hashes, snapshot_paths
from benchmarks.v1.cuda_glm_preflight import PROTOCOL
from benchmarks.v1.freeze_comparison_run8 import collect, collect_snapshot

ROOT = Path(__file__).resolve().parents[2]


def artifact_names():
    comparisons = [
        "normal/glm-comparisons/" + hashlib.sha256(c["id"].encode()).hexdigest()[:16] + ".json"
        for c in CASES
    ]
    ptx = [f"normal/glm-comparisons/{family}-ptx.json" for family in ("binary", "poisson")]
    recipes = [
        "normal/glm-recipes/"
        + hashlib.sha256(f"{family}/{depth}/{step}/{rate}".encode()).hexdigest()[:16]
        + ".json"
        for family in ("binary", "poisson")
        for depth in (1, 2)
        for step, rate in SETTINGS
    ]
    return sorted(comparisons + ptx + recipes)


def build_protocol():
    path = ROOT / PROTOCOL
    if path.exists():
        current = json.loads(path.read_text())
        if current["authorization"] != "pending" or current["upload_authorization"] != "pending":
            raise ValueError("only a pending freeze may be regenerated")
        if (ROOT / current["output"]).exists():
            raise ValueError("an attempted run cannot be regenerated")
    previous = json.loads((ROOT / "v1-sprints/105-validation-run11.json").read_text())
    scalar = json.loads((ROOT / "v1-sprints/089-symmetry-run5.json").read_text())
    groups = dict(
        scalar=scalar["test_files"],
        normal=[
            f"tests/v1/{name}.py"
            for name in (
                "test_device_normal_cuda",
                "test_device_normal_comparison_cuda",
                "test_device_comparison_consumers_cuda",
                "test_device_comparison_recipe_cuda",
            )
        ],
        validation=["tests/v1/test_parallel_validation_cuda.py"],
        glm=[
            f"tests/v1/{name}.py"
            for name in (
                "test_device_glm_cuda",
                "test_device_glm_rounds_cuda",
                "test_device_glm_comparison_cuda",
                "test_device_glm_recipes_cuda",
            )
        ],
    )
    tests = [p for group in groups.values() for p in group]
    support = [
        "benchmarks/v1/cuda_glm_preflight.py",
        "benchmarks/v1/normal_build_cpu_env.py",
        "benchmarks/v1/evidence/normal-comparison-092/study.json",
        "tests/v1/run4_score_kernel.py",
        "tests/v1/glm_artifacts.py",
        *[
            f"tests/v1/{name}.py"
            for name in (
                "test_device_split_reference",
                "test_device_round_reference",
                "test_device_normal_reference",
                "test_device_glm_reference",
                "test_loss_change",
            )
        ],
        *[
            f"tests/v1/reference/{name}.py"
            for name in (
                "device_splits",
                "device_rounds",
                "device_normal",
                "normal_precision",
                "normal_acceptance",
                "normal_comparison",
                "coupled",
                "tree",
                "scalar",
                "data",
                "classification",
                "positive",
                "device_glm",
                "glm_comparison",
                "glm_recipe",
            )
        ],
    ]
    expected = collect(tests)
    counts = {
        name: sum(n.split("::")[0] in files for n in expected) for name, files in groups.items()
    }
    artifacts = artifact_names()
    if counts != dict(scalar=212, normal=167, validation=39, glm=153):
        raise ValueError(f"declared cohort accounting differs: {counts}")
    if len(set(artifacts)) != len(artifacts) or len(artifacts) != 77:
        raise ValueError("declared artifact accounting differs")
    protocol = dict(
        scope="108 bounded resident binary/Poisson objective, comparison and recipe validation with relevant scalar/Normal/field regressions. No full R1/R4, Normal, E4, external quality/speed or adoption claim.",
        authorization="pending",
        upload_authorization="pending",
        require_upload_authorization=True,
        upload_destination="Modal",
        app_name="openboost-v1-glm-validation",
        output="benchmarks/v1/evidence/cuda-glm-108",
        budget=dict(
            run=12,
            previous_runs_consumed=11,
            additional_runs=1,
            gpu="T4",
            function_seconds=900,
            test_seconds=600,
            retries=0,
        ),
        resources=dict(
            cpu=2, memory_mib=8192, gpu="T4", timeout_seconds=900, retries=0, max_containers=1
        ),
        image=previous["image"],
        packages=previous["packages"],
        normal_cpu_environment=True,
        cpu_environment_contract="Fresh NumPy 2.3.5 plus installed core only; the sixteen GLM recipe trajectories replay saved models through this interpreter with CUDA/training imports additionally denied.",
        test_files=tests,
        support_files=support,
        expected_cases=expected,
        groups=groups,
        group_counts=counts,
        cohorts=dict(candidate=dict(test_files=tests, expected_cases=expected)),
        retained_artifacts=artifacts,
        retained_artifact_bytes=64 * 1024**2,
        glm_float32=dict(rtol=1e-4, atol=1e-5, metric_rtol=1e-3, metric_atol=1e-3),
        comparison_contract="59 original stored-input cases and every comparison in sixteen recipe trajectories must enclose the independent direct 220-digit objective difference. Frozen expected signs/unchanged controls and six directed-double PTX instructions per family are mandatory. No post-run tolerance or verdict edits.",
        compilation_policy="One pytest process, no explicit prewarm; first-use JIT occurs within the shared test deadline. Existing regressions can warm shared kernels. High-precision audits, exports and saved-model replay are instrumented correctness checks, not end-to-end cost evidence.",
        accounting="571 cases must pass without skips or duplicates, with exact source/package identity and all 77 JSON artifacts. One invocation; 600-second pytest and 900-second function limits; no retries. Image building precedes the function cap, so these resource/invocation bounds are not a guaranteed dollar cap. Preserve partial outputs and failures and stop for retrospective.",
        regression_scope="212 scalar/storage/split/tree/runtime cases, 167 Normal operation/comparison/consumer/recipe cases and 39 field-validation cases. Existing assertions and tolerances remain unchanged. Old compared Normal runtime/installed D2/trajectory and cost cohorts are not repeated; immutable historical evidence remains separate. Regression outcomes are retained in JUnit, not a new complete Normal trace archive.",
    )
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    sources = snapshot_hashes(ROOT, snapshot_paths(ROOT, PROTOCOL, protocol))
    protocol["upload_file_count"] = len(sources)
    protocol["frozen_sources"] = {p: h for p, h in sources.items() if p != PROTOCOL}
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-snapshot", action="store_true")
    args = parser.parse_args()
    protocol = build_protocol()
    if args.collect_snapshot:
        report = collect_snapshot(protocol, protocol_path=PROTOCOL)
        (ROOT / "v1-sprints/108-isolated-collection.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print(
        f"Pending: {protocol['upload_file_count']} files, {len(protocol['expected_cases'])} cases; no dispatch."
    )
