"""Predeclared P7 comparison gates; failures are data, not missing evidence."""

import math
import statistics

CONFIG = dict(
    n_trees=30,
    max_depth=3,
    learning_rate=0.05,
    n_bins=64,
    min_child_weight=1.0,
    reg_lambda=1.0,
    subsample=1.0,
    colsample_bytree=1.0,
)

STRATEGIES = ("legacy_cpu", "legacy_cuda", "experimental_cuda", "extensions_cuda")


def summarize(cells, frozen):
    expected = {(seed, name) for seed in (0, 1, 2) for name in STRATEGIES}
    indexed = {(c["seed"], c["strategy"]): c for c in cells}
    if len(cells) != len(expected) or set(indexed) != expected:
        raise ValueError("Incomplete or duplicate value matrix")
    frozen_index = {(c["seed"], c["backend"], c["mode"]): c for c in frozen}
    if any((seed, "cuda", "resident") not in frozen_index for seed in (0, 1, 2)):
        raise ValueError("Missing frozen reference")
    comparisons = []
    for seed in (0, 1, 2):
        old = next(
            c
            for c in frozen
            if c["seed"] == seed and c["backend"] == "cuda" and c["mode"] == "resident"
        )["records"][1]["metrics"]
        for strategy in STRATEGIES:
            cell = indexed[seed, strategy]
            if len(cell.get("records", [])) != 4 or cell.get("error"):
                raise ValueError("Missing first plus three warm fits")
            for record in cell["records"]:
                values = [
                    record["fit_s"],
                    record["predict_s"],
                    *(record["metrics"][k] for k in ("nll", "crps", "coverage90")),
                ]
                if (
                    any(not math.isfinite(v) for v in values)
                    or record["fit_s"] <= 0
                    or record["predict_s"] <= 0
                ):
                    raise ValueError("Invalid timings/metrics")
                if record["fallback_warnings"]:
                    raise ValueError("Unexpected fallback")
            cell["warm_fit_median_s"] = statistics.median(r["fit_s"] for r in cell["records"][1:])
            cell["warm_predict_median_s"] = statistics.median(
                r["predict_s"] for r in cell["records"][1:]
            )
        legacy, candidate = (indexed[seed, s] for s in ("legacy_cuda", "experimental_cuda"))

        def quality(reference, actual):
            return (
                abs(actual["nll"] - reference["nll"]) <= 0.01 * max(1, abs(reference["nll"]))
                and actual["crps"] <= 1.01 * reference["crps"]
                and abs(actual["coverage90"] - reference["coverage90"]) <= 0.01
            )

        comparisons.append(
            {
                "seed": seed,
                "quality_pass": all(
                    quality(a["metrics"], b["metrics"])
                    for a, b in zip(legacy["records"], candidate["records"], strict=True)
                ),
                "legacy_matches_frozen_quality": all(
                    quality(old, a["metrics"]) for a in legacy["records"]
                ),
                "warm_fit_ratio": candidate["warm_fit_median_s"] / legacy["warm_fit_median_s"],
            }
        )
    medians = {
        s: statistics.median(indexed[seed, s]["warm_fit_median_s"] for seed in (0, 1, 2))
        for s in STRATEGIES
    }
    ratio = medians["experimental_cuda"] / medians["legacy_cuda"]
    return {
        "per_seed": comparisons,
        "warm_fit_medians_s": medians,
        "default_fit_ratio": ratio,
        "profiling_triggered": ratio > 1.2,
        "quality_pass": all(
            c["quality_pass"] and c["legacy_matches_frozen_quality"] for c in comparisons
        ),
        "performance_budget_pass": ratio <= 1.2,
        "scope": "resident fit only; eval/callbacks unsupported on strict GPU; T4 seconds are not billed dollars",
    }


def validate_profiles(cells, *, profile_only=False):
    """Reject a green matrix without the promised separate profile evidence."""
    if profile_only and (
        len(cells) != 4
        or {(c["seed"], c["strategy"]) for c in cells} != {(0, s) for s in STRATEGIES}
    ):
        raise ValueError("Incomplete isolated profile matrix")
    for cell in cells:
        if (
            cell.get("config") != CONFIG
            or cell.get("mode") != "resident"
            or cell.get("split_sizes") != [12384, 4128, 4128]
        ):
            raise ValueError("Value configuration changed")
        expected_phases = (
            ["untimed_warmup"] if profile_only else ["process_first", "warm_1", "warm_2", "warm_3"]
        )
        if [r.get("phase") for r in cell["records"]] != expected_phases:
            raise ValueError("Value repetition order changed")
        profile = cell.get("profile", {})
        if (
            not profile.get("top_host_functions")
            or not math.isfinite(profile.get("wall_s", float("nan")))
            or profile["wall_s"] <= 0
        ):
            raise ValueError("Missing separate profile")
        if profile_only and (
            profile.get("isolated_host_profile") is not True
            or not profile.get("synchronized_inclusive_timers")
        ):
            raise ValueError("Missing isolated host profile")
        expected_device = "cpu" if cell["strategy"] == "legacy_cpu" else "cuda"
        if any(r.get("actual_device") != expected_device for r in cell["records"]):
            raise ValueError("Unexpected execution device")
        if expected_device == "cuda":
            counts = {
                (r["file"], r["function"]): r["calls"] for r in profile.get("path_functions", [])
            }
            if cell["strategy"] == "legacy_cuda":
                if (
                    counts.get(("_tree.py", "fit_tree_gpu_native")) != 60
                    or counts.get(("_objectives.py", "step")) != 30
                ):
                    raise ValueError("Missing native CUDA path")
            elif (
                counts.get(("_device.py", "step")) != 30
                or counts.get(("_levelwise.py", "build")) != 60
                or counts.get(("_tree.py", "fit_tree_gpu_native"), 0)
            ):
                raise ValueError("Missing strict CUDA path")
            memory = profile.get("memory", {})
            if memory.get("errors") != [] or memory.get("samples", 0) < 2:
                raise ValueError("Missing sampled CUDA memory")
            initial, peak, total = (
                memory.get(k, -1)
                for k in ("initial_used_bytes", "sampled_peak_used_bytes", "total_bytes")
            )
            if (
                not (0 <= initial <= peak <= total)
                or memory.get("sampled_peak_delta_bytes") != peak - initial
            ):
                raise ValueError("Invalid sampled CUDA memory")
