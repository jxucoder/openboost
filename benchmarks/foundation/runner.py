"""Validate returned smoke evidence; usable offline without Modal installed."""

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REQUIRED_TESTS = {"test_device_interop", "test_normal_gpu_fit"}


def validate_result(manifest, result):
    if result.get("returncode") != 0 or result.get("timed_out", True):
        raise ValueError("Remote test process failed or timed out")
    if manifest.get("source_dirty") is not False:
        raise ValueError("Source must be committed and clean")
    for key in ("wheel_sha256", "source_sha"):
        if not manifest.get(key) or result.get(key) != manifest[key]:
            raise ValueError(f"Provenance mismatch: {key}")
    try:
        root = ET.fromstring(result.get("junit", ""))
    except ET.ParseError as exc:
        raise ValueError("Missing or invalid JUnit report") from exc
    required = set(REQUIRED_TESTS)
    suite = manifest.get("suite", "smoke")
    if suite in ("correctness", "boundaries", "baseline"):
        required.update({"test_weighted_newton", "test_weighted_distribution[normal]", "test_weighted_distribution[poisson]"})
    elif suite in ("histograms", "splits", "leaves", "builder", "trainer"):
        required.add("test_batch_histogram_device_oracle")
        if suite in ("splits", "leaves", "builder", "trainer"):
            required.add("test_batch_split_routing_oracle")
        if suite in ("leaves", "builder", "trainer"):
            required.add("test_batch_leaf_rule_oracle")
        if suite in ("builder", "trainer"):
            required.add("test_levelwise_builder_device_oracle")
        if suite == "trainer":
            required.add("test_strict_extension_trainer")
    elif suite != "smoke":
        raise ValueError("Unknown evidence suite")
    if suite in ("boundaries", "baseline"):
        required.update({"test_visible_fallback[custom]", "test_visible_fallback[exposure]", "test_visible_fallback[generic]", "test_device_error_rolls_back", "test_device_sampling_preflight[subsample]", "test_device_sampling_preflight[colsample_bytree]", "test_eval_callback_persistence[normal]", "test_eval_callback_persistence[poisson]"})
    if suite == "baseline":
        required.add("test_baseline_matrix")
    cases = list(root.iter("testcase"))
    names = [case.get("name") for case in cases]
    if len(names) != len(required) or set(names) != required:
        raise ValueError("Required GPU cases missing or duplicated")
    if any(list(root.iter(tag)) for tag in ("failure", "error", "skipped")):
        raise ValueError("GPU cases failed, errored, or skipped")
    env = result.get("environment", {})
    checks = result.get("checks", {})
    if not env.get("cuda_available") or not env.get("gpu_name"):
        raise ValueError("No real CUDA environment evidence")
    if not (
        checks.get("interop") is True
        and checks.get("native_tree_calls") == 4
        and checks.get("device_objective_calls") == 2
        and checks.get("installed_files_verified", 0) > 0
        and checks.get("dataset_sha256")
    ):
        raise ValueError("Missing installed-wheel/device-path checks (possible fallback)")
    if suite in ("histograms", "splits", "leaves", "builder", "trainer"):
        batch = checks.get("batch_histograms", {})
        if batch.get("device_arrays") is not True or batch.get("legacy_download_wrappers_blocked") is not True or len(batch.get("cases", [])) != 3:
            raise ValueError("Missing batch histogram device/oracle checks")
    if suite in ("splits", "leaves", "builder", "trainer"):
        split = checks.get("batch_splits", {})
        if split.get("device_arrays") is not True or split.get("routed_child_oracle") is not True or split.get("exact_ties_and_gain_boundary") is not True or len(split.get("cases", [])) != 4:
            raise ValueError("Missing split/routing oracle checks")
    if suite in ("leaves", "builder", "trainer"):
        leaf = checks.get("batch_leaves", {})
        if leaf.get("device_arrays") is not True or leaf.get("row_sum_oracle") is not True or leaf.get("bounded_changes_next_gradient") is not True or len(leaf.get("cases", [])) != 3 or len(leaf.get("two_rounds", [])) != 2:
            raise ValueError("Missing leaf rule/reduction oracle checks")
    if suite in ("builder", "trainer"):
        builder = checks.get("levelwise_builder", {})
        if builder.get("device_cache_survives_owner_release") is not True or builder.get("cpu_load_prediction") is not True or builder.get("compact_transfer_calls") != 85 or len(builder.get("two_channel_cases", [])) != 4:
            raise ValueError("Missing level-wise builder evidence")
    if suite == "trainer":
        execution = checks.get("strict_extension_trainer", {})
        if not all(execution.get(k) is True for k in ("actual_fit", "legacy_dispatch_blocked", "rollback", "cpu_load_prediction")) or execution.get("compact_transfer_calls") != 40 or len(execution.get("cases", [])) != 2 or len(execution.get("adapter_cases", [])) != 4 or execution.get("additional_invalid_statistics") != 3:
            raise ValueError("Missing strict extension trainer evidence")
    if suite == "baseline":
        validate_baseline(checks.get("baseline_cells", []))


def validate_baseline(cells):
    expected = {(seed, mode, backend) for seed in (0, 1, 2) for mode in ("resident", "eval") for backend in ("cpu", "cuda")}
    if len(cells) != len(expected):
        raise ValueError("Incomplete baseline matrix")
    indexed = {(c.get("seed"), c.get("mode"), c.get("backend")): c for c in cells}
    if set(indexed) != expected:
        raise ValueError("Missing or duplicated baseline cells")
    for (_, _, backend), cell in indexed.items():
        records = cell.get("records", [])
        if [r.get("phase") for r in records] != ["first_fit", "repeat_fit"]:
            raise ValueError("Missing first/repeated fit")
        for r in records:
            if r.get("fallback_warnings") != []:
                raise ValueError("Baseline fallback")
            path = r.get("fit_path", {})
            if path.get("objective_calls") != 30 or path.get("native_tree_calls") != (60 if backend == "cuda" else 0):
                raise ValueError("Baseline device path mismatch")
            for name in ("fit_s", "predict_params_s"):
                if not math.isfinite(r.get(name, float("nan"))) or r[name] <= 0:
                    raise ValueError("Invalid baseline timing")
            for name in ("nll", "crps", "coverage90"):
                if not math.isfinite(r.get("metrics", {}).get(name, float("nan"))):
                    raise ValueError("Invalid baseline metric")
    for seed in (0, 1, 2):
        for mode in ("resident", "eval"):
            a, b = (indexed[seed, mode, backend]["records"][1]["metrics"] for backend in ("cpu", "cuda"))
            if (abs(b["nll"] - a["nll"]) > .01 * max(1, abs(a["nll"]))
                    or b["crps"] > a["crps"] * 1.01
                    or abs(b["coverage90"] - a["coverage90"]) > .01):
                raise ValueError("Baseline quality gate failed")


def main():
    directory = Path(sys.argv[1])
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        result = json.loads((directory / "results.json").read_text())
        validate_result(manifest, result)
    except (OSError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Foundation {manifest.get('suite', 'smoke')} evidence passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
