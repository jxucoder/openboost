"""105 exact-algorithm GPU comparison; partial repetitions never pass cost gates."""

from statistics import median

import numpy as np

from benchmarks.v1.performance_evidence import load_array, verify_report


def judge(baseline, candidate, inputs, *, baseline_sources, candidate_sources, limit, cpu=None):
    result = dict(
        measurement_complete=False,
        quality_passed=False,
        cost_passed=False,
        candidate_over_baseline_warm_ratio=None,
        reasons=[],
    )
    try:
        for name, report, sources in (
            ("baseline", baseline, baseline_sources),
            ("candidate", candidate, candidate_sources),
        ):
            if report.get("profile", False):
                raise ValueError(name + " profile is ineligible for cost ratios")
            if report.get("backend") != "cuda":
                raise ValueError(name + " CUDA report missing")
            checked = verify_report(
                report, inputs, inputs["sha256"], expected_repetitions=4, expected_sources=sources
            )
            if not checked["complete"]:
                raise ValueError(name + ": " + report["status"])
        if cpu is not None:
            if cpu.get("backend") != "cpu":
                raise ValueError("CPU control missing")
            checked = verify_report(
                cpu,
                inputs,
                inputs["sha256"],
                expected_repetitions=2,
                expected_sources=candidate_sources,
            )
            if not checked["complete"]:
                raise ValueError("cpu: " + cpu["status"])
        result["measurement_complete"] = True
        for left, right in zip(baseline["fits"], candidate["fits"], strict=True):
            if left["model_sha256"] != right["model_sha256"] or not np.array_equal(
                load_array(left["validation_raw"]), load_array(right["validation_raw"])
            ):
                raise ValueError("validation scheduling changed model/predictions")
            for key in ("kernel_launches", "validation_export_bytes"):
                if (
                    left["measurement"]["final_metrics"][key]
                    != right["measurement"]["final_metrics"][key]
                ):
                    raise ValueError("validation launch/flag contract changed")
        if cpu is not None:
            reference, actual = cpu["fits"][-1], candidate["fits"][-1]
            values = load_array(reference["validation_raw"])
            error = np.sqrt(np.mean((load_array(actual["validation_raw"]) - values) ** 2, axis=0))
            error /= np.maximum(1, np.std(values, axis=0))
            deltas = {
                k: abs(actual["quality"][k] - value) / max(abs(value), 1e-12)
                for k, value in reference["quality"].items()
            }
            result.update(
                cpu_relative_metric_differences=deltas,
                cpu_normalized_prediction_rmse=error.tolist(),
            )
            if max(error) > 0.01 or max(deltas.values()) > 0.01:
                raise ValueError("CPU control exceeds frozen 1% quality/prediction difference")
        result["quality_passed"] = True
        ratio = median(f["measurement"]["fit_seconds"] for f in candidate["fits"][1:]) / median(
            f["measurement"]["fit_seconds"] for f in baseline["fits"][1:]
        )
        result.update(candidate_over_baseline_warm_ratio=ratio, cost_passed=ratio <= limit)
        if ratio > limit:
            result["reasons"].append("warm fit exceeds frozen cost limit")
    except (ValueError, KeyError, TypeError) as error:
        result["reasons"].append(str(error))
    return result
