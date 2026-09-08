"""104 quality-qualified internal timing comparisons; incomplete work remains explicit."""

from statistics import median

import numpy as np

from benchmarks.v1.performance_checkpoint import CONFIG, quality, workload


def judge(cpu, cuda, *, expected_sources=None):
    reasons = []
    for backend, result in (("cpu", cpu), ("cuda", cuda)):
        if result.get("status") != "complete":
            reasons.append(f"{backend}: {result.get('status', 'missing')}")
        elif not result.get("measurements"):
            reasons.append(backend + ": measurements missing")
    if reasons:
        return dict(
            measurement_complete=False,
            quality_comparable=False,
            reasons=reasons,
            cpu_over_cuda_warm_fit_ratio=None,
        )
    rows, recipe = cpu["train_rows"], cpu["recipe"]
    train, validation = workload(rows, recipe)
    for backend, result in (("cpu", cpu), ("cuda", cuda)):
        if (
            expected_sources is not None
            and result["environment"]["core_sources"] != expected_sources
        ):
            reasons.append(backend + ": installed sources differ")
        if (result["backend"], result["recipe"], result["train_rows"]) != (backend, recipe, rows):
            reasons.append(backend + ": workload/backend differs")
        if result["config"] != CONFIG or result["profile"]:
            reasons.append(backend + ": changed settings or profiled timing")
        if (result["train_identity"], result["validation_identity"]) != (
            train.identity,
            validation.identity,
        ):
            reasons.append(backend + ": inputs differ")
        expected = CONFIG["cpu_repetitions" if backend == "cpu" else "repetitions"]
        measurements = result["measurements"]
        if len(measurements) != expected or [m["repeat"] for m in measurements] != list(
            range(expected)
        ):
            reasons.append(backend + ": missing/duplicate repetitions")
        if len(result["model_identities"]) != expected or len(set(result["model_identities"])) != 1:
            reasons.append(backend + ": repeat model identities differ")
        if not result["replay_exact"]:
            reasons.append(backend + ": model replay differs")
        for m in measurements:
            if not np.isfinite(m["fit_seconds"]) or m["fit_seconds"] <= 0:
                reasons.append(backend + ": invalid timing")
            width = 1 if recipe == "squared" else 2
            if (
                m["state"]["version"] != CONFIG["rounds"]
                or m["state"]["terms"] != width * CONFIG["rounds"]
            ):
                reasons.append(backend + ": incomplete round/term budget")
            if backend == "cuda" and (
                not m["final_metrics"]
                or m["final_metrics"]["live_bytes"] != 0
                or m["live_bytes_after_run_close"] != 0
            ):
                reasons.append("cuda: cleanup incomplete")
        try:
            actual = quality(validation, result["validation_raw"], recipe)
            if any(
                not np.isclose(actual[k], v, rtol=1e-12, atol=1e-12)
                for k, v in result["quality"].items()
            ) or set(actual) != set(result["quality"]):
                reasons.append(backend + ": recorded quality differs from predictions")
        except (ValueError, FloatingPointError):
            reasons.append(backend + ": invalid predictions")
    if cpu["measurements"][0]["binning_identity"] != cuda["measurements"][0]["binning_identity"]:
        reasons.append("binning differs")
    if cpu["environment"]["core_sources"] != cuda["environment"]["core_sources"]:
        reasons.append("CPU/CUDA core sources differ")
    measurement_complete = not reasons
    deltas = {}
    prediction_error = None
    if measurement_complete:
        for key, value in cpu["quality"].items():
            deltas[key] = abs(cuda["quality"][key] - value) / max(abs(value), 1e-12)
        if max(deltas.values()) > 0.01:
            reasons.append("task metrics differ by more than 1%")
        reference = np.asarray(cpu["validation_raw"])
        difference = np.asarray(cuda["validation_raw"]) - reference
        prediction_error = np.sqrt(np.mean(difference**2, axis=0)) / np.maximum(
            1, np.std(reference, axis=0)
        )
        if np.max(prediction_error) > 0.01:
            reasons.append("normalized prediction RMSE exceeds 1%")
    comparable = measurement_complete and not reasons
    ratio = None
    if comparable:
        ratio = median(m["fit_seconds"] for m in cpu["measurements"][1:]) / median(
            m["fit_seconds"] for m in cuda["measurements"][1:]
        )
    return dict(
        measurement_complete=measurement_complete,
        quality_comparable=comparable,
        reasons=reasons,
        relative_metric_differences=deltas,
        normalized_prediction_rmse=None if prediction_error is None else prediction_error.tolist(),
        cpu_over_cuda_warm_fit_ratio=ratio,
    )
