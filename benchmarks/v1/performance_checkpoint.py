"""104 bounded CPU/CUDA measurement child; generated data and public recipes only."""

import argparse
import cProfile
import gc
import hashlib
import json
import math
import platform
import pstats
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from openboost import NumericData, Problem, RunContext
from openboost.artifacts import Model
from openboost.binning import Binning, PreparedData

CONFIG = dict(
    features=16,
    bins=32,
    rounds=20,
    max_depth=3,
    learning_rate=0.1,
    reg_lambda=1.0,
    step="fixed",
    seed=7,
    repetitions=4,
    cpu_repetitions=2,
    pool_bytes=512 * 1024**2,
)


def workload(rows, recipe):
    """Same float32-origin train/validation inputs on every backend; no test-set tuning."""
    rng = np.random.default_rng(CONFIG["seed"])
    n = rows + rows // 5
    x = rng.normal(size=(n, CONFIG["features"])).astype(np.float32)
    mean = 2 * np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2 + 1.5 * (x[:, 2] > 0)
    scale = 0.5 + 0.1 * np.abs(x[:, 4])
    target = (mean + 0.2 * x[:, 3] + scale * rng.normal(size=n)).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    weight[::23] = 0
    width = 1 if recipe == "squared" else 2
    offset = np.zeros((n, width), np.float32)
    offset[:, 0] = 0.2 * x[:, 3]
    x[rng.random(x.shape) < 0.01] = np.nan
    problems = []
    for sl in (slice(0, rows), slice(rows, n)):
        ids = np.arange(n)[sl]
        data = NumericData(x[sl], ids, tuple(f"f{i}" for i in range(x.shape[1])))
        problems.append(
            Problem(
                data, target[sl, None], ids, weight=weight[sl], offset=offset[sl], raw_width=width
            )
        )
    return tuple(problems)


def quality(problem, raw, recipe):
    """Direct held-out formulas, independent of the production objective loss methods."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != problem.offset.shape or not np.isfinite(raw).all():
        raise ValueError("finite aligned predictions required")
    value = raw + problem.offset
    y, w = problem.target[:, 0], problem.weight / problem.weight.sum()
    residual = value[:, 0] - y
    if recipe == "squared":
        return dict(half_mse=float(w @ (residual**2 / 2)))
    scale = np.exp(value[:, 1])
    z = -residual / scale
    cdf = (1 + np.fromiter((math.erf(v / math.sqrt(2)) for v in z), float)) / 2
    density = np.exp(-(z**2) / 2) / math.sqrt(2 * math.pi)
    return dict(
        nll=float(w @ (np.log(scale) + z**2 / 2 + math.log(2 * math.pi) / 2)),
        crps=float(w @ (scale * (z * (2 * cdf - 1) + 2 * density - 1 / math.sqrt(math.pi)))),
    )


def fit(train, validation, recipe, backend):
    """Timing includes fresh preparation, fit, model export and resource cleanup."""
    from openboost import recipes

    options = {
        k: CONFIG[k] for k in ("rounds", "max_depth", "bins", "learning_rate", "reg_lambda", "step")
    }
    context = result = None
    live_bytes_after_run_close = None
    timer = perf_counter()
    context_seconds = 0.0
    try:
        if backend == "cuda":
            from openboost import device_recipes
            from openboost.device import DeviceOperations
            from openboost.execution import ExecutionContext

            context = ExecutionContext(max_bytes=CONFIG["pool_bytes"])
            context_seconds = perf_counter() - timer
            start = perf_counter()
            prepared = Binning.fit(train.data, bins=CONFIG["bins"])
            preparation = perf_counter() - start
            start = perf_counter()
            device_options = {key: value for key, value in options.items() if key != "bins"}
            result = getattr(device_recipes, recipe)(
                DeviceOperations(context),
                train,
                validation,
                run_id="104-cost",
                seed=CONFIG["seed"],
                binning=prepared,
                **device_options,
            )
            context.synchronize()
            training = perf_counter() - start
            start = perf_counter()
            model = result.run.export(result.state, best=True)
            record = model.record()
            export = perf_counter() - start
            state = dict(
                version=result.state.version,
                terms=result.state.n_terms,
                best_terms=result.state.best_n_terms,
            )
            metrics = dict(context.metrics)
            binning_identity = prepared.identity
        else:
            start = perf_counter()
            prepared = PreparedData(train.data, CONFIG["bins"])
            preparation = perf_counter() - start
            start = perf_counter()
            result = getattr(recipes, recipe)(
                train,
                validation,
                context=RunContext("104-cost", CONFIG["seed"]),
                retention="summary",
                prepared=prepared,
                **options,
            )
            training = perf_counter() - start
            start = perf_counter()
            record = result.state.best_model.record()
            export = perf_counter() - start
            state = dict(
                version=result.state.version,
                terms=len(result.state.model.terms),
                best_terms=len(result.state.best_model.terms),
            )
            metrics = None
            binning_identity = prepared.binned.binning.identity
        stop = dict(reason=result.stop.reason, completed_rounds=result.stop.completed_rounds)
    finally:
        cleanup_start = perf_counter()
        if context is not None:
            if result is not None:
                result.run.close()
                live_bytes_after_run_close = context.metrics["live_bytes"]
            context.close()
        result = None
        cleanup = perf_counter() - cleanup_start
    elapsed = perf_counter() - timer
    return dict(
        fit_seconds=elapsed,
        context_seconds=context_seconds,
        host_preparation_seconds=preparation,
        training_seconds=training,
        export_seconds=export,
        cleanup_seconds=cleanup,
        metrics=metrics,
        final_metrics=None if context is None else dict(context.metrics),
        live_bytes_after_run_close=live_bytes_after_run_close,
        state=state,
        stop=stop,
        binning_identity=binning_identity,
    ), record


def profile_rows(profiler):
    rows = []
    for (file, line, name), (_, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
        normalized = file.replace("\\", "/")
        selected = (
            "openboost/" in normalized
            and name in {"_launch", "upload", "export", "synchronize", "release"}
        ) or ("numba/cuda/dispatcher.py" in normalized and name == "compile")
        if selected:
            rows.append(
                dict(
                    file=normalized.split("site-packages/")[-1],
                    line=line,
                    function=name,
                    calls=calls,
                    own_seconds=own,
                    inclusive_seconds=cumulative,
                )
            )
    return sorted(rows, key=lambda row: -row["inclusive_seconds"])


def run(rows, recipe, backend, *, profile=False, checkpoint=None):
    start = perf_counter()
    train, validation = workload(rows, recipe)
    construction = perf_counter() - start
    measurements, identities, profiling = [], [], []
    record = None
    repetitions = 2 if profile else CONFIG["cpu_repetitions" if backend == "cpu" else "repetitions"]
    for repeat in range(repetitions):
        gc.collect()
        profiler = cProfile.Profile() if profile else None
        if profiler:
            profiler.enable()
        try:
            measured, record = fit(train, validation, recipe, backend)
        finally:
            if profiler:
                profiler.disable()
        measured.update(repeat=repeat, phase="first" if repeat == 0 else "warm")
        measurements.append(measured)
        identities.append(hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest())
        if profiler:
            profiling.append(dict(repeat=repeat, calls=profile_rows(profiler)))
        if checkpoint:
            checkpoint(
                dict(
                    status="running",
                    backend=backend,
                    recipe=recipe,
                    train_rows=rows,
                    config=CONFIG,
                    measurements=measurements,
                    profile=profile,
                    profile_calls=profiling,
                )
            )
    start = perf_counter()
    model = Model.from_record(record)
    load_seconds = perf_counter() - start
    predictions, prediction_times = {}, {}
    single = NumericData(
        validation.data.values[:1], validation.row_ids[:1], train.data.feature_names
    )
    for name, data in (("single", single), ("batch", validation.data)):
        times = []
        for _ in range(4):
            start = perf_counter()
            raw = model.predict(data)
            times.append(perf_counter() - start)
        prediction_times[name] = times
        predictions[name] = raw
    replay = Model.from_record(record).predict(validation.data)
    core = Path(sys.modules["openboost"].__file__).parent
    return dict(
        status="complete",
        backend=backend,
        recipe=recipe,
        train_rows=rows,
        validation_rows=len(validation.target),
        config=CONFIG,
        train_identity=train.identity,
        validation_identity=validation.identity,
        construction_seconds=construction,
        measurements=measurements,
        model_identities=identities,
        model=record,
        model_load_seconds=load_seconds,
        cpu_prediction_seconds=prediction_times,
        validation_raw=predictions["batch"].tolist(),
        quality=quality(validation, predictions["batch"], recipe),
        replay_exact=bool(np.array_equal(replay, predictions["batch"])),
        profile=profile,
        profile_calls=profiling,
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            installed_path=str(core),
            core_sources={
                "src/openboost/" + str(p.relative_to(core)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in sorted(core.rglob("*.py"))
            },
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--recipe", choices=("squared", "normal"), required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(result):
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    result = run(args.rows, args.recipe, args.backend, profile=args.profile, checkpoint=save)
    save(result)
    print(
        json.dumps(
            dict(
                status=result["status"],
                backend=args.backend,
                recipe=args.recipe,
                rows=args.rows,
                fit_seconds=[m["fit_seconds"] for m in result["measurements"]],
            )
        )
    )
