"""Fresh-process P7 cell; profiling is a separate, untimed fifth fit."""

import argparse
import cProfile
import hashlib
import json
import pstats
import shutil
import threading
import time
import warnings
from pathlib import Path

import numpy as np

if __package__:
    from .baseline_worker import CONFIG, normal_metrics
    from .dataset import load_housing, split_indices
else:
    from baseline_worker import CONFIG, normal_metrics
    from dataset import load_housing, split_indices


def run_cell(strategy, seed, archive, output=None, profile_only=False):
    import openboost as ob
    from openboost.experimental import Booster, DistributionObjectiveAdapter, TrainerConfig

    backend = "cpu" if strategy == "legacy_cpu" else "cuda"
    X, y = load_housing(archive)
    train, val, test = split_indices(len(y), seed)
    X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    objective = None
    extra = {}
    if strategy == "experimental_cuda":
        objective = DistributionObjectiveAdapter("normal", natural=True)
    elif strategy == "extensions_cuda":
        from bounded_leaves import BoundedNewton
        from normal_fisher import ChannelDecay, NormalFisher

        from openboost.experimental import LevelWiseBuilder

        objective = NormalFisher()
        extra = dict(
            tree_builder=LevelWiseBuilder(leaf_rule=BoundedNewton(0.5)),
            step_schedule=ChannelDecay(tau=1),
        )

    def create():
        if objective is None:
            return ob.NaturalBoostNormal(**CONFIG, random_state=seed)
        return Booster(
            objective=objective,
            device=backend,
            config=TrainerConfig(**CONFIG, random_state=seed),
            **extra,
        )

    def sync():
        if backend == "cuda":
            from numba import cuda

            cuda.synchronize()

    def predict(model):
        if objective is None:
            return model.predict_params(X_test)
        params = objective.constrain(model.predict_raw(X_test))
        return (
            dict(loc=params["mu"], scale=params["sigma"])
            if strategy == "extensions_cuda"
            else params
        )

    records = []
    result = dict(
        strategy=strategy,
        seed=seed,
        config=CONFIG,
        mode="resident",
        split_sizes=[len(train), len(val), len(test)],
        records=records,
    )
    previous = None
    with ob.backend_context(backend):
        phases = (
            ("untimed_warmup",) if profile_only else ("process_first", "warm_1", "warm_2", "warm_3")
        )
        for phase in phases:
            model = create()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sync()
                started = time.perf_counter()
                model.fit(X_train, y_train)
                sync()
                fit_s = time.perf_counter() - started
                started = time.perf_counter()
                params = predict(model)
                sync()
                predict_s = time.perf_counter() - started
            records.append(
                dict(
                    phase=phase,
                    fit_s=fit_s,
                    predict_s=predict_s,
                    metrics=normal_metrics(y_test, params),
                    fallback_warnings=[
                        str(w.message) for w in caught if "fallback" in str(w.message).lower()
                    ],
                    prediction_sha256=hashlib.sha256(
                        b"".join(
                            np.asarray(params[k], dtype="<f4").tobytes() for k in ("loc", "scale")
                        )
                    ).hexdigest(),
                    actual_device=model.fit_report_["actual_device"]
                    if objective
                    else ob.get_backend(),
                )
            )
            if output:
                Path(output).write_text(json.dumps(result, indent=2) + "\n")
            if previous is not None:
                for k in params:
                    np.testing.assert_allclose(params[k], previous[k], rtol=2e-5, atol=2e-6)
            previous = {k: v.copy() for k, v in params.items()}
        profile = profile_fit(create(), X_train, y_train, backend, sync)
    result["profile"] = profile
    return result


def profile_fit(model, X, y, backend, sync):
    """Host attribution and sampled device-wide memory, never a CUDA trace."""
    # No sampling thread while cProfile runs. CUDA/Cython tracing can mix thread
    # events into its call accounting; separate fit avoids that contamination.
    from contextlib import ExitStack
    from unittest.mock import patch

    import openboost._trainer as trainer
    from openboost._objectives import DistributionObjective
    from openboost.experimental import LevelWiseBuilder
    from openboost.experimental._device import DeviceExtensionSession, DeviceObjectiveBridge

    wall_timers = {}

    def timed(label, original):
        def wrapper(*args, **kwargs):
            sync()
            before = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                sync()
                entry = wall_timers.setdefault(label, dict(calls=0, inclusive_s=0.0))
                entry["calls"] += 1
                entry["inclusive_s"] += time.perf_counter() - before

        return wrapper

    profiler = cProfile.Profile()
    with ExitStack() as stack:
        for owner, name, label in (
            (DistributionObjective, "step", "legacy_objective"),
            (DeviceObjectiveBridge, "step", "extension_objective_boundary"),
            (DeviceExtensionSession, "build", "extension_tree_boundary"),
            (LevelWiseBuilder, "build", "levelwise_builder"),
            (trainer, "fit_tree_gpu_native", "legacy_native_tree"),
        ):
            stack.enter_context(patch.object(owner, name, timed(label, getattr(owner, name))))
        sync()
        started = time.perf_counter()
        try:
            profiler.enable()
            model.fit(X, y)
            sync()
        finally:
            profiler.disable()
        elapsed = time.perf_counter() - started
    memory = sample_memory_fit(model, X, y, backend, sync)
    stats = pstats.Stats(profiler).stats
    rows = [
        dict(
            file=Path(file).name,
            line=line,
            function=name,
            primitive_calls=cc,
            calls=nc,
            self_s=tt,
            cumulative_s=ct,
        )
        for (file, line, name), (cc, nc, tt, ct, _) in stats.items()
    ]
    rows.sort(key=lambda r: r["cumulative_s"], reverse=True)
    path_functions = [
        r
        for r in rows
        if (r["file"], r["function"])
        in {
            ("_tree.py", "fit_tree_gpu_native"),
            ("_objectives.py", "step"),
            ("_device.py", "step"),
            ("_device.py", "build"),
            ("_levelwise.py", "build"),
        }
    ]
    transfer_functions = [
        r for r in rows if r["function"] in ("asnumpy", "copy_to_host", "_to_host")
    ]
    return dict(
        wall_s=elapsed,
        isolated_host_profile=True,
        synchronized_inclusive_timers=wall_timers,
        timer_scope="separate profile fit with synchronized nested boundaries; overlaps and synchronization overhead, not production timing",
        memory=memory,
        top_host_functions=rows[:40],
        path_functions=path_functions,
        named_transfer_functions=transfer_functions,
        transfer_scope="cProfile named wrappers only; nested calls overlap, scalar/internal copies excluded; not a total transfer audit",
        scope="cProfile inclusive host times overlap; asynchronous launches are not kernel durations",
        nsys_available=shutil.which("nsys") is not None,
        cuda_trace_captured=False,
    )


def sample_memory_fit(model, X, y, backend, sync):
    if backend != "cuda":
        return {"scope": "not applicable to CPU"}
    import cupy as cp

    sync()
    free, total = cp.cuda.runtime.memGetInfo()
    initial = total - free
    stop = threading.Event()
    samples, errors = [initial], []

    def sample():
        try:
            with cp.cuda.Device(0):
                while not stop.is_set():
                    f, t = cp.cuda.runtime.memGetInfo()
                    samples.append(t - f)
                    stop.wait(0.005)
        except Exception as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        model.fit(X, y)
        sync()
    finally:
        stop.set()
        thread.join(timeout=5)
    if errors or thread.is_alive():
        raise RuntimeError("Memory sampler failed")
    return dict(
        scope="separate memory-only warm fit, no cProfile; 5 ms device-wide samples including contexts/caches; lower bound, not exact per-fit peak",
        initial_used_bytes=initial,
        total_bytes=total,
        sampled_peak_used_bytes=max(samples),
        sampled_peak_delta_bytes=max(samples) - initial,
        samples=len(samples),
        errors=errors,
        cupy_pool_used_bytes=cp.get_default_memory_pool().used_bytes(),
        cupy_pool_total_bytes=cp.get_default_memory_pool().total_bytes(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "strategy", choices=("legacy_cpu", "legacy_cuda", "experimental_cuda", "extensions_cuda")
    )
    parser.add_argument("seed", type=int)
    parser.add_argument("archive")
    parser.add_argument("output")
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    Path(args.output).write_text(
        json.dumps(
            run_cell(args.strategy, args.seed, args.archive, args.output, args.profile_only),
            indent=2,
        )
        + "\n"
    )
