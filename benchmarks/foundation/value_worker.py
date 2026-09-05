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


def run_cell(strategy, seed, archive, output=None):
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
        for phase in ("process_first", "warm_1", "warm_2", "warm_3"):
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
    memory = {"scope": "not applicable to CPU"}
    stop = threading.Event()
    samples, errors = [], []
    thread = None
    if backend == "cuda":
        import cupy as cp

        sync()
        free, total = cp.cuda.runtime.memGetInfo()
        initial = total - free
        samples.append(initial)

        def sample():
            try:
                with cp.cuda.Device(0):
                    while not stop.is_set():
                        free_now, total_now = cp.cuda.runtime.memGetInfo()
                        samples.append(total_now - free_now)
                        stop.wait(0.005)
            except Exception as exc:
                errors.append(type(exc).__name__ + ": " + str(exc))

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
    profiler = cProfile.Profile()
    started = time.perf_counter()
    try:
        profiler.enable()
        model.fit(X, y)
        sync()
    finally:
        profiler.disable()
        stop.set()
        if thread:
            thread.join(timeout=5)
    elapsed = time.perf_counter() - started
    if backend == "cuda":
        memory = dict(
            scope="5 ms sampled device-wide usage during separate warm profile fit; includes contexts/other allocations; lower bound, not exact per-fit peak",
            initial_used_bytes=initial,
            total_bytes=total,
            sampled_peak_used_bytes=max(samples),
            sampled_peak_delta_bytes=max(samples) - initial,
            samples=len(samples),
            errors=errors,
            cupy_pool_used_bytes=cp.get_default_memory_pool().used_bytes(),
            cupy_pool_total_bytes=cp.get_default_memory_pool().total_bytes(),
        )
        if errors or thread.is_alive():
            raise RuntimeError("Memory sampler failed")
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
        memory=memory,
        top_host_functions=rows[:40],
        path_functions=path_functions,
        named_transfer_functions=transfer_functions,
        transfer_scope="cProfile named wrappers only; nested calls overlap, scalar/internal copies excluded; not a total transfer audit",
        scope="cProfile inclusive host times overlap; asynchronous launches are not kernel durations",
        nsys_available=shutil.which("nsys") is not None,
        cuda_trace_captured=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "strategy", choices=("legacy_cpu", "legacy_cuda", "experimental_cuda", "extensions_cuda")
    )
    parser.add_argument("seed", type=int)
    parser.add_argument("archive")
    parser.add_argument("output")
    args = parser.parse_args()
    Path(args.output).write_text(
        json.dumps(run_cell(args.strategy, args.seed, args.archive, args.output), indent=2) + "\n"
    )
