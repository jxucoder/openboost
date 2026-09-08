"""105 child entry point; complete fits or a separate instrumented validation probe."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.v1.performance_evidence import environment, load_inputs, run, write_json


def profile(inputs):
    from openboost.device import DeviceOperations
    from openboost.execution import ExecutionContext

    train, _ = load_inputs(inputs, inputs["sha256"])
    values = np.column_stack((train.target[:, 0], train.weight)).astype(np.float32)
    calls = {}

    class Counted(DeviceOperations):
        def _launch(self, name, *args):
            entry = calls.setdefault(name, dict(calls=0, host_seconds=0.0))
            start = perf_counter()
            try:
                return super()._launch(name, *args)
            finally:
                entry["calls"] += 1
                entry["host_seconds"] += perf_counter() - start

    samples = []
    with ExecutionContext(max_bytes=inputs["config"]["pool_bytes"]) as context:
        ops = Counted(context)
        buffer = context.upload(values)
        for repeat in range(11):
            start, stop = context._cp.cuda.Event(), context._cp.cuda.Event()
            tick = perf_counter()
            start.record(context._stream)
            ops._validate(context._array(buffer))
            stop.record(context._stream)
            stop.synchronize()
            samples.append(
                dict(
                    repeat=repeat,
                    wall_seconds=perf_counter() - tick,
                    stream_interval_ms=float(context._cp.cuda.get_elapsed_time(start, stop)),
                )
            )
        context.release(buffer)
        owned_after_release = context.metrics["live_bytes"]
    return dict(
        status="complete",
        profile=True,
        input_sha256=inputs["sha256"],
        scope="Validation-operation stream intervals include host enqueue gaps and blocking flags; not exclusive kernel time or fit ratios.",
        shape=list(values.shape),
        samples=samples,
        launches_by_name=calls,
        live_bytes_after_release=owned_after_release,
        final_metrics=dict(context.metrics),
        environment=environment(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--input-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    load_inputs(inputs, args.input_sha)
    if args.profile:
        write_json(args.output, profile(inputs))
    else:
        run(inputs, args.backend, args.output)
