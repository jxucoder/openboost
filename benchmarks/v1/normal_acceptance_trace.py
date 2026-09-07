"""Loss analysis of retained observations; never executes or emulates CUDA."""

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
from tests.v1.reference.device_normal import geometry, rounds
from tests.v1.reference.normal_acceptance import compare_precisions

SCHEMA = "openboost-normal-acceptance-trace-v1"


def pack(values):
    """Store finite IEEE values and their exact bits, including scalar metrics."""
    values = np.asarray(values)
    if values.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("float32 or float64 observations required")
    if not np.isfinite(values).all():
        raise ValueError("finite observations required")
    bits = values.view(np.uint32 if values.dtype.itemsize == 4 else np.uint64)
    return dict(
        dtype=values.dtype.name,
        shape=list(values.shape),
        values=values.tolist(),
        bits=bits.tolist(),
    )


def unpack(record):
    if record["dtype"] not in ("float32", "float64"):
        raise ValueError("float32 or float64 observations required")
    values = np.array(record["values"], dtype=record["dtype"])
    if pack(values) != record:
        raise ValueError("observation shape or bits do not match values")
    return values


def state_metrics(state):
    return dict(
        identity=state.identity,
        version=state.version,
        n_terms=state.n_terms,
        best_n_terms=state.best_n_terms,
        loss=pack(np.float64(state.loss)),
        validation_score=pack(np.float64(state.validation_score)),
        best_score=pack(np.float64(state.best_score)),
    )


def trial_analysis(before, proposal, inputs):
    """Compare full losses and high-precision differences at identical stored rows."""
    result = {}
    for name, metric in (("training", "loss"), ("validation", "validation_score")):
        data = inputs[name]
        target, offset, weight = (unpack(data[k]) for k in ("target", "offset", "weight"))
        target = target[:, 0]
        old, new = (unpack(record[name + "_raw"]) for record in (before, proposal))
        old_nll = geometry(old, target, offset, weight)[0]
        new_nll = geometry(new, target, offset, weight)[0]
        measured_old, measured_new = (
            float(unpack(record[metric])) for record in (before, proposal)
        )
        result[name] = dict(
            raw_changed=not np.array_equal(old, new),
            measured_full_loss_difference=measured_new - measured_old,
            original_row_float64_before=old_nll,
            original_row_float64_after=new_nll,
            original_row_float64_difference=new_nll - old_nll,
            measured_before_minus_original_row=measured_old - old_nll,
            measured_after_minus_original_row=measured_new - new_nll,
            high_precision=compare_precisions(old, new, target, offset, weight),
        )
    return result


def analyze(trace):
    if trace["schema"] != SCHEMA or trace["update"] not in ("forward", "reverse"):
        raise ValueError("recognized acceptance trace required")
    if not trace.get("initial") or not trace.get("steps"):
        raise ValueError("incomplete trace")

    def verify_bits(value):
        if isinstance(value, dict):
            if "dtype" in value:
                unpack(value)
            else:
                for item in value.values():
                    verify_bits(item)
        elif isinstance(value, list):
            for item in value:
                verify_bits(item)

    verify_bits(trace)
    _, expected = rounds("conflict", mode="ordinary", update=trace["update"], depth=0, rate=8)
    data = trace["inputs"]["training"]
    target, offset, weight = (unpack(data[k]) for k in ("target", "offset", "weight"))
    target = target[:, 0]
    rows = []
    previous = trace["initial"]
    for index, step in enumerate(trace["steps"]):
        if index >= len(expected):
            raise ValueError("too many observed steps")
        ref = expected[index]
        if (step["round"], step["channels"]) != (ref["round"], list(ref["channels"])):
            raise ValueError("unexpected round/channel order")
        if not 1 <= len(step["trials"]) <= 6 or any(
            len(step[name]) != 1 for name in ("roots", "terms", "fields")
        ):
            raise ValueError("incomplete step observations")
        if step["before"] != previous:
            raise ValueError("observed state continuity differs")
        old = unpack(step["before"]["training_raw"])
        _, gradient, fisher = geometry(old, target, offset, weight)
        channel = step["channels"][0]
        root, fields = step["roots"][0], step["fields"][0]
        field_values = unpack(fields["values"])
        # Compare two distinct questions: ideal rows vs rounded device fields,
        # then measured reduction vs exactly the fields that entered that reduction.
        expected_total = [math.fsum(float(v) for v in col) for col in field_values.T]
        total = unpack(root["total"])
        trials = []
        for j, trial in enumerate(step["trials"]):
            before, proposal, resolved = step["before"], trial["proposal"], trial["resolved"]
            accepted = float(unpack(proposal["loss"])) < float(unpack(before["loss"]))
            if (
                trial["coefficient"] != 8 * 0.5**j
                or proposal["coefficient"] != trial["coefficient"]
                or proposal["parent_identity"] != before["identity"]
                or type(trial["accepted"]) is not bool
                or trial["accepted"] != accepted
                or (accepted and j != len(step["trials"]) - 1)
                or resolved["version"] != before["version"] + int(accepted)
                or resolved["n_terms"] != before["n_terms"] + int(accepted)
            ):
                raise ValueError("inconsistent observed transaction")
            if not accepted and resolved != before:
                raise ValueError("rejection changed observed state")
            if accepted:
                for key in ("loss", "validation_score", "training_raw", "validation_raw"):
                    if resolved[key] != proposal[key]:
                        raise ValueError("acceptance changed proposal snapshot")
                improved = float(unpack(proposal["validation_score"])) < float(
                    unpack(before["best_score"])
                )
                if resolved["best_n_terms"] != (
                    resolved["n_terms"] if improved else before["best_n_terms"]
                ):
                    raise ValueError("inconsistent best prefix")
                if resolved["best_score"] != (
                    proposal["validation_score"] if improved else before["best_score"]
                ):
                    raise ValueError("inconsistent best score")
            comparison = trial_analysis(step["before"], trial["proposal"], trace["inputs"])
            trials.append(
                dict(
                    coefficient=trial["coefficient"],
                    accepted=trial["accepted"],
                    comparison=comparison,
                )
            )
        g = fields["names"].index("gradient")
        h = fields["names"].index("curvature")
        rows.append(
            dict(
                round=step["round"],
                channels=step["channels"],
                original_float64_trajectory_attempts=ref["attempts"],
                original_float64_trajectory_accepted=ref["accepted"],
                gradient_max_abs_error=float(np.max(np.abs(unpack(step["gradient"]) - gradient))),
                fisher_max_abs_error=float(np.max(np.abs(unpack(step["fisher"]) - fisher))),
                ordinary_direction_max_abs_error=float(
                    np.max(np.abs(unpack(step["direction"]) + unpack(step["gradient"])))
                ),
                ideal_weighted_gradient_sum=math.fsum(
                    float(w) * float(v) for w, v in zip(weight, gradient[:, channel], strict=True)
                ),
                stored_field_total=expected_total,
                measured_total_minus_stored_field_total=(total - expected_total).tolist(),
                leaf_from_measured_total=-float(total[g]) / (float(total[h]) + root["reg_lambda"]),
                measured_leaf=float(unpack(root["leaf"])[0]),
                trials=trials,
            )
        )
        previous = step["trials"][-1]["resolved"]
    return dict(
        schema="openboost-normal-acceptance-analysis-v1",
        device_execution=False,
        update=trace["update"],
        observed_conformance=trace["conformance"],
        steps=rows,
        limitation="Offline math on retained observations. Diagnostics do not change acceptance or establish full conformance. Precision agreement is not an interval bound.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(json.loads(args.trace.read_text()))
    report["trace_sha256"] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[2]
    report["analysis_sources"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            Path(__file__),
            *(
                root / "tests/v1/reference" / name
                for name in (
                    "normal_acceptance.py",
                    "device_normal.py",
                    "device_rounds.py",
                    "device_splits.py",
                    "device_histogram.py",
                )
            ),
        )
    }
    report["environment"] = dict(
        python=platform.python_version(), numpy=np.__version__, os=platform.platform()
    )
    report["argv"] = sys.argv
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
