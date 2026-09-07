"""CPU checks of observation encoding/analysis, using labeled analytic records.

No device operations are replaced or run. These records are not GPU evidence.
"""

import copy
import json

import numpy as np
import pytest
from benchmarks.v1.normal_acceptance_trace import SCHEMA, analyze, pack, trial_analysis, unpack

from .reference.device_normal import geometry


@pytest.mark.parametrize(
    "value", [np.float64(-0.0), np.float32(1e-30), np.array([[1, 2]], np.float32)]
)
def test_json_round_trip_preserves_exact_bits(value):
    record = json.loads(json.dumps(pack(value)))
    assert pack(unpack(record)) == pack(value)


@pytest.mark.parametrize("broken", ["bits", "shape", "dtype", "nonfinite"])
def test_corrupted_observation_is_rejected(broken):
    record = pack(np.array([1], np.float32))
    if broken == "bits":
        record["bits"][0] += 1
    elif broken == "shape":
        record["shape"] = [2]
    elif broken == "dtype":
        record["dtype"] = "int32"
    else:
        record["values"] = [float("inf")]
    with pytest.raises(ValueError):
        unpack(record)


def analytic_trace():
    """One analytically known mean update; data deliberately differ from GPU cases."""
    inputs = {}
    for name, y in (("training", 1), ("validation", -1)):
        inputs[name] = dict(
            target=pack(np.array([[y]], np.float32)),
            offset=pack(np.zeros((1, 2), np.float32)),
            weight=pack(np.ones(1, np.float32)),
        )

    def values(mean):
        result = {}
        raw = np.array([[mean, 0]], np.float32)
        for name, metric in (("training", "loss"), ("validation", "validation_score")):
            data = inputs[name]
            target, offset, weight = (unpack(data[k]) for k in ("target", "offset", "weight"))
            result[name + "_raw"] = pack(raw)
            result[metric] = pack(np.float64(geometry(raw, target[:, 0], offset, weight)[0]))
        return result

    initial = dict(
        **values(0),
        identity="synthetic-initial",
        version=0,
        n_terms=0,
        best_n_terms=0,
        best_score=values(0)["validation_score"],
    )
    trials = []
    for coefficient in (8, 4, 2):
        proposal = dict(
            **values(coefficient / 2),
            identity=f"synthetic-proposal-{coefficient}",
            parent_identity=initial["identity"],
            coefficient=coefficient,
        )
        accepted = coefficient == 2
        resolved = (
            dict(initial, **values(1), identity="synthetic-accepted", version=1, n_terms=1)
            if accepted
            else copy.deepcopy(initial)
        )
        trials.append(
            dict(coefficient=coefficient, accepted=accepted, proposal=proposal, resolved=resolved)
        )
    return dict(
        schema=SCHEMA,
        source_kind="analytic CPU test; not device evidence",
        update="forward",
        inputs=inputs,
        initial=initial,
        conformance={"status": "synthetic"},
        steps=[
            dict(
                round=0,
                channels=[0],
                before=initial,
                trials=trials,
                gradient=pack(np.array([[-1, 0]], np.float32)),
                fisher=pack(np.array([[1, 2]], np.float32)),
                direction=pack(np.array([[1, 0]], np.float32)),
                fields=[
                    dict(
                        names=["gradient", "curvature"],
                        roles=["weighted", "weighted"],
                        values=pack(np.array([[-1, 1]], np.float32)),
                    )
                ],
                roots=[
                    dict(
                        total=pack(np.array([-1, 1], np.float32)),
                        reg_lambda=1,
                        row_positions=[0],
                        leaf=pack(np.array([0.5], np.float32)),
                    )
                ],
                terms=[dict(mapping=pack(np.array([[1, 0]], np.float32)))],
            )
        ],
    )


def test_analysis_distinguishes_training_acceptance_and_validation_best():
    report = analyze(analytic_trace())
    (step,) = report["steps"]
    assert (
        report["device_execution"] is False
        and report["observed_conformance"]["status"] == "synthetic"
    )
    assert step["gradient_max_abs_error"] == step["fisher_max_abs_error"] == 0
    assert step["leaf_from_measured_total"] == step["measured_leaf"] == 0.5
    assert step["stored_field_total"] == [-1, 1]
    assert [t["comparison"]["training"]["high_precision"]["signs"] for t in step["trials"]] == [
        [1, 1],
        [0, 0],
        [-1, -1],
    ]
    assert step["trials"][-1]["comparison"]["validation"]["high_precision"]["signs"] == [1, 1]


def test_names_determine_root_fields_independently_of_role_and_column_order():
    trace = analytic_trace()
    step = trace["steps"][0]
    fields = step["fields"][0]
    fields["names"].reverse()
    fields["values"] = pack(unpack(fields["values"])[:, ::-1].copy())
    step["roots"][0]["total"] = pack(np.array([1, -1], np.float32))
    assert analyze(trace)["steps"][0]["leaf_from_measured_total"] == 0.5


@pytest.mark.parametrize(
    "bad", ["round", "missing", "coefficient", "parent", "accept", "version", "best", "bits"]
)
def test_incomplete_or_inconsistent_transactions_cannot_be_counted_as_diagnostics(bad):
    trace = analytic_trace()
    step = trace["steps"][0]
    trial = step["trials"][-1]
    if bad == "round":
        step["round"] = 2
    elif bad == "missing":
        step["roots"] = []
    elif bad == "coefficient":
        trial["coefficient"] = 1
    elif bad == "parent":
        trial["proposal"]["parent_identity"] = "foreign"
    elif bad == "accept":
        trial["accepted"] = False
    elif bad == "version":
        trial["resolved"]["version"] = 0
    elif bad == "best":
        trial["resolved"]["best_n_terms"] = 1
    else:
        step["gradient"]["bits"][0][0] += 1
    with pytest.raises(ValueError):
        analyze(trace)


def test_same_stored_inputs_reveal_rounded_full_loss_disagreement():
    trace = analytic_trace()
    before = trace["initial"]
    proposal = copy.deepcopy(before)
    proposal["loss"] = pack(np.nextafter(unpack(before["loss"]), -np.inf))
    result = trial_analysis(before, proposal, trace["inputs"])["training"]
    assert result["raw_changed"] is False
    assert result["measured_full_loss_difference"] < 0
    assert result["high_precision"]["signs"] == [0, 0]
