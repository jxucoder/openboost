"""Derive a retrospective from committed run-6 artifacts, without device execution."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from tests.v1.reference.device_normal import base, fixture, geometry, rounds

from benchmarks.v1.cuda_aggregation_preflight import judge_run

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks/v1/evidence/cuda-normal-090"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(directory=EVIDENCE):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    sources = {name: digest(directory / name) for name in manifest["artifacts"]}
    assert sources == manifest["artifacts"]
    verdict = judge_run(
        manifest["result"],
        (directory / "junit.xml").read_text(),
        manifest["protocol"],
        manifest["sources"],
    )
    assert verdict == json.loads((directory / "verdict.json").read_text())
    groups = defaultdict(Counter)
    for c in ET.parse(directory / "junit.xml").getroot().iter("testcase"):
        status = next((s for s in ("failure", "error", "skipped") if c.find(s) is not None), "pass")
        groups[c.get("classname")][status] += 1

    measured = {}
    for path in sorted((directory / "normal").glob("*/measurement.json")):
        rows = json.loads(path.read_text())
        if not rows:  # The missing-input case measures inference correctness only.
            continue
        measured[path.parent.name] = [
            dict(
                repeat=r["repeat"],
                fit_seconds=r["fit_including_context_preparation_export_seconds"],
                cpu_prediction_seconds=r["cpu_prediction_seconds"],
                trials=len(r["trials"]),
                rejected_trials=sum(not t["accepted"] for t in r["trials"]),
                invalid_trials=sum(t["failure"] is not None for t in r["trials"]),
                metrics=r["fit_metrics"],
                final_live_bytes=r["final_metrics"]["live_bytes"],
                validation_nll_crps=r["validation_nll_crps"],
            )
            for r in rows
        ]
    measurements = [r for rows in measured.values() for r in rows]
    first = [r["fit_seconds"] for r in measurements if r["repeat"] == 0]
    repeated = [r["fit_seconds"] for r in measurements if r["repeat"] == 1]

    # Original-row CPU math on its own optimum and an actual saved GPU D2 base.
    # D2 and conflict have equal training arrays and different validation targets.
    # This is not a reconstruction of the missing failed-device trial trace.
    f, d2 = fixture("conflict"), fixture("d2")
    assert all(np.array_equal(f[k], d2[k]) for k in ("x", "target", "offset", "weight"))
    model_path = directory / "normal/d2-ordinary-0-forward-backtracking/model.json"
    recorded_base = np.array(json.loads(model_path.read_text())["base"])
    reference_base = base(f["target"], f["offset"], f["weight"])
    stationary = {}
    for name, values in (
        ("reference_base", reference_base),
        ("saved_d2_device_base", recorded_base),
    ):
        loss, gradient, _ = geometry(
            np.broadcast_to(values, f["offset"].shape), f["target"], f["offset"], f["weight"]
        )
        stationary[name] = dict(
            raw=values.tolist(),
            cpu_reference_loss=loss,
            cpu_reference_gradient_sum=gradient.sum(axis=0).tolist(),
        )
    stationary["saved_base_source"] = str(model_path.relative_to(directory))
    stationary["reference_trace"] = {}
    for update in ("forward", "reverse"):
        _, steps = rounds("conflict", mode="ordinary", update=update, depth=0, rate=8)
        stationary["reference_trace"][update] = [
            dict(
                round=s["round"],
                channels=s["channels"],
                before_loss=s["loss_before"],
                root_values=[float(n[0]["value"]) for n in s["nodes"]],
                attempts=s["attempts"],
                accepted=s["accepted"],
                version=s["version"],
            )
            for s in steps
        ]
    stationary["limitation"] = (
        "Reference-only math and a separate passing D2 artifact. Failed GPU trials did not retain round/channel, raw values, gradients or loss bits; cause is not proven."
    )
    diagnostics = [
        json.loads(line.split("NORMAL_NEAR_TIE_DIAGNOSTIC=", 1)[1])
        for line in (directory / "pytest.log").read_text().splitlines()
        if "NORMAL_NEAR_TIE_DIAGNOSTIC=" in line
    ]
    assert len(diagnostics) == 1
    tie = diagnostics[0]
    elapsed = (
        datetime.fromisoformat(manifest["finished"]) - datetime.fromisoformat(manifest["started"])
    ).total_seconds()
    return dict(
        schema="openboost-normal-run6-retrospective-v1",
        device_execution=False,
        dispatch_revision=manifest["revision"],
        artifact_manifest_sha256=digest(directory / "manifest.json"),
        input_artifact_hashes=sources,
        analysis_source_sha256=digest(Path(__file__)),
        reference_sources={
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                ROOT / "tests/v1/reference" / name
                for name in (
                    "device_normal.py",
                    "device_rounds.py",
                    "device_splits.py",
                    "device_histogram.py",
                )
            )
        },
        overall_passed=verdict["passed"],
        groups=dict(groups),
        failures=[c["case"] for c in verdict["cases"] if c["status"] != "pass"],
        retained_model_count=len(list((directory / "normal").glob("*/model.json"))),
        timing=dict(
            dispatch_wall_seconds=elapsed,
            worker_wall_seconds=manifest["result"]["worker_wall_seconds"],
            first_in_case_fit_range_seconds=[min(first), max(first)],
            repeated_fit_range_seconds=[min(repeated), max(repeated)],
            measured_fits=len(measurements),
            rejected_trials=sum(r["rejected_trials"] for r in measurements),
            interpretation="Tiny six-row correctness fixtures, no matched-quality comparator. First/repeated in-case timing is not process-cold. Fixture creation and supplied binning precede the clock; context/device preparation, trials and export are timed.",
        ),
        measured_trajectories=measured,
        stationary_reference=stationary,
        near_tie=dict(
            device_gains=tie["device_gains"],
            device_gain_bits=tie["device_gain_bits"],
            selected_key=tie["selected_key"],
            max_prediction_difference=float(np.max(np.abs(tie["prediction_difference"]))),
            structural_parity_repaired=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output
    output.write_text(json.dumps(analyze(), indent=2, allow_nan=False) + "\n")
