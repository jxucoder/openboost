"""Every historical requirement has a collected revised counterpart, never a claimed pass."""

import hashlib
import json
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest
from benchmarks.v1.comparison_cohorts import (
    BINDINGS,
    HISTORICAL,
    REFERENCE,
    ROOT,
    bindings,
    trajectories,
)


def assert_trajectory_replay(saved, current):
    """Exact structure/decisions; at most one ULP in recomputed reporting scores."""
    records, scores = [], []
    for source in (saved, current):
        record, values = deepcopy(source), []
        for case in record["cases"]:
            for cohort in ("historical", "revised"):
                for step in case[cohort]:
                    for key in ("loss", "validation_score", "best_score"):
                        values.append(step.pop(key))
        records.append(record)
        scores.append(np.asarray(values, dtype=np.float64))
    assert records[0] == records[1]
    assert all(np.all(np.isfinite(values)) for values in scores)
    np.testing.assert_array_max_ulp(scores[0], scores[1], maxulp=1)


def test_one_ulp_report_replay_does_not_modify_the_frozen_record():
    saved = json.loads((ROOT / REFERENCE).read_text())
    current = deepcopy(saved)
    step = current["cases"][0]["revised"][0]
    step["best_score"] = np.nextafter(step["best_score"], np.inf)
    assert_trajectory_replay(saved, current)
    assert saved == json.loads((ROOT / REFERENCE).read_text())


@pytest.mark.parametrize(
    "fault", ["two-ulp", "nan", "infinity", "acceptance", "best-prefix", "coefficient", "source"],
)
def test_reporting_budget_cannot_mask_changed_decisions_or_sources(fault):
    saved = json.loads((ROOT / REFERENCE).read_text())
    current = deepcopy(saved)
    step = current["cases"][0]["revised"][0]
    if fault == "two-ulp":
        step["loss"] = np.nextafter(np.nextafter(step["loss"], np.inf), np.inf)
    elif fault in ("nan", "infinity"):
        step["loss"] = float("nan" if fault == "nan" else "inf")
    elif fault == "acceptance":
        step["accepted"] = not step["accepted"]
    elif fault == "best-prefix":
        step["best_terms"] += 1
    elif fault == "coefficient":
        step["coefficients"][0] = np.nextafter(step["coefficients"][0], np.inf)
    else:
        current["sources"][next(iter(current["sources"]))] = "changed"
    with pytest.raises(AssertionError):
        assert_trajectory_replay(saved, current)


def test_binding_file_covers_all_historical_cases_and_actual_collection():
    saved = json.loads((ROOT / BINDINGS).read_text())
    assert saved == bindings()
    assert (
        saved["historical_mapping_sha256"]
        == hashlib.sha256((ROOT / HISTORICAL).read_bytes()).hexdigest()
    )
    nodes = [c["revised_node"] for c in saved["cases"]]
    assert len(nodes) == len(set(nodes)) == 383
    files = list(dict.fromkeys(n.split("::")[0] for n in nodes))
    result = subprocess.check_output(
        [sys.executable, "-m", "pytest", *files, "--collect-only", "-o", "addopts=", "-q"],
        cwd=ROOT,
        text=True,
    )
    collected = [line for line in result.splitlines() if line.startswith("tests/")]
    assert collected == nodes
    assert all(c["revised_status"] == "collected_not_run" for c in saved["cases"])
    assert [c["historical_outcome"] for c in saved["cases"]].count("failed") == 2


def test_reference_freeze_reproduces_all_ninety_trajectories_and_preserves_old_sources():
    raw = (ROOT / REFERENCE).read_bytes()
    # Pin the original bytes separately from cross-platform recomputation.
    assert hashlib.sha256(raw).hexdigest() == (
        "f127780bca3465d69387e5bf93d723df60220085d155bba49e200154d546a766"
    )
    saved = json.loads(raw)
    assert_trajectory_replay(saved, trajectories())
    assert len(saved["cases"]) == 90 and saved["device_execution"] is False
    old = json.loads((ROOT / HISTORICAL).read_text())
    for case in old["cases"]:
        file = case["historical_node"].split("::")[0]
        assert (
            hashlib.sha256((ROOT / file).read_bytes()).hexdigest()
            == case["historical_source_sha256"]
        )
    sources = json.loads(
        (ROOT / "benchmarks/v1/evidence/normal-comparison-092/study.json").read_text()
    )["sources"]
    for file, digest in sources.items():
        if file.startswith("tests/v1/reference/"):
            assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest
