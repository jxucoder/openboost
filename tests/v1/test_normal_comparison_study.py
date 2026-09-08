"""Full historical accounting and independent comparison-cohort preregistration."""

import hashlib
import json
import subprocess
from collections import Counter
from decimal import Decimal

from benchmarks.v1.normal_comparison_study import ROOT, mapping, study


def test_archived_clean_study_reproduces_every_case_and_source():
    saved = json.loads(
        (ROOT / "benchmarks/v1/evidence/normal-comparison-092/study.json").read_text()
    )
    assert saved["dirty"] is False and saved["source_revision"].startswith("a5967b5")
    current = study()
    for key in ("cases", "counts", "sources", "historical_mapping_sha256", "device_execution"):
        assert saved[key] == current[key]
    for path, digest in saved["sources"].items():
        committed = subprocess.check_output(
            ["git", "show", saved["source_revision"] + ":" + path], cwd=ROOT
        )
        assert hashlib.sha256(committed).hexdigest() == digest


def test_every_original_case_has_one_explicit_disposition_and_no_revised_pass():
    result = mapping()
    old = json.loads((ROOT / "v1-sprints/090-normal-run6.json").read_text())
    assert [row["historical_node"] for row in result["cases"]] == old["expected_cases"]
    assert len({row["planned_requirement"] for row in result["cases"]}) == 383
    assert all(row["revised_status"] == "not_implemented_or_run" for row in result["cases"])
    assert Counter(row["historical_outcome"] for row in result["cases"]) == {
        "passed": 381,
        "failed": 2,
    }
    assert result["counts"] == {
        "unchanged-operations": 236,
        "mapped-transactions": 96,
        "normal-recipes": 31,
        "installed-d2-inference": 20,
    }
    failures = [row for row in result["cases"] if row["predicate_supersession"]]
    assert [row["historical_node"] for row in failures] == [
        f"tests/v1/test_device_normal_runtime_cuda.py::test_frozen_three_round_transactions"
        f"[False-8.0-{order}-ordinary-0-conflict-0-None]"
        for order in ("forward", "reverse")
    ]
    assert all(row["historical_outcome"] == "failed" for row in failures)
    assert result["original_tolerances"] == {key: old[key] for key in ("float32", "normal_float32")}
    saved = json.loads((ROOT / "v1-sprints/092-historical-case-mapping.json").read_text())
    assert saved == result


def test_frozen_study_covers_actual_traces_and_cpu_learners_with_original_row_oracles():
    result = study()
    assert result["device_execution"] is False
    assert len(result["cases"]) == len({row["id"] for row in result["cases"]}) == 106
    assert Counter(row["origin"] for row in result["cases"]) == {
        "stored_gpu_inputs_reanalyzed_on_cpu": 20,
        "declared_analytic_case": 5,
        "frozen_original_row_cpu_proposal": 81,
    }
    for row in result["cases"]:
        comparison, oracle = row["comparison"], row["high_precision"]
        if comparison["status"] in ("improvement", "worsening"):
            assert row["oracle_enclosed"] is True
            sign = -1 if comparison["status"] == "improvement" else 1
            assert oracle["signs"] == [sign, sign]
    cases = {row["id"]: row for row in result["cases"]}
    tiny = cases["analytic/tiny-improvement"]
    assert tiny["comparison"]["status"] == "improvement"
    assert Decimal(tiny["high_precision"]["decimal100"]) == Decimal.from_float(-(2.0**-61))
    cancellation = cases["analytic/scale-cancellation"]
    assert cancellation["comparison"]["status"] == "unresolved"
    assert not cancellation["high_precision"]["estimates_agree"]
    assert cancellation["high_precision"]["higher_precision"]["estimates_agree"]
    assert cases["analytic/outside-exponent-range"]["comparison"]["reason"] == "exponent_range"
    for order in ("forward", "reverse"):
        for alpha in (8.0, 4.0):
            assert (
                cases[f"run7/{order}/channel0/alpha{alpha}/training"]["comparison"]["status"]
                == "worsening"
            )
    for case in ("weighted", "d2"):
        for mode, damping in (("ordinary", 0), ("natural", 0), ("natural", 0.25)):
            for update in ("joint", "forward", "reverse"):
                assert any(
                    key.startswith(f"cpu-reference/{case}/{mode}/{damping}/{update}/")
                    for key in cases
                )
