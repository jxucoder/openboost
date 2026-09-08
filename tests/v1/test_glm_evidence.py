"""Offline result integrity and numerical-audit counterexamples; no CUDA work."""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "benchmarks/v1/evidence/cuda-glm-108/analyze.py"
SPEC = importlib.util.spec_from_file_location("glm_evidence_108", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_raw_verdict_and_retained_evidence_reproduce_offline_analysis():
    assert audit.analyze() == audit.read(audit.ROOT / "analysis.json")


@pytest.mark.parametrize("fault", ["dirty", "approval", "source", "package", "artifact-index"])
def test_mutated_provenance_cannot_pass_audit(fault, monkeypatch):
    read = audit.read
    manifest = copy.deepcopy(read(audit.ROOT / "manifest.json"))
    if fault == "dirty":
        manifest["dirty"] = True
    elif fault == "approval":
        manifest["protocol"]["authorization"] = "pending"
    elif fault == "source":
        manifest["sources"]["src/openboost/device_glm.py"] = "changed"
    elif fault == "package":
        manifest["result"]["packages"]["numpy"] = "0.0"
    else:
        manifest["artifacts"].pop("pytest.log")
    monkeypatch.setattr(
        audit, "read", lambda path: manifest if path.name == "manifest.json" else read(path)
    )
    with pytest.raises(ValueError):
        audit.analyze()


def test_raw_artifact_tampering_is_rejected(monkeypatch):
    original = (audit.ROOT / "pytest.log").read_bytes()
    digest = audit.digest
    monkeypatch.setattr(
        audit, "digest", lambda value: "changed" if value == original else digest(value)
    )
    with pytest.raises(ValueError, match="raw artifact changed"):
        audit.analyze()


@pytest.mark.parametrize("fault", [None, "bounds", "identity", "method"])
def test_independent_difference_rejects_invalid_comparison_claims(fault):
    arrays = ([0], [0.01], [1], [0], [1], [1])
    exact = str(audit.direct_difference("binary", *arrays))
    record = dict(
        lower=-1,
        upper=0,
        unchanged=False,
        method="binary-convex-taylor18-interval-v1",
        reason="test",
    )
    if fault == "bounds":
        record.update(lower=1, upper=2)
    elif fault == "identity":
        record.update(lower=0, upper=0, unchanged=True)
    elif fault == "method":
        record["method"] = "unverified"
    result = audit.comparison("binary", arrays, record, exact)
    assert all(result[k] for k in ("encloses", "identity_matches", "method_matches")) is (
        fault is None
    )
    with pytest.raises(ValueError, match="direct likelihood differs"):
        audit.comparison("binary", arrays, record, "1")
