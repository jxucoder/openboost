"""Local source, budget and installed-snapshot guards; no CUDA execution."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from benchmarks.v1.cuda_aggregation_preflight import (
    check_dispatch,
    check_frozen_sources,
    snapshot_hashes,
    snapshot_paths,
)
from benchmarks.v1.cuda_validation_preflight import PROTOCOL
from benchmarks.v1.freeze_validation_run11 import budget_seconds

ROOT = Path(__file__).resolve().parents[2]


def packet_sources(protocol):
    if protocol["authorization"] != "consumed":
        return snapshot_hashes(ROOT, snapshot_paths(ROOT, PROTOCOL, protocol))
    # A consumed packet describes its execution revision, not today's production.
    manifest = json.loads((ROOT / protocol["output"] / "manifest.json").read_text())
    assert manifest["protocol"]["frozen_sources"] == protocol["frozen_sources"]
    sources = {
        name: hashlib.sha256(
            subprocess.check_output(["git", "show", manifest["revision"] + ":" + name], cwd=ROOT)
        ).hexdigest()
        for name in manifest["sources"]
    }
    assert sources == manifest["sources"]
    return sources


def test_local_baseline_install_matches_original_and_replays_saved_models():
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    report = json.loads((ROOT / "v1-sprints/105-baseline-install-local.json").read_text())
    assert report["passed"] and report["installed"]["sources"] == protocol["baseline_sources"]
    assert report["replay"] == dict(complete=True, verified_fits=2)
    for name, sha in report["support_sources"].items():
        assert hashlib.sha256((ROOT / "benchmarks/v1" / name).read_bytes()).hexdigest() == sha
    assert report["installed"]["packages"]["numpy"] == "2.3.5"


def test_frozen_payload_baseline_and_isolated_collection_agree():
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    sources = packet_sources(protocol)
    check_frozen_sources(PROTOCOL, protocol, sources)
    assert len(sources) == protocol["upload_file_count"] == 88
    collected = json.loads((ROOT / "v1-sprints/105-isolated-collection.json").read_text())
    assert collected["cohorts"]["candidate"]["cases"] == protocol["expected_cases"]
    assert collected["cohorts"]["candidate"]["collected"] == 474
    assert collected["cohorts"]["candidate"]["executed"] == 0
    assert {p: h for p, h in collected["snapshot_sources"].items() if p != PROTOCOL} == protocol[
        "frozen_sources"
    ]
    core = {p: h for p, h in sources.items() if p.startswith("src/openboost/")}
    assert collected["installed_sources"] == core
    assert {p for p in core if core[p] != protocol["baseline_sources"][p]} == set(
        protocol["baseline_overlays"]
    )
    for target, original in protocol["baseline_overlays"].items():
        assert sources[original] == protocol["baseline_sources"][target]
    assert budget_seconds(protocol) == protocol["budget"]["test_seconds"] == 600
    assert protocol["resources"]["timeout_seconds"] == 900
    assert len(set(protocol["retained_artifacts"])) == 17


def test_unapproved_packet_cannot_dispatch(tmp_path):
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    protocol.update(authorization="pending", upload_authorization="pending")
    with pytest.raises(ValueError, match="allowance is pending"):
        check_dispatch(ROOT, tmp_path / "not-a-run", protocol)


def test_post_freeze_edit_cannot_dispatch():
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    sources = packet_sources(protocol)
    sources["src/openboost/device.py"] = "changed"
    with pytest.raises(ValueError, match="source freeze changed"):
        check_frozen_sources(PROTOCOL, protocol, sources)


def test_consumed_packet_never_reads_current_source_tree(monkeypatch):
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    assert protocol["authorization"] == "consumed"

    def forbidden(*args):
        raise AssertionError("consumed evidence must not bind later source files")

    monkeypatch.setattr("tests.v1.test_validation_freeze.snapshot_paths", forbidden)
    assert len(packet_sources(protocol)) == 88
