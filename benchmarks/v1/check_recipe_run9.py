"""Offline run-9 provenance and isolated wheel collection; never dispatch Modal."""

import hashlib
import json
import subprocess
from pathlib import Path

from benchmarks.v1.cuda_aggregation_preflight import (
    check_frozen_sources,
    snapshot_hashes,
    snapshot_paths,
)
from benchmarks.v1.cuda_recipe_preflight import PROTOCOL
from benchmarks.v1.freeze_comparison_run8 import collect_snapshot

ROOT = Path(__file__).resolve().parents[2]


def main():
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    sources = snapshot_hashes(ROOT, snapshot_paths(ROOT, PROTOCOL, protocol))
    check_frozen_sources(PROTOCOL, protocol, sources)
    assert len(sources) == protocol["upload_file_count"]
    prior = protocol["prior_evidence"]
    for path, digest in prior["artifacts"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    manifest = json.loads((ROOT / prior["manifest"]).read_text())
    verdict = json.loads((ROOT / prior["verdict"]).read_text())
    assert manifest["revision"] == prior["revision"] and not manifest["dirty"]
    assert manifest["status"] == "fail" and verdict["passed"] is False
    assert verdict["historical"]["expected_disagreements_match"] is True
    assert protocol["packages"] == manifest["protocol"]["packages"]
    core = {p: h for p, h in sources.items() if p.startswith("src/openboost/")}
    assert core == {
        p: h for p, h in manifest["sources"].items() if p.startswith("src/openboost/")
    }
    changed = sorted(
        p for p, h in sources.items() if p in manifest["sources"] and h != manifest["sources"][p]
    )
    assert changed == protocol["test_files"]
    (fixture,) = protocol["test_files"]
    corrected = subprocess.check_output(
        ["git", "show", protocol["fixture_revision"] + ":" + fixture], cwd=ROOT
    )
    assert hashlib.sha256(corrected).hexdigest() == sources[fixture]
    expected = [
        node for node in manifest["protocol"]["cohorts"]["revised"]["expected_cases"]
        if node.startswith(fixture + "::")
    ]
    assert expected == protocol["expected_cases"] and len(expected) == 15
    recipe_class = fixture[:-3].replace("/", ".") + "::"
    earlier = verdict["revised"]["cases"]
    retained = [case for case in earlier if not case["case"].startswith(recipe_class)]
    assert len(earlier) == 529 and len(retained) == 514
    assert all(case["status"] == "pass" for case in retained)
    assert [case["case"] for case in earlier if case["status"] != "pass"] == [
        recipe_class
        + "test_recipe_current_best_and_patience_have_distinct_validation_anchors[forward]"
    ]
    report = collect_snapshot(
        dict(protocol, cohorts={"recipe": protocol}), protocol_path=PROTOCOL
    )
    report["prior_evidence"] = dict(
        revision=prior["revision"],
        original_verdict=False,
        earlier_other_cases_passed=len(retained),
        changed_reused_sources=changed,
        production_sources_match=True,
    )
    (ROOT / "v1-sprints/103-isolated-collection.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"Verified {len(sources)} frozen files; 15 isolated cases collected; zero GPU executions.")


if __name__ == "__main__":
    main()
