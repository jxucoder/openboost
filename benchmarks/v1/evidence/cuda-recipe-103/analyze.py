"""Audit run 9 and combined recipe coverage without rewriting run 8's verdict."""

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(path.read_text())


def cases(path):
    return [
        dict(
            case=case.get("classname", "") + "::" + case.get("name", ""),
            status="fail"
            if any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
            else "pass",
        )
        for case in ET.parse(path).getroot().iter("testcase")
    ]


def verify_sources(manifest):
    check(manifest["dirty"] is False, "dispatch must be clean")
    for name, expected in manifest["sources"].items():
        value = subprocess.check_output(
            ["git", "show", manifest["revision"] + ":" + name], cwd=REPO
        )
        check(digest(value) == expected, f"dispatch source differs: {name}")
    result = manifest["result"]
    core = {p: h for p, h in manifest["sources"].items() if p.startswith("src/openboost/")}
    check(result["installed_sources"] == core, "installed core differs")
    check(result["snapshot_sources"] == manifest["sources"], "uploaded snapshot differs")
    expected_packages = dict(p.split("==") for p in manifest["protocol"]["packages"])
    check(result["packages"] == expected_packages, "installed package versions differ")


def analyze():
    manifest = read(ROOT / "manifest.json")
    verdict = read(ROOT / "verdict.json")
    protocol = manifest["protocol"]
    for name, expected in manifest["artifacts"].items():
        check(digest((ROOT / name).read_bytes()) == expected, f"raw artifact changed: {name}")
    verify_sources(manifest)
    check(len(manifest["sources"]) == 46, "upload closure differs")
    prior = protocol["prior_evidence"]
    for name, expected in prior["artifacts"].items():
        check(digest((REPO / name).read_bytes()) == expected, f"prior artifact changed: {name}")
    earlier_root = (REPO / prior["manifest"]).parent
    earlier = read(earlier_root / "manifest.json")
    original = read(earlier_root / "verdict.json")
    verify_sources(earlier)
    check(earlier["revision"] == prior["revision"], "prior revision differs")
    check(original["passed"] is False and earlier["status"] == "fail", "old failure erased")
    check(original["historical"]["expected_disagreements_match"], "historical outcomes differ")
    archive = read(earlier_root / "archive-index.json")["files"]
    for name, expected in archive.items():
        check(
            digest((earlier_root / name).read_bytes()) == expected, f"old archive changed: {name}"
        )
    check(
        manifest["result"]["installed_sources"] == earlier["result"]["installed_sources"],
        "production changed between runs",
    )
    check(protocol["packages"] == earlier["protocol"]["packages"], "package freeze changed")
    changed = sorted(
        p
        for p, h in manifest["sources"].items()
        if p in earlier["sources"] and h != earlier["sources"][p]
    )
    check(changed == protocol["test_files"], "unexpected reused source change")
    (fixture,) = changed
    corrected = subprocess.check_output(
        ["git", "show", protocol["fixture_revision"] + ":" + fixture], cwd=REPO
    )
    check(digest(corrected) == manifest["sources"][fixture], "corrected fixture differs")
    current_cases = cases(ROOT / "junit.xml")
    old_cases = cases(earlier_root / "revised/junit.xml")
    check(current_cases == verdict["cases"], "run-9 JUnit/verdict disagree")
    check(old_cases == original["revised"]["cases"], "run-8 JUnit/verdict disagree")
    expected = [
        node.replace(".py::", "::").replace("/", ".") for node in protocol["expected_cases"]
    ]
    check(len(expected) == len(set(expected)) == 15, "recipe matrix differs")
    check(sorted(row["case"] for row in current_cases) == sorted(expected), "recipe cases differ")
    recipe_class = fixture[:-3].replace("/", ".") + "::"
    retained = [row for row in old_cases if not row["case"].startswith(recipe_class)]
    check(len(old_cases) == 529 and len(retained) == 514, "earlier case coverage differs")
    check(all(row["status"] == "pass" for row in retained), "earlier carried case failed")
    check(
        sorted(row["case"] for row in old_cases if row["case"].startswith(recipe_class))
        == sorted(expected),
        "recipe identities changed between runs",
    )
    passed = all(row["status"] == "pass" for row in current_cases)
    passed &= manifest["result"]["exit_code"] == 0
    check(verdict["passed"] == passed, "run-9 pass flag disagrees")
    check(manifest["status"] == ("pass" if passed else "fail"), "manifest outcome disagrees")
    return dict(
        manifest_sha256=digest((ROOT / "manifest.json").read_bytes()),
        execution_revision=manifest["revision"],
        raw_artifacts_verified=len(manifest["artifacts"]),
        dispatch_sources_verified=len(manifest["sources"]),
        run9_passed=passed,
        recipe_cases=current_cases,
        earlier_revision=earlier["revision"],
        earlier_raw_verdict=False,
        earlier_archive_files_verified=len(archive),
        earlier_dispatch_sources_verified=len(earlier["sources"]),
        production_sources_match=True,
        changed_reused_sources=changed,
        earlier_other_cases_passed=len(retained),
        new_recipe_cases_passed=sum(row["status"] == "pass" for row in current_cases),
        combined_revised_coverage_passed=passed,
        combined_revised_cases=len(retained) + len(current_cases),
        dispatch_seconds=(
            datetime.fromisoformat(manifest["finished"])
            - datetime.fromisoformat(manifest["started"])
        ).total_seconds(),
        worker_seconds=manifest["result"]["worker_wall_seconds"],
        scope="514 earlier passes plus fifteen new recipe cases with identical production. "
        "Not one 529-case passing invocation or full Normal/application/performance conformance.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = analyze()
    if args.check:
        check(result == read(ROOT / "analysis.json"), "derived analysis differs")
    else:
        (ROOT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "run9_passed",
                    "earlier_other_cases_passed",
                    "new_recipe_cases_passed",
                    "combined_revised_coverage_passed",
                    "earlier_raw_verdict",
                )
            },
            indent=2,
        )
    )
