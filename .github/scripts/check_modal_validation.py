#!/usr/bin/env python3
"""Check committed Modal receipts and byte bindings; never run validation jobs."""

import hashlib
import json
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath

EVIDENCE = "docs/v1/evidence/pr27-modal-validation/"
POLICY = ".github/modal-validation-policy.json"
PHASES = {"cpu310": {"cpu310", "cpu310-build", "lint"},
          "cpu312": {"cpu312", "build", "docs", "lint", "gpu-prerequisites"},
          "gpu": {"gpu", "gpu-build", "gpu-prerequisites"}}
REQUIRED_JOBS = {"cpu310", "cpu312", "gpu", "build", "docs"}
BOOTSTRAP = "import pathlib,sys;sys.path.insert(0,sys.argv.pop(1));import openboost;assert 'site-packages' in pathlib.Path(openboost.__file__).parts;import pytest;raise SystemExit(pytest.main(sys.argv[1:]))"
MAX_JSON = 8 * 1024**2
MAX_ARTIFACT = 128 * 1024**2
MAX_TOTAL = 512 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def parse(raw):
    require(len(raw) <= MAX_JSON, "JSON report exceeds byte limit")
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def safe(path):
    require(type(path) is str and path and ":" not in path and "\\" not in path
            and not any(ord(c) < 32 or ord(c) == 127 for c in path), "unsafe artifact path")
    value = PurePosixPath(path)
    require(not value.is_absolute() and str(value) == path
            and all(part not in {".", ".."} for part in value.parts), "unsafe artifact path")
    return path


def sha(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def validate_gpu_tests(paths):
    require(type(paths) is list and paths and all(type(path) is str for path in paths)
            and len(set(paths)) == len(paths)
            and all(re.fullmatch(r"tests/v1/test_[A-Za-z0-9_]+\.py(?:::[A-Za-z0-9_]+)?", path)
                    for path in paths), "frozen GPU selectors differ")
    return paths


def git(*args):
    return subprocess.check_output(["git", *args], stderr=subprocess.PIPE, timeout=30)


def inventory(revision):
    result = {}
    for entry in git("ls-tree", "-rz", "--full-tree", revision).split(b"\0"):
        if not entry:
            continue
        header, path = entry.split(b"\t", 1)
        path = safe(path.decode("utf-8"))
        if path.startswith(EVIDENCE):
            continue
        mode, kind, oid = header.decode("ascii").split()
        require(mode in {"100644", "100755"} and kind == "blob", "nonregular tracked input: " + path)
        result[path] = dict(mode=mode, type=kind, oid=oid)
    require(result, "empty candidate inventory")
    return result


def blob(path, maximum=MAX_JSON):
    safe(path)
    spec = "HEAD:" + path
    entry = git("ls-tree", "-z", "HEAD", "--", path).split(b"\0")
    require(len(entry) == 2 and entry[0].split(b"\t", 1)[0].split()[0] in {b"100644", b"100755"},
            "tracked regular artifact required: " + path)
    require(git("cat-file", "-t", spec).strip() == b"blob", "tracked regular blob required: " + path)
    size = int(git("cat-file", "-s", spec))
    require(0 <= size <= maximum, "artifact exceeds byte limit: " + path)
    value = git("cat-file", "blob", spec)
    require(len(value) == size, "artifact size changed: " + path)
    return value


def bound(path, descriptor, maximum=MAX_ARTIFACT):
    require(type(descriptor) is dict and set(descriptor) == {"sha256", "bytes"}, "artifact descriptor fields differ")
    require(type(descriptor["bytes"]) is int and descriptor["bytes"] >= 0, "artifact size must be an integer")
    raw = blob(path, maximum)
    require(len(raw) == descriptor["bytes"] and hashlib.sha256(raw).hexdigest() == sha(descriptor["sha256"]),
            "artifact bytes differ: " + path)
    return raw


def junit(raw, *, allowed_skips, gpu, nodeids):
    require(len(raw) <= 16 * 1024**2 and b"<!DOCTYPE" not in raw and b"<!ENTITY" not in raw,
            "bounded plain JUnit XML required")
    root = ET.fromstring(raw)
    require(root.tag in {"testsuites", "testsuite"}, "JUnit root differs")
    cases = list(root.iter("testcase"))
    require(cases and not list(root.iter("failure")) and not list(root.iter("error")),
            "JUnit has failures/errors or no executed cases")
    seen, skipped = set(), []
    for case in cases:
        name = case.attrib.get("classname", "") + "::" + case.attrib.get("name", "")
        require(name not in seen and case.attrib.get("name"), "duplicate or unnamed JUnit case")
        seen.add(name)
        skips = case.findall("skipped")
        require(len(skips) <= 1, "duplicate JUnit skip")
        if skips:
            skipped.append(dict(id=name, reason=skips[0].attrib.get("message", "")))
    require(len(skipped) < len(cases), "all validation cases were skipped")
    expected = set()
    require(type(nodeids) is list and nodeids and len(set(nodeids)) == len(nodeids), "unique collected node IDs required")
    for nodeid in nodeids:
        path, separator, tail = nodeid.partition(".py::")
        require(separator and path and tail, "invalid collected node ID")
        head, bracket, parameters = tail.partition("[")
        parts = head.split("::")
        classname = path.replace("/", ".") + ("." + ".".join(parts[:-1]) if len(parts) > 1 else "")
        expected.add(classname + "::" + parts[-1] + bracket + parameters)
    require(seen == expected, "executed JUnit population differs from original collection")
    require(not gpu or not skipped, "CUDA validation cannot skip cases")
    require(sorted(skipped, key=lambda x: x["id"]) == allowed_skips, "JUnit skips differ from frozen policy")
    for suite in root.iter("testsuite"):
        for field in ("failures", "errors"):
            require(int(suite.attrib.get(field, "0")) == 0, "JUnit suite reports " + field)
        require(int(suite.attrib.get("tests", "-1")) == len(list(suite.iter("testcase"))),
                "JUnit testcase count differs")
    return len(cases)


def expected_command(name, protocol):
    python = "/tmp/pr27-environment/bin/python"
    if name == "build" or name.endswith("-build"):
        return ["uv", "build", "--python", python, "--no-build-isolation", "--out-dir", "/tmp/pr27-results/dist"]
    if name == "docs":
        return ["/tmp/pr27-environment/bin/mkdocs", "build", "--strict", "--site-dir", "/tmp/pr27-site"]
    if name == "lint":
        return ["/tmp/pr27-environment/bin/ruff", "check", "src/openboost", "tests/v1", "tests/conftest.py"]
    targets = protocol["gpu_tests"] if name == "gpu" else ["tests/v1/test_model_identity_cache.py"] if name == "gpu-prerequisites" else ["tests/"]
    extra = ["-m", "gpu"] if name == "gpu" else [] if name == "gpu-prerequisites" else ["-m", "not gpu and not benchmark"]
    if name == "gpu":
        extra.append("--basetemp=/tmp/pr27-results/pytest-tmp")
    xml = "gpu.xml" if name == "gpu" else "prerequisites.xml" if name == "gpu-prerequisites" else "cpu.xml"
    return [python, "-I", "-c", BOOTSTRAP, "/tmp/pr27-source", *targets, "-o", "addopts=", "-n", "0", "-q", *extra, "--junitxml=/tmp/pr27-results/" + xml]


def verify():
    current = inventory("HEAD")
    index = parse(blob(EVIDENCE + "report.json"))
    require(type(index) is dict and set(index) == {"schema", "source_commit", "inventory_sha256", "phases"},
            "validation index fields differ")
    require(index["schema"] == "openboost-pr27-modal-validation-index-v1", "validation index version differs")
    source = index["source_commit"]
    require(type(source) is str and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source), "full source commit required")
    require(inventory(source) == current, "candidate changed since Modal validation")
    expected_digest = digest(current)
    require(sha(index["inventory_sha256"]) == expected_digest, "candidate inventory digest differs")
    require(type(index["phases"]) is dict and set(index["phases"]) == set(PHASES), "all three validation phases required")
    policy_raw = blob(POLICY)
    policy = parse(policy_raw)
    require(type(policy) is dict and set(policy) == {"schema", "jobs", "gpu_tests", "expected_gpu_cases"}
            and policy["schema"] == "openboost-pr27-modal-validation-policy-v1", "validation policy differs")
    require(set(policy["jobs"]) == REQUIRED_JOBS, "required policy jobs differ")
    validate_gpu_tests(policy["gpu_tests"])
    require(type(policy["expected_gpu_cases"]) is int and policy["expected_gpu_cases"] > 0, "positive frozen GPU population required")
    expected_installed = {path: hashlib.sha256(blob(path)).hexdigest() for path in current
                          if path.startswith("src/openboost/") and path.endswith(".py")}
    require(expected_installed, "production package missing")
    total, counts = 0, {}
    for phase, expected_jobs in PHASES.items():
        prefix = EVIDENCE + phase + "/"
        result = parse(bound(prefix + "manifest.json", index["phases"][phase], MAX_JSON))
        require(result["schema"] == "openboost-pr27-modal-validation-v1"
                and result["phase"] == phase and result["passed"] is True, phase + " did not pass")
        require(result["source_commit"] == source and result["inventory"] == current
                and result["inventory_sha256"] == expected_digest, phase + " source binding differs")
        protocol = result["protocol"]
        require(digest(protocol) == sha(result["protocol_sha256"])
                and protocol["source_commit"] == source and protocol["inventory_sha256"] == expected_digest
                and protocol["policy"] == dict(path=POLICY, sha256=hashlib.sha256(policy_raw).hexdigest())
                and result["policy_sha256"] == protocol["policy"]["sha256"], phase + " frozen execution protocol differs")
        require(protocol["gpu_tests"] == policy["gpu_tests"] and protocol["expected_gpu_cases"] == policy["expected_gpu_cases"], "selected CUDA scope changed")
        require(result["platform"]["provider"] == "Modal" and result["platform"]["os"] == "Linux"
                and result["collection_complete"] is True and result["git_clean"] is True,
                phase + " completed Modal execution observations missing")
        require(result["installed_sources"] == expected_installed, phase + " installed source bytes differ")
        if phase == "gpu":
            require("T4" in result["gpu"]["hardware"], "actual T4 observation missing")
        require(type(result["jobs"]) is dict and set(result["jobs"]) == expected_jobs, phase + " job population differs")
        require(type(result["artifacts"]) is dict and result["artifacts"], phase + " raw artifacts missing")
        artifacts = {}
        for path, descriptor in result["artifacts"].items():
            safe(path)
            raw = bound(prefix + path, descriptor)
            total += len(raw)
            require(total <= MAX_TOTAL, "aggregate receipt artifacts exceed bound")
            # Keep only JUnit/short logs in memory after hashing other raw artifacts.
            if path.endswith((".xml", ".json")):
                artifacts[path] = raw
        for name, job in result["jobs"].items():
            expected_python = "3.10" if phase == "cpu310" else "3.12"
            kind = "build" if name.endswith("-build") else "pytest" if name == "gpu-prerequisites" else "lint"
            rule = policy["jobs"].get(name, dict(python=expected_python, kind=kind, allowed_skips=[]))
            require(set(rule) == {"python", "kind", "allowed_skips"}, "job policy fields differ")
            kind = "pytest" if name in {"cpu310", "cpu312", "gpu", "gpu-prerequisites"} else "build" if name.endswith("-build") else name
            require(rule["python"] == expected_python and rule["kind"] == kind,
                    "required CPU/CUDA/build/docs policy differs")
            require(job["status"] == "pass" and type(job["exit_code"]) is int and job["exit_code"] == 0,
                    name + " has no passing process result")
            require(job["source_commit"] == source and job["inventory_sha256"] == expected_digest,
                    name + " source binding differs")
            require(job["command"] == expected_command(name, protocol), name + " original command differs from required execution")
            require(type(job["wall_seconds"]) in {int, float} and math.isfinite(job["wall_seconds"]) and job["wall_seconds"] > 0,
                    name + " runtime observation missing")
            require(result["platform"]["python"].startswith(expected_python + "."), name + " actual Python version differs")
            require(type(job["artifacts"]) is list and job["artifacts"] and len(set(job["artifacts"])) == len(job["artifacts"]),
                    name + " artifact inventory missing or duplicated")
            require(all(path in result["artifacts"] for path in job["artifacts"]), name + " job artifact missing")
            require(name + ".stdout.txt" in job["artifacts"] and name + ".stderr.txt" in job["artifacts"], name + " raw process logs missing")
            if rule["kind"] == "pytest":
                require(job["junit"] in job["artifacts"] and job["junit"] in artifacts, name + " JUnit artifact missing")
                require(job["collection"] in job["artifacts"] and job["collection"] in artifacts, name + " original collection artifact missing")
                collected = parse(artifacts[job["collection"]])
                require(collected["command"] == [value for value in job["command"] if not value.startswith("--junitxml=")] + ["--collect-only"], name + " original collection command differs")
                counts[name] = junit(artifacts[job["junit"]], allowed_skips=rule["allowed_skips"], gpu=name == "gpu", nodeids=collected["nodeids"])
                require(type(collected["case_count"]) is int and collected["case_count"] == counts[name] == job["cases"], name + " collected/observed case count differs")
                if name == "gpu":
                    require(counts[name] == policy["expected_gpu_cases"], "selected GPU population differs")
    return dict(source_commit=source, inventory_files=len(current), artifact_bytes=total, testcase_counts=counts)


if __name__ == "__main__":
    try:
        result = verify()
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError, ET.ParseError, RecursionError) as error:
        print("Modal validation gate failed: " + str(error), file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, sort_keys=True))
    print("Committed Modal receipts match this candidate; no runtime work executed by this check.")
