"""Adversarial checks for evidence integrity, not synthetic quality claims."""

import hashlib
import json

import pytest
from benchmarks.v1.judge import cache_key, judge


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    manifest = {
        "schema": "openboost-integrity-v0",
        "protocol_sha256": "a" * 64,
        "provenance": {"code_sha": "b" * 40, "dirty": False, "environment": {"os": "fixture"}},
        "expected": [],
    }
    cases = []
    for i in range(1, 14):
        cell = {
            "id": f"A{i}/cpu/0",
            "application": f"A{i}",
            "required": True,
            "backend": "cpu",
            "dataset_sha256": "c" * 64,
            "split_sha256": "d" * 64,
            "preprocessing_sha256": "e" * 64,
            "config": {"depth": 2},
            "seed": 0,
            "model": "fixture",
            "fold": "0",
        }
        manifest["expected"].append(cell)
    # Compute keys only after the complete matrix is frozen.
    for cell in manifest["expected"]:
        artifacts = {}
        for role, data in {
            "predictions": b"[1.0, 2.0]",
            "model": b"fixture model",
            "log": b"ok",
        }.items():
            name = cell["application"] + "-" + role + ".json"
            (tmp_path / name).write_bytes(data)
            artifacts[role] = {"path": name, "sha256": digest(data)}
        cases.append(
            {
                "id": cell["id"],
                "status": "pass",
                "cache_key": cache_key(manifest, cell),
                "backend": "cpu",
                "fallback": False,
                "exit_code": 0,
                "artifacts": artifacts,
                "metrics": {"loss": 0.2},
                "reason": "",
            }
        )
    return tmp_path, manifest, cases


def test_missing_required_fold_fails(bundle):
    root, manifest, cases = bundle
    result = judge(manifest, cases[:-1], root)
    assert not result["integrity_pass"]
    assert any("missing case" in e for e in result["errors"])


def test_valid_bundle_is_only_integrity_not_quality(bundle):
    root, manifest, cases = bundle
    result = judge(manifest, cases, root)
    assert result["integrity_pass"]
    assert result["gate_results"] == {}
    assert len(result["statuses"]) == 13


@pytest.mark.parametrize("status", ["not_run", "fail", "unsupported", "error", "timeout"])
def test_every_required_nonpass_fails(bundle, status):
    root, manifest, cases = bundle
    cases[0].update(status=status, reason="intentional failure", exit_code=None)
    assert not judge(manifest, cases, root)["integrity_pass"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "cuda"),
        ("fallback", True),
        ("exit_code", 2),
        ("exit_code", None),
        ("exit_code", False),
        ("metrics", {"loss": float("nan")}),
        ("metrics", {"loss": float("inf")}),
        ("metrics", {}),
        ("status", "skipped"),
    ],
)
def test_false_pass_claims_fail(bundle, field, value):
    root, manifest, cases = bundle
    cases[0][field] = value
    assert not judge(manifest, cases, root)["integrity_pass"]


@pytest.mark.parametrize(
    "mutation",
    ["config", "split", "dataset", "preprocessing", "protocol", "code", "environment", "seed"],
)
def test_cache_binds_every_input(bundle, mutation):
    root, manifest, cases = bundle
    cell = manifest["expected"][0]
    if mutation in {"split", "dataset", "preprocessing"}:
        cell[mutation + "_sha256"] = "f" * 64
    elif mutation == "config":
        cell["config"]["depth"] = 3
    elif mutation == "protocol":
        manifest["protocol_sha256"] = "f" * 64
    elif mutation == "code":
        manifest["provenance"]["code_sha"] = "f" * 40
    elif mutation == "environment":
        manifest["provenance"]["environment"]["threads"] = 8
    else:
        cell["seed"] = 1
    report = judge(manifest, cases, root)
    assert not report["integrity_pass"]
    assert any("cache identity" in e for e in report["errors"])


@pytest.mark.parametrize(
    "mutation", ["duplicate", "unknown", "missing_predictions", "missing_log", "corrupt", "deleted"]
)
def test_missing_or_conflicting_evidence_fails(bundle, mutation):
    root, manifest, cases = bundle
    if mutation == "duplicate":
        cases.append(cases[0])
    elif mutation == "unknown":
        cases[0]["id"] = "unlisted"
    elif mutation.startswith("missing_"):
        del cases[0]["artifacts"][mutation.removeprefix("missing_")]
    else:
        path = root / cases[0]["artifacts"]["predictions"]["path"]
        if mutation == "deleted":
            path.unlink()
        else:
            path.write_text("[42]")
    assert not judge(manifest, cases, root)["integrity_pass"]


@pytest.mark.parametrize(
    "data", [b"[NaN]", b"[1e999]", b"[]", b"[true]", b"[[1],[1,2]]", b'"not predictions"']
)
def test_rehashed_invalid_predictions_still_fail(bundle, data):
    root, manifest, cases = bundle
    entry = cases[0]["artifacts"]["predictions"]
    (root / entry["path"]).write_bytes(data)
    entry["sha256"] = digest(data)
    assert not judge(manifest, cases, root)["integrity_pass"]


@pytest.mark.parametrize("path", ["../outside", "/tmp/outside"])
def test_path_escape_is_rejected(bundle, path):
    root, manifest, cases = bundle
    cases[0]["artifacts"]["predictions"]["path"] = path
    report = judge(manifest, cases, root)
    assert any("unsafe artifact path" in e for e in report["errors"])


def test_symlink_escape_is_rejected(bundle, tmp_path_factory):
    root, manifest, cases = bundle
    outside = tmp_path_factory.mktemp("outside") / "prediction.json"
    outside.write_bytes(b"[1.0, 2.0]")
    link = root / "escape.json"
    link.symlink_to(outside)
    cases[0]["artifacts"]["predictions"]["path"] = link.name
    assert any("escapes run directory" in e for e in judge(manifest, cases, root)["errors"])


@pytest.mark.parametrize(
    "mutation", ["empty", "duplicate", "optional_only", "dirty", "nan_config", "schema"]
)
def test_invalid_manifest_fails(bundle, mutation):
    root, manifest, cases = bundle
    if mutation == "empty":
        manifest["expected"] = []
    elif mutation == "duplicate":
        manifest["expected"].append(manifest["expected"][0])
    elif mutation == "optional_only":
        manifest["expected"][0]["required"] = False
    elif mutation == "dirty":
        manifest["provenance"]["dirty"] = True
    elif mutation == "nan_config":
        manifest["expected"][0]["config"]["rate"] = float("nan")
    else:
        manifest["schema"] = "future"
    assert not judge(manifest, cases, root)["integrity_pass"]


def test_optional_gpu_unsupported_is_visible_and_missing_is_failure(bundle):
    from copy import deepcopy

    root, manifest, cases = bundle
    gpu = deepcopy(manifest["expected"][0])
    gpu.update(id="A1/cuda/0", backend="cuda", required=False)
    manifest["expected"].append(gpu)
    for cell, record in zip(manifest["expected"], cases, strict=False):
        record["cache_key"] = cache_key(manifest, cell)
    record = deepcopy(cases[0])
    record.update(
        id=gpu["id"],
        status="unsupported",
        backend="none",
        exit_code=None,
        cache_key=cache_key(manifest, gpu),
        reason="no CUDA device",
        metrics={},
    )
    record["artifacts"] = {"log": record["artifacts"]["log"]}
    assert not judge(manifest, cases, root)["integrity_pass"]
    cases.append(record)
    report = judge(manifest, cases, root)
    assert report["integrity_pass"] and report["statuses"][gpu["id"]] == "unsupported"
    assert report["gate_results"] == {}
    gpu["required"] = True
    for cell, record in zip(manifest["expected"], cases, strict=True):
        record["cache_key"] = cache_key(manifest, cell)
    assert not judge(manifest, cases, root)["integrity_pass"]


def test_cli_exit_and_strict_json(bundle):
    import subprocess
    import sys

    root, manifest, cases = bundle
    (root / "manifest.json").write_text(json.dumps(manifest))
    path = root / "cases.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    command = [sys.executable, "-m", "benchmarks.v1.judge", str(root)]
    success = subprocess.run(command, capture_output=True, text=True)
    assert success.returncode == 0
    assert json.loads(success.stdout)["gate_results"] == {}
    path.write_text("\n".join(json.dumps(c) for c in cases[:-1]))
    failure = subprocess.run(command, capture_output=True, text=True)
    assert failure.returncode == 1
    assert not json.loads(failure.stdout)["integrity_pass"]
    path.write_text('{"id": "first", "id": "second"}\n')
    duplicate = subprocess.run(command, capture_output=True, text=True)
    assert duplicate.returncode == 1
    assert "duplicate JSON field" in duplicate.stdout


def test_missing_second_fold_fails_even_with_all_applications_present(bundle):
    from copy import deepcopy

    root, manifest, cases = bundle
    second_fold = deepcopy(manifest["expected"][0])
    second_fold.update(id="A1/cpu/1", fold="1", seed=1, split_sha256="f" * 64)
    manifest["expected"].append(second_fold)
    for cell, record in zip(manifest["expected"], cases, strict=False):
        record["cache_key"] = cache_key(manifest, cell)
    report = judge(manifest, cases, root)
    assert report["errors"] == ["missing case: A1/cpu/1"]
