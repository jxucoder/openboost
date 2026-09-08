"""Local closure/status checks do not count as a passing remote isolation smoke."""

import copy
import json
import shutil
from pathlib import Path

import pytest
from benchmarks.v1.authoring import modal_worker as worker
from benchmarks.v1.authoring.linux_probe import CASES

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def snapshot(tmp_path):
    shutil.copytree(ROOT / worker.PACKET, tmp_path / worker.PACKET)
    probe = tmp_path / worker.PROBE
    probe.parent.mkdir(parents=True)
    probe.write_bytes((ROOT / worker.PROBE).read_bytes())
    packet = json.loads((tmp_path / worker.PACKET / "manifest.json").read_text())
    uploads = {
        f"{worker.PACKET}/author/{name}": f"/materials/{name}" for name in packet["author_files"]
    }
    uploads[worker.PROBE] = "/opt/probe.py"
    files = {
        name: worker.digest((tmp_path / name).read_bytes())
        for name in [*uploads, f"{worker.PACKET}/manifest.json"]
    }
    return tmp_path, dict(
        schema="openboost-linux-worker-smoke-v1",
        files=files,
        uploads=uploads,
        cases=list(CASES),
        resources=worker.RESOURCES,
        modal_version=worker.SDK,
        authorization="pending",
    )


def test_real_packet_is_staged_file_by_file_and_sdk_can_construct_image(snapshot, tmp_path):
    modal = pytest.importorskip("modal")
    root, freeze = snapshot
    packet, uploads = worker.inputs(root, freeze)
    delivery = worker.stage(root, uploads, freeze["files"], tmp_path / "delivery")
    assert len(delivery) == 13 and len(packet["wheel_sources"]) == 31
    assert set(delivery) == set(uploads.values())
    assert not any("evaluator" in path or "expected.json" in path for path in delivery)
    assert isinstance(worker.image_for(modal, delivery), modal.Image)
    # Constructing this public SDK object does not build/upload/start an image.


@pytest.mark.parametrize(
    "change", ["unlisted", "declared_answer", "changed", "missing", "symlink", "parent_symlink"]
)
def test_packet_mutations_cannot_expand_or_change_delivery(snapshot, change):
    root, freeze = snapshot
    author = root / worker.PACKET / "author"
    if change == "unlisted":
        (author / "answers.json").write_text("{}")
    elif change == "declared_answer":
        name = f"{worker.PACKET}/author/README.md"
        freeze["uploads"][name] = "/evaluator/expected.json"
    elif change == "changed":
        (author / "README.md").write_text("changed")
    elif change in ("missing", "symlink"):
        target = author / "README.md"
        original = target.read_bytes()
        target.unlink()
        if change == "symlink":
            (root / "answer-copy").write_bytes(original)
            target.symlink_to(root / "answer-copy")
    elif change == "parent_symlink":
        (author / "docs").rename(root / "docs")
        (author / "docs").symlink_to(root / "docs", target_is_directory=True)
    with pytest.raises(ValueError):
        worker.inputs(root, freeze)


@pytest.mark.parametrize(
    "name", ["../answer", "/tmp/answer", "docs/../answer", "docs//answer", "./answer"]
)
def test_noncanonical_paths_are_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="canonical"):
        worker.regular(tmp_path, name)


def test_changed_file_during_staging_is_rejected(snapshot, tmp_path):
    root, freeze = snapshot
    _, uploads = worker.inputs(root, freeze)
    (root / worker.PROBE).write_text("changed after check")
    with pytest.raises(ValueError, match="while staging"):
        worker.stage(root, uploads, freeze["files"], tmp_path / "delivery")


def test_pending_execution_stops_before_any_service_call(snapshot, tmp_path):
    root, freeze = snapshot
    with pytest.raises(ValueError, match="pending user authorization"):
        worker.execute(root, freeze, b"{}", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_missing_protected_file_still_retains_failed_gate_and_raw_logs(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "stdout.jsonl").write_text('{"kind":"partial"}\n')
    report = dict(passed=True)
    worker.retain(output, report, tmp_path, {"missing-evaluator.json": worker.digest(b"{}")})
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["passed"] is False
    assert saved["protected_after"] == {"missing-evaluator.json": None}
    assert "missing-evaluator.json" in saved["integrity_errors"]
    assert saved["artifacts"]["stdout.jsonl"] == worker.digest(b'{"kind":"partial"}\n')


def result_stream():
    packet = json.loads((ROOT / worker.PACKET / "manifest.json").read_text())
    core = {name.removeprefix("openboost/"): sha for name, sha in packet["wheel_sources"].items()}
    core["py.typed"] = worker.digest(b"")
    records = [
        dict(
            kind="runtime",
            core_files=core,
            packages={"openboost": "1.0.0.dev0", "numpy": "2.3.5", "uv": "0.12.1"},
        )
    ]
    records += [dict(kind="case", name=name, passed=True) for name in CASES]
    records += [dict(kind="ready_for_timeout", child_pid=12, child_session=12)]
    return packet, records


def test_classifier_accepts_complete_shape_only_with_provider_timeout():
    packet, records = result_stream()
    stdout = "\n".join(map(json.dumps, records))
    assert worker.classify(stdout, True, packet)["passed"]
    assert not worker.classify(stdout, False, packet)["passed"]


@pytest.mark.parametrize(
    "change",
    [
        "startup",
        "missing",
        "duplicate",
        "failed",
        "truthy",
        "core",
        "marker",
        "session",
        "packages",
    ],
)
def test_incomplete_or_failed_stream_never_passes_as_timeout(change):
    packet, records = result_stream()
    records = copy.deepcopy(records)
    if change == "startup":
        records = []
    elif change == "missing":
        records.pop(1)
    elif change == "duplicate":
        records[2] = records[1]
    elif change == "failed":
        records[1]["passed"] = False
    elif change == "truthy":
        records[1]["passed"] = 1
    elif change == "core":
        records[0]["core_files"]["__init__.py"] = "changed"
    elif change == "marker":
        records.pop()
    elif change == "session":
        records[-1]["child_session"] = 10
    elif change == "packages":
        records[0]["packages"]["numpy"] = "wrong"
    assert not worker.classify("\n".join(map(json.dumps, records)), True, packet)["passed"]
