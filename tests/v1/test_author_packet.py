"""The author view must be closed, auditable and free of evaluator traversal."""

import json
import re
import subprocess
import zipfile
from pathlib import Path

import pytest
from benchmarks.v1.prepare_author_packet import AUTHOR_FILES, audit_wheel, copy_author_files

ROOT = Path(__file__).resolve().parents[2]


def test_current_view_has_only_resolving_internal_links(tmp_path):
    result = copy_author_files(ROOT, tmp_path)
    assert result["omitted_links"]
    assert set(result["original_sources"]) == set(AUTHOR_FILES)
    assert {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.md")} == set(AUTHOR_FILES) | {
        "README.md"
    }
    for path in tmp_path.rglob("*.md"):
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            assert (path.parent / target.split("#")[0]).resolve().is_file()
    assert not (tmp_path / "examples").exists()
    assert not (tmp_path / "tests").exists()


def test_links_cannot_expand_allowlist_to_evaluator(tmp_path):
    repo, author = tmp_path / "repo", tmp_path / "author"
    for name in AUTHOR_FILES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[outside](../../tests/v1/private.py)\n")
    result = copy_author_files(repo, author)
    assert len(result["omitted_links"]) == len(AUTHOR_FILES)
    assert all(
        "outside this packet" in p.read_text()
        for p in author.rglob("*.md")
        if p.name != "README.md"
    )
    assert not (author / "tests").exists()


@pytest.mark.parametrize("change", ["missing", "changed", "extra", "extra_data", "duplicate"])
def test_wheel_source_closure_rejects_mismatch(tmp_path, change):
    source = tmp_path / "src/openboost"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("# public core\n")
    wheel = tmp_path / "core.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        if change != "missing":
            archive.writestr(
                "openboost/__init__.py", "# changed\n" if change == "changed" else "# public core\n"
            )
        if change == "extra":
            archive.writestr("tests/answer.py", "# must not be delivered\n")
        if change == "extra_data":
            archive.writestr("tests/answer.json", "{}\n")
        if change == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr("openboost/__init__.py", "# public core\n")
    with pytest.raises(ValueError):
        audit_wheel(tmp_path, wheel)


def test_failed_build_is_retained_as_incomplete(tmp_path, monkeypatch):
    from benchmarks.v1 import prepare_author_packet as packet

    monkeypatch.setattr(
        packet.subprocess,
        "check_output",
        lambda command, **kwargs: b"" if "--porcelain" in command else "test-revision\n",
    )
    monkeypatch.setattr(
        packet.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="missing cached dependency"
        ),
    )
    destination = tmp_path / "failed"
    with pytest.raises(subprocess.CalledProcessError):
        packet.prepare(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["status"] == "incomplete" and not manifest["dispatch_ready"]
    assert manifest["build"]["exit_code"] == 1
    assert "missing cached dependency" in manifest["build"]["stderr"]
    assert manifest["author_files"] and manifest["attempts"] == []


@pytest.mark.parametrize("marker", ["valid", "missing", "changed"])
def test_legitimate_package_marker_is_verified(tmp_path, marker):
    source = tmp_path / "src/openboost"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("# core\n")
    (source / "py.typed").write_bytes(b"")
    wheel = tmp_path / "core.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("openboost/__init__.py", "# core\n")
        if marker != "missing":
            archive.writestr("openboost/py.typed", b"" if marker == "valid" else b"unexpected")
    if marker == "valid":
        assert set(audit_wheel(tmp_path, wheel)) == {"openboost/__init__.py"}
    else:
        with pytest.raises(ValueError, match="package marker"):
            audit_wheel(tmp_path, wheel)
