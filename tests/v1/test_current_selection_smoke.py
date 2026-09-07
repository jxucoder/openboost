"""Selection integration must reject unsupported permission modes before writing."""

import pytest
from benchmarks.v1 import current_selection_smoke as smoke


def test_protected_selection_rejects_unsupported_host_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke.sys, "platform", "darwin")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Linux root evaluator"):
        smoke.run(output, protected=True)
    assert not output.exists()


def test_protected_selection_rejects_nonroot_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke.sys, "platform", "linux")
    monkeypatch.setattr(smoke.os, "geteuid", lambda: 1000)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Linux root evaluator"):
        smoke.run(output, protected=True)
    assert not output.exists()


@pytest.mark.skipif(
    smoke.sys.platform != "linux" or smoke.os.geteuid() != 0,
    reason="requires a real Linux root evaluator and traversable installed runtime",
)
def test_protected_selection_runs_real_workers_and_seals_outputs(linux_workspace, monkeypatch):
    import json

    actual_execute = smoke.execute
    checked = False

    def inspect(command, directory, **options):
        nonlocal checked
        root = directory.parent
        if not checked:
            for name, target, operation in (
                ("read", root / "evaluator/test-features.npz", "read_bytes()"),
                ("write", root / "evaluator/protocol.json", 'write_bytes(b"bad")'),
            ):
                denied = actual_execute(
                    [
                        smoke.sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(target)!r}).{operation}",
                    ],
                    root / f"denied-{name}",
                    **options,
                )
                assert denied["status"] == "error"
                assert "PermissionError" in (root / f"denied-{name}/worker.log").read_text()
            checked = True
        return actual_execute(command, directory, **options)

    monkeypatch.setattr(smoke, "execute", inspect)
    root = linux_workspace / "protected"
    report = smoke.run(root, protected=True)
    assert report["passed"] and checked
    assert len(report["trials"]) == 16
    for trial in report["trials"]:
        assert trial["status"] == "pass"
        assert trial["worker_identity"] == "uid_gid_65534_no_new_privs"
        assert trial["address_limit_bytes"] == 8 * 1024**3
        assert trial["timeout_s"] == 1800
    for directory in root.glob("trial-*"):
        assert directory.stat().st_uid == 0
        assert directory.stat().st_mode & 0o777 == 0o700
    assert (root / "evaluator").stat().st_mode & 0o777 == 0o700
    assert (root / "candidate/worker-input.npz").stat().st_mode & 0o777 == 0o444
    assert json.loads((root / "evaluator/summary.json").read_text())["passed"]


@pytest.fixture
def linux_workspace():
    import tempfile

    with tempfile.TemporaryDirectory(prefix="openboost-selection-", dir="/tmp") as directory:
        root = smoke.Path(directory)
        root.chmod(0o755)
        yield root
