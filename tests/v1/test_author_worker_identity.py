"""The root entrypoint must enforce identity before any worker code executes."""

import copy
import errno
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.v1.authoring import linux_launcher as launcher
from benchmarks.v1.authoring import modal_worker as worker

from .test_author_linux_worker import result_stream

ROOT = Path(__file__).resolve().parents[2]


def test_worker_command_cannot_bypass_privilege_launcher():
    command = worker.worker_command(["/private/expected.json"])
    assert command[:5] == ["python", "-I", "-B", "/opt/launcher.py", "drop"]
    assert command[5:9] == ["python", "-I", "-B", "/opt/probe.py"]
    assert command[9:] == ["/materials", '["/private/expected.json"]']


class ExecReached(Exception):
    pass


@pytest.fixture
def system(monkeypatch):
    """Model syscall ordering only; this is not Linux privilege-drop evidence."""
    calls = []
    state = dict(uids=[0] * 3, gids=[0] * 3, groups=[0], no_new_privs=0)

    def set_field(name, value):
        calls.append(name)
        state[name] = list(value)

    def prctl(option, value=0):
        calls.append(f"prctl:{option}")
        if option == 38:
            state["no_new_privs"] = value
            return 0
        assert option == 39
        return state["no_new_privs"]

    def execute(path, argv):
        calls.append((path, argv))
        raise ExecReached

    fake = SimpleNamespace(
        setgroups=lambda groups: set_field("groups", groups),
        setresgid=lambda *gids: set_field("gids", gids),
        setresuid=lambda *uids: set_field("uids", uids),
        getresuid=lambda: tuple(state["uids"]),
        getresgid=lambda: tuple(state["gids"]),
        getgroups=lambda: state["groups"],
        execv=execute,
        execvp=execute,
        path=os.path,
    )
    monkeypatch.setattr(launcher, "os", fake)
    monkeypatch.setattr(
        launcher, "sys", SimpleNamespace(platform="linux", executable=sys.executable)
    )
    monkeypatch.setattr(launcher, "prctl", prctl)
    return calls, state, fake


def test_drop_checks_saved_identity_before_fresh_exec(system, capsys):
    calls, state, _ = system
    command = ["python", "-I", "-c", "print('worker')"]
    with pytest.raises(ExecReached):
        launcher.run("drop", command)
    assert calls[:5] == ["prctl:38", "groups", "gids", "uids", "prctl:39"]
    assert calls[5] == (
        sys.executable,
        [sys.executable, "-I", "-B", str(Path(launcher.__file__).resolve()), "verify", *command],
    )
    assert json.loads(capsys.readouterr().out) == dict(
        kind="identity", phase="before_exec", **state
    )


@pytest.mark.parametrize("operation", ["prctl", "setgroups", "setresgid", "setresuid"])
def test_failed_drop_syscall_never_executes_worker(system, monkeypatch, operation, capsys):
    calls, _, fake = system

    def fail(*args):
        raise OSError(errno.EPERM, "deliberate syscall failure")

    monkeypatch.setattr(launcher if operation == "prctl" else fake, operation, fail)
    with pytest.raises(OSError, match="deliberate"):
        launcher.run("drop", ["worker"])
    assert not any(isinstance(call, tuple) for call in calls)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "field,value",
    [("uids", [1000, 1000, 0]), ("gids", [1000, 1000, 0]), ("groups", [0]), ("no_new_privs", 0)],
)
@pytest.mark.parametrize("phase", ["drop", "verify"])
def test_wrong_saved_identity_groups_or_privilege_flag_stops_exec(
    system, monkeypatch, field, value, phase, capsys
):
    calls, _, _ = system
    observed = dict(uids=[1000] * 3, gids=[1000] * 3, groups=[], no_new_privs=1)
    observed[field] = value
    monkeypatch.setattr(launcher, "identity", lambda: observed)
    with pytest.raises(RuntimeError, match="identity check failed"):
        launcher.run(phase, ["worker"])
    assert not any(isinstance(call, tuple) for call in calls)
    assert capsys.readouterr().out == ""


def test_post_exec_guard_rechecks_identity_before_worker(system, capsys):
    calls, state, _ = system
    state.update(uids=[1000] * 3, gids=[1000] * 3, groups=[], no_new_privs=1)
    with pytest.raises(ExecReached):
        launcher.run("verify", ["worker", "argument with spaces"])
    assert calls == ["prctl:39", ("worker", ["worker", "argument with spaces"])]
    assert json.loads(capsys.readouterr().out) == dict(kind="identity", phase="after_exec", **state)


@pytest.mark.skipif(sys.platform == "linux", reason="requires an unsupported host")
def test_actual_unsupported_host_stops_before_command(tmp_path):
    sentinel = tmp_path / "worker-ran"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(ROOT / worker.LAUNCHER),
            "drop",
            sys.executable,
            "-c",
            f"open({str(sentinel)!r}, 'w').close()",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0 and "requires Linux" in result.stderr
    assert not sentinel.exists() and not result.stdout


def guarded_stream():
    packet, records = result_stream()
    records[0].update(uid=1000, gid=1000)
    guards = [
        dict(
            kind="identity",
            phase=phase,
            uids=[1000] * 3,
            gids=[1000] * 3,
            groups=[],
            no_new_privs=1,
        )
        for phase in ("before_exec", "after_exec")
    ]
    return packet, guards + records


def test_corrected_verdict_keeps_original_cases_and_actual_timeout():
    packet, records = guarded_stream()
    stdout = "\n".join(map(json.dumps, records))
    assert worker.classify_identity(stdout, True, packet) == dict(passed=True, records=records)
    assert not worker.classify_identity(stdout, False, packet)["passed"]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "reordered",
        "saved_root",
        "groups",
        "privileges",
        "probe_root",
        "original_failure",
        "no_marker",
    ],
)
def test_identity_records_cannot_hide_a_bad_or_incomplete_smoke(change):
    packet, records = guarded_stream()
    records = copy.deepcopy(records)
    if change == "missing":
        records.pop(1)
    elif change == "duplicate":
        records.insert(1, records[0])
    elif change == "reordered":
        records[0], records[1] = records[1], records[0]
    elif change == "saved_root":
        records[1]["uids"][-1] = 0
    elif change == "groups":
        records[1]["groups"] = [0]
    elif change == "privileges":
        records[1]["no_new_privs"] = 0
    elif change == "probe_root":
        records[2]["uid"] = 0
    elif change == "original_failure":
        records[3]["passed"] = False
    elif change == "no_marker":
        records.pop()
    assert not worker.classify_identity("\n".join(map(json.dumps, records)), True, packet)["passed"]


def test_original_failed_run_and_probe_remain_unchanged():
    archive = ROOT / "benchmarks/v1/evidence/author-linux-isolation-096"
    assert (ROOT / worker.PROBE).read_bytes() == (archive / "upload/opt/probe.py").read_bytes()
    packet = json.loads((ROOT / worker.PACKET / "manifest.json").read_text())
    stdout = (archive / "stdout.jsonl").read_text()
    for timed_out in (False, True):
        assert not worker.classify(stdout, timed_out, packet)["passed"]
        assert not worker.classify_identity(stdout, timed_out, packet)["passed"]
