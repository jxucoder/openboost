"""Bounded worker execution with durable logs and explicit failure propagation."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

_LINUX_EXEC = """
import ctypes, os, resource, sys
limit, drop = int(sys.argv[1]), int(sys.argv[2])
if limit:
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
if drop:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    os.setgroups([])
    os.setresgid(65534, 65534, 65534)
    os.setresuid(65534, 65534, 65534)
os.execvp(sys.argv[3], sys.argv[3:])
"""


def execute(
    command, directory, *, timeout_s, threads=2, address_limit_bytes=None, unprivileged=False
):
    """Run one fresh worker; workers must emit predictions.npz and model.bin.

    Files are required before success is possible. This is execution integrity,
    not a quality judgment. The caller supplies an empty dedicated output directory.
    Optional Linux address-space limits are stricter than resident-memory caps.
    Unprivileged mode requires a root evaluator, drops to UID/GID 65534 with no
    supplementary groups/no_new_privs, and passes only a minimal worker environment.
    Evaluator files must have separate ownership/permissions. This is not a complete
    hostile-code sandbox: network and separate sessions/namespaces are not restricted.
    Use distinct containers for independent attempts; do not share UID-owned outputs.
    """
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(s, str) or not s for s in command)
    ):
        raise ValueError("nonempty argv required")
    if (
        type(timeout_s) not in (int, float)
        or not 0 < timeout_s <= 7200
        or type(threads) is not int
        or threads < 1
    ):
        raise ValueError("invalid resource budget")
    if address_limit_bytes is not None and (
        type(address_limit_bytes) is not int or address_limit_bytes <= 0
    ):
        raise ValueError("positive integer address limit required")
    if type(unprivileged) is not bool:
        raise ValueError("unprivileged mode must be boolean")
    if (address_limit_bytes is not None or unprivileged) and sys.platform != "linux":
        raise ValueError("verified address/identity mode requires Linux")
    if unprivileged and (os.geteuid() != 0 or address_limit_bytes is None):
        raise ValueError("unprivileged mode requires a root evaluator and explicit address limit")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("worker output directory must be empty")
    if unprivileged:
        os.chown(root, 65534, 65534)
        root.chmod(0o700)
    env = (
        dict(PATH="/usr/local/bin:/usr/bin:/bin", LANG="C.UTF-8", HOME=str(root), TMPDIR=str(root))
        if unprivileged
        else os.environ.copy()
    )
    launch = command
    if address_limit_bytes is not None or unprivileged:
        launch = [
            sys.executable,
            "-c",
            _LINUX_EXEC,
            str(address_limit_bytes or 0),
            str(int(unprivileged)),
            *command,
        ]
    env.update(
        {
            k: str(threads)
            for k in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
            ]
        }
    )
    started = time.monotonic()
    record = {
        "command": command,
        "launch_command": launch,
        "address_limit_bytes": address_limit_bytes,
        "worker_identity": "uid_gid_65534_no_new_privs" if unprivileged else "inherited",
        "environment_policy": "minimal_explicit" if unprivileged else "inherited",
        "timeout_s": timeout_s,
        "threads": threads,
        "status": "error",
        "exit_code": None,
        "reason": "",
    }
    with (root / "worker.log").open("wb") as log:
        try:
            process = subprocess.Popen(
                launch,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                record["exit_code"] = process.wait(timeout=timeout_s)
                if unprivileged:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                if record["exit_code"] != 0:
                    record["reason"] = "worker exited nonzero"
                elif not all(
                    (root / f).is_file()
                    and not (root / f).is_symlink()
                    and (root / f).stat().st_size > 0
                    for f in ["predictions.npz", "model.bin"]
                ):
                    record["reason"] = "missing or invalid worker artifact"
                else:
                    record["status"] = "pass"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                record.update(
                    status="timeout",
                    exit_code=process.returncode,
                    reason="wall budget exceeded; process group killed",
                )
        except OSError as exc:
            record["reason"] = str(exc)
    record["wall_s"] = time.monotonic() - started
    record["artifacts"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.iterdir()
        if p.is_file() and not p.is_symlink()
    }
    (root / "execution.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
