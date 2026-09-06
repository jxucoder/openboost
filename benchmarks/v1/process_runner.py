"""Bounded worker execution with durable logs and explicit failure propagation."""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def execute(command, directory, *, timeout_s, threads=2):
    """Run one fresh worker; workers must emit predictions.npz and model.bin.

    Files are required before success is possible. This is execution integrity,
    not a quality judgment. The caller supplies an empty dedicated output directory.
    Memory caps must be enforced by the container/host and recorded separately.
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
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("worker output directory must be empty")
    env = os.environ.copy()
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
        "timeout_s": timeout_s,
        "threads": threads,
        "status": "error",
        "exit_code": None,
        "reason": "",
    }
    with (root / "worker.log").open("wb") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                record["exit_code"] = process.wait(timeout=timeout_s)
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
