"""Generic, solution-free probe run inside the prospective Linux author Sandbox."""

import errno
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

DOCS = ("cpu-state", "numeric-ops", "preparation", "squared", "stopping", "trees")
CHECKS = (
    "nonroot",
    "work_write",
    "private_read",
    "private_write",
    "child_read",
    "symlink_read",
    "core_write",
    "materials_write",
    "regain_root",
    "credentials_absent",
    "loopback_usable",
    "outbound_unavailable",
    "core_unchanged",
)
CASES = tuple(f"doc:{name}" for name in DOCS) + CHECKS


def emit(record):
    print(json.dumps(record, allow_nan=False), flush=True)


def denied(operation, allowed):
    """An unrelated exception or interpreter crash is not an access denial."""
    try:
        operation()
    except OSError as error:
        return dict(passed=error.errno in allowed, errno=error.errno)
    return dict(passed=False, reason="operation unexpectedly succeeded")


def python(code):
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return dict(
        passed=result.returncode == 0,
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def core_files(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def run(materials, private_paths):
    import openboost

    if sys.platform != "linux":
        raise RuntimeError("this probe requires a real Linux Sandbox")
    core = Path(openboost.__file__)
    before = core_files(core.parent)
    emit(
        dict(
            kind="runtime",
            python=sys.version,
            os=platform.platform(),
            machine=platform.machine(),
            uid=os.getuid(),
            gid=os.getgid(),
            cpu_count=os.cpu_count(),
            packages={
                name: importlib.metadata.version(name) for name in ("openboost", "numpy", "uv")
            },
            core_files=before,
        )
    )
    results = {}

    def check(name, operation):
        try:
            result = operation()
        except Exception as error:
            result = dict(passed=False, error=f"{type(error).__name__}: {error}")
        results[name] = result
        emit(dict(kind="case", name=name, **result))

    for name in DOCS:

        def example(name=name):
            text = (materials / "docs/v1" / f"{name}.md").read_text()
            code = "\n\n".join(re.findall(r"```python\n(.*?)```", text, re.S))
            if not code:
                raise ValueError("missing positive public example")
            return python(code)

        check(f"doc:{name}", example)
    check("nonroot", lambda: dict(passed=os.getuid() == os.geteuid() == 1000))

    def work_write():
        path = Path("author-module.py")
        path.write_text("print('author workspace is writable')\n")
        return python("exec(open('author-module.py').read())")

    check("work_write", work_write)
    unavailable = (errno.ENOENT, errno.EACCES, errno.EPERM)
    readonly = (errno.EACCES, errno.EPERM, errno.EROFS)

    def private_access(write=False):
        records = [
            denied(lambda p=p: p.write_bytes(b"tamper") if write else p.read_bytes(), unavailable)
            for p in private_paths
        ]
        return dict(passed=bool(records) and all(r["passed"] for r in records), paths=records)

    check("private_read", private_access)
    check("private_write", lambda: private_access(write=True))
    # Pass path names only. No evaluator bytes enter this image or any command.
    child_code = (
        "import errno\nfrom pathlib import Path\n"
        f"for name in {list(map(str, private_paths))!r}:\n"
        " try: Path(name).read_bytes()\n"
        " except OSError as error: assert error.errno in (errno.ENOENT, errno.EACCES, errno.EPERM)\n"
        " else: raise AssertionError('private read succeeded')\n"
    )
    check("child_read", lambda: python(child_code))

    def symlink_read():
        link = Path("private-link")
        link.symlink_to(private_paths[0])
        return denied(link.read_bytes, unavailable)

    check("symlink_read", symlink_read)
    check("core_write", lambda: denied(lambda: core.write_bytes(b"tamper"), readonly))
    check(
        "materials_write",
        lambda: denied(lambda: (materials / "README.md").write_bytes(b"tamper"), readonly),
    )
    check("regain_root", lambda: denied(lambda: os.setresuid(0, 0, 0), (errno.EPERM,)))
    keys = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
    check("credentials_absent", lambda: dict(passed=all(not os.environ.get(k) for k in keys)))

    def loopback():
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            with socket.create_connection(server.getsockname(), timeout=2):
                client, _ = server.accept()
                client.close()
        return dict(passed=True)

    check("loopback_usable", loopback)

    def outbound():
        try:
            with socket.create_connection(("1.1.1.1", 443), timeout=2):
                return dict(passed=False, reason="outbound TCP connection succeeded")
        except OSError as error:
            # One failed external connection is only a scoped observation. The
            # controller also requires the provider's block_network configuration.
            allowed = (
                errno.ETIMEDOUT,
                errno.EACCES,
                errno.EPERM,
                errno.ENETUNREACH,
                errno.EHOSTUNREACH,
                errno.ECONNREFUSED,
            )
            return dict(
                passed=isinstance(error, TimeoutError) or error.errno in allowed,
                error=type(error).__name__,
                errno=error.errno,
            )

    check("outbound_unavailable", outbound)
    after = core_files(core.parent)
    check("core_unchanged", lambda: dict(passed=before == after, core_files=after))
    if set(results) != set(CASES) or not all(r["passed"] for r in results.values()):
        raise RuntimeError("isolation smoke failed; do not count a later timeout as a pass")
    # A different session defeats a process-group-only supervisor. The provider
    # must end this entire Sandbox at its frozen lifetime, not just this process.
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(600)"], start_new_session=True
    )
    emit(dict(kind="ready_for_timeout", child_pid=child.pid, child_session=os.getsid(child.pid)))
    while True:
        time.sleep(1)


if __name__ == "__main__":
    run(Path(sys.argv[1]), [Path(p) for p in json.loads(sys.argv[2])])
