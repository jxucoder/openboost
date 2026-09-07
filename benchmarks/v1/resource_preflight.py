"""Bounded CPU resource probes before Sprint 066 fits; no datasets are uploaded."""

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def bounded(command, *, seconds, address_bytes, directory):
    """Linux child address-space ceiling plus a parent-enforced process-group deadline.

    RLIMIT_AS is stricter than an RSS cap. The enclosing container must separately
    enforce/record its actual memory ceiling; virtual and resident memory differ.
    Logs and outcomes survive expected child allocation failure and timeout.
    """
    if sys.platform != "linux":
        raise ValueError("verified address-space enforcement requires Linux")
    if (
        type(address_bytes) is not int
        or address_bytes <= 0
        or isinstance(seconds, bool)
        or not 0 < seconds <= 120
    ):
        raise ValueError("positive address-space limit and at most 120 seconds required")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    wrapper = (
        "import os,resource,sys; n=int(sys.argv[1]); "
        "resource.setrlimit(resource.RLIMIT_AS,(n,n)); os.execv(sys.argv[2],sys.argv[2:])"
    )
    argv = [sys.executable, "-c", wrapper, str(address_bytes), *command]
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    start = time.monotonic()
    with (root / "worker.log").open("wb") as log:
        process = subprocess.Popen(
            argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=seconds)
            status = "pass" if code == 0 else "error"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            status = "timeout"
    record = dict(
        command=argv,
        timeout_s=seconds,
        address_limit_bytes=address_bytes,
        threads=2,
        status=status,
        exit_code=code,
        wall_s=time.monotonic() - start,
        log=(root / "worker.log").read_text(),
    )
    (root / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def probe(directory):
    """Exercise limits, rather than infer enforcement from requested resources."""
    root = Path(directory)
    memory_path = Path("/sys/fs/cgroup/memory.max")
    cpu_path = Path("/sys/fs/cgroup/cpu.max")
    limits = {
        str(p): p.read_text().strip() if p.exists() else None for p in (memory_path, cpu_path)
    }
    memory = (
        "import resource,sys; print('address_limit',resource.getrlimit(resource.RLIMIT_AS),flush=True);"
        "\ntry: data=bytearray(256*1024**2)"
        "\nexcept MemoryError: print('expected allocation rejected',flush=True); sys.exit(73)"
        "\nprint('ERROR allocation escaped limit',flush=True)"
    )
    allocation = bounded(
        [sys.executable, "-c", memory],
        seconds=10,
        address_bytes=128 * 1024**2,
        directory=root / "allocation",
    )
    timeout = bounded(
        [sys.executable, "-c", "import time; print('started',flush=True); time.sleep(10)"],
        seconds=1,
        address_bytes=128 * 1024**2,
        directory=root / "timeout",
    )
    success = bounded(
        [sys.executable, "-c", "print('success')"],
        seconds=10,
        address_bytes=128 * 1024**2,
        directory=root / "success",
    )
    memory_limit = limits[str(memory_path)]
    cpu_limit = limits[str(cpu_path)]
    container_memory_pass = memory_limit == str(8192 * 1024**2)
    cpu_parts = cpu_limit.split() if cpu_limit else []
    container_cpu_pass = (
        len(cpu_parts) == 2 and cpu_parts[0] != "max" and int(cpu_parts[0]) == 2 * int(cpu_parts[1])
    )
    checks = dict(
        allocation_rejected=allocation["exit_code"] == 73
        and "expected allocation rejected" in allocation["log"],
        timeout_killed=timeout["status"] == "timeout" and timeout["exit_code"] == -signal.SIGKILL,
        partial_log_retained="started" in timeout["log"],
        valid_child_succeeds=success["status"] == "pass",
        container_8gib_enforced=container_memory_pass,
        container_two_cpus_enforced=container_cpu_pass,
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        cgroup_limits=limits,
        environment=dict(
            os=platform.platform(),
            python=platform.python_version(),
            cpu=platform.processor(),
            cpu_count=os.cpu_count(),
            gpu=None,
        ),
        cases=dict(allocation=allocation, timeout=timeout, success=success),
        scope="resource preflight only; RLIMIT_AS probe is not an RSS measurement",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--modal", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output path required")
    source = Path(__file__).resolve()
    manifest = dict(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        argv=sys.argv,
        requested=dict(cpu=[2, 2], memory_mib=[8192, 8192], timeout_s=60, retries=0),
    )
    if args.modal:
        import modal

        app = modal.App("openboost-v1-resource-preflight")
        image = modal.Image.debian_slim(python_version="3.12").add_local_file(
            source, "/opt/resource_preflight.py", copy=True
        )

        @app.function(
            image=image,
            cpu=(2, 2),
            memory=(8192, 8192),
            timeout=60,
            retries=0,
            max_containers=1,
            serialized=True,
            include_source=False,
        )
        def remote_probe():
            import runpy
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                return runpy.run_path("/opt/resource_preflight.py")["probe"](directory)

        with modal.enable_output(), app.run():
            result = remote_probe.remote()
            manifest["image_id"] = image.object_id
        manifest["modal_version"] = modal.__version__
    else:
        with tempfile.TemporaryDirectory() as directory:
            result = probe(directory)
    args.output.write_text(json.dumps(dict(manifest=manifest, result=result), indent=2) + "\n")
    print(json.dumps(result["checks"], indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
