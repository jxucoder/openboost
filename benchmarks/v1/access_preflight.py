"""Real Linux permission/resource probes using synthetic evaluator-only fixtures."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


def probe(directory):
    import runpy

    if sys.platform != "linux" or os.geteuid() != 0:
        raise ValueError("probe requires a Linux root evaluator inside the isolated container")
    source = Path(__file__).resolve().parent
    execute = runpy.run_path(str(source / "process_runner.py"))["execute"]
    root = Path(directory)
    root.chmod(0o755)
    private = root / "evaluator"
    private.mkdir(mode=0o700)
    labels = private / "test-labels.json"
    verifier = private / "verifier.py"
    labels.write_text("[101, 202, 303]\n")
    verifier.write_text("# Synthetic verifier sentinel, not an author-task verifier.\n")
    protected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (labels, verifier)}
    os.environ["OPENBOOST_EVALUATOR_CANARY"] = "must-not-reach-worker"
    measurement = """
import ctypes,json,os,resource,runpy
from pathlib import Path
import numpy as np
measurement=runpy.run_path('/opt/resource_preflight.py')['worker_measurement']()
assert os.getresuid()==(65534,65534,65534) and os.getresgid()==(65534,65534,65534)
assert not os.getgroups()
assert ctypes.CDLL(None).prctl(39,0,0,0,0)==1
assert 'OPENBOOST_EVALUATOR_CANARY' not in os.environ
try:
    os.setuid(0)
except PermissionError:
    pass
else:
    raise AssertionError('worker regained root')
try:
    resource.setrlimit(resource.RLIMIT_AS,(9*1024**3,9*1024**3))
except (ValueError,PermissionError):
    pass
else:
    raise AssertionError('worker raised hard address limit')
np.savez('predictions.npz', prediction=np.array([1.,2.]))
Path('model.bin').write_bytes(b'permission/resource probe, not a trained model')
Path('measurement.json').write_text(json.dumps(dict(worker=measurement,uid=os.getresuid(),gid=os.getresgid(),groups=os.getgroups(),no_new_privs=True,root_restore_rejected=True,hard_limit_raise_rejected=True,evaluator_env_absent=True)))
"""
    commands = {
        "numerical": (measurement, 1800),
        "read_labels": ("from pathlib import Path; import sys; Path(sys.argv[1]).read_bytes()", 5),
        "tamper_verifier": (
            'from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b"tampered")',
            5,
        ),
        "parent_environment": (
            "from pathlib import Path; import sys; Path(sys.argv[1]).read_bytes()",
            5,
        ),
        "memory": ('import mmap; print("allocating",flush=True); mmap.mmap(-1,9*1024**3)', 5),
        "error": ('raise RuntimeError("intentional worker failure")', 5),
        "timeout": ('import time; print("started",flush=True); time.sleep(10)', 0.25),
    }
    extra = dict(
        read_labels=str(labels),
        tamper_verifier=str(verifier),
        parent_environment=f"/proc/{os.getpid()}/environ",
    )
    cases = {}
    for name, (code, seconds) in commands.items():
        command = [sys.executable, "-c", code]
        if name in extra:
            command.append(extra[name])
        cases[name] = execute(
            command,
            root / name,
            timeout_s=seconds,
            threads=2,
            address_limit_bytes=8 * 1024**3,
            unprivileged=True,
        )
    measured_path = root / "numerical/measurement.json"
    measured = json.loads(measured_path.read_text()) if measured_path.exists() else None
    checks = dict(
        numerical_pass=cases["numerical"]["status"] == "pass",
        labels_read_denied=cases["read_labels"]["exit_code"] != 0
        and "PermissionError" in (root / "read_labels/worker.log").read_text()
        and str(labels) in (root / "read_labels/worker.log").read_text(),
        verifier_write_denied=cases["tamper_verifier"]["exit_code"] != 0
        and "PermissionError" in (root / "tamper_verifier/worker.log").read_text()
        and str(verifier) in (root / "tamper_verifier/worker.log").read_text(),
        parent_environment_denied=cases["parent_environment"]["exit_code"] != 0
        and "PermissionError" in (root / "parent_environment/worker.log").read_text()
        and extra["parent_environment"] in (root / "parent_environment/worker.log").read_text(),
        evaluator_files_unchanged=all(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in protected.items()
        ),
        oversized_mapping_rejected=cases["memory"]["status"] == "error"
        and "Cannot allocate memory" in (root / "memory/worker.log").read_text(),
        worker_error_rejected=cases["error"]["status"] == "error"
        and cases["error"]["exit_code"] != 0,
        timeout_killed=cases["timeout"]["status"] == "timeout"
        and cases["timeout"]["exit_code"] == -9,
        timeout_partial_log_retained="started" in (root / "timeout/worker.log").read_text(),
    )
    if measured is not None:
        checks.update({"numerical_" + k: v for k, v in measured["worker"]["checks"].items()})
    artifacts = {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and not p.is_relative_to(private)
    }
    return dict(
        passed=all(checks.values()),
        checks=checks,
        cases=cases,
        measurement=measured,
        protected_hashes=protected,
        environment=dict(python=platform.python_version(), os=platform.platform()),
        scope="UID-based file permissions and bounded numerical subprocesses; synthetic protected fixtures, not full search or complete hostile-code isolation",
        artifacts=artifacts,
    )


def main(output):
    import modal

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    paths = [
        source,
        source.with_name("process_runner.py"),
        source.with_name("resource_preflight.py"),
    ]
    manifest = dict(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        argv=sys.argv,
        requested=dict(cpu=[2, 2], memory_mib=[8192, 8192], function_seconds=120, retries=0),
        status="running",
        modal_version=modal.__version__,
    )

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    app = modal.App("openboost-v1-access-preflight")
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "numpy==2.3.5", "threadpoolctl==3.6.0", uv_version="0.12.1"
    )
    for path in paths:
        image = image.add_local_file(path, "/opt/" + path.name, copy=True)

    @app.function(
        image=image,
        cpu=(2, 2),
        memory=(8192, 8192),
        timeout=120,
        retries=0,
        max_containers=1,
        serialized=True,
        include_source=False,
    )
    def remote():
        import runpy

        with tempfile.TemporaryDirectory() as directory:
            return runpy.run_path("/opt/access_preflight.py")["probe"](directory)

    try:
        with modal.enable_output(), app.run():
            result = remote.remote()
            manifest["image_id"] = image.object_id
        for name, data in result.pop("artifacts").items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        manifest.update(
            status="complete",
            result=result,
            artifacts={
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )
    except Exception as error:
        manifest.update(status="error", error=str(error))
        raise
    finally:
        save()
    print(json.dumps(result["checks"], indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
