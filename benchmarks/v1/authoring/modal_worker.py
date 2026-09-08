"""Check or execute the frozen CPU isolation smoke; never dispatch a model."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path, PurePosixPath

from benchmarks.v1.authoring.linux_probe import CASES
from benchmarks.v1.prepare_author_packet import AUTHOR_FILES

ROOT = Path(__file__).resolve().parents[3]
PACKET = "benchmarks/v1/evidence/author-preparation-095/packet"
EVALUATOR = "benchmarks/v1/evidence/author-preparation-095/verifiers/evaluator"
PROBE = "benchmarks/v1/authoring/linux_probe.py"
LAUNCHER = "benchmarks/v1/authoring/linux_launcher.py"
FREEZE = ROOT / "v1-sprints/097-worker-identity-smoke.json"
SDK = "1.3.0.post1"
RESOURCES = dict(
    sandboxes=1,
    cpu=2,
    memory_mib=2048,
    timeout_s=90,
    gpu=None,
    application_retries=0,
    max_output_bytes=2**20,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def regular(root, name):
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or path.as_posix() != name
        or any(p in (".", "..") for p in name.split("/"))
    ):
        raise ValueError("snapshot paths must be canonical relative paths")
    target = root
    for part in path.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError(f"snapshot symlink: {name}")
    if not target.is_file():
        raise ValueError(f"missing regular snapshot file: {name}")
    return target


def inputs(root, freeze):
    """Validate explicit local sources without initializing a Modal client."""
    if (
        freeze["schema"] != "openboost-linux-worker-smoke-v2"
        or freeze["resources"] != RESOURCES
        or freeze["modal_version"] != SDK
    ):
        raise ValueError("worker environment or resource freeze differs")
    for name, expected in freeze["files"].items():
        if digest(regular(root, name).read_bytes()) != expected:
            raise ValueError(f"frozen input changed: {name}")
    manifest_name = f"{PACKET}/manifest.json"
    if manifest_name not in freeze["files"]:
        raise ValueError("packet manifest is not frozen")
    packet = json.loads(regular(root, manifest_name).read_text())
    files = packet["author_files"]
    (wheel,) = [name for name in files if name.startswith("wheels/") and name.endswith(".whl")]
    expected = set(AUTHOR_FILES) | {"README.md", "wheels/.gitignore", wheel}
    if packet["status"] != "prepared" or set(files) != expected:
        raise ValueError("unexpected author file closure")
    author = root / PACKET / "author"
    actual = {
        p.relative_to(author).as_posix() for p in author.rglob("*") if p.is_file() or p.is_symlink()
    }
    if actual != expected:
        raise ValueError("unlisted or missing author file")
    uploads = {f"{PACKET}/author/{name}": f"/materials/{name}" for name in files}
    uploads[PROBE] = "/opt/probe.py"
    uploads[LAUNCHER] = "/opt/launcher.py"
    if freeze["uploads"] != uploads:
        raise ValueError("upload closure differs from author materials, probe and launcher")
    for name, expected_hash in files.items():
        source = f"{PACKET}/author/{name}"
        if freeze["files"].get(source) != expected_hash:
            raise ValueError("author delivery digest differs from packet")
    if not {PROBE, LAUNCHER}.issubset(freeze["files"]):
        raise ValueError("probe or launcher is not frozen")
    if freeze["cases"] != list(CASES):
        raise ValueError("case set changed")
    return packet, uploads


def stage(root, uploads, hashes, output):
    """Copy bytes individually; never upload a repository or packet directory."""
    output.mkdir(parents=True, exist_ok=False)
    delivery = {}
    for name, remote in uploads.items():
        data = regular(root, name).read_bytes()
        if digest(data) != hashes[name]:
            raise ValueError("input changed while staging")
        target = output / remote.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        delivery[remote] = target
    return delivery


def image_for(modal, delivery):
    image = modal.Image.debian_slim(python_version="3.12")
    for remote, local in sorted(delivery.items()):
        image = image.add_local_file(local, remote, copy=True)
    (wheel,) = [name for name in delivery if name.endswith(".whl")]
    return (
        image.uv_pip_install(
            "numpy==2.3.5", "uv==0.12.1", wheel, uv_version="0.12.1", extra_options="--no-deps"
        )
        .dockerfile_commands("RUN mkdir -p /work && chown 1000:1000 /work")
        .env(
            {
                "HOME": "/work",
                "OPENBOOST_BACKEND": "cpu",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        .workdir("/work")
    )


def worker_command(private_paths):
    """The trusted root entrypoint always precedes the original probe."""
    return [
        "python",
        "-I",
        "-B",
        "/opt/launcher.py",
        "drop",
        "python",
        "-I",
        "-B",
        "/opt/probe.py",
        "/materials",
        json.dumps(private_paths),
    ]


def classify(stdout, timed_out, packet):
    """An actual provider timeout alone does not establish usable isolation."""
    records = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    runtime = [r for r in records if r.get("kind") == "runtime"]
    markers = [r for r in records if r.get("kind") == "ready_for_timeout"]
    cases = [r for r in records if r.get("kind") == "case"]
    expected_core = {
        name.removeprefix("openboost/"): sha for name, sha in packet["wheel_sources"].items()
    }
    expected_core["py.typed"] = digest(b"")
    passed = (
        timed_out is True
        and len(runtime) == 1
        and len(markers) == 1
        and len(cases) == len(CASES)
        and len(records) == len(CASES) + 2
        and [r["name"] for r in cases] == list(CASES)
        and all(r.get("passed") is True for r in cases)
        and records[-1].get("kind") == "ready_for_timeout"
        and type(markers[0].get("child_pid")) is int
        and markers[0]["child_pid"] > 0
        and markers[0].get("child_session") == markers[0]["child_pid"]
        and runtime[0].get("core_files") == expected_core
        and runtime[0].get("packages")
        == {"openboost": "1.0.0.dev0", "numpy": "2.3.5", "uv": "0.12.1"}
    )
    return dict(passed=passed, records=records)


def classify_identity(stdout, timed_out, packet):
    """Keep the original 096 verdict and additionally require both identity guards."""
    records = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    expected = [
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
    guarded = records[:2] == expected
    original = classify("\n".join(map(json.dumps, records[2:])), timed_out, packet)
    runtime = [r for r in records[2:] if r.get("kind") == "runtime"]
    probe_identity = len(runtime) == 1 and runtime[0].get("uid") == runtime[0].get("gid") == 1000
    return dict(passed=guarded and probe_identity and original["passed"], records=records)


def retain(output, report, root, protected):
    """Retain a failed gate even if a protected file vanished during the run."""
    after = {}
    for name in protected:
        try:
            after[name] = digest(regular(root, name).read_bytes())
        except (OSError, ValueError) as error:
            after[name] = None
            report.setdefault("integrity_errors", {})[name] = f"{type(error).__name__}: {error}"
    report["protected_after"] = after
    if after != protected:
        report["passed"] = False
    report["artifacts"] = {
        p.relative_to(output).as_posix(): digest(p.read_bytes())
        for p in sorted(output.rglob("*"))
        if p.is_file() and p != output / "manifest.json"
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


def execute(root, freeze, freeze_bytes, output):
    if freeze["authorization"] != "approved":
        raise ValueError("this CPU upload/run request is pending user authorization")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root):
        raise ValueError("commit the frozen inputs before execution")
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("use a fresh output outside the repository")
    packet, uploads = inputs(root, freeze)
    if importlib.metadata.version("modal") != SDK:
        raise ValueError("installed Modal SDK differs from freeze")
    import modal

    output.mkdir(parents=True, exist_ok=False)
    (output / "freeze.json").write_bytes(freeze_bytes)
    report = dict(
        schema="openboost-linux-author-isolation-v2",
        passed=False,
        dispatch_ready=False,
        attempts=[],
        generated_tokens=None,
        freeze_sha256=digest(freeze_bytes),
        modal_version=SDK,
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        dirty=False,
        status="incomplete",
        resources=freeze["resources"],
    )
    sandbox = None
    # These are real evaluator files on the controller. Neither bytes nor a
    # containing directory are passed to Modal; only path names enter the probe.
    protected = {
        name: sha for name, sha in freeze["files"].items() if name.startswith(EVALUATOR + "/")
    }
    report["protected_before"] = protected
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        if set(protected) != {
            f"{EVALUATOR}/{name}"
            for name in ("inputs.json", "expected.json", "judge.py", "manifest.json")
        }:
            raise ValueError("actual evaluator closure is not frozen")
        delivery = stage(root, uploads, freeze["files"], output / "upload")
        image = image_for(modal, delivery)
        app = modal.App.lookup("openboost-v1-author-isolation-097", create_if_missing=True)
        private_paths = [str(root / name) for name in protected]
        private_paths += [
            str(
                root
                / "benchmarks/v1/evidence/author-preparation-095/isolation/development-narrow/unlisted-counterexample.json"
            )
        ]
        command = worker_command(private_paths)
        report["command"] = command
        started = time.monotonic()
        with (
            (output / "build.log").open("w") as build_log,
            contextlib.redirect_stdout(build_log),
            contextlib.redirect_stderr(build_log),
            modal.enable_output(),
        ):
            sandbox = modal.Sandbox.create(
                *command,
                app=app,
                image=image,
                cpu=(2, 2),
                memory=(2048, 2048),
                timeout=90,
                block_network=True,
                secrets=[],
                volumes={},
            )
        report.update(
            sandbox_id=sandbox.object_id,
            image_id=image.object_id,
            create_wall_s=time.monotonic() - started,
        )
        timed_out = False
        started = time.monotonic()
        try:
            sandbox.wait()
        except modal.exception.SandboxTimeoutError:
            timed_out = True
        report.update(
            wait_wall_s=time.monotonic() - started,
            provider_timeout=timed_out,
            returncode=sandbox.poll(),
        )
        stdout, stderr = sandbox.stdout.read(), sandbox.stderr.read()
        (output / "stdout.jsonl").write_text(stdout)
        (output / "stderr.txt").write_text(stderr)
        if len(stdout.encode()) + len(stderr.encode()) > RESOURCES["max_output_bytes"]:
            raise ValueError("smoke output exceeds the frozen 1-MiB bound")
        report.update(classify_identity(stdout, timed_out, packet))
        report["status"] = "complete"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if sandbox is not None:
            try:
                sandbox.terminate()
            except Exception as error:
                report["cleanup_error"] = f"{type(error).__name__}: {error}"
                report["passed"] = False
        retain(output, report, root, protected)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--execute", type=Path, metavar="FRESH_OUTPUT")
    args = parser.parse_args()
    data = args.freeze.read_bytes()
    freeze = json.loads(data)
    if args.execute:
        result = execute(ROOT, freeze, data, args.execute)
        if not result["passed"]:
            raise SystemExit("Linux isolation gate failed; see retained manifest")
    else:
        _, uploads = inputs(ROOT, freeze)
        print(
            json.dumps(
                dict(
                    status="local_inputs_verified",
                    remote_run=False,
                    dispatch_ready=False,
                    uploads=len(uploads),
                    cases=len(CASES),
                )
            )
        )
