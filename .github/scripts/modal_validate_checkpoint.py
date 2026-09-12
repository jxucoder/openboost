"""One explicitly frozen public-commit Modal validation phase; never dispatch on import."""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

SCHEMA = "openboost-pr27-modal-validation-v1"
REPOSITORY = "https://github.com/jxucoder/openboost.git"
EVIDENCE = "docs/v1/evidence/pr27-modal-validation/"
BUILD_PACKAGES = ["hatchling==1.27.0", "pathspec==1.1.1", "trove-classifiers==2026.6.1.19"]
RETURN_LIMIT = 64 * 1024**2


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def inventory(raw):
    result = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        info, name = item.split(b"\t", 1)
        mode, kind, oid = info.decode().split()
        name = name.decode("utf-8")
        if name.startswith(EVIDENCE):
            continue
        if mode not in ("100644", "100755") or kind != "blob" or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsupported tracked source entry")
        result[name] = dict(mode=mode, type=kind, oid=oid)
    return result


def validate_protocol(p):
    if p["schema"] != SCHEMA or p["repository"] != REPOSITORY:
        raise ValueError("public PR validation protocol required")
    if not re.fullmatch(r"[0-9a-f]{40}", p["source_commit"]):
        raise ValueError("full immutable public source commit required")
    if not re.fullmatch(r"[0-9a-f]{64}", p["inventory_sha256"]):
        raise ValueError("exact public source inventory digest required")
    if p["phase"] not in ("cpu310", "cpu312", "gpu"):
        raise ValueError("unknown single-use phase")
    expected_python = "3.10" if p["phase"] == "cpu310" else "3.12"
    if p["python"] != expected_python or p["build_packages"] != BUILD_PACKAGES or p["uv"] != "0.12.1":
        raise ValueError("frozen interpreter/build dependencies differ")
    if p["resources"] != dict(cpu=2, memory_mib=8192, timeout_seconds=2100 if p["phase"] == "gpu" else 1200,
                             work_seconds=1800 if p["phase"] == "gpu" else 1080, max_containers=1, retries=0):
        raise ValueError("bounded single-worker phase resources required")
    if p["maximum_return_bytes"] != RETURN_LIMIT or p["maximum_log_bytes"] != 16 * 1024**2:
        raise ValueError("fixed return/log bounds required")
    if p["policy"]["path"] != ".github/modal-validation-policy.json" or not re.fullmatch(r"[0-9a-f]{64}", p["policy"]["sha256"]):
        raise ValueError("candidate-tracked execution policy required")
    if p["expected_core_modules"] != 63 or p["allowed_skips"] != []:
        raise ValueError("exact current installed module count and no waived skips required")
    if type(p["gpu_tests"]) is not list or not p["gpu_tests"] or len(set(p["gpu_tests"])) != len(p["gpu_tests"]):
        raise ValueError("explicit complete selected GPU module list required")
    if any(not re.fullmatch(r"tests/v1/test_[A-Za-z0-9_]+\.py(?:::[A-Za-z0-9_]+)?", name) for name in p["gpu_tests"]):
        raise ValueError("bounded public GPU test paths required")
    if type(p["expected_gpu_cases"]) is not int or not 0 < p["expected_gpu_cases"] < 10000:
        raise ValueError("frozen positive GPU case count required")


def remote_phase(p):
    """Remote body serializes only this script's stdlib code plus protocol metadata."""
    import platform
    import selectors
    import signal
    import stat
    import time
    import xml.etree.ElementTree as ET
    from contextlib import suppress
    from datetime import datetime, timezone

    validate_protocol(p)
    started = time.monotonic()
    deadline = started + p["resources"]["work_seconds"]
    out, repo = Path("/tmp/pr27-results"), Path("/tmp/pr27-source")
    out.mkdir()
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null",
               UV_PYTHON_DOWNLOADS="never", UV_NO_PROGRESS="1", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_CORETYPE="HASWELL",
               NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR")
    env.pop("PYTHONPATH", None)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[name] = "1"
    report = dict(schema=SCHEMA, source_commit=p["source_commit"], inventory_sha256=p["inventory_sha256"],
                  protocol=p, protocol_sha256=digest(p), phase=p["phase"], passed=False, jobs={}, commands=[],
                  started_at=datetime.now(timezone.utc).isoformat(), platform=dict(python=platform.python_version(),
                  system=platform.platform(), machine=platform.machine(), os=platform.system(), provider="Modal"), installed_sources={}, inventory={},
                  packages={}, observed_limits={}, artifacts={}, collection_complete=False)
    log_bytes = 0

    def write(name, value):
        raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        if len(raw) > 4 * 1024**2:
            raise ValueError("bounded report exceeded")
        (out / name).write_bytes(raw)

    def command(name, argv, *, cwd=None, maximum_seconds=None):
        nonlocal log_bytes
        limit = min(deadline, time.monotonic() + maximum_seconds) if maximum_seconds is not None else deadline
        if time.monotonic() >= limit:
            raise TimeoutError("phase work deadline exhausted before " + name)
        stdout, stderr = out / (name + ".stdout.txt"), out / (name + ".stderr.txt")
        record = dict(name=name, argv=argv, started_at=datetime.now(timezone.utc).isoformat(),
                      exit_code=None, status="error", timeout_seconds=limit-time.monotonic(), artifacts=[stdout.name, stderr.name])
        report["commands"].append(record)
        begin = time.monotonic()
        process = None
        try:
            with stdout.open("xb") as a, stderr.open("xb") as b, selectors.DefaultSelector() as selector:
                process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                for stream, target in ((process.stdout, a), (process.stderr, b)):
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, target)
                while selector.get_map() or process.poll() is None:
                    if time.monotonic() >= limit:
                        raise TimeoutError("subprocess/pipe-EOF deadline exceeded")
                    for key, _ in selector.select(min(0.2, max(0.001, limit - time.monotonic()))):
                        block = os.read(key.fd, 65536)
                        if not block:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        left = p["maximum_log_bytes"] - log_bytes
                        key.data.write(block[:left])
                        key.data.flush()
                        log_bytes += min(left, len(block))
                        if len(block) > left:
                            raise ValueError("whole-phase log cap exceeded")
                record.update(exit_code=process.wait(), status="pass" if process.returncode == 0 else "fail")
        except Exception as error:
            record.update(reason=type(error).__name__ + ": " + str(error)[:1024])
        finally:
            if process is not None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                for pipe in (process.stdout, process.stderr):
                    if pipe is not None and not pipe.closed:
                        pipe.close()
                if record["exit_code"] is None:
                    record["exit_code"] = process.returncode
            record["wall_seconds"] = time.monotonic() - begin
            write("progress.json", report)
        if record["status"] != "pass":
            raise RuntimeError("failed command: " + name)
        return stdout.read_bytes()

    def job(name, argv, *, junit=None, expected=None, maximum_seconds=None):
        row = dict(status="not_run", source_commit=p["source_commit"], inventory_sha256=p["inventory_sha256"],
                   command=argv, junit=junit, allowed_skips=p["allowed_skips"], artifacts=[],
                   started_at=datetime.now(timezone.utc).isoformat(), wall_seconds=0, exit_code=None)
        report["jobs"][name] = row
        begin = time.monotonic()
        try:
            if junit:
                collection_argv = [value for value in argv if not value.startswith("--junitxml=")] + ["--collect-only"]
                raw_collection = command(name + "-collection", collection_argv, cwd=repo)
                nodeids = [line for line in raw_collection.decode().splitlines() if line.startswith("tests/") and "::" in line]
                if not nodeids or len(set(nodeids)) != len(nodeids) or (expected is not None and len(nodeids) != expected):
                    raise ValueError("nonempty exact installed collection required")
                row["collection"] = name + "-collection.json"
                write(row["collection"], dict(case_count=len(nodeids), nodeids=nodeids, command=collection_argv))
                expected = len(nodeids)
            command(name, argv, cwd=repo, maximum_seconds=maximum_seconds)
            row.update(status="pass", exit_code=0)
            if junit:
                cases = ET.parse(out / junit).findall(".//testcase")
                row.update(cases=len(cases), failures=sum(len(c.findall("failure")) + len(c.findall("error")) for c in cases),
                           skipped=sum(len(c.findall("skipped")) for c in cases))
                if not cases or row["failures"] or row["skipped"] or (expected is not None and len(cases) != expected):
                    raise ValueError("exact nonempty passing JUnit population required")
        except Exception as error:
            row.update(status="fail", reason=type(error).__name__ + ": " + str(error)[:1024])
            raise
        finally:
            row["artifacts"] = [name + ".stdout.txt", name + ".stderr.txt"] + ([junit] if junit else [])
            if "collection" in row:
                row["artifacts"].extend([row["collection"], name + "-collection.stdout.txt", name + "-collection.stderr.txt"])
            row["wall_seconds"] = time.monotonic() - begin
            row["exit_code"] = report["commands"][-1]["exit_code"]
            write("progress.json", report)

    def pytest_command(targets, extra=()):
        # Only add repository benchmark/test namespaces, never repo/src. The
        # current core stays installed; explicit historical snapshots are separate.
        code = "import pathlib,sys;sys.path.insert(0,sys.argv.pop(1));import openboost;assert 'site-packages' in pathlib.Path(openboost.__file__).parts;import pytest;raise SystemExit(pytest.main(sys.argv[1:]))"
        return [str(repo / ".venv/bin/python"), "-I", "-c", code, str(repo), *targets,
                "-o", "addopts=", "-n", "0", "-q", *extra]

    def installed_check(name):
        code = """
import hashlib,importlib.metadata,json,pathlib,sys
import openboost
root=pathlib.Path(openboost.__file__).parent
assert 'site-packages' in root.parts and not root.is_relative_to(pathlib.Path(sys.argv[1])/'src')
actual={'src/openboost/'+str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')}
expected={'src/openboost/'+str(p.relative_to(pathlib.Path(sys.argv[1])/'src/openboost')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (pathlib.Path(sys.argv[1])/'src/openboost').rglob('*.py')}
assert actual == expected and len(actual)==63
print(json.dumps({'installed_sources':actual,'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'installed_path':str(root)}))
"""
        value = json.loads(command(name, [str(repo / ".venv/bin/python"), "-I", "-c", code, str(repo)], cwd=repo))
        report.update(value)

    def prepare_gpu_consumers(python, wheels):
        command("gpu-build-tools", ["uv", "pip", "install", "--python", python, "--no-deps", *p["build_packages"]], cwd=repo)
        command("extension-build", ["uv", "build", "--wheel", "--python", python, "--no-build-isolation", "--out-dir", str(out / "extensions"), str(repo / "examples/v1_extensions/cohort_splits")], cwd=repo)
        extension_wheels = list((out / "extensions").glob("ob_cohort_splits-*.whl"))
        if len(extension_wheels) != 1:
            raise ValueError("one current extension wheel required")
        command("extension-install", ["uv", "pip", "install", "--python", python, "--no-deps", str(extension_wheels[0])], cwd=repo)
        command("fresh-venv", ["uv", "venv", "--python", sys.executable, "/tmp/pr27-fresh"], cwd=repo)
        command("fresh-install", ["uv", "pip", "install", "--python", "/tmp/pr27-fresh/bin/python", "--no-deps", "numpy==2.3.5", str(wheels[0])], cwd=repo)
        env["OPENBOOST_FRESH_CPU_PYTHON"] = "/tmp/pr27-fresh/bin/python"
        ready = """
import importlib.metadata,json,pathlib,sys
import openboost
import ob_cohort_splits.device as extension
assert hasattr(extension,'DeviceCohortLearner')
assert 'site-packages' in pathlib.Path(extension.__file__).parts
source=pathlib.Path(sys.argv[1])/'examples/v1_extensions/cohort_splits/src/ob_cohort_splits/device.py'
assert pathlib.Path(extension.__file__).read_bytes()==source.read_bytes()
assert importlib.metadata.version('ob-cohort-splits')=='0.1.0'
print(json.dumps({'public_core':openboost.__file__,'installed_extension':extension.__file__}))
"""
        command("installed-entrypoints", [python, "-I", "-c", ready, str(repo)], cwd=repo)
        fresh = """
import importlib.metadata,json,pathlib,sys
import openboost
actual={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
assert actual=={'numpy':'2.3.5','openboost':'1.0.0.dev0'}
assert 'site-packages' in pathlib.Path(openboost.__file__).parts
for name in ('cupy','numba','ob_cohort_splits'):
    assert __import__('importlib.util').util.find_spec(name) is None
print(json.dumps({'packages':actual,'installed_core':openboost.__file__}))
"""
        command("fresh-entrypoint", ["/tmp/pr27-fresh/bin/python", "-I", "-c", fresh], cwd=repo)
        job("gpu-prerequisites", pytest_command(["tests/v1/test_model_identity_cache.py"], ["--junitxml=" + str(out / "prerequisites.xml")]), junit="prerequisites.xml", expected=48)

    try:
        for name in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max"):
            report["observed_limits"][name] = Path(name).read_text().strip() if Path(name).is_file() else None
        command("clone", ["git", "clone", "--no-checkout", "--no-tags", REPOSITORY, str(repo)], maximum_seconds=240)
        command("fetch", ["git", "fetch", "--no-tags", "origin", p["source_commit"]], cwd=repo, maximum_seconds=120)
        command("checkout", ["git", "checkout", "--detach", p["source_commit"]], cwd=repo, maximum_seconds=60)
        observed = command("revision", ["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
        if observed != p["source_commit"]:
            raise ValueError("public commit differs")
        source_inventory = inventory(command("inventory", ["git", "ls-tree", "-rz", "--full-tree", "HEAD"], cwd=repo))
        if digest(source_inventory) != p["inventory_sha256"]:
            raise ValueError("public source inventory differs")
        report["inventory"] = source_inventory
        policy_raw = (repo / p["policy"]["path"]).read_bytes()
        if hashlib.sha256(policy_raw).hexdigest() != p["policy"]["sha256"]:
            raise ValueError("candidate execution policy differs")
        report["policy_sha256"] = p["policy"]["sha256"]
        write("execution-policy.json", json.loads(policy_raw))
        report["lock_sha256"] = hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest()
        if command("clean-before", ["git", "status", "--porcelain"], cwd=repo).strip():
            raise ValueError("clone unexpectedly dirty")
        command("sync", ["uv", "sync", "--locked", "--extra", "test", "--no-install-project", "--python", sys.executable], cwd=repo)
        python = str(repo / ".venv/bin/python")
        command("build-tools", ["uv", "pip", "install", "--python", python, "--no-deps", *p["build_packages"]], cwd=repo)
        job("build" if p["phase"] == "cpu312" else p["phase"] + "-build",
            ["uv", "build", "--python", python, "--no-build-isolation", "--out-dir", str(out / "dist")])
        wheels = list((out / "dist").glob("openboost-*.whl"))
        if len(wheels) != 1 or len(list((out / "dist").glob("openboost-*.tar.gz"))) != 1:
            raise ValueError("one wheel and one sdist required")
        command("install-wheel", ["uv", "pip", "install", "--python", python, "--no-deps", str(wheels[0])], cwd=repo)
        installed_check("installed-before")
        command("pip-freeze", ["uv", "pip", "freeze", "--python", python], cwd=repo)
        if p["phase"] != "gpu":
            job("lint", [str(repo / ".venv/bin/ruff"), "check", "src/openboost", "tests/v1", "tests/conftest.py"])
            job(p["phase"], pytest_command(["tests/"], ["-m", "not gpu and not benchmark", "--junitxml=" + str(out / "cpu.xml")]), junit="cpu.xml")
            if p["phase"] == "cpu312":
                job("docs", [str(repo / ".venv/bin/mkdocs"), "build", "--strict", "--site-dir", "/tmp/pr27-site"])
                site = Path("/tmp/pr27-site")
                write("docs-files.json", {str(path.relative_to(site)):hashlib.sha256(path.read_bytes()).hexdigest() for path in site.rglob("*") if path.is_file()})
                # This changes only optional packages, then reinstalls the same
                # wheel: uv sync would otherwise remove the installed project.
                command("sync-cuda-collection", ["uv", "sync", "--locked", "--extra", "test", "--extra", "cuda", "--no-install-project", "--python", sys.executable], cwd=repo)
                command("reinstall-wheel", ["uv", "pip", "install", "--python", python, "--no-deps", str(wheels[0])], cwd=repo)
                prepare_gpu_consumers(python, wheels)
                raw = command("gpu-collection", pytest_command(p["gpu_tests"], ["-m", "gpu", "--collect-only"]), cwd=repo)
                collected = [line for line in raw.decode().splitlines() if line.startswith("tests/") and "::" in line]
                if len(collected) != p["expected_gpu_cases"] or len(set(collected)) != len(collected):
                    raise ValueError("installed GPU readiness population differs")
                write("gpu-collection.json", dict(cases=len(collected), nodeids=collected, tests=p["gpu_tests"]))
                report["gpu_readiness"] = dict(passed=True, cases=len(collected), artifact="gpu-collection.json")
        else:
            command("sync-cuda", ["uv", "sync", "--locked", "--extra", "test", "--extra", "cuda", "--no-install-project", "--python", sys.executable], cwd=repo)
            command("reinstall-wheel", ["uv", "pip", "install", "--python", python, "--no-deps", str(wheels[0])], cwd=repo)
            hardware = command("gpu-hardware", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], cwd=repo, maximum_seconds=30).decode().strip()
            if "T4" not in hardware:
                raise ValueError("required T4 hardware differs")
            report["gpu"] = dict(hardware=hardware, artifact="gpu-hardware.stdout.txt")
            prepare_gpu_consumers(python, wheels)
            env["OPENBOOST_NORMAL_ARTIFACTS"] = str(out / "normal")
            env["OPENBOOST_REVISED_NORMAL_ARTIFACTS"] = str(out / "revised-normal")
            env["OPENBOOST_COMPARISON_TRAJECTORIES"] = str(out / "comparison-trajectories")
            job("gpu", pytest_command(p["gpu_tests"], ["-m", "gpu", "--basetemp=" + str(out / "pytest-tmp"), "--junitxml=" + str(out / "gpu.xml")]), junit="gpu.xml", expected=p["expected_gpu_cases"], maximum_seconds=1080)
        installed_check("installed-after")
        report["git_clean"] = not command("clean-after", ["git", "status", "--porcelain"], cwd=repo).strip()
        if not report["git_clean"]:
            raise ValueError("candidate changed during validation")
        report["passed"] = True
    except Exception as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error)[:2048])
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["wall_seconds"] = time.monotonic() - started
    report["retained_log_bytes"] = log_bytes
    # Transport retains all declared artifacts, bounded independently of gzip
    # ratios. Oversized outputs produce a false manifest and explicit omission.
    files, total, omitted = [], 10240, []
    for index, path in enumerate(sorted(out.rglob("*"))):
        if not path.is_file():
            continue
        size = path.lstat().st_size
        if index > 8192 or not stat.S_ISREG(path.lstat().st_mode) or len(str(path.relative_to(out)).encode()) > 512 or total + size + 4608 > 58 * 1024**2:
            omitted.append(dict(path=str(path.relative_to(out)), bytes=size))
            continue
        total += size + 4608  # PAX/header/data padding conservative framing allowance.
        files.append(path)
    if omitted:
        report.update(passed=False, omitted=omitted, collection_complete=False)
    else:
        report["collection_complete"] = True
    for path in files:
        relative = str(path.relative_to(out))
        report["artifacts"][relative] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    raw_report = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(raw_report) > 4 * 1024**2:
        raise ValueError("phase metadata exceeds reserved return cap")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            info = archive.gettarinfo(str(path), arcname=str(path.relative_to(out)))
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            with path.open("rb") as source:
                archive.addfile(info, source)
    packed = stream.getvalue()
    if len(packed) + len(raw_report) > RETURN_LIMIT:
        raise ValueError("returned archive+metadata exceeds fixed cap")
    return dict(manifest=raw_report, archive=packed, archive_sha256=hashlib.sha256(packed).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    if len(raw) > 1024**2:
        raise ValueError("small frozen protocol required")
    p = json.loads(raw)
    validate_protocol(p)
    if os.environ.get("MODAL_PROFILE") != "edamame-labs":
        raise ValueError("explicit authorized Modal profile required")
    if args.output.exists():
        raise ValueError("phase output consumed; no automatic retries")
    if p["phase"] == "gpu":
        for phase in ("cpu310", "cpu312"):
            item = p["cpu_prerequisites"][phase]
            manifest = Path(item["path"]).read_bytes()
            if hashlib.sha256(manifest).hexdigest() != item["sha256"]:
                raise ValueError("CPU prerequisite receipt changed")
            value = json.loads(manifest)
            if not value["passed"] or value["source_commit"] != p["source_commit"] or value["inventory_sha256"] != p["inventory_sha256"]:
                raise ValueError("exact-candidate CPU prerequisite did not pass")
            if phase == "cpu312" and (not value.get("gpu_readiness", {}).get("passed")
                    or value["protocol"]["gpu_tests"] != p["gpu_tests"]
                    or value["gpu_readiness"]["cases"] != p["expected_gpu_cases"]):
                raise ValueError("installed GPU readiness required before allocation")
    import modal
    from modal._serialization import serialize

    args.output.mkdir(parents=True)
    launch = dict(protocol_sha256=hashlib.sha256(raw).hexdigest(), canonical_protocol_sha256=digest(p),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), status="starting",
                  serialized_function_bytes=len(serialize(remote_phase)))
    (args.output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    if launch["serialized_function_bytes"] > 65536:
        raise ValueError("remote function serialization exceeds bound")
    image = (modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
             if p["phase"] == "gpu" else modal.Image.debian_slim(python_version=p["python"]))
    image = image.apt_install("git").uv_pip_install("uv==" + p["uv"], uv_version=p["uv"])
    app = modal.App("openboost-pr27-" + p["phase"])
    options = dict(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=p["resources"]["timeout_seconds"],
                   max_containers=1, retries=0, serialized=True, include_source=False)
    if p["phase"] == "gpu":
        options["gpu"] = "T4"
    remote = app.function(**options)(remote_phase)
    try:
        with modal.enable_output(), app.run():
            value = remote.remote(p)
        if len(value["manifest"]) + len(value["archive"]) > RETURN_LIMIT:
            raise ValueError("received phase exceeds cap")
        if hashlib.sha256(value["archive"]).hexdigest() != value["archive_sha256"]:
            raise ValueError("returned archive hash differs")
        (args.output / "manifest.json").write_bytes(value["manifest"])
        (args.output / "artifacts.tar").write_bytes(value["archive"])
        result = json.loads(value["manifest"])
        with tarfile.open(fileobj=io.BytesIO(value["archive"]), mode="r:") as archive:
            members = archive.getmembers()
            if len(members) > 8192 or sum(member.size for member in members) > 58 * 1024**2:
                raise ValueError("returned member count/bytes exceed cap")
            names = set()
            for member in members:
                if not member.isfile() or member.name in names or member.name in {"manifest.json", "launch.json", "artifacts.tar"} or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                    raise ValueError("unsafe returned artifact")
                names.add(member.name)
                payload = archive.extractfile(member).read()
                descriptor = result["artifacts"][member.name]
                if len(payload) != descriptor["bytes"] or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
                    raise ValueError("returned artifact differs")
                path = args.output / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            if names != set(result["artifacts"]):
                raise ValueError("returned archive inventory differs")
        launch["status"] = "pass" if result["passed"] and result["collection_complete"] else "fail"
    except Exception as error:
        launch.update(status="error", error=type(error).__name__ + ": " + str(error)[:2048])
        raise
    finally:
        (args.output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    raise SystemExit(0 if launch["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
