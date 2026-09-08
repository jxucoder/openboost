"""Bounded macOS file-access smoke against an actual author/evaluator snapshot."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-read*)
(allow file-read-metadata)
(allow file-read* (subpath (param "WORK")) (subpath (param "MATERIALS"))
    (subpath (param "RUNTIME")) (subpath (param "PYTHON_RUNTIME"))
    (subpath "/System") (subpath "/usr") (subpath "/Library")
    (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random"))
(deny file-write*)
(allow file-write* (subpath (param "WORK")))
(deny file-read* (subpath (param "EVALUATOR")) (subpath (param "REPOSITORY")))
"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(command, cwd, env, seconds):
    """Supervise the real subprocess and retain timeout output; no token claim."""
    record = dict(argv=command, cwd=str(cwd), wall_limit_s=seconds)
    started = time.monotonic()
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=seconds)
            record["status"] = "complete"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            record["status"] = "timeout"
        record.update(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            wall_s=time.monotonic() - started,
        )
    return record


def probe(packet, evaluator, output):
    if sys.platform != "darwin":
        raise ValueError("this smoke requires actual macOS sandbox-exec")
    packet, evaluator, output = (Path(p).resolve() for p in (packet, evaluator, output))
    if output.is_relative_to(ROOT):
        raise ValueError("use an output outside the repository, which the worker cannot read")
    packet_record = json.loads((packet / "manifest.json").read_text())
    if packet_record["status"] != "prepared":
        raise ValueError("a prepared author view is required")
    for name, expected in packet_record["author_files"].items():
        if digest(packet / "author" / name) != expected:
            raise ValueError(f"author packet changed: {name}")
    evaluator_record = json.loads((evaluator / "manifest.json").read_text())
    for name, expected in evaluator_record["runtime_files"].items():
        if digest(evaluator / name) != expected:
            raise ValueError(f"evaluator changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    work, private, materials = output / "work", output / "evaluator", output / "materials"
    work.mkdir()
    shutil.copytree(evaluator, private)
    shutil.copytree(packet / "author", materials)
    unlisted = output / "unlisted-answers.json"
    shutil.copyfile(private / "expected.json", unlisted)
    profile = output / "worker.sb"
    profile.write_text(PROFILE)
    report = dict(
        schema="openboost-local-file-isolation-v1",
        passed=False,
        dispatch_ready=False,
        attempts=[],
        scope="macOS worker file-access and wall-time smoke; not complete agent isolation",
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        os=platform.platform(),
        python=platform.python_version(),
        source_sha256=digest(Path(__file__)),
        profile_sha256=digest(profile),
        packet_manifest_sha256=digest(packet / "manifest.json"),
        evaluator_manifest_sha256=digest(evaluator / "manifest.json"),
        commands=[],
        cases={},
    )
    env = dict(
        PATH=os.environ["PATH"],
        HOME=str(work),
        TMPDIR=str(work),
        UV_CACHE_DIR=os.environ.get("UV_CACHE_DIR", "/tmp/openboost-research-uv-cache"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )

    def setup(command):
        result = invoke(command, output, env, 120)
        report["commands"].append(result)
        if result["exit_code"] != 0:
            raise RuntimeError(result["stderr"])
        return result["stdout"]

    try:
        setup(["uv", "venv", "--python", sys.executable, str(output / "runtime")])
        python = str(output / "runtime/bin/python")
        (wheel,) = (materials / "wheels").glob("*.whl")
        setup(
            [
                "uv",
                "pip",
                "install",
                "--offline",
                "--link-mode",
                "copy",
                "--python",
                python,
                "numpy==2.3.5",
                str(wheel),
            ]
        )
        core = Path(
            setup([python, "-I", "-c", "import openboost; print(openboost.__file__)"]).strip()
        )
        protected = [*sorted(p for p in private.rglob("*") if p.is_file()), core, wheel, unlisted]
        report["protected_before"] = {str(p.relative_to(output)): digest(p) for p in protected}
        launcher = [
            "/usr/bin/sandbox-exec",
            "-D",
            f"WORK={work}",
            "-D",
            f"EVALUATOR={private}",
            "-D",
            f"REPOSITORY={ROOT}",
            "-D",
            f"MATERIALS={materials}",
            "-D",
            f"RUNTIME={output / 'runtime'}",
            "-D",
            f"PYTHON_RUNTIME={Path(sys.base_prefix).resolve()}",
            "-f",
            str(profile),
            python,
            "-I",
            "-B",
        ]
        # Public docs are executable author materials; no expected answers are supplied.
        blocks = []
        for path in sorted((materials / "docs/v1").glob("*.md")):
            code = "\n\n".join(re.findall(r"```python\n(.*?)```", path.read_text(), re.S))
            if code:
                blocks.append((path.stem, code))
        for name, code in blocks:
            record = invoke([*launcher, "-c", code + "\nprint('public-doc-passed')"], work, env, 15)
            report["cases"][f"public_{name}"] = record
            if record["exit_code"] != 0 or "public-doc-passed" not in record["stdout"]:
                raise RuntimeError(
                    f"public installed-wheel check failed: {name}: {record['stderr']}"
                )
        read = "import pathlib,sys; pathlib.Path(sys.argv[1]).read_bytes()"
        write = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('deliberate tamper')"
        cases = {
            "read_unlisted_answer_copy": (["-c", read, str(unlisted)], 5),
            "read_answers": (["-c", read, str(private / "expected.json")], 5),
            "read_judge": (["-c", read, str(private / "judge.py")], 5),
            "read_repository_solution": (
                [
                    "-c",
                    read,
                    str(ROOT / "examples/v1_extensions/expectile/src/ob_expectile/__init__.py"),
                ],
                5,
            ),
            "write_judge": (["-c", write, str(private / "judge.py")], 5),
            "write_manifest": (["-c", write, str(private / "manifest.json")], 5),
            "write_core": (["-c", write, str(core)], 5),
            "write_wheel": (["-c", write, str(wheel)], 5),
            "symlink_read": (
                [
                    "-c",
                    "import os,pathlib,sys; os.symlink(sys.argv[1], 'alias'); pathlib.Path('alias').read_bytes()",
                    str(private / "expected.json"),
                ],
                5,
            ),
            "hardlink_read": (
                [
                    "-c",
                    "import os,pathlib,sys; os.link(sys.argv[1], 'hard-alias'); pathlib.Path('hard-alias').read_bytes()",
                    str(private / "expected.json"),
                ],
                5,
            ),
            "child_read": (
                [
                    "-c",
                    "import subprocess,sys; sys.exit(subprocess.run([sys.executable,'-I','-c',sys.argv[1],sys.argv[2]]).returncode)",
                    read,
                    str(private / "expected.json"),
                ],
                5,
            ),
            "network": (["-c", "import socket; socket.socket().connect(('127.0.0.1', 9))"], 5),
            "wall_timeout": (
                ["-c", "import time; print('started',flush=True); time.sleep(30)"],
                0.25,
            ),
        }
        for name, (arguments, seconds) in cases.items():
            report["cases"][name] = invoke([*launcher, *arguments], work, env, seconds)
        checks = {}
        for name in cases:
            case = report["cases"][name]
            checks[name] = (
                case["status"] == "timeout"
                and case["exit_code"] == -9
                and "started" in case["stdout"]
                if name == "wall_timeout"
                else case["status"] == "complete"
                and case["exit_code"] != 0
                and "PermissionError" in case["stderr"]
            )
        report["protected_after"] = {str(p.relative_to(output)): digest(p) for p in protected}
        checks["protected_files_unchanged"] = (
            report["protected_before"] == report["protected_after"]
        )
        report.update(checks=checks, public_examples=len(blocks), passed=all(checks.values()))
        if not report["passed"]:
            raise RuntimeError("file isolation smoke did not meet every declared check")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("evaluator", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    probe(args.packet, args.evaluator, args.output)
