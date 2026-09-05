"""Build and verify three wheels in a fresh, non-editable, outside-repo venv."""

import ast
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run(args, cwd, env):
    try:
        return subprocess.check_output(args, cwd=cwd, env=env, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        print(exc.output, flush=True)
        raise


def verify(output):
    env = dict(
        os.environ,
        OPENBOOST_BACKEND="cpu",
        NUMBA_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    source_files = (
        list(HERE.rglob("*.py"))
        + list(HERE.rglob("pyproject.toml"))
        + list(HERE.rglob("README.md"))
        + [HERE / "requirements-cpu.txt"]
    )
    source_files = [p for p in source_files if "__pycache__" not in p.parts]
    public_imports = set()
    for p in HERE.glob("*/src/**/*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            names = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [n.name for n in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for name in names:
                if name.startswith("openboost"):
                    assert name in {"openboost", "openboost.experimental"}, (p, name)
                    public_imports.add(name)
    with tempfile.TemporaryDirectory(prefix="openboost-extension-") as temp:
        work = Path(temp)
        wheels = work / "wheels"
        wheels.mkdir()
        for project in [ROOT, HERE / "normal_fisher", HERE / "bounded_leaves"]:
            run(["uv", "build", "--wheel", "--out-dir", str(wheels), str(project)], work, env)
        venv = work / "venv"
        run(["uv", "venv", "--python", sys.executable, str(venv)], work, env)
        python = str(venv / "bin/python")
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                *map(str, sorted(wheels.glob("*.whl"))),
                "-r",
                str(HERE / "requirements-cpu.txt"),
            ],
            work,
            env,
        )
        frozen = json.loads(
            run(
                [
                    python,
                    "-c",
                    "import json,importlib.metadata as m; print(json.dumps(sorted(d.metadata['Name']+'=='+d.version for d in m.distributions())))",
                ],
                work,
                env,
            )
        )
        installed = json.loads(
            run(
                [
                    python,
                    "-c",
                    "import json,sys,openboost,normal_fisher,bounded_leaves; from pathlib import Path; print(json.dumps({m.__name__:str(Path(m.__file__).relative_to(sys.prefix)) for m in (openboost,normal_fisher,bounded_leaves)}))",
                ],
                work,
                env,
            )
        )
        assert all("site-packages/" in p for p in installed.values())
        shutil.copy(HERE / "normal_fisher/tests/test_normal.py", work / "test_normal.py")
        shutil.copy(HERE / "bounded_leaves/tests/test_leaf.py", work / "test_leaf.py")
        shutil.copy(HERE / "test_composition.py", work / "test_composition.py")
        shutil.copy(HERE / "check_inference.py", work / "check_inference.py")
        shutil.copy(HERE / "demo.py", work / "demo.py")
        demo = json.loads(run([python, "demo.py"], work, env))
        test_output = run(
            [
                python,
                "-m",
                "pytest",
                "-q",
                "--junitxml=junit.xml",
                "test_normal.py",
                "test_leaf.py",
                "test_composition.py",
            ],
            work,
            env,
        )
        run(
            [
                "uv",
                "pip",
                "uninstall",
                "--python",
                python,
                "openboost-example-normal-fisher",
                "openboost-example-bounded-leaves",
            ],
            work,
            env,
        )
        inference = json.loads(run([python, "check_inference.py"], work, env))
        result = {
            "source_sha": run(["git", "rev-parse", "HEAD"], ROOT, env).strip(),
            "source_dirty": bool(run(["git", "status", "--porcelain"], ROOT, env).strip()),
            "files": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source_files)
            },
            "wheel_hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(wheels.glob("*.whl"))
            },
            "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "os": platform.system(),
            "machine": platform.machine(),
            "threads": 1,
            "uv_version": run(["uv", "--version"], work, env).strip(),
            "packages": frozen,
            "module_paths_relative_to_venv": installed,
            "extension_openboost_imports": sorted(public_imports),
            "method_source_lines": {
                p.parent.name: len(p.read_text().splitlines())
                for p in HERE.glob("*/src/*/__init__.py")
            },
            "inference_after_uninstall": inference,
            "demo": demo,
            "composition": json.loads((work / "composition.json").read_text()),
            "command": "uv run --no-sync python examples/extensions/verify_wheels.py OUTPUT",
            "scope": "CPU installation and mathematical conformance; no external adoption or GPU claim",
        }
        # Only sanitized evidence leaves the disposable environment.
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        junit = ET.fromstring((work / "junit.xml").read_text())
        for suite in junit.iter("testsuite"):
            suite.attrib.pop("hostname", None)
        ET.ElementTree(junit).write(output / "junit.xml", encoding="unicode", xml_declaration=True)
        print(test_output.replace(str(work), "<temporary-workdir>"))
        print(json.dumps(inference))
        print("Evidence:", output)


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
