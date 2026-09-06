"""Build isolated development wheels, check them, then remove training plugins."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "examples/v1_extensions"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "openboost-v1-development-extensions-v1",
        "passed": False,
        "claim": "repository-authored D2/D3/D4 installation and correctness only",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "device": "cpu",
        "threads": 1,
        "commands": [],
        "sources": {
            str(p.relative_to(ROOT)): digest(p)
            for p in sorted(SOURCE.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".toml")
        },
    }
    report["reference_sources"] = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted((ROOT / "tests/v1/reference").glob("*.py"))
    }
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(command, cwd):
        report["commands"].append({"argv": command, "cwd": str(cwd)})
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout

    try:
        with tempfile.TemporaryDirectory(prefix="openboost-v1-extensions-") as temporary:
            work = Path(temporary)
            wheels = work / "wheels"
            for project in (
                ROOT,
                SOURCE / "cohort_splits",
                SOURCE / "penalized_leaves",
                SOURCE / "ordered_updates",
            ):
                run(["uv", "build", "--wheel", "--offline", "--out-dir", str(wheels)], project)
            paths = sorted(wheels.glob("*.whl"))
            report["wheels"] = {p.name: digest(p) for p in paths}
            run(["uv", "venv", "--python", sys.executable, str(work / "env")], work)
            python = str(work / "env/bin/python")
            run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--offline",
                    "--python",
                    python,
                    "numpy==2.3.5",
                    *map(str, paths),
                ],
                work,
            )
            for name in ("checks.py", "core_inference.py", "ordered_checks.py"):
                shutil.copyfile(SOURCE / name, work / name)
            run([python, "-I", str(work / "checks.py"), str(output)], work)
            run(
                [
                    sys.executable,
                    "-m",
                    "examples.v1_extensions.ordered_oracle",
                    str(output / "ordered-expected.json"),
                ],
                ROOT,
            )
            run([python, "-I", str(work / "ordered_checks.py"), str(output)], work)
            report["versions"] = json.loads(
                run(
                    [
                        python,
                        "-I",
                        "-c",
                        "import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in "
                        "['openboost','numpy','ob-cohort-splits','ob-penalized-leaves','ob-ordered-updates']}))",
                    ],
                    work,
                )
            )
            run(
                [
                    "uv",
                    "pip",
                    "uninstall",
                    "--python",
                    python,
                    "ob-cohort-splits",
                    "ob-penalized-leaves",
                    "ob-ordered-updates",
                ],
                work,
            )
            run([python, "-I", str(work / "core_inference.py"), str(output)], work)
            report["plugin_free_inference"] = True
            report["artifacts"] = {
                p.name: digest(p)
                for p in sorted(output.glob("*.json"))
                if p.name != "manifest.json"
            }
            report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Installed D2/D3/D4 checks and plugin-free inference passed.")


if __name__ == "__main__":
    main(sys.argv[1])
