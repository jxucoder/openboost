"""Build an installed run-10 core from the candidate snapshot plus two frozen originals."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from benchmarks.v1.performance_evidence import digest


def build(root, destination, protocol):
    root, destination = Path(root), Path(destination)
    source = destination / "source"
    source.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root / "src/openboost", source / "src/openboost")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copyfile(root / name, source / name)
    for target, original in protocol["baseline_overlays"].items():
        shutil.copyfile(root / original, source / target)
    sources = {
        str(p.relative_to(source)): digest(p.read_bytes())
        for p in sorted((source / "src/openboost").glob("*.py"))
    }
    if sources != protocol["baseline_sources"]:
        raise ValueError("baseline source overlay differs from run 10")
    python = destination / "venv/bin/python"
    report = dict(passed=False, python=str(python), sources=sources, commands=[])
    deadline = perf_counter() + protocol["bootstrap_seconds"]

    def command(argv):
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise TimeoutError("baseline installation deadline exhausted")
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=remaining)
        report["commands"].append(
            dict(
                argv=argv,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        return completed.stdout

    command(
        [
            "uv",
            "build",
            "--python",
            sys.executable,
            "--wheel",
            "--no-build-isolation",
            "--out-dir",
            str(destination / "wheels"),
            str(source),
        ]
    )
    command(
        [
            "uv",
            "venv",
            "--system-site-packages",
            "--python",
            sys.executable,
            str(destination / "venv"),
        ]
    )
    # system-site-packages includes the base interpreter, not a parent virtual
    # environment. Append its resolved site directories after our own core;
    # including resolved paths also handles uv's dependency environment .pth.
    site = Path(
        command(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"]
        ).strip()
    )
    dependencies = [p for p in sys.path if Path(p).name in ("site-packages", "dist-packages")]
    (site / "openboost-benchmark-dependencies.pth").write_text("\n".join(dependencies) + "\n")
    report["dependency_paths"] = dependencies
    wheel = next((destination / "wheels").glob("*.whl"))
    command(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", "--reinstall", str(wheel)]
    )
    code = """import hashlib,importlib.metadata,json,pathlib,openboost,sys
core=pathlib.Path(openboost.__file__).parent
assert str(core).startswith(sys.argv[1]) and 'site-packages' in core.parts
print(json.dumps(dict(installed_path=str(core),sources={'src/openboost/'+p.name:
    hashlib.sha256(p.read_bytes()).hexdigest() for p in core.glob('*.py')},
    packages={name:importlib.metadata.version(name) for name in json.loads(sys.argv[2])})))
"""
    packages = dict(p.split("==") for p in protocol["packages"])
    installed = json.loads(
        command([str(python), "-c", code, str(destination), json.dumps(list(packages))])
    )
    if installed["sources"] != sources or installed["packages"] != packages:
        raise ValueError("installed baseline sources/packages differ")
    report.update(passed=True, installed=installed, wheel_sha256=digest(wheel.read_bytes()))
    return report
