"""Reconstruct consumed run-12 sources for offline audit in a fresh process."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path("benchmarks/v1/evidence/cuda-glm-108")


def snapshot(destination):
    """Materialize and verify executed files without reading current core sources."""
    destination = Path(destination)
    manifest = json.loads((ROOT / EVIDENCE / "manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        content = subprocess.check_output(
            ["git", "show", manifest["revision"] + ":" + name], cwd=ROOT
        )
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("dispatch source differs: " + name)
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    shutil.copytree(ROOT / EVIDENCE, destination / EVIDENCE)
    git_dir = (
        subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"], cwd=ROOT)
        .decode()
        .strip()
    )
    (destination / ".git").write_text("gitdir: " + git_dir + "\n")
    return destination
