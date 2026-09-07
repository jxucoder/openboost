"""Prepare a narrow OpenBoost author view; no agent dispatch or isolation claim."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

AUTHOR_FILES = tuple(
    f"v1-sprints/069-author-packet/{name}.md" for name in ("README", "D1", "D2")
) + tuple(f"docs/v1/{name}.md" for name in ("cpu-state", "numeric-ops", "trees", "squared"))
EVALUATOR_FILES = (
    "planning/foundation-tasks.md",
    "examples/v1_extensions/checks.py",
    "examples/v1_extensions/expectile_oracle.py",
    "examples/v1_extensions/expectile_checks.py",
    "benchmarks/v1/search-design.json",
)


def prepare(output):
    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo):
        raise ValueError("commit the packet inputs before building the author view")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    author = output / "author"
    for name in AUTHOR_FILES:
        target = author / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, target)
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(author / "wheels")], cwd=repo, check=True
    )
    record = dict(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        scope="OpenBoost author-view preparation only",
        dispatch_ready=False,
        author_files={
            str(p.relative_to(author)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(author.rglob("*"))
            if p.is_file()
        },
        evaluator_inputs={
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in EVALUATOR_FILES
        },
        budget=dict(wall_s=1800, generated_tokens=20000),
        missing=[
            "observable token/time enforcement",
            "actual evaluator isolation test",
            "independent runner and model/settings freeze",
            "incumbent path audit and arm freeze",
            "standalone D1/D2 evaluator invocation and dependency closure",
        ],
        attempts=[],
    )
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
