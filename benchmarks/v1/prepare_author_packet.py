"""Prepare a narrow OpenBoost author view; no agent dispatch or isolation claim."""

import argparse
import hashlib
import json
import posixpath
import re
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

AUTHOR_FILES = tuple(
    f"v1-sprints/069-author-packet/{name}.md" for name in ("README", "D1", "D2")
) + tuple(
    f"docs/v1/{name}.md"
    for name in ("cpu-state", "numeric-ops", "trees", "squared", "preparation", "stopping")
)
EVALUATOR_FILES = ("planning/foundation-tasks.md",) + tuple(
    f"benchmarks/v1/authoring/{name}.py"
    for name in ("export", "development", "judge", "verify", "isolation")
)


def copy_author_files(repo, author):
    """Copy only explicit inputs; links never expand the author allowlist."""
    omitted, originals = [], {}
    for name in AUTHOR_FILES:
        source = repo / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"author input must be a regular file: {name}")
        originals[name] = hashlib.sha256(source.read_bytes()).hexdigest()

        def link(match, origin=name):
            label, target = match.groups()
            parsed = urlsplit(target)
            resolved = posixpath.normpath(
                posixpath.join(posixpath.dirname(origin), unquote(parsed.path))
            )
            if (
                not parsed.scheme
                and not parsed.netloc
                and (not parsed.path or resolved in AUTHOR_FILES)
            ):
                return match.group(0)
            omitted.append(dict(source=origin, target=target, label=label))
            return label + " (outside this packet)"

        content = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, source.read_text())
        target = author / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    (author / "README.md").write_text(
        "# OpenBoost D1/D2 CPU author view\n\n"
        "Start with [D1](v1-sprints/069-author-packet/D1.md) and "
        "[D2](v1-sprints/069-author-packet/D2.md), then "
        "[state](docs/v1/cpu-state.md), [operations](docs/v1/numeric-ops.md), "
        "[trees](docs/v1/trees.md), [squared training](docs/v1/squared.md), "
        "[preparation](docs/v1/preparation.md) and [stopping](docs/v1/stopping.md).\n\n"
        "Only selected public CPU documentation is bundled. Links to other pages "
        "are explicitly rendered as plain labels; they do not expose evaluator "
        "or example-solution files. The core wheel contains its public Python "
        "implementation, including experimental CUDA modules. This CPU packet "
        "does not establish their current hardware validation or supply CUDA dependencies.\n\n"
        "Install with uv in a fresh environment using NumPy 2.3.5 and the wheel in "
        "`wheels/`. Existing extension solutions and mathematical judges are not "
        "part of this author view. Model, tools, incumbent arms and budget enforcement "
        "are not frozen; no independent attempt is dispatched by this export.\n"
    )
    return dict(original_sources=originals, omitted_links=omitted)


def audit_wheel(repo, wheel):
    expected = {
        str(p.relative_to(repo / "src")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((repo / "src/openboost").rglob("*.py"))
    }
    marker = repo / "src/openboost/py.typed"
    support = {"openboost/py.typed": marker.read_bytes()} if marker.is_file() else {}
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("wheel contains duplicate entries")
        for name in names:
            if name not in expected and name not in support and not (
                name.startswith("openboost-")
                and ".dist-info/" in name
                and ".." not in name.split("/")
            ):
                raise ValueError("wheel includes a file outside the public core and metadata")
        if any(name not in names or archive.read(name) != data for name, data in support.items()):
            raise ValueError("wheel package marker differs from the source")
        actual = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in names
            if name.endswith(".py")
        }
    if not expected or actual != expected:
        raise ValueError("wheel Python sources differ from the current public core")
    return actual


def prepare(output):
    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo):
        raise ValueError("commit the packet inputs before building the author view")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    author = output / "author"
    record = dict(
        schema="openboost-author-view-v2",
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        scope="OpenBoost author-view preparation only",
        dispatch_ready=False,
        status="incomplete",
        evaluator_inputs={
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in EVALUATOR_FILES
        },
        budget=dict(wall_s=1800, generated_tokens=20000),
        missing=[
            "observable token/time enforcement",
            "actual evaluator isolation test",
            "independent runner and model/settings freeze",
            "incumbent path audit and arm freeze",
            "complete author task adapter, invalid-input and edit accounting freeze",
            "bind the standalone verifier cohort and its installed dependency closure",
        ],
        attempts=[],
    )
    try:
        record.update(copy_author_files(repo, author))
        command = ["uv", "build", "--wheel", "--offline", "--out-dir", str(author / "wheels")]
        record["build_command"] = command
        result = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=120)
        record["build"] = dict(
            exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr
        )
        result.check_returncode()
        wheels = list((author / "wheels").glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("exactly one core wheel required")
        record["wheel_sources"] = audit_wheel(repo, wheels[0])
        record["status"] = "prepared"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["author_files"] = {
            str(p.relative_to(author)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(author.rglob("*"))
            if p.is_file()
        }
        (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
