#!/usr/bin/env python
"""Execute the ``python`` code blocks in an allowlist of docs pages.

For each allowlisted markdown file, every fenced ```python block is extracted,
the blocks are concatenated in order into one script, and that script is run in
a subprocess with ``OPENBOOST_BACKEND=cpu`` from a temporary working directory
(so examples that write files never pollute the repo).

A block can be excluded from execution by placing an HTML comment marker on the
nearest non-blank line above its opening fence:

    <!-- docs-ci: skip -->
    ```python
    model.fit(X_gpu, y)  # needs CUDA / external data; not run in CI
    ```

Exit status is non-zero if any file's script fails, so this can gate CI.

Usage:
    OPENBOOST_BACKEND=cpu uv run python scripts/run_doc_examples.py
    uv run python scripts/run_doc_examples.py docs/getting-started/quickstart.md   # subset
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Only files on this allowlist are checked. Add a page here once its examples
# are self-contained (each file's blocks run top-to-bottom as one script).
DOC_FILES = [
    "docs/cookbook/custom-loss.md",
    "docs/cookbook/custom-distribution.md",
    "docs/cookbook/custom-growth-strategy.md",
    "docs/cookbook/device-loss.md",
    "docs/getting-started/quickstart.md",
    "docs/user-guide/models/gradient-boosting.md",
]

SKIP_MARKER = "docs-ci: skip"
FENCE_OPEN = re.compile(r"^```python\s*$")
FENCE_CLOSE = re.compile(r"^```\s*$")
PER_FILE_TIMEOUT = 1200  # seconds


def extract_blocks(markdown: str) -> list[tuple[int, str]]:
    """Return [(start_line, code)] for each runnable ```python block.

    Blocks whose nearest preceding non-blank line is an HTML comment containing
    ``docs-ci: skip`` are dropped.
    """
    lines = markdown.splitlines()
    blocks: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        if FENCE_OPEN.match(lines[i]):
            # Look upward for a skip marker on the nearest non-blank line.
            skip = False
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j >= 0:
                prev = lines[j].strip()
                if prev.startswith("<!--") and SKIP_MARKER in prev:
                    skip = True

            start = i + 1
            body: list[str] = []
            i += 1
            while i < len(lines) and not FENCE_CLOSE.match(lines[i]):
                body.append(lines[i])
                i += 1
            if not skip:
                blocks.append((start + 1, "\n".join(body)))  # 1-indexed line
        i += 1
    return blocks


def build_script(path: Path, blocks: list[tuple[int, str]]) -> str:
    parts = [f'"""Auto-generated from {path} by scripts/run_doc_examples.py."""']
    for line_no, code in blocks:
        parts.append(f"# --- {path.name}: block starting at line {line_no} ---")
        parts.append(code)
    return "\n\n".join(parts) + "\n"


def run_file(rel_path: str, verbose: bool = False) -> tuple[str, str]:
    """Run one docs file's examples. Returns (status, detail).

    status is one of 'PASS', 'FAIL', 'NO BLOCKS', 'MISSING'.
    """
    path = REPO_ROOT / rel_path
    if not path.is_file():
        return "MISSING", "file not found"

    blocks = extract_blocks(path.read_text(encoding="utf-8"))
    if not blocks:
        return "NO BLOCKS", "no runnable python blocks"

    script = build_script(Path(rel_path), blocks)
    env = dict(os.environ)
    env["OPENBOOST_BACKEND"] = "cpu"
    env.setdefault("MPLBACKEND", "Agg")  # never open GUI windows

    with tempfile.TemporaryDirectory(prefix="openboost-docs-") as tmpdir:
        script_path = Path(tmpdir) / "example.py"
        script_path.write_text(script, encoding="utf-8")
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=PER_FILE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return "FAIL", f"timed out after {PER_FILE_TIMEOUT}s"
        elapsed = time.monotonic() - start

    detail = f"{len(blocks)} blocks, {elapsed:.1f}s"
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-30:])
        return "FAIL", f"{detail}\n{tail}"
    if verbose and proc.stdout.strip():
        detail += "\n" + "\n".join(f"    | {ln}" for ln in proc.stdout.strip().splitlines())
    return "PASS", detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional subset of repo-relative docs paths (default: full allowlist).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show example stdout on success."
    )
    args = parser.parse_args()

    targets = args.files or DOC_FILES
    failures = 0
    for rel_path in targets:
        status, detail = run_file(rel_path, verbose=args.verbose)
        first, *rest = detail.splitlines()
        print(f"[{status:>9}] {rel_path} ({first})")
        for line in rest:
            print(f"           {line}")
        if status in ("FAIL", "MISSING"):
            failures += 1

    print(f"\n{len(targets) - failures}/{len(targets)} docs files passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
