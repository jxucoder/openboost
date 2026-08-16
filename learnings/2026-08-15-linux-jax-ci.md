# 2026-08-15: Linux JAX CI Isolation

## Context

PR #19's fast-test matrix passed on macOS/Python 3.10 and 3.12, while both
Ubuntu jobs remained in the pytest step for more than 15 minutes. Lint and
dependency installation had already passed. The test extra installs JAX only
on Linux, and two JAX compilation tests were running inside the repository-wide
`pytest-xdist -n auto` process pool.

## Decision or Result

Mark the two JAX-dependent tests explicitly and run them once, serially, on
Linux/Python 3.12. The regular fast and full suites exclude that marker. This
separates JAX compilation/runtime behavior from xdist worker behavior and avoids
duplicating the same optional-backend test across two Python versions.

Add job-level timeouts and workflow concurrency cancellation so a future hang
cannot consume the default six-hour GitHub Actions limit or leave superseded PR
runs executing indefinitely.

## Changes

- `tests/test_distribution_gradients.py`: mark the two true JAX tests.
- `pyproject.toml`: register the `jax` pytest marker.
- `.github/workflows/unit-tests.yml`: exclude JAX from parallel suites, add one
  serial Linux/Python 3.12 step, add bounded job timeouts, and cancel superseded
  runs on the same ref.

## Verification

- Non-JAX distribution gradient selection: 49 passed, 2 deselected.
- JAX selection on macOS Intel: 2 selected and correctly skipped because no JAX
  wheel is installed on that platform.
- Workflow YAML parsing and `git diff --check`: passed.
- The decisive verification is the replacement PR workflow on Ubuntu; local
  macOS cannot reproduce the Linux-only JAX installation.

## Failed Attempts

- GitHub does not publish downloadable logs for an in-progress job; the log
  endpoint returned 404, so there was no responsible way to name a specific
  test from partial output.
- Repeated status polling confirmed the Ubuntu pair was symmetric and isolated
  to pytest but could not distinguish JAX compilation from another Linux-only
  stall. The new split is designed to make the next run diagnostic as well as
  faster.

## Risks and Follow-ups

- If the non-JAX Ubuntu suite still stalls, JAX was not the cause; use the job
  timeout's completed logs to identify the last test/file and fix that path.
- If only the serial JAX step stalls, pin or revise the Linux JAX test/runtime
  rather than weakening the core suite.

## Commits

- `7e26d0e` — `ci: isolate JAX tests from xdist`
