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

The replacement Ubuntu/Python 3.12 job falsified the original performance
hypothesis: the non-JAX core suite passed 729 tests in 937.02 seconds, while the
two isolated JAX tests passed in 3.59 seconds. JAX was not the source of the long
step. Keep the isolation because it makes the optional path explicit, but treat
the 15-minute cold core suite as the CI cost to optimize.

The downstream `full-tests` job was also rerunning the entire core suite after
all four matrix jobs had passed. It now runs only tests marked `slow` plus the
documentation examples. Numba's on-disk cache is restored per OS and Python
version so later commits can reuse compiled kernels; the first run for a new
source hash remains a cold run.

## Changes

- `tests/test_distribution_gradients.py`: mark the two true JAX tests.
- `pyproject.toml`: register the `jax` pytest marker.
- `.github/workflows/unit-tests.yml`: exclude JAX from parallel suites, add one
  serial Linux/Python 3.12 step, add bounded job timeouts, and cancel superseded
  runs on the same ref. A follow-up avoids the duplicate core suite, restores a
  versioned Numba cache, and moves JavaScript actions off deprecated Node 20
  releases.

## Verification

- Non-JAX distribution gradient selection: 49 passed, 2 deselected.
- JAX selection on macOS Intel: 2 selected and correctly skipped because no JAX
  wheel is installed on that platform.
- Workflow YAML parsing and `git diff --check`: passed.
- The decisive verification is the replacement PR workflow on Ubuntu; local
  macOS cannot reproduce the Linux-only JAX installation.
- Replacement Ubuntu/Python 3.12 job: 729 passed, 32 skipped in 937.02 seconds;
  isolated JAX step: 2 passed in 3.59 seconds.
- Revised local downstream selection: 2 slow tests passed, 762 deselected in
  10.91 seconds; all 6 executable documentation files passed.

## Failed Attempts

- GitHub does not publish downloadable logs for an in-progress job; the log
  endpoint returned 404, so there was no responsible way to name a specific
  test from partial output.
- Repeated status polling confirmed the Ubuntu pair was symmetric and isolated
  to pytest but could not distinguish JAX compilation from another Linux-only
  stall. The new split is designed to make the next run diagnostic as well as
  faster.

## Risks and Follow-ups

- Measure one warm-cache PR run before claiming the cache improved latency.
- The two slow tests and documentation examples still need to pass in the
  revised downstream job before the duplicate-suite removal is accepted.

## Commits

- `7e26d0e` — `ci: isolate JAX tests from xdist`
