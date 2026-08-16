# 2026-08-15: Performance Regression Gate

## Context

The performance CI expected `benchmarks/results/performance_baselines.json`,
but that file was ignored and absent from fresh checkouts. On a missing file,
the script benchmarked the current commit, saved that same result as its own
baseline, and exited successfully. The advertised regression gate was therefore
a no-op on every fresh GitHub runner.

## Decision or Result

Compare the code before a push with the code after the push on the same GitHub
runner instead of committing an absolute timing baseline from unrelated
hardware. The baseline revision is `github.event.before`, not `HEAD^`, so one
workflow covers every commit in a multi-commit push to main.

Both revisions run through the current fixed harness with separate Numba cache
directories. Raw parent/current JSON files are uploaded and include commit,
runtime versions, platform, backend, and relevant Numba environment fields.
Missing baselines now exit with status 2 before doing expensive work; creating a
local baseline requires an explicit flag.

## Changes

- `benchmarks/check_performance.py`: add explicit baseline/output/source-root
  options, benchmark-only mode, provenance, and fail-closed missing-baseline
  behavior.
- `.github/workflows/unit-tests.yml`: fetch history, create a detached worktree
  at the previous remote main, benchmark old/current code on one runner with
  isolated caches, and upload both artifacts.
- `tests/test_performance_check.py`: cover equal results, runtime/quality
  regressions, explicit baseline loading, and provenance.

## Verification

- Missing-baseline CLI check exited 2 in 0.2 seconds and created no baseline.
- Unit suite: 4 passed.
- Ruff and workflow YAML parsing: passed.
- End-to-end temporary-worktree simulation:
  - baseline commit `6440e31143e4cbd56e9d523a25a8f48bca302670`;
  - current commit `077211066af1956f38e2a7183cd77bdd0ace140c`;
  - both artifacts contained provenance and comparison returned no regressions.
- The temporary worktree was removed after verification.

## Failed Attempts

- A committed baseline generated on this Intel macOS host was rejected as a CI
  design because absolute timings are not portable to GitHub's Linux runners.
- Comparing only `HEAD^` was rejected because a push containing several commits
  would test only the final commit. `github.event.before` represents the actual
  remote-main baseline for the pushed range.

## Risks and Follow-ups

- Shared hosted runners remain noisy. The current median-of-three and 20%
  threshold are a regression alarm, not publication-quality performance proof.
- Parent and current code use dependencies installed from the current checkout;
  this isolates source regressions but does not detect dependency-only speed
  changes.
- The workflow must run on GitHub once to validate hosted-runner behavior and
  artifact upload. External ScoringBench results remain the value proof; this CI
  microbenchmark is maintenance infrastructure.

## Commits

- `30b9ab5` — `ci: compare performance across pushed revisions`
