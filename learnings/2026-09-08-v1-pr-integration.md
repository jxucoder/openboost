# 2026-09-08: Prepare the validated foundation branch for PR review

## Context

The user requests another PR and preparation for merge after run 12 passes. The
branch contains 154 commits since PR 24; retained source SHA references make the
Git history part of reproducibility. GPU/model allowances remain unchanged.

## Decision or Result

Use the [109 integration plan](../v1-sprints/109-pr-integration.md). Create an
English PR for the full branch and prepare a merge commit after checks/review.
The current request authorizes the branch push and PR creation, not the merge.

## Changes

- Fetch full Git history in CPU CI so historical execution-source checks work in
  a clean checkout; select only CPU, non-benchmark tests on hosted runners.
- Replace the broken generated-docs link with the verified immutable GitHub
  evidence path at `abd5b1a`, preserving strict warning handling.
- Update the manual GPU workflow explanation to acknowledge the implemented CUDA
  foundation and its separately approved evidence packets. It still dispatches
  no hardware and cannot claim GPU verification from a hosted CPU job.

## Verification

Initial full production/test Ruff passes. The exact strict MkDocs command fails
before the link fix, identifying the smallest reproducible documentation blocker.
Source/workflow inspection identifies shallow history and GPU selection as CI
blockers. No production, frozen oracle or raw artifact changes are proposed.
Record completed local and hosted verification before marking the PR ready.

The final local serial CPU entry point passes 2,277 tests with one Linux-only skip
and 880 deselections in 95.04 seconds. Full production/test Ruff, strict MkDocs
and offline wheel/sdist builds pass. The exact failed docs command now passes
without suppressing warnings. The changed tracked text has no CJK prose or
credential-shaped literals. Hosted matrix results must be read from the new PR
after publication; local macOS 3.12 checks do not stand in for those platforms.

## Failed Attempts

The branch-protection API reports main is unprotected; this is repository state,
not an authorization or test failure. It does not remove the validation/review
requirements. Existing PR 19 is unrelated and remains untouched.

The first [hosted CPU run](https://github.com/jxucoder/openboost/actions/runs/34240154310)
at `7b18667` fails on Linux/Python 3.12: nine failures, 2,266 passes, three skips and
880 deselections. Two consumer cases incorrectly require the original host's
rounded loss dip; this host reports a tie. Seven old trajectory cases directly
use full-loss subtraction, bypassing the current recipe's objective comparison.
Documentation passes, while matrix fail-fast cancels the three sibling jobs.

Changing the old CPU trajectory test in place passes the current comparison
assertions but trips the run-12 replay source guard. Restore that source exactly;
archive identity is not relaxed to permit the change.

## Hosted counterexample correction

- Keep the original full-loss module and all frozen sources byte-identical. Add
  the current comparison cohort in a separate file, reusing the independent 092
  oracle and retaining all ninety gradient/split/leaf/prediction/metric settings.
- Default collection explicitly deselects those ninety historical full-loss
  trajectories; `--include-historical-normal` remains an opt-in historical probe.
  A real collection test checks the exact one-to-one parameter mapping and proves
  the original cases still collect with the flag. These are not hidden passes or
  expected failures; the active suite tests the current public contract.
- Run captured worsening through native reporting and injected lower/equal/higher
  scores, requiring real objective rejection and containment of the independent
  high-precision value. Reporting is a controlled test seam, never production code.
- Disable matrix fail-fast so failures retain independent observations from all
  supported host/version combinations. Document the historical/current distinction
  in `tests/README.md`.

The final focused command runs `test_comparison_consumers.py`,
`test_compared_normal_transactions.py`, `test_normal_cohort_collection.py` and
`test_glm_evidence.py` with `-n 0 -q --tb=short`: **121 pass in 26.13 seconds**.
This includes the unchanged run-12 source/retained-artifact replay and its tamper
controls. Full production/test Ruff and `git diff --check` pass.

Final serial regression (`pytest tests/ -m 'not gpu and not benchmark' -n 0 -q
--tb=short`) passes **2,284 checks**, with one platform skip and 970 deselections,
in 102.11 seconds. Ninety deselections are explicitly historical full-loss cases;
all ninety replacement settings pass. Strict docs also pass. The first failed
hosted run is preserved; the correction still needs a complete hosted matrix.

## Risks and Follow-ups

Use a merge commit: squash/rebase would replace execution SHAs needed by audits.
The broad branch includes deferred author-study infrastructure; its presence does
not authorize resuming studies. Required multiclass/AFT/vector topology, compatible
train-many, application quality and formal E4 remain open after this PR.

## Commits

- `c965cb6`: completed run-12 evidence and retrospective.
- Base `47108db`: remote main after merged PR 24.
