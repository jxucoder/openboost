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

## Risks and Follow-ups

Use a merge commit: squash/rebase would replace execution SHAs needed by audits.
The broad branch includes deferred author-study infrastructure; its presence does
not authorize resuming studies. Required multiclass/AFT/vector topology, compatible
train-many, application quality and formal E4 remain open after this PR.

## Commits

- `c965cb6`: completed run-12 evidence and retrospective.
- Base `47108db`: remote main after merged PR 24.
