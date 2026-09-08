# Sprint 109: PR integration checkpoint

Status: local integration checks complete; publish and inspect the PR for branch
`codex/v1-retrospective-plan`. Live hosted-check and merge status belong to that PR;
merge is not yet requested.
This delivery checkpoint follows the completed 108 retrospective and precedes
further multiclass construction. No GPU upload, model study or release is included.

## Scope and plan

1. Compare the current branch to the latest remote main, inspect existing PRs and
   merge rules, and review production/evidence/workflow boundaries. Preserve all
   execution commits and raw artifacts. The starting branch has 154 commits since
   merged PR 24, covering the resident CUDA foundation, reliable comparisons,
   bounded squared/Normal/GLM validation, field-validation cost evidence and the
   recorded deferral of author studies.
2. Reproduce the CI entry points before publication. Correct demonstrated workflow
   and documentation blockers without altering numerical code, frozen test cases
   or historical evidence. Verify CPU regression, full production/test lint,
   strict documentation and packaging. Record changes in learnings and commit.
3. Push the branch and open an English PR summarizing the whole change, evidence,
   remaining required scope and merge strategy. Inspect GitHub checks and address
   actionable integration failures. Leave the PR ready for review; do not merge,
   enable auto-merge, publish a release or dispatch another hardware packet.

## Acceptance

- The PR targets current main with no conflict or missing raw artifact files.
- CPU CI explicitly excludes GPU/benchmark execution. Real CUDA evidence remains
  the frozen, separately authorized hardware runs; a hosted CPU job cannot pass it.
- CI fetches full Git history because evidence replay resolves original execution
  SHAs. Use a merge commit to preserve those SHAs; squash/rebase merging would
  break historical source provenance and future clean-checkout replay.
- Public docs build strictly, and workflow labels describe current capabilities.
- Required local checks and the PR's configured hosted checks pass, or a concrete
  remaining blocker is recorded without calling the PR ready to merge.
- Preserve all R/C/A requirements and deferred author studies. The PR adds bounded
  foundation evidence, not full v1 completion, external-library parity or E4.

## Initial findings

Remote main is `47108db` (merged PR 24), with no newer commits and no PR for this
branch. Repository merge commits are allowed. The branch adds 2,086 changed files,
including 1,695 evidence files and 24 production modules; raw artifact volume must
not obscure the smaller executable-code review boundary.

Full production/test Ruff passes. The exact strict docs command fails on the
known Normal run-6 evidence link, which points outside the generated docs tree.
CPU CI currently uses a shallow checkout and selects real GPU and benchmark cases
on hosted CPU runners. Correct all three before the new PR. The manual GPU workflow
also incorrectly says no CUDA implementation exists; retain its non-executing
boundary while updating that explanation.

## Local integration result

The exact serial CPU workflow command passes 2,277 tests with one Linux-only skip
and 880 GPU/benchmark deselections in 95.04 seconds. Full production/test Ruff,
strict MkDocs and offline wheel/sdist builds pass. The broken evidence link now
targets its verified immutable repository revision; no warning is suppressed.
All source changes in this slice are workflow/documentation changes, with no
alteration to production or frozen numerical/persistence evidence. Changed text
contains no CJK prose or credential-shaped literals. Hosted Linux/macOS and Python
3.10/3.12 checks remain required observations on the published PR.

## Hosted numerical counterexample and follow-up

[PR 25](https://github.com/jxucoder/openboost/pull/25) is open. The first Linux
3.12 job reports nine failures, 2,266 passes and three platform/privilege skips;
matrix fail-fast cancels its siblings. Documentation passes. Two consumer tests
assume that re-evaluating captured inputs reproduces a rounded loss decrease on
every host; Linux reports equal losses. Seven trajectory failures use the obsolete
`full_loss_after < full_loss_before` rule in the test itself, although the current
Normal recipe explicitly uses `Normal.compare`.

1. Replace the consumer's host-specific reporting assumption with controlled
   lower/equal/higher reports plus native reporting. Keep the captured input bytes,
   independent high-precision worsening and actual comparison rejection checks.
2. Add a separate revised CPU trajectory cohort and exercise objective
   comparisons against the existing independent revised oracle for all ninety
   settings. Preserve gradient, split, leaf, prediction and metric assertions.
   Keep the original module byte-identical: default collection deselects its ninety
   historical full-loss trajectories, with `--include-historical-normal` available
   for explicit historical study. A collection check proves all ninety parameter
   settings have active replacements. No frozen source, oracle, raw verdict or
   production code changes.
3. Re-run focused tests, archival audits and CPU regression, then push the cohesive
   fix. Disable matrix fail-fast so each supported host/version yields a result.
   Do not call the PR ready until the hosted matrix passes.

An initial in-place test revision passed the new trajectory assertions but failed
the offline run-12 source-identity guard. That attempt was reverted. The final
layout keeps the full consumed module intact and puts the revised CPU cohort in
`tests/v1/test_compared_normal_transactions.py`. No historical result is rejudged
as a current pass, and no tolerance or comparison policy is weakened.

Final local verification passes 121 focused comparison/collection/archive checks
in 26.13 seconds, followed by 2,284 CPU regression checks with one platform skip
and 970 deselections in 102.11 seconds. The deselections comprise 880 GPU/benchmark
cases and ninety historical full-loss trajectories, each with an active revised
counterpart. Full production/test Ruff, strict docs and whitespace checks pass.
Hosted checks must now verify the pushed correction on all four matrix cells.

## Reporting replay on hosted macOS

At `8df852c`, both Linux matrix cells pass. Both macOS cells pass the current
comparison cohorts but fail the exact JSON trajectory replay: the same four
`best_score`/`validation_score` fields differ by one ULP, from
`2.3308416471007947` to `2.3308416471007942`. All non-report fields match exactly;
each macOS cell reports 2,282 passes, two platform/privilege skips and one failure.

Keep the original trajectory artifact at `c881259` byte-identical and explicitly
pin its SHA256 in the replay test. Separate that integrity check from recomputed
reporting values: allow at most one float64 ULP for `loss`, `validation_score` and
`best_score` only, require finite values, and keep every decision, coefficient,
prefix, setting, initial value and source hash exact. The old artifact and all
production comparisons remain unchanged. Ten focused checks pass, including
one-ULP replay and failures for two ULPs, nonfinite reports, changed acceptance,
prefixes, coefficients and sources. Verify the full hosted matrix again.

Final local regression passes 2,292 tests with one platform skip in 18.23 seconds
using the configured parallel CPU suite. Full lint passes. Commit the correction
and let the PR record the next complete hosted matrix; leave the actual merge to
the user's next instruction.
