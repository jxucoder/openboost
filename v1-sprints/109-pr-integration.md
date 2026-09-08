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
