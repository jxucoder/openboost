# 2026-09-12: Bind checkpoint merge to installed Modal results

## Context

The user requested merging the curated foundation PR. Draft guards skipped CI,
but marking ready and main push would both launch heavy hosted CPU/build jobs,
contrary to the standing Modal-only compute direction.

## Decision or Result

Run the exact public candidate on bounded Modal workers and make hosted checks
verify its committed source-bound receipts. Preserve all original source-specific
historical tests; use the corrected current D2 trajectory consumers for the current
exact-ordering policy. The original production checkpoint bytes do not change.

## Changes

- The validation runner builds actual wheels and checks installed sources before
  numerical consumers. CPU phases and installed CUDA collection precede GPU use.
- The lightweight gate checks the entire tracked input inventory, retained raw
  artifacts and JUnit results. Missing or stale evidence fails.
- Current D2 tests consume the same run's independently checked Normal outputs;
  obsolete topology-only transfer assertions remain in historical tests.

## Verification

Construction verification: all 64 production files match the checkpoint; four
derived extension function bodies are unchanged; 441 GPU cases collect in 1.16 s
with none executed. Nine tiny receipt controls and source/test/helper Ruff checks
pass. The initial helper policy-path failure and overly narrow selector validation
were corrected before dispatch and are recorded in the construction artifact.

The [merge record](../v1-sprints/checkpoint-merge-validation.md) identifies the
runtime protocol and result index. Initial AST/lint, collection and tiny synthetic
receipt checks validate construction only. Consult actual receipts for numerical,
packaging, strict documentation and CUDA outcomes.

## Failed Attempts

The initial draft's skipped checks did not validate the candidate. Any remote
failure is retained under its one-use phase output; no automatic retry is allowed.

## Risks and Follow-ups

Any tracked input change requires refreshed validation. This intentionally bounded
manual Modal workflow is not a general cloud CI service. Public checkpoint merge
does not close full-budget train-many, remaining application evaluations or E5.

## Commits

This slice follows the curated public checkpoint commit `f143eb2`.


## First remote attempts

Retain the original `c95dc201` failures. Modal rejects Python 3.12 serialized
controller bytecode in a Python 3.10 image before worker creation. Match controller
and image versions, then select the tested child interpreter explicitly and record
both identities. The Python 3.12 CPU run passes 4,630 cases but exposes a historical
collection check that did not opt into the newly explicit historical selection.
Restore the original 383-case collection through the historical flag. Record the
optional SDK and unsupported-host skips precisely; do not promote them to passes
or install authoring infrastructure solely to remove a deferred control's skip.
