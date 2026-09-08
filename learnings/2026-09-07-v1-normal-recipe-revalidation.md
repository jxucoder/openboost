# 2026-09-07: Bounded revalidation of the Normal CUDA recipe correction

## Context

Run 8 at `469ca0e` has 528/529 revised passes. The single failure is a fixture
expectation, corrected at `6026ebb`: forward order has ten accepted terms but
best prefix nine because the tenth term is zero. Later assertions in that case
were not reached. Local semantic coverage cannot substitute for real execution
of the corrected device test. All eight existing GPU allowances are consumed.

## Decision or Result

Prepare [Sprint 103](../v1-sprints/103-normal-cuda-revalidation.md) for the full
fifteen-case recipe file, covering the corrected forward case, joint/reverse
siblings, tiny improvements, rejection and ownership/failure handling. Reuse the
existing installed-source single-cohort runner. Production is unchanged, so a
new two-cohort 914-execution run is unnecessary to resolve this fixture failure.
Report any combined evidence explicitly; preserve the raw failed result.

## Changes

- A small dispatch entry point selects the new pending protocol and fixed output.
  The protocol freezes 46 uploads (about 1.01 MB), the unchanged eighteen pinned
  packages, fifteen case IDs and one proposed T4 invocation with existing runner
  limits: two CPUs, 8192 MiB, 900 function seconds, 600 test seconds, zero retries.
- An offline checker verifies the earlier manifest/verdict hashes and 514 other
  passing cases, exact production identity, the corrected fixture's commit/hash,
  and all reused source identities. The protocol does not regrade run 8.
- The existing local wheel-collection helper accepts an explicit protocol path;
  its default run-8 behavior remains unchanged. This helper is not part of run 8's
  frozen upload. The new check collects the exact copied closure from an extracted
  offline wheel. No environment mutation, network request or device test occurs.
- Current guidance records this pending packet. Agent/model studies stay paused.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.check_recipe_run9`:
  46 frozen sources verify, all fifteen isolated recipe cases collect, zero execute.
  [Report](../v1-sprints/103-isolated-collection.json) records all source hashes,
  installed wheel identity, prior evidence and local Python version.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_cuda_aggregation_manifest.py tests/v1/test_cuda_normal_manifest.py tests/v1/test_cuda_comparison_manifest.py -n 0 -q`:
  57 pass in 0.33 seconds. Existing guards cover missing, duplicate, skipped/error
  cases, provenance mismatches and unapproved dispatch.
- Production and all three changed support scripts pass Ruff. Full CPU coverage
  remains `6026ebb`'s 1949 passes and one Linux-only skip; this slice changes no
  production or recipe assertions.
- The actual pending dispatch CLI exits before importing Modal or creating output.
  The raw run-8 analyzer still verifies 416 artifacts and 86 measured sources;
  all 420 archive-index hashes remain unchanged. New Markdown links resolve,
  whitespace checks pass and MkDocs builds with its existing 090 evidence-link
  warning. No source from the consumed upload is modified in this slice.

## Failed Attempts

No new external invocation is attempted. The earlier automatic approval rejection
and subsequent explicit run-8 approval remain in the 102 record. A general
continuation prepares this separate packet; it does not reuse consumed authority.

## Risks and Follow-ups

Run 9 remains unexecuted; neither local collection nor 57 CPU harness checks pass
its device gate. Approval must cover the concrete 46-file Modal upload and the
single bounded GPU invocation. Image construction is outside the function deadline.
Retain any failure and stop for retrospective after execution. Required CUDA
recipes, compatible train-many, practical quality/cost and all R/C/A scope remain.
No push, model request or additional invocation is authorized by this preparation.

## Commits

- `c36f96a` preserves run 8's raw failure and analysis.
- `6026ebb` corrects the fixture and adds CPU no-op semantic coverage.
- This commit prepares the independent run-9 packet; no hardware result is claimed.

## Explicit approval after preparation

The user replies "approve" to the concrete request after `7fe4937`. This authorizes
the 46-file Modal upload and one T4 invocation with two CPUs, 8192 MiB, a 900-second
function cap, 600-second test cap and zero retries. Only the protocol's two pending
authorization fields become approved; all prefrozen sources and case/resource
constraints remain identical. The original isolated collection report is preserved.
Verify the dispatch guard and source closure locally, commit a clean execution
revision, then execute once and retain failures or success before retrospective.
The local approved dispatch guard passes with all 45 prefrozen source hashes
unchanged and the fixed output absent. The same 57 harness checks pass in 0.36
seconds before the authorization commit.
