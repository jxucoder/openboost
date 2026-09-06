# 2026-09-06: Share result semantics without prescribing diagnostic record classes

## Context

Sprint 041 found run_many rejecting OrderedResult despite valid context/problem
state. Parent b99376d. The ordered package's nested parameter records should not
require an objective-specific scheduler branch or a built-in trace record.

## Decision or Result

Introduce structural RecipeResult and validate its accepted-state identity,
completed stop metadata and one trace entry per completed outer round. Preserve
the original result and recipe-owned diagnostic content. Ordered parameter commits
need not match outer-round counts. Remove the scheduler's dependency on recipes.
Malformed, unfinished or foreign results fail their individual run.

This intentionally changes the foundation in response to exploratory evidence;
it cannot be counted as a frozen no-core-edit E5 attempt. The extension algorithm
source did not change. Earlier failed artifacts are preserved.

## Changes

- results.py: public protocol and validation; runs.py: use the shared contract.
- tests/v1/test_public_results.py: eleven cases covering the counterexample,
  invalid results and mixed built-in/ordered M=1/8/32 scheduling.
- Installed verifier: ordered equality and mixed shared-preparation, stopping,
  failures, permutation/regroup/retry and scoped-RNG checks.
- [Sprint 042](../v1-sprints/042-result-contract.md) and
  [raw artifacts](../benchmarks/v1/evidence/recipe-results-042/README.md).

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- Before editing, `uv run --no-sync pytest tests/v1/test_public_results.py::test_external_ordered_result_is_preserved -q -o addopts=''`
  failed with the misleading foreign-state ValueError for a matching external result.
- `uv run --no-sync pytest tests/v1/test_public_results.py -q -o addopts=''`:
  eleven passed after implementation.
- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`:
  794 passed.
- `uv run --no-sync ruff check src/openboost examples/v1_extensions tests/v1/test_public_results.py`:
  passed. `uv run --no-sync mkdocs build --strict` and `uv build --offline`: passed.
- `uv run --no-sync python examples/v1_extensions/verify.py /tmp/openboost-v1-results-042`:
  isolated four-wheel installation, all development checks, and exact inference
  of eight models after removal of all plugins passed.
- Verified recorded source/reference/artifact hashes against the final files.
  macOS x86_64, Python 3.12.12, NumPy 2.3.5; one BLAS/OpenMP thread for installed checks.

## Failed Attempts

The first focused test reproduced the concrete-class rejection. No compatibility
shim or external result conversion was added; the shared contract addresses the
dependency directly. Historical Sprint 041 evidence retains the rejected result.

## Risks and Follow-ups

Runtime structural validation does not isolate arbitrary Python callbacks or copy
their diagnostic objects. Authors must honor input/state ownership. This is a
completed in-memory result contract, not a resume/serialization format. Next D1
and remaining D5 author probes, with real worker integration; formal E2/E5/E6,
GPU cost and external adoption remain unverified. No push or publication.

## Commits

- This result-contract implementation; parent `b99376d`.
