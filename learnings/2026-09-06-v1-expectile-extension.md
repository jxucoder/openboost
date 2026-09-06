# 2026-09-06: Author an expectile objective outside the foundation

## Context

Sprint 038 M2 still required D1 public objective authoring. Parent b7fdb60.
D1 is intentionally an incumbent-friendly control, not an assumed advantage.

## Decision or Result

A separate ob-expectile wheel implements analytic geometry, weighted initialization
and a small loop entirely through public operations. No production changes were
needed. Initialization uses bisection, independently checked by stationary-interval
enumeration. Unweighted derivatives enter newton, which applies sample weights
once. Offsets enter geometry and initialization but are excluded from raw artifacts.

## Changes

- [Package](../examples/v1_extensions/expectile/README.md), independent trace
  generator, installed checker and five focused source tests.
- Installed verifier now builds five wheels and removes four training plugins
  before checking nine saved raw models in another interpreter.
- [Sprint 043](../v1-sprints/043-expectile-extension.md) and
  [raw evidence](../benchmarks/v1/evidence/expectile-043/README.md).

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- `uv run --no-sync pytest tests/v1/test_public_expectile.py -q -o addopts=''`:
  initial test failed because the package did not exist; implementation resolves it.
- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`:
  799 passed, including five new checks.
- `uv run --no-sync ruff check src/openboost examples/v1_extensions tests/v1/test_public_expectile.py`:
  passed; `uv run --no-sync mkdocs build --strict`: passed.
- `uv run --no-sync python examples/v1_extensions/verify.py /tmp/openboost-v1-expectile-043`:
  all installed extension, scheduling and plugin-free inference checks passed.
- macOS x86_64/Python 3.12.12/NumPy 2.3.5; no CUDA or real-data claims.

## Failed Attempts

The initial missing-module failure establishes the absent extension, not a
foundation limitation. No mathematical mismatch or core revision was required.

## Risks and Follow-ups

Loop wiring remains author-owned and duplicated; do not introduce a general
trainer without further evidence. The fixed-step example has no line search and
claims only its explicit signature. Raw prediction callers supply offsets.
Continue remaining D5 development and real worker integration. Formal E2/E5/E6,
quality/cost, CUDA and adoption remain open. No push or publication.

## Commits

- This D1 extension and verification slice; parent b7fdb60.
