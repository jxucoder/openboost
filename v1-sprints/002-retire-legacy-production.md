# Sprint 002: Retire old production code and reset v1 implementation

Date: 2026-09-05. Starting revision: `50acfc6`. Status: **this sprint complete; F0.2 ongoing**.
Trigger: the user explicitly selected retiring old production code and rebuilding v1.
This moves retirement forward from F5 at the user's request; F0.2/F0.3/F1 dependencies,
all required cases and acceptance gates remain unchanged.

## Plan and boundaries

1. Remove old trainers/models/core/backends/experimental/distributed implementations from
   `src/openboost/`. Recreate only an under-construction namespace and typing marker, with no API shim.
2. Preserve `tests/v1/`, historical mathematical tests, raw benchmark artifacts, learnings and designs.
   Default tests cover v1 only; historical tests require a fixed revision and are not current passes/skips.
3. Update README, package description/dependencies, test discovery, CI and documentation entry points.
   Retired examples cannot advertise current APIs. Retire GPU/release workflows; an oracle-only namespace is not a releasable product.
4. Check imports/package contents, 55 reference tests, lint, builds and documentation; record reflection.

## Acceptance

- No old training/model/device modules remain; only the new namespace imports.
- Complete old sources remain at `50acfc6` and earlier; historical experiments and tests/v1 are preserved.
- Default tests and `pytest tests/` do not depend on old production; results describe a v1 subset only.
- Wheels exclude old modules; README/docs do not advertise removed runnable training APIs.
- Retirement adds no algorithm/performance claims and does not complete F1 or any E-gate.

## Verification

The old package had 47 tracked files: 46 Python modules and one typing marker.
After retirement, only recreated `__init__.py` and empty `py.typed` remain. Bytecode/Numba
caches were removed. Version became `1.0.0.dev0` so an empty namespace is not described as the old RC.

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv lock --check --offline
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv build --offline
```

- **55 passed, no skips**. Default discovery excludes historical suites from the count.
- Ruff, lock check and strict docs passed; sdist and a wheel built from sdist succeeded.
- Wheel contains only two package files. `python -I` imports directly from it; no old
  trainer/model/core/backend/distributed/experimental modules can be discovered.
- Fourteen changed Markdown files, 76 local links, fences and `git diff --check` passed.
- YAML structure checks passed for four workflows. Old GPU/release entry points retain only
  explicitly failing manual status messages; no scheduled GPU or automatic release remains.
  Docs workflow builds only `docs/v1/`, without deployment.
- `git diff HEAD -- benchmarks tests/v1` was empty: neither historical benchmarks nor committed references changed.
- Initial offline locking failed because cross-Python metadata was not cached. Online locking
  succeeded, then offline checking passed with 96 resolved packages; no hashes were edited to bypass failure.
- These are local checks; GitHub CI matrix, CUDA, real quality and formal E-gates were not run.

## Reflection

The user wanted to leave the old production structure behind, rather than maintain a transition.
The 55 independent references were committed; old code still included global backends, old
weight/gain conventions, multiple model/experimental entry points and out-of-scope distributed code.
Decision: retire production as one slice, preserve counterexamples/evidence, and explicitly state
that there is temporarily no training API. Return to remaining F0.2 work; cleanup is not construction.

After removal, all 55 references run unchanged. The reference diff is empty; isolated imports
and default tests pass. Preserve this independence. Future conformance requires separate
comparisons, not packaging the reference as an allegedly optimized product. Next: typed data,
binning/category/classification, then remaining mathematics and state/run probes. No application
implementation or E1/E3/E4 gate passed through retirement.

## Commits

- `50acfc6`: last revision containing full old production and v1 independent references.
- Retirement slice: `refactor: retire legacy production code for the v1 rebuild`.
