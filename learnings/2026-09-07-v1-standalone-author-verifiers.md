# 2026-09-07: Standalone author-verifier preparation

## Context

The [093 checkpoint](../v1-sprints/093-foundation-progress-and-next-steps.md)
prioritizes actual authoring evidence alongside the pending Normal CUDA run.
Existing D1 checks shared their process/directory with expected answers; the D2
entrypoint also required the unrelated D3 extension. Neither was a complete
standalone author-evaluation boundary.

## Decision or Result

Audit the real runner before more accounting design. `codex-cli 0.153.2` exposes
token usage events and goal budgets, but no verified mapping/enforcement for the
frozen generated-token limit. The [retained audit](../benchmarks/v1/evidence/author-runner-audit-094/README.md)
records this concrete gap. No model call, independent attempt or synthetic budget
test ran. Actual OS isolation remains unverified on this local macOS environment.

Separate mathematical export, known-extension observations and core-only judging.
Two D1 cases and nine D2 cases now have independent expectations for derivatives,
constrained growth and two rounds. The installed smoke runs the judge in a fresh
venv with only NumPy and the core wheel, without candidate/reference imports.
These are designer development results, not authoring-cost or complete task passes.

## Changes

- [Authoring tools](../benchmarks/v1/authoring/README.md): independent D1/D2 commands,
  explicit reference provenance, hashed evaluator files, strict observation
  structure and fresh saved-model inference. Runtime package files and wheels are
  retained by the installed smoke. Existing public production code is unchanged.
- Negative controls reject changed derivatives, missing rounds, reweighted cohort
  information, nonfinite/duplicate-key observations, wrong shapes, modified
  evaluator files, absent/symlinked models and models inconsistent with observations.
- [Documentation overview](../docs/v1/index.md): remove obsolete CPU-only wording
  and state the scalar hardware result, Normal failures and pending run-8 correction.
- [Sprint 094](../v1-sprints/094-author-verifier-preparation.md): local scope, acceptance
  and remaining independent-runner obligations. Original author/device cohorts stay intact.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/v1/test_author_verifier.py -n 0 -q`: 18 passed.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' --tb=short`: 1812 passed, one Linux-only skip, 16.49 seconds on macOS/Python 3.12.12. This duration describes a regression run, not a model speed result.
- Production/changed-support Ruff checks pass. `uv run --no-sync mkdocs build`
  succeeds with the existing execution-page link warning for evidence outside the
  documentation tree.
- Initial offline installed smoke passes all 11 cases with NumPy 2.3.5 and
  OpenBoost 1.0.0.dev0; a standalone command rejects the deliberate changed model.
  [Committed-source reproduction](../benchmarks/v1/evidence/author-verifiers-094/README.md)
  at clean `0a85320` passes all 11 cases, the fresh dependency-absence probe and the
  standalone wrong-model failure. All 36 retained artifact hashes verify; no files
  were changed during archive copying. No broader test rerun was needed after the
  documentation/evidence-only closure.
- All 85 source digests in the run-8 freeze match; the freeze file itself is
  unchanged. No upload, CUDA run, independent author or CPU search launched.

## Failed Attempts

Initial collector code assumed `TreeTerm.predict`, a shared `geometry` method and
a top-level `Model` export. Focused tests rejected these assumptions. The collector
now uses public `Model` prefix replay, the actual squared gradient/loss methods and
`openboost.artifacts.Model`. No production compatibility shim was introduced.

## Risks and Follow-ups

Hash checks need a protected manifest. Separate Python processes/venvs do not
deny an author's filesystem access, and unisolated observations are not trusted
execution evidence. Invalid-input/identity checks, core/private edit accounting,
appropriate incumbent arms, model/settings, enforced token/time budgets and the
actual isolation smoke remain required before independent attempts. Refresh the
author-view export in a new cohort; preserve cfca092 and sealed H1/H2.

## Commits

- `0a85320` — standalone D1/D2 development verifier, real runner audit, tests and docs.
- Subsequent evidence commit — clean-source installed reproduction and Sprint 094 reflection.
