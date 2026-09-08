# 2026-09-06: Verify installed custom completion and independent runs

## Context

Sprint 065 follows structural stopping commit `2871f0f`. Source checks alone did
not demonstrate the common contract outside the checkout or verify all planned
RNG and stale-preparation cases. Existing behavior was tested before considering
any additional core changes.

## Decision or Result

All installed probes pass without core edits. Keep the existing public preparation,
run identity and structural result boundaries. The new evidence closes N1's
development checks, not formal authoring, real selected quality, GPU or adoption.

## Changes

- Preserve original M=1/8/32 patience checks; add custom/built-in/ordered schedules,
  distinct keyed streams, malformed-stop recovery and all three preparation cases.
- Copy the genuine threshold loop into the isolated check directory, then remove
  it before fresh inference. Add its model to the nine plugin-free models.
- Record production source hashes alongside existing verifier/reference/wheel hashes.
- Commit [raw evidence](../benchmarks/v1/evidence/scheduling-065/README.md), docs and
  Sprint 065 reflection; no production module changes.

## Verification

Commands use `UV_CACHE_DIR=/tmp/openboost-research-uv-cache`.

- `uv run --no-sync python examples/v1_extensions/verify.py /tmp/openboost-v1-sprint065`:
  passed, fresh five-wheel build/install, Python -I outside checkout, installed D1–D4
  oracles, six M suites, mutation/fault probes, ten fresh inference models.
- `uv run --no-sync ruff check src/openboost examples/v1_extensions/scheduler_checks.py examples/v1_extensions/verify.py examples/v1_extensions/core_inference.py`: passed before execution; formatting applied.
- Independently verified all 17 artifact hashes and all verifier/reference/core
  source hashes; checked M=1/8/32 coverage, exact threshold recurrence predictions,
  injected ValueErrors and preparation acceptance [false, false, true].
- Strict docs, final Ruff and local link/diff checks passed at commit closure.
  The last full CPU suite is Sprint 064's 943 passes; no core changed afterward.

The manifest records parent revision plus dirty verifier state; source hashes match
the committed implementation. The data are deterministic synthetic fixtures. Offline
dependencies were available; this proves only the declared CPU installation environment.

## Failed Attempts

Ruff caught an unbound loop variable in a fault-injection closure; bound it before
running installed checks. An edit patch failed context matching and was reapplied
without changing scope. No core semantic failure occurred in the installed run.

## Risks and Follow-ups

Installed internal checks do not measure independent author effort. Preparation
ownership is covered for these mutations, not arbitrary hostile callbacks. The
next uncertainty is practical CPU runtime/memory under enforced budgets (066).
Retain the profile-first dependency before incremental execution and trace changes.

## Commits

- Installed isolation verifier and evidence; parent `2871f0f`.
