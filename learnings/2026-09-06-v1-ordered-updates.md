# 2026-09-06: Ordered updates compose; external result interoperability remains open

## Context

Sprint 038 M2/D4 requires changed parameter order and adaptive acceptance through
public operations. Parent 6b2385f. The built-in Normal and Formula recipes update
both channels jointly; more built-ins alone would not establish external authoring.

## Decision or Result

A separately installed package composes Normal ordinary/Fisher and Formula full
GGN through the same objective-independent ordered loop. Each parameter observes
the latest accepted state, with six bounded rates and finite strict descent.
StopState advances once per outer round; validation chooses best snapshots after
each accepted parameter. No core edits or private imports were necessary.

run_many rejects the extension's OrderedResult despite matching run/problem state.
This is a concrete shared-result abstraction gap, not an algorithm failure. Keep
it as a D5 follow-up before claiming complete E2/CPU expressiveness. Do not add a
special-case scheduler branch or count this rejected run as successful scheduling.

## Changes

- examples/v1_extensions/ordered_updates/: installed public ordered-sweep package.
- ordered_oracle.py / ordered_checks.py: separate reference generation and isolated
  installed comparison, including the scheduler counterexample.
- verify.py / core_inference.py: four-wheel builds and eight raw model roundtrips
  after uninstalling all three plugins; reference/source/artifact hashes retained.
- tests/v1/test_public_ordered.py: sixteen ordered/rejection/stopping checks.
- [Sprint 041](../v1-sprints/041-ordered-updates.md) and
  [raw evidence](../benchmarks/v1/evidence/ordered-updates-041/README.md).

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- `uv run --no-sync pytest tests/v1/test_public_ordered.py -q -o addopts=''`:
  fourteen passed before adding partial-rejection/recovery cases; all sixteen
  pass in the full regression.
- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`:
  783 passed.
- `uv run --no-sync ruff check src/openboost examples/v1_extensions tests/v1/test_public_ordered.py`:
  passed; `uv run --no-sync mkdocs build --strict`: passed.
- `uv run --no-sync python examples/v1_extensions/verify.py /tmp/openboost-v1-ordered-041`:
  installed development checks and eight fresh-process raw roundtrips pass.
  Maximum installed raw/reference difference: 2.220446049250313e-16.
- macOS x86_64, Python 3.12.12, NumPy 2.3.5, one BLAS/OpenMP thread for installed
  checks. Core wheel matches Sprint 039. No CUDA, quality or performance result.

## Failed Attempts

- Initial extension import failed before implementation.
- The first reference comparison used the reference helper's default step sizes,
  while the extension correctly used D4's fixed six rates. Bound rates explicitly.
- A depth-one extension learner then disagreed with D4's depth-two reference;
  aligned the extension default to the declared reference task. No oracle or
  tolerance relaxation was used to mask those configuration differences.
- Installed run_many rejects OrderedResult; preserved in raw evidence and next plan.

## Risks and Follow-ups

Substep diagnostic records retain full immutable before/after states and are not
memory optimized or resumable checkpoints. Geometry's returned loss and the search
loss callback must describe the same objective. Raw inference passing does not
complete Normal/Formula output/dependency or real-data workflows. Next D5 result
interoperability, D1/D5 installed probes and real worker integration. E5/E7 remain
unverified; no push, publication or external contact.

## Commits

- This development slice; parent `6b2385f`.
