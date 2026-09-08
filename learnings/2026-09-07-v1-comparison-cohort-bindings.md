# 2026-09-07: Bind revised comparison conformance before hardware

## Context

After `2c5927c`, all three consumers were constructed, but the 27 added CUDA
distinctions did not replace the original 383 required cases. The user approved
completing bindings and preparing the next hardware freeze, without authorizing
an upload or another invocation.

## Decision or Result

Keep every old source, outcome, tolerance and the 092-A historical mapping intact.
A separate original-row loop uses the already frozen comparison mathematics for
training and validation best. Across all ninety original settings its complete
reference summaries match the historical float64 trajectories exactly. This
does not predict a GPU pass: stored float32 inputs and device execution still
require verification. The two known failing settings still require six rejections
per substep in the new independent reference.

## Changes

- `tests/v1/reference/compared_normal.py` reuses the frozen geometry and exhaustive
  row-based grower while changing the loop's comparison policy and best anchor.
- Three explicitly named revised CUDA cohorts retain 96 transaction, 31 recipe
  and twenty installed-D2/inference cases with their original parameters and
  metric/raw/leaf tolerances. Recipe retained bytes include the owned best raw.
- `tests/v1/comparison_audit.py` checks actual completed device comparisons against
  the original-row interval and high-precision expressions. It records raw bits,
  prepared inputs and operation counters; partial records survive failed tests.
  Its exports/synchronizations make these timings diagnostic, not product timing.
- The new 383-case collected binding file preserves the original mapping hash.
  The 404982-byte reference artifact retains both summaries for all ninety cases.

## Verification

- The new reference test initially failed on its missing implementation. The
  binding tests initially failed on missing binding/reference artifacts.
- 101 initial reference/audit checks pass; the added equal-reporting best case
  verifies that the new reference loop actually exercises independent best.
- Exact collection finds all 383 bound cases: 236 unchanged plus 147 revised.
  The two binding/source-preservation tests pass. CUDA cases remain unrun.
- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short` — 1757 pass, one Linux-only skip.
- Ruff passes production and all new Python files; `git diff --check` passes.

## Failed Attempts

A first controlled best test passed a list to the frozen array-only geometry
oracle. It was corrected to supply its documented ndarray input; the oracle was
not modified. No failure prompted a tolerance, original-case or source change.

## Risks and Follow-ups

Device comparison lowering, all collected CUDA cases and fresh installed replay
remain unverified. Complete the separate historical/revised verdict protocol,
bounded retained artifacts, lowering observation and uninstrumented fit-cost
checks before requesting one concrete allowance. No push or hardware invocation.

## Commits

- This entry accompanies the revised-cohort construction and binding commit.
