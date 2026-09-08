# 2026-09-07: Current author packet and a native isolation counterexample

## Context

After approval to continue [094](../v1-sprints/094-author-verifier-preparation.md),
remaining work included current author materials, invalid-input/identity checks
and actual access denial. The original export had unresolved documentation links
and no check that its wheel matched the public source tree.

## Decision or Result

The refreshed author view copies explicit CPU materials, records rather than
follows omitted links, checks wheel contents and retains incomplete builds.
Verifier v2 preserves the two D1/nine D2 numerical cases and adds rejection
observations. Correct numerical output cannot compensate for permissive invalid
tau or foreign-problem behavior in the tested development adapters.

Native macOS sandboxing works outside the tool's existing nested sandbox. Its
first profile passed six installed documentation examples, eleven access/network
denial probes, a forced wall timeout and unchanged protected hashes. An unlisted
copy of the actual expected answers remained readable, disproving broader isolation.
The stronger read-allowlist aborts this x86_64 Python with exit -6 before its first
print. Bounded system runtime-path probes did not resolve this; no root cause such
as Rosetta is established. A PID-specific log query returned no matching entries.
Stop speculative policy widening and retain the failure. No author was dispatched.

## Changes

- Author-view builder: original/delivered hashes, omitted-link ledger, offline
  diagnostics, wheel source closure and current evaluator-tool hashes.
- Standalone verifier: explicit v2 rejection observations; false/missing values fail.
- Experimental native file probe: copied actual inputs, child/alias denial checks,
  real timeout and retained startup failures. Installed files use copy mode to
  avoid mutating shared package-cache hardlinks during negative controls.
- Squared documentation: current incremental trial evaluation, validation patience
  and the actual full/summary trace options.

## Verification

- Focused packet/verifier/supervisor tests: 31 passed, including permissive
  extension failures, missing dependency/build failure and real subprocess timeout.
- Full CPU suite: `UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' --tb=short -q`:
  1825 passed, one Linux-only skip in 14.05 seconds. Ruff passes. MkDocs builds with
  the existing execution-page external-evidence link warning. All 85 run-8 source
  hashes match; no production source changed.
- First native policy passes its thirteen declared checks; an additional unlisted
  answer read succeeds. Stronger policy fails during the first positive example.
- Committed-source reproduction, final regression and artifact hashes are recorded
  at closure. Original author/device archives and the run-8 freeze stay unchanged.
- Marker correction: all 11 packet tests pass; the existing installed wheel's
  31 Python modules and exact source marker pass the corrected audit. Ruff passes.

## Failed Attempts

- The initial wheel audit at `40dd684` rejected the legitimate `openboost/py.typed`
  marker. The wheel-level positive check exposed this after the initial focused
  suite. Verify the marker against source bytes and add positive/missing/changed
  marker regression cases; do not broadly allow arbitrary package data.
- Nested sandbox application failed with Operation not permitted; the standard tool
  escalation allowed the no-op capability check and bounded local experiments.
- Denying only named evaluator locations misses other copies. A manifest hash is
  insufficient when the author can read a previous answer copy elsewhere.
- A stricter profile aborts Python startup. Interpreter aborts cannot count as
  successful access denials: usable positive execution is also required.

## Risks and Follow-ups

This is not a complete hostile-code boundary. Worker timeout is not full author
time accounting. Generated-token semantics/enforcement, core/private edit tracking,
fair incumbent paths, model/settings and a supported isolated environment remain
open. Decide that environment before more native-policy work. No model, GPU,
upload, independent author result or cost advantage was produced.

## Commits

- `40dd684` — packet/rejection/probe construction; the subsequent marker correction
  is required before the clean packet export can pass.
