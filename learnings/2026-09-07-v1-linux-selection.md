# 2026-09-07: Validate protected selection on Linux

## Context

The protected selection call path is implemented at `176d556`, but its actual
Linux test has only been skipped locally. Standalone UID probes are insufficient.

## Decision or Result

Use a bounded allowlisted Modal harness with installed public CPU code. A new
snapshot Git commit is explicitly distinct from the original source revision;
both identities and source hashes remain observable. Retain process failures.

## Changes

- Run the focused Linux tests and preserve an additional real 16-trial bundle.
- Attempt protected feature/model reads and protocol modification after selection.
- Keep synthetic scope and one-thread recipe execution explicit.

## Verification

Source lint passes before the clean remote run. Actual results follow the harness
commit; no new Linux selection success is claimed at this point.

## Failed Attempts

None yet. Local macOS cannot verify privilege transitions.

## Risks and Follow-ups

No full search, real quality gate, network sandbox or independent author evidence.
The 1800-second child policy is configured, not observed expiring in this short run.

## Commits

- Bounded Linux harness; parent `176d556`.

## Actual results and counterexample

[Committed evidence](../benchmarks/v1/evidence/protected-selection-070/README.md):
three Linux tests pass; the extra 16-trial protected search and selected replay
pass. All five checks pass and 115 returned hashes/38 original source hashes verify.
The container snapshot has separate literal provenance. Local CPU regression is
1021 passed, one Linux-only skip; docs and lint pass.

Cross-platform local re-audit rejects the unchanged Linux receipt because 12
score values differ by up to 3.552713678800501e-15. Winner and all non-score fields
match exactly. The raw diagnostic is preserved. The exact receipt check remains
unchanged: define its numerical portability contract and test tampering/near-tie
cases before deciding on a fix. This newly exercised boundary takes priority over
expanding the search. Do not mistake fail-closed rejection for a quality failure.
