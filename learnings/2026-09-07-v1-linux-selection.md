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
