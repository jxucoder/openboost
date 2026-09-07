# Sprint 070 protected selection on Linux

Original clean source: `a0bc47337a91ed1c58653820c1bb7ab7e4a96d3a`.
Container-only snapshot: `0801c1a1d73a5a3dfbf6e137d1c1e809f132617d` (clean).
The snapshot contains only 38 allowlisted public source/test/packaging files;
it is not the original repository history. The outer manifest records all hashes,
Modal image/version, exact invocation and requested resources.

## Observed results

- All three focused tests run on Linux and pass, including the previously skipped
  privilege integration test (`pytest.log`, 12.63 seconds).
- The additional retained selection bundle has 16 successful four-round fits,
  followed by validation-only audit, sealed receipt and fresh selected inference.
  Winner: `openboost:15`. Trial wall times range from 0.474 to 1.135 seconds.
- Actual known-path attempts to read held-back test features, modify the protocol,
  and read a completed trial model return PermissionError. Target hashes stay fixed.
- All five harness checks pass. Local inspection verifies 38 original source hashes,
  all 115 returned artifact hashes and the smoke's matching source hashes.

Each fit uses UID/GID 65534, an 8-GiB address ceiling, one thread and a configured
1800-second child deadline. The Modal function requests two CPUs, 8192 MiB and
180 seconds, with zero application retries. No preemption/retry was reported.
The input and labels are synthetic; no real test labels or sealed tasks were used.

## Cross-platform receipt counterexample

The same receipt fails exact re-audit on macOS. All non-score receipt fields and
the winner agree; 12 score values differ, at most 3.552713678800501e-15 in absolute
value. `cross-platform-recheck.json` records the values and rejection. This is
not a changed-file failure: all raw hashes match. Linux selection/release passes
within its environment; cross-platform receipt release is unresolved.

Reproduce by loading `selection/evaluator/protocol.json` and `selection/evaluator/
records.json`, then calling `benchmarks.v1.selection.audit` against `selection/`
and comparing with the saved receipt. `release_test` with the original protocol
and receipt pins rejects the macOS recomputation. Do not regenerate the receipt
or loosen its hash check to hide this counterexample.

## Scope and next action

This is synthetic selection integration, not full 300/1000-round search, real-data
quality, cost acceptance or a complete hostile-code sandbox. The receipt portability
counterexample needs an explicit numerical contract and regression before expansion.
Complete coverage, author accounting and full resource qualification remain open.

Re-run the bounded Linux experiment:

```bash
uv run --no-sync python -m benchmarks.v1.selection_preflight /tmp/openboost-linux-selection-070
```
