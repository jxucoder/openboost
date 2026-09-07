# Normal CUDA run 6: 381 passed, two acceptance failures

The single approved T4 run at clean
`4143d188b9635749308ef52382a1562928ae2612` completed all **383** preregistered
cases: **381 passed, two failed, none skipped or errored**. This is a failed
acceptance run. Its raw verdict remains false; no fixture or tolerance changed.

All 212 run-5 regressions pass. Normal operations pass 23/23, mapped runtime
94/96, recipes 31/31, and installed D2/fresh inference 20/20. The remaining
passing case is the separately labelled near-tie diagnostic; it establishes
measurement capture, not repaired structural parity.

The failures are `test_frozen_three_round_transactions` with
`[False-8.0-forward-ordinary-0-conflict-0-None]` and
`[False-8.0-reverse-ordinary-0-conflict-0-None]`. Both observed trial coefficients
`(8,4)` where the reference requires all six `(8,4,2,1,.5,.25)`. Full Normal
acceptance remains open. No rerun was made; all six device allowances are consumed.

## Unmodified raw evidence

- [Manifest](manifest.json): dispatch revision, exact commands, 67 uploaded source
  hashes, installed core/extension hashes, 18 pinned packages, environment and
  all 79 raw artifact hashes.
- [Verdict](verdict.json): exact case accounting and failing overall outcome.
- [JUnit](junit.xml) and [pytest output](pytest.log): both failures, passing cases,
  score/PTX diagnostics, Normal measurements and 407 warnings.
- [Saved inference artifacts](normal/): 76 JSON files for eighteen D2 trajectories
  and a separate missing-input Normal model, each with model, inputs, measurements
  and a fresh CPU replay without CUDA or the training extension.
- [Frozen protocol](../../../../v1-sprints/090-normal-run6.json) and
  [request](../../../../v1-sprints/090-normal-run6-request.md).

All 67 source hashes match Git at the dispatch revision; all 79 raw artifact
hashes verify. Recomputing the verdict reproduces the stored failure. Installed
core, extension and package versions match, the CPU environment built, and all
76 declared inference artifacts are present. Only the live protocol marks its
allowance consumed; the raw manifest retains the approved dispatch protocol.

Local archival checks replay all nineteen saved models and compare exact fresh
CPU predictions plus the frozen device-to-CPU tolerance. These do not rerun CUDA.
The detailed retrospective is recorded in Sprint 090 and the learning log.
