# 2026-09-05: Refocus value evidence and repair benchmark configuration

## Context

The user asked to review the bigger impact/adoption/value goal and continue.
The latest small GPU optimization leaves a 12.888x experimental/legacy fit ratio.
Continuing internal optimization alone does not validate a useful product.

## Decision or Result

Pause general GPU foundation optimization as the immediate queue. Prioritize
trustworthy external quality evaluation and independent-author use, then a real
exposure-aware risk case. Preserve the experimental API and all negative data.
See [next investment gate](../planning/impact-adoption-value-next.md).

## Changes

- ScoringBench CLI seed now reaches OpenBoost and XGBLSS; NGBoost receives depth
  and seed through a fresh clone of its installed default tree learner per factory.
  Previously OpenBoost/XGBLSS omitted model seeds, and NGBoost ignored CLI depth.
- Provenance records constructed wrapper/base parameters, excluding private state,
  so the manifest can be audited beyond CLI intent. This is configuration evidence,
  not execution-device or benchmark-completion proof.
- Tests isolate optional upstream imports but exercise real OpenBoost fit/predict
  with row sampling; no heavy PyTorch import on unsupported Intel macOS.

## Verification

- Initial two tests failed on missing OpenBoost model_params and NGBoost Base.
- Four focused tests pass: all constructor seed/depth forwarding, independent
  cloned base learners, JSON configuration, and actual seeded OpenBoost fitting.
  Same model seed survives changed global NumPy state; changing the seed changes
  sampled predictions. Global test RNG state is restored.
- Inspected clean local ScoringBench a938a667b7839b41e9272929010573410301c0b4.
- NGBoost [default learner source](https://raw.githubusercontent.com/stanfordmlgroup/ngboost/master/ngboost/learners.py)
  confirms a clonable sklearn tree; clone the installed learner rather than
  guessing or hard-coding its other defaults. No upstream repository modified.

## Failed Attempts

- No complete Linux run attempted in this slice. Local constructor doubles do
  not establish actual competitor fitting or external metric integration.

## Risks and Follow-ups

- Audit cached result reuse against config/source and account for failed folds
  before treating a full suite as complete. The upstream runner catches dataset
  failures and reuses existing result files; a manifest alone is insufficient.
- Next is a bounded, pinned Linux smoke, then a preregistered real quality shard
  and full suite. No quality ranking/acceptance/adoption claim is made here.
- Author materials exist, but no outside attempt/contact occurred. No push,
  publishing, leaderboard submission or external messaging was performed.

## Commits

- `00dd67a` — previous fixed-slot performance evidence, budget still failed.

- Final verification: 5 focused tests plus 47 distributional regressions =
  52 passed; the additional test verifies constructed configuration is actually
  persisted in the manifest. Production/changed-file lint and MkDocs build pass
  (existing deprecation/griffe warnings). Full Linux/competitor metrics remain
  explicitly unverified; no benchmark-quality result was produced this slice.
