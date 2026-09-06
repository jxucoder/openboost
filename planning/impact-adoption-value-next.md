# Impact, adoption and value: next investment gate

> Superseded by the user's foundation-first, no-backward-compatibility direction
> and [the active F0–F5 plan](agent-boosting-foundation-plan.md) on 2026-09-05.
> The text below records the earlier strategic interpretation, not the current
> work queue or mission. P7 falsifies its scoped performance target; it does not
> establish that the foundation product hypothesis has failed. The completed
> ScoringBench configuration fix remains useful evidence infrastructure.

Review date: 2026-09-05. Replaces continued GPU micro-optimization as the immediate
work queue; the experimental API and its negative results remain available.

## What the evidence changes

The original goal is useful distributional risk modeling and independently usable
research tools. A high Python implementation ratio, more extension points and
passing our own tests are means, not measures of impact or adoption.

- [P7 follow-up](../benchmarks/results/foundation/20260905T193308Z-5ebd75ab/README.md)
  preserves quality and records a modest speed improvement, but the experimental
  path still costs 12.888x the paired legacy CUDA fit. The general GPU foundation
  investment gate failed; repeated small optimizations do not validate demand.
- CPU/CUDA agreement proves execution correctness. Housing's nominal 90%
  intervals covering about 96% does not prove useful calibration.
- Independent package installation works, but repository-authored examples are
  not external authors or evidence that anyone will depend on the package.

## Priority queue

| Objective | Next evidence | Decision enabled |
|---|---|---|
| Value | ScoringBench proper scores/calibration on a frozen external suite, with failures and reproducible configuration | Whether NaturalBoost offers useful predictive distributions beyond one Housing example |
| Adoption | Two outside developers attempt the prepared task; at least one completes their method without core edits | Whether the abstraction reduces their work and merits independent-package support |
| Impact | A reproducible exposure-aware risk case study after the benchmark integration is trustworthy | Whether the API addresses a real domain problem and can support independent research |

Continue the verified legacy NaturalBoost CPU/CUDA paths for these experiments.
Keep the strict extension API experimental. Resume GPU foundation optimization
when a real method/workload needs it, with an explicit quality/cost target; do
not expand multi-GPU, train-many or API surface to compensate for missing users.

## Immediate work: trustworthy third-party comparison

1. **Constructor configuration (this slice):** forward CLI seed to OpenBoost and
   XGBLSS, and depth/seed to a fresh clone of NGBoost's installed default learner.
   Record constructed wrapper/base-learner parameters, not merely CLI intent.
   Local tests must exercise seeded OpenBoost sampling, not only compare kwargs.
2. **Completion/provenance gate (next):** inspect upstream result-cache reuse,
   reject incompatible prior configurations, and account for every requested
   dataset/model/fold failure. Verify effective dataset row caps, hashes, split
   identities and the pinned upstream commit. An empty or partial result is not
   an official-quality completion. Keep upstream code unmodified.
3. **Bounded Linux integration run:** isolated installed wheel plus a pinned
   ScoringBench environment; OpenBoost CPU and NGBoost, two-fold diabetes smoke.
   Verify distribution/metric and result-schema behavior. This is not a ranking.
4. **Real quality shard, then full suite:** preregister parameters, seeds, full
   expected matrix and resource budget; run one five-fold real-data shard, then
   the remaining fixed suite after the pipeline passes. No test-fold tuning or
   hiding failed datasets. Keep scale extensions separate from the official
   protocol, and local results separate from upstream acceptance.

The local ScoringBench checkout inspected for this review is clean at
`a938a667b7839b41e9272929010573410301c0b4`; that is the inspected revision, not
an assertion about the current upstream HEAD. The existing launcher is not yet
claimed ready for publication-quality full-suite evidence. Intel macOS must not
run its complete PyTorch/NumPy stack; use Linux for the full launcher.

## Adoption boundary

[The author task and record](../examples/extensions/AUTHOR_TASK.md) are ready.
Measure active/setup/blocked time, assistance, private imports/core edits,
mathematical correctness, GPU results if available and willingness to depend on
OpenBoost. Failed attempts remain evidence. No outside author has attempted the
trial in this work, and no invitation or message has been sent. Contacting authors,
publishing results and submitting a leaderboard entry require user authorization.

Do not conflate the three gates: full internal correctness does not pass adoption;
a benchmark win does not establish user demand; a pleasant API does not establish
calibration or risk-model value. The mission in AGENTS.md is unchanged.
