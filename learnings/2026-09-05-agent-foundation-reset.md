# 2026-09-05: Foundation is the product; compatibility does not constrain design

## Context

After repeated bigger-goal reviews, the user clarified that OpenBoost should
expose algorithm components for researchers and agents. GBDT, NaturalBoost,
FormulaBoost and train-many are use cases for finding the right abstraction.
The user then requested a second thinking pass and a plan, explicitly permitting
breaking APIs and rebuilding the architecture without backward compatibility.

## Decision or Result

The product hypothesis is lower total cost from algorithm change to a verified,
reusable result. NaturalBoost risk modeling remains a useful application; it
does not define the entire product or force ScoringBench to precede every other
foundation task. The new plan uses F0–F5 rather than restarting historical P0.

Existing libraries already have important extension capabilities. Reviewed
primary documentation for XGBoost objectives/parameters, LightGBM update/leaf
mutation, NGBoost distributions/scores, and Py-Boost's Python GPU customization;
links and limits are in the new plan. Task-based comparisons must allow these
capabilities and source changes rather than assume competitors are black boxes.

Recommend ordinary Python recipes owning algorithm order, explicit run and
candidate state, reusable bulk operations, and a device execution layer.
Formula and train-many must probe the design early. This is a design hypothesis,
not implemented functionality. No requirement to adapt or retain the existing
trainer, public API, fixed tree slots, global backend or persistence format.

## Changes

- [Active plan](../planning/agent-boosting-foundation-plan.md): reasoning,
  architecture semantics, four use-case probes, task/evaluation protocol,
  bounded implementation stages and decision gates.
- [AGENTS.md](../AGENTS.md): align mission and priorities with the user's explicit
  direction; distinguish current implementation facts from redesign constraints.
- Previous GPU design/execution and ScoringBench priority plan: mark historical
  or superseded, preserving their raw evidence and original failed budgets.
- Learning index: make this user correction discoverable before earlier strategy.

## Verification

- Read code paths for the trainer's channel loop and fixed schedule interface,
  experimental contracts/builders/device tree export, FormulaObjective and
  `fit_trees_batch`; inspected public extension docs and formula/batch tests.
- Reviewed [P7 result](../benchmarks/results/foundation/20260905T193308Z-5ebd75ab/README.md):
  12.888x warm-fit ratio remains a failed 1.2 budget on that workload. No new run.
- Documentation-only validation passed: 49 local links across seven Markdown
  files exist, fenced blocks balance, and `git diff --check` is clean. The link
  checker excludes code spans/blocks so Python indexing is not read as a link.
  No new runtime tests, GPU jobs, external contacts or publishing.

## Failed Attempts

- Earlier strategy interpreted a scoped GPU overhead failure as grounds for
  prioritizing the distributional vertical over the foundation. The user corrected
  that ordering; do not repeat it by treating the old plan as current guidance.
- Repository-authored objective/leaf/schedule packages do not yet establish
  general algorithm expressivity, agent efficiency, independent adoption or
  predictive improvement. In particular, easy tasks overlap incumbent features.

## Risks and Follow-ups

- No clean redesign implemented or evaluated. No agent comparison or external
  adoption measured. Competitor documentation is not a frozen runtime experiment.
- Breaking changes remove migration obligations, not the need for mathematical
  oracles, explicit unsupported inputs, independent evidence or new-format round trips.
- Next implementation task: F0.1 task cards, then independent references and a
  frozen protocol. No wholesale rewrite or further kernel tuning in this slice.

## Commits

- `3ac1552` — prior ScoringBench configuration fix and code-review baseline.
- This documentation slice: `docs: redesign the agent boosting foundation plan`.
