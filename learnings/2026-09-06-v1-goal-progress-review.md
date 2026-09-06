# 2026-09-06: Recenter execution on verified algorithm changes

## Context

The user requested a review of the goal, progress and planning after shared CPU
preparation. Reviewed clean revision `8afce35`. Sprint 035's audit was partly
superseded by multi-output recipes/scaling and M=1/8/32 preparation reuse, while
several current-status paragraphs still described early B03 or missing AFT code.

## Decision or Result

The CPU implementation now has substantial algorithm breadth and shared semantic
components. That supports further testing of the foundation hypothesis; it does
not establish lower independent authoring cost, real quality, GPU cost or adoption.
The next correctness slice is independent validation-driven stopping. Installed
public extension trials and current OpenBoost real-data integration should follow
early, rather than another sequence of built-in objectives.

Preserve every required family and E0–E7 threshold. F0.3 and formal F1–F5 remain
open. The CPU overlap does not authorize interface freeze or formal comparisons
without their explicit prerequisite ledgers. Keep exploratory authoring separate
from E5 and external adoption; keep held-out contents outside foundation design.

## Changes

- [Sprint 038](../v1-sprints/038-goal-progress-and-plan.md): goal/progress review,
  complete application status, six dependency groups, acceptance and immediate work.
- [Agent guide](../AGENTS.md), [sprint index](../v1-sprints/README.md) and
  [main plan](../planning/agent-boosting-foundation-plan.md): current execution
  pointers and status corrected without rewriting historical evidence or gates.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_public*.py --collect-only -q -o addopts=''`
  collected 247 public tests. This is collection, not a new regression pass.
- Last full regression: 735 passed, recorded in
  [Sprint 037](../v1-sprints/037-shared-preparation.md), with 19 installed-wheel
  examples, lint, strict documentation build and packaging on macOS/Python 3.12.12/
  NumPy 2.3.5. No broader environment or performance claim.
- Checked relative Markdown targets in all five changed files with pathlib;
  `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict`
  and `git diff --check` passed. Documentation-only change; no fresh model fits.

## Failed Attempts

No implementation attempt in this review. The audit found stale status text;
historical sprint findings must be interpreted at their recorded revisions.
Test counts and preparation equivalence cannot be promoted into product gates.

## Risks and Follow-ups

- Private recipe helpers and repeated policy loops may obstruct independent
  authoring; actual D2/D3/D4 trials must determine the minimum public changes.
- Prediction recomputation may dominate cost and impede device residency. This
  is a call-path risk, not a measured regression. Profile before cache redesign.
- Close global evaluation matrix, source/protocol gaps and current OpenBoost
  worker integration. Use the Sprint 017 ledger rather than repeating baseline smokes.
- GPU and independent repeated adoption remain unverified. Do not push or contact
  external authors without the user's explicit authorization.

## Commits

- This review commit — review goal and evidence; order remaining v1 acceptance work.
- `654a3f2` — shared preparation implementation reviewed here.
- `8afce35` — audited parent, including its whitespace correction.
