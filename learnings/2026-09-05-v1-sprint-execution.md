# 2026-09-05: Execute v1 through scoped sprints and explicit reflection

## Context

The user authorized execution and requested `v1-sprints/` for planning, records
and periodic reflection against the existing v1 plan.

## Decision or Result

Use [the sprint index](../v1-sprints/README.md) as the execution entry point.
The architecture and E-gates stay in planning; do not create competing scope.
Reflect at sprint closure, every three implementation commits, phase transitions
and architectural/correctness counterexamples. Keep concrete evidence and decisions.

## Changes

- [Sprint 001](../v1-sprints/001-scalar-tree-reference.md) scopes the first B01/F0.2
  scalar/tree reference slice and names independent failure cases before implementation.
- Agent guide and main plan point execution and reflection to the sprint directory.

## Verification

- Read the clean `9700845` starting state, plan, contracts, old split implementation,
  public experimental docs and tests. No `tests/v1/` or `v1-sprints/` existed.
- Sprint bootstrap checks passed: six Markdown files, 43 resolving local links,
  balanced fences and `git diff --check`. No production behavior changed.

## Failed Attempts

- Reusing the normal root test setup would load production OpenBoost through
  `tests/conftest.py`; reference tests need an isolated entry point and import check.

## Risks and Follow-ups

- Sprint 001 is only part of F0.2. Its results cannot mark the full reference
  matrix, production foundation, quality, GPU performance or adoption complete.

## Commits

- `9700845` — preceding architecture design.
- Sprint bootstrap: `docs: start v1 sprint execution and reflection log`.
