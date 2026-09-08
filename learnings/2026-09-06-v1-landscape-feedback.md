# 2026-09-06: Use external research to challenge specific foundation contracts

## Context

After Sprint 063 (`961b6f9`), the user supplied an optional AI-authored landscape
survey. Its arguments needed source verification and comparison with actual code,
not wholesale conversion into more required features or another benchmark matrix.

## Decision or Result

The [addendum](../v1-sprints/063-landscape-feedback.md) selectively verifies GBNet,
NGBoost releases, XGBoost 3.4 and three research directions. Add GBNet/current
NGBoost authoring paths where they are relevant opponents. Preserve Py-Boost and
strong engine controls. No frozen versions, thresholds or required cases change.

The stopping suggestion exposes a concrete restriction: structural recipe results
still require the concrete StopState record. A custom completed record carrying
reason=score_test is rejected by run_many. This is an explicit current-contract
limitation, not a silent numerical failure or proof custom loops cannot stop.
Add it to N1 as a development counterexample before interface freeze.

Shared vector topology, per-row full-metric directions and aggregate coupled leaf
optimization are different semantics. A two-row SPD quadratic example confirms
that averaging row Newton directions differs from the coupled leaf solution.
Named fields/custom scoring/leaf callbacks may already support a small external
matrix leaf implementation; verify that before introducing a new core abstraction.
Existing L-by-K term mappings likewise are not proof of an adaptive common-direction
algorithm. All research ideas reviewed here are development, never held-out tasks.

## Changes

- Sprint 063 addendum: sources, observed stopping rejection, reproducible command,
  coupling distinction, bounded probes and scope decisions.
- N1/N3 execution text and current navigation incorporate the specific findings.
- No production implementation or new mandatory paper reproduction.

## Verification

- Read primary GBNet/NGBoost/XGBoost documentation and three arXiv full texts.
  AlphaXiv overview reports were unavailable; full-text fallbacks succeeded.
- Executed the addendum's equivalent tiny stopping-result probe: expected
  ValueError, `recipe result requires AcceptedState and StopState`.
- Independent NumPy quadratic solves: average row directions `(-0.75, 0)`;
  aggregate leaf solution `(-9/17, 3/17)`, explicitly different.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict`:
  passed. All 99 local Markdown targets in changed documents exist;
  `git diff --check` passed.

No ScoreStop/coupled-leaf/parallel-boosting implementation, new real-data fit,
GPU job, formal author attempt or held-out inspection occurred. The last complete
CPU regression remains Sprint 063's 923 passes; no new full test run is claimed.

## Failed Attempts

AlphaXiv overview web opens failed; direct public requests confirmed all three
reports return 404. The three full-text endpoints returned 200, and arXiv HTML
provided primary-source verification. No unavailable overview was treated as evidence.

## Risks and Follow-ups

The supplied survey is only selectively verified. Documentation availability is
not an installed comparator check. New viewed examples must not contaminate sealed
H1/H2 or replace the required five D types. Actual statistical stopping needs its
own calibration/dependence/validation checks; generic stopping extensibility does
not confer those guarantees. GPU overlap remains a proposal. No push or outreach.

## Commits

- This feedback review; parent `961b6f9`.
