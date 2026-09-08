# 2026-09-08: GLM comparison and recipe integration

## Context

Sprint 106 constructs binary/Poisson device geometry and prescribed rounds, but
automatic acceptance/best/patience still lack reliable loss-change operations.
The user asks to continue. All eleven hardware allowances remain consumed.

## Decision or Result

Use scalar convexity to bound the exact loss difference: old derivative times
step plus half the squared step times an interval for segment curvature. Binary
uses signed-margin sigmoid/curvature; Poisson uses separate exposure times exp.
Unresolved bounds conservatively retain anchors or reject backtracking trials.
See the [107-A derivation](../v1-sprints/107-glm-comparison-mathematics.md).

## Changes

Freeze 59 stored-input cases and a structurally different 160/220-digit direct
likelihood oracle before production comparison code. The independent prototype
reuses the prior interval arithmetic and Taylor exponential through range reduction;
it does not import production or simulate CUDA. Its all-row 106 geometry checks
precede unchanged/zero-weight shortcuts. Normal sources and evidence stay intact.

## Verification

The first distinguishing case is a stationary Poisson step: rounded reporting
losses are equal while direct high-precision loss increases. The bound proves
worsening. Other cases cover tiny improvements, both binary tails, unchanged and
equal-loss different raw, zero weights, nonuniform exposure/offsets, large steps,
deterministic random rows, reversed directions and permutations.
All 73 focused mathematical tests pass in 1.28 seconds; changed-file Ruff passes.
No production comparator or new CUDA case has executed at this checkpoint.

## Failed Attempts

Two extra range-reduction halvings placed outward-rounded +/-256 endpoints just
outside the old reference exponential's +/-64 limit. A third halving leaves room
for expansion. The original endpoint tests and old reference implementation are
unchanged. This correction precedes all production/device observations.

## Risks and Follow-ups

Convex bounds can be wide for finite steps and near cancellation. Their enclosure
validity is conditional on explicit arithmetic; high-precision agreement alone
does not verify device lowering. GPU execution, recipe consumers and a concrete
hardware request follow local construction. No speed/adoption claim is made.

## Commits

- `5943f4f`: preceding class-aware export and prescribed-round construction.
