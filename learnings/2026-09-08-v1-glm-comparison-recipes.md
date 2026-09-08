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

## Slice B result

`95b5232` freezes the numerical contract before production. The private scalar
factory reuses the established enclosing +/*/division operations and constructs
the full-range exponential directly. CUDA supplies directed-double primitives;
its row kernels reuse the 106 geometry-domain helpers and its reduction validates
every row before any weight-zero or unchanged shortcut. Public callbacks return
LossChange, retaining separate resident snapshots and releasing all local scratch.

All 59 shared-expression cases contain the independent 220-digit likelihood
difference and retain required signs; two private range controls also pass. Combined
with the independent study and configuration checks: 160 passed in 1.95 seconds.
Seventy-seven GPU tests collect, including both allocation failures, dispatch after
row computation, no host/reporting fallback and PTX directed-double checks. None
has executed. The future numerical case artifacts include exact float32 inputs,
bounds, high-precision value and transfer counters. No old Normal file or archive
was modified; recipe consumers and their actual device validation remain next.

## Slice C result and reflection

`67788ba` commits the resident comparisons. The scalar recipe now uses the same
grower and transaction/search operations, requiring explicit fields/comparison
callbacks. Accepted raw, best validation raw and last-qualifying patience raw have
separate owners. Temporary fields/trees, superseded states, proposals and patience
storage are released on success and tested failure paths. Result history retains
scalar trials/comparisons only. Binary/Poisson wrappers add no separate trainer.

The initial forward union annotation used a quoted member inside an evaluated
union, which fails at module import. Quoting the complete deferred annotation
fixes import without changing old record behavior. Focused preflight and independent
trajectory controls pass: 42 in 1.15 seconds. Sixteen reference settings include
fixed steps, backtracking retries and full rejection; two additional sequences
exercise distinct best/patience anchors. Thirty-two new GPU cases collect,
including actual-comparison Decimal audits, equal-score anchor replacement,
default grower use, saved CPU inference and five failure/cleanup phases.

All 153 GLM device cases are unexecuted. Full CPU regression passes 2,241 tests
with one Linux-only skip and 880 deselections in 77.50 seconds. Production and
changed-file Ruff pass; offline wheel/sdist construction succeeds. CPU/native
device-result parity, PTX and execution cost remain unverified. The next slice
must freeze one exact retained-artifact/regression request before any new upload
or device allowance. No application family or formal quality/cost gate is closed.
