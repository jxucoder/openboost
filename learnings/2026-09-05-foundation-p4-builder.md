# 2026-09-05: P4.4 level-wise builder assembly

## Context

All three primitive slices have real T4 evidence. Assemble the smallest builder
needed for the Normal/bounded-leaf independent-package experiment, without
adding a second trainer loop to the library or changing legacy model defaults.

## Decision or Result

LevelWiseBuilder composes histogram, split, routing and leaf rules over fixed
slots. It is explicit opt-in on the CPU facade and callable directly on CUDA.
The existing CPUHistogramBuilder default is retained because it supports a wider
feature/parameter boundary; the new builder is the candidate experimental GPU
path, not a validated default replacement or speed optimization.

## Changes

- Maintain frontier and leaf masks across levels. Route actual samples, retain
  early leaves, and apply leaf rules only to the final routed leaves. Release
  each histogram before allocating the next; use one declared histogram budget.
- Return standard host TreeStructure plus an owned same-device training cache.
  Only five O(nodes) arrays download after CUDA growth; no per-level histogram
  or sample-ID download. BuiltTree annotation now admits either array backend.
- Preflight numeric/nonmissing metadata, L2/full sampling, depth/array/device,
  default CUDA stream, leaf capability and histogram budget. Validate via the
  existing config contract. No implicit fallback or parameter clamping.
- CPU whole-tree oracle scans original row masks recursively, independently
  computing splits and final leaves. CPU trainer/persistence exercises two
  channels, nonconstant coefficients and bounded versus default leaves.
- GPU suite checks input-view lifetime, compact copy shapes/count, two-round
  Normal mean/log-scale composition at 16/4097 rows, optional clipping and CPU
  load prediction. GPU Booster.fit remains P5, distinctly unimplemented.

## Verification

- Initial collection failed because LevelWiseBuilder was not exported.
- Depth 0/1/2/3 whole-tree reference, early leaf/zero curvature, strict preflight,
  two-channel actual CPU trainer and prediction save/load tests passed.
- Focused CPU regression: 120 passed across builder, primitives, evidence runner,
  objective, dispatch and persistence tests. Changed production/support lint passed.
- Public Normal adapter example executed with loc/scale channels. MkDocs build
  passed (existing griffe docstring warnings). Real-device results follow below.

## Failed Attempts

- Lint rejected ambiguous row-mask variable names in the independent oracle;
  renamed them without changing the reference computation.

## Risks and Follow-ups

- CUDA assembly is pending real T4 verification at the implementation commit.
- Only numeric L2/full sampling/default stream; fixed full slots may cost time.
- Synthetic CPU/CUDA NLL/CRPS agreement is a numerical test, not external quality,
  speed or adoption evidence. CPU default is intentionally not broadened/replaced.
- After P4.4, advance independent CPU extension wheel examples before P5, as
  agreed in the goal review; record concrete install/API obstacles.

## Commits

- `7dfeba4` — P4.3 frozen T4 evidence.
