# 2026-09-05: Foundation P2 correctness

## Context

P2 starts with the suspected weighted constant-Hessian optimization conflict.
The unified objective weights both gradients and Hessians; the native hint
currently depends only on the unweighted objective's unit-Hessian property.

## Decision or Result

First commit the regression harness so real CUDA failures and fixes can both
be traced to clean wheel builds. No production fix in this initial slice.

## Changes

- Add an isolated correctness suite alongside the existing smoke. Require all
  smoke cases plus fixed-bin weighted Newton and Normal/Poisson weighted fits.
- Fixed-bin oracle includes zero weights and nonuniform positive weights,
  inspects the actual native histogram hint and checks analytic Newton values.
- Compare device and CPU distribution gradients, split, raw scores and NLL.

## Verification

- Host evidence-gate tests and lint run before commit.
- Local GPU skips are not validation; real T4 execution follows this commit.

## Failed Attempts

None yet; the constant-Hessian issue remains a hypothesis until the GPU run.

## Risks and Follow-ups

This initial suite does not freeze a real-data baseline or cover fallback,
seed propagation, callback/eval or persistence boundaries. These remain P2.

## Commits

- `1669974` — previous passing P1 evidence.
