# Sprint 022: B05 Normal geometry and shared raw state

Starting revision: 6491948. Status: complete for this slice. Approved B03–B06 overlap applies;
F0.3 and all required application evaluations remain open.

## Plan and acceptance

1. Separate observed target width from explicit raw parameter width in Problem
   and state, preserving existing squared behavior and strict offset alignment.
2. Expose Normal ordinary/Fisher geometry and least-squares direction fitting;
   compose joint mean/log-scale updates through existing trees and transactions.
3. Compare geometry, base, multiple updates, NLL and persistence with independent
   references. Exercise bounded backtracking, nonfinite trials and full rejection.
4. Run regression/lint/docs/build, record evidence and reflect before local commit.

The first failing check constructs a scalar-target problem with two raw columns.
This slice implements joint Normal updates; ordered updates and Formula remain
separate required probes. No CUDA or predictive-quality parity claim is intended.

## Result and reflection

Joint Normal boosting and independent target/raw widths are implemented. All 572
CPU tests pass; ordinary and natural three-round intermediates match independent
references. Lint, strict docs, build and five installed-wheel examples pass.
See the [learning record](../learnings/2026-09-06-v1-b05-normal-recipe.md).

Observation: Normal required changing the shape contract, but reused numeric
preparation, scalar growth, mapped terms and transactions. Its likelihood Fisher
and direction-regression curvature are distinct and explicitly tested. This is
support for exposing the geometry-to-fit boundary; it does not establish the
full-metric Formula boundary. Keep that as the next B06 probe, alongside
heterogeneous runs. Do not freeze the current interfaces or claim all F1 work done.

The squared and joint Normal construction paths now exist. Ordered Normal, all
other application coverage, real evaluation, agent/adoption evidence and CUDA
remain required. F0.3 is still open. Nothing was pushed.
