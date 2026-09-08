# Sprint 094: Standalone author-verifier preparation

Status: in progress. Local continuation of [093](093-foundation-progress-and-next-steps.md)
and [069](069-authoring-pilot.md). No agent dispatch, device run, upload, budget
change or independent author claim. The run-8 source freeze stays unchanged.

## Plan and acceptance

1. Record the installed runner's actual accounting interface and concrete gaps.
   Do not invent token counts or infer enforcement from a field named budget.
2. Export D1/D2 mathematical fixtures separately from candidate observations.
   Run a standalone judge with a hashed runtime dependency closure, without
   importing candidate extensions or the repository's reference package.
3. Verify installed development extensions as positive controls and deliberately
   incorrect observations/models as negative controls. Preserve raw artifacts,
   update author-facing capability documentation, and commit verified slices.

First failing check: a standalone D2 invocation must not require the D3 extension;
the existing combined verifier does. The new checks must additionally reject
wrong derivatives, reweighted independent information, missing rounds, nonfinite
values, altered saved models and modified evaluator inputs.

## Boundaries

Known designer-authored extensions are development controls, never independent
attempts. A separate Python process is not an OS sandbox. Actual author access
denial, token-budget exhaustion, model/settings and incumbent-arm freezes remain
required before dispatch. Keep the original cfca092 author-view cohort immutable.
Do not inspect sealed H1/H2 or expand the paused CPU search.

## Results and reflection

The installed `codex-cli 0.153.2` exposes usage events and a goal token budget,
but exact generated-token accounting/enforcement remains unverified. The
[concrete audit](../benchmarks/v1/evidence/author-runner-audit-094/README.md) records
the gap without an author attempt or a synthetic accounting substitute.

[Standalone development tooling](../benchmarks/v1/authoring/README.md) separates
public fixture export, known-extension observation collection and core-only
judging. Two D1 and nine D2 cases have independent mathematical expectations.
This closes a numerical invocation dependency, not the entire task-card or
dispatch gate. Invalid-input/identity checks and actual author isolation still
need to join the final runner. The initial installed smoke passed; committed
source reproduction and regression results are recorded at closure.

The author-facing overview now distinguishes implemented scalar/Normal CUDA
paths from their historical hardware evidence and current pending correction.
No frozen run-8 source changed. The next hardware action remains the pending
run-8 request; no hardware allowance is granted by this sprint.
