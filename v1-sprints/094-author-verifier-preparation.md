# Sprint 094: Standalone author-verifier preparation

Status: local preparation slice complete; independent dispatch remains open.
Local continuation of [093](093-foundation-progress-and-next-steps.md)
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

### Closure evidence and next boundary

At clean `0a85320`, the [installed reproduction](../benchmarks/v1/evidence/author-verifiers-094/README.md)
passes both standalone commands: 2 D1 cases and 9 D2 cases, with eleven saved models
replayed in an environment where the training plugins and reference package are
absent. The deliberately changed model fails through the standalone CLI. All 36
artifact hashes verify after archival. The CPU suite passes 1812 tests with one
Linux-only skip; focused verifier checks pass 18/18. Ruff and documentation build
pass, retaining the existing external-evidence documentation link warning.

Reflection: the foundation's public operations already support both development
changes. This slice needed no production change. It removes evaluator coupling
but does not test whether an independent author can discover those operations or
use them with less work. Prioritize the remaining invalid-input/identity and edit
checks, a new current author packet, fair incumbent paths and real accounting/
isolation over adding more numerical fixture families. No synthetic runner
simulation should be counted as progress on actual token enforcement. The next
device action is unchanged; any independent model smoke needs explicit authority.
