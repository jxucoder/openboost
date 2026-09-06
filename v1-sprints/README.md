# OpenBoost v1 execution and reflection

This directory records the user-requested sprint plans, results and reflections. The goal is
**to help researchers and agents make correct algorithm changes using public composable components,
and verify their cost and practical value.**

Sources: [main plan](../planning/agent-boosting-foundation-plan.md),
[construction design](../planning/foundation-construction-design.md), [tasks](../planning/foundation-tasks.md),
[evaluation](../planning/openboost-v1-evaluation.md). These define scope/architecture/gates; this
folder manages execution, not a competing roadmap. R1–R9/C1–C7/A1–A13 all remain required.

## Current execution position

| Sprint | Plan mapping | Status | Deliverable and record |
|---|---|---|---|
| 001 | B01/F0.2 scalar/tree subset | Complete; 55 tests | [Scalar/tree references](001-scalar-tree-reference.md) |
| 002 | User-requested early production retirement | Complete; namespace/build/docs checked | [Clean v1 starting point](002-retire-legacy-production.md) |
| 003 | B01/F0.2 transforms/classification | Complete; see record | [Transforms/classification](003-data-classification-reference.md) |
| 004 | B01/F0.2 A4–A6 probes | Complete; 122 total tests | [Ranking/quantile/vector](004-ranking-quantile-vector-reference.md) |
| 005 | B01/F0.2 A7–A10 probes | Complete; 183 total tests | [Positive targets/AFT](005-positive-aft-reference.md) |
| 006 | B01/F0.2 A11/A12/D4 probes | Complete; 208 total tests | [Normal/Formula](006-normal-formula-reference.md) |
| 007 | B01/F0.2 identity/A13/D5 | Complete; 229 total tests | [Identity/runs](007-identity-runs-reference.md) |
| 008 | B01/F0.2 exact D1/D3/D4 | Complete; 255 total tests | [Author-task references](008-author-mutation-reference.md) |
| 009 | B01/F0.2 full mixed/vector growth | Complete; 279 total tests | [Mixed/vector trees](009-mixed-vector-growth-reference.md) |
| 010 | B01/F0.2 finite compositions and exit | Complete; 288 total tests | [Compositions and exit](010-reference-integration-exit.md) |
| 011 | B02/F0.3 integrity subset | Complete; 336 total tests | [Integrity judge](011-artifact-integrity-judge.md) |
| 012 | B02/F0.3 A5 data/date windows | Complete; 360 total tests | [Bike freeze](012-bike-data-freeze.md) |
| 013 | B02/F0.3 A1/A11 five splits | Complete; 380 total tests | [Housing splits](013-housing-five-splits.md) |
| 014 | B02/F0.3 A2 official test/stratification | Complete; 396 total tests | [Adult freeze](014-adult-data-freeze.md) |
| 015 | Cross-cutting English prose | Complete; no phase advancement | [English repository](015-english-repository.md) |

Next complete B02/F0.3 frozen evaluation. Public F1 implementation begins at B03 and connects
components/all algorithms through B04–B10. References are not product implementation. Select
bounded future sprints by dependency without bypassing phase exits. The user reaffirmed evaluation first.

## Execution rules

1. Start each sprint with purpose, F/B/A/C/E mapping, an independent failing example, deliverables and acceptance.
2. Commit every independently verified slice; record commands/results/unverified scope/commit, not just file counts.
3. **Reflect at every sprint closure, every three implementation commits, phase transitions,
   and architectural/correctness counterexamples.** Record it in the current sprint; one reflection can satisfy multiple triggers.
4. Record evidence/reasons and update design before changing architecture/order. Never silently
   relax gates, discard failures, remove cases or switch to multi-GPU/feature catalog work.
   Routine small fixes do not require replanning the whole project.
5. `learnings/` stores durable cross-sprint conclusions linking here; execution details live here.
6. All repository prose must be English, per the user's instruction.

## Reflection checklist

- Which difficulty in making a correct algorithm change did this reduce? If preparation only, which component does it justify?
- Does execution follow construction dependencies? Did old API restrictions, specialized trainers or premature optimization slip in?
- Is there independent math/state evidence? Which internal simulations cannot support performance, quality or adoption claims?
- Can structurally different cases reuse the boundary, or is it being designed around one example?
- What required scope remains? What is the next smallest verifiable deliverable?

Use observation → evidence → decision → next step, not an unsupported conclusion that the direction is right.

## Overall completion

F0.1 specifications/construction and F0.2 references are delivered. F0.3 and F1–F5 remain incomplete.
Sprint 001 delivered scalar/tree; 002 retired production; 003 transforms/classification; 004 ranking/
quantile/vector; 005 positive/count/policy joins/event-right-censored AFT; 006 Normal/Formula;
007 identity/isolation/selection; 008 D1/D3/D4; 009 full mixed/vector growth; 010 finite model/state
compositions. See [F0.2 exit mapping](f0-2-acceptance-ledger.md).

Sprint 011 adds integrity judging. Sprint 012 freezes A5 data/calendar/five date windows.
Sprint 013 adds Housing inputs/five splits for A1/A11, with license unresolved. Sprint 014 adds A2
Adult data and official-test-preserving splits. Other required data, capabilities, budgets,
held-out tasks, runner and quality judge remain incomplete; all real quality results are pending.
No E0–E6 gate passed through creating these files. E7 has no new independent-adoption evidence.
Every application's production and real evaluation still require individual completion.

Current sequencing review: [Sprint 017 audit](017-f0-sequencing-audit.md).
It distinguishes existing F0.3 prerequisites from later evaluation results and
proposes a bounded CPU-construction overlap. The phase gate has not been changed.

The user approved the Sprint 017 overlap on 2026-09-06. Active construction:
[Sprint 018 / B03](018-b03-cpu-state.md). F0.3 remains incomplete; no scope or
acceptance threshold was removed.

[Sprint 019 / B04 operations](019-b04-numeric-operations.md) delivers numeric
preparation and shared scalar split operations.
[Sprint 020 / B04 trees](020-b04-depthwise-tree.md) adds depthwise assembly and
validated numeric tree inference/persistence.
[Sprint 021 / B05 squared](021-b05-squared-recipe.md) adds mapped tree transactions
and the first complete squared CPU recipe.
[Sprint 022 / B05 Normal](022-b05-normal-recipe.md) separates target/raw widths and
adds joint Normal ordinary/Fisher updates.
[Sprint 023 / B06](023-b06-formula-runs.md) adds saturation Formula/full GGN and
sequential heterogeneous execution probes.
[Sprint 024 / B07 growth](024-b07-growth-policies.md) adds best-first and symmetric
numeric policies. [Sprint 025 / B07 categories](025-b07-categorical.md) adds mixed
input, category equality and typed dictionary persistence.
[Sprint 026 / B08 binary](026-b08-binary.md) adds explicit class order and binary
logistic inference/training.
[Sprint 027 / B08 vectors](027-b08-vector-multiclass.md) adds joint multiclass,
vector leaves, separate split/leaf statistics and output mappings (630 tests).
B09 ranking/quantile/penalized leaves are next; full A6 workflows, CUDA and
quality/performance evaluation remain incomplete.

[Sprint 028 / B09 ranking](028-b09-ranking.md) adds query-local pairwise/lambda
geometry and fixed-step ranking (642 tests). Quantile/penalized routed leaves
are next; real A4 evaluation and the broader incomplete gates remain open.

[Sprint 029 / B09 quantiles](029-b09-quantile-leaves.md) adds routed residual
views, weighted quantile and anchored penalized leaves (655 tests). B10 positive
target/exposure and AFT construction is next; real application gates remain open.

[Sprint 030 / B10 Poisson](030-b10-poisson.md) adds explicit exposure/count
geometry and rate/count inference (665 tests). Gamma/A8 is next, followed by
Tweedie/composition/A9 and AFT/A10. Real A7 evaluation remains open.

[Sprint 031 / B10 Gamma](031-b10-gamma.md) adds weighted positive-target means
(674 tests). Tweedie/frequency-severity composition and AFT are next; real A8
evaluation remains open.

[Sprint 032 / B10 Tweedie](032-b10-tweedie.md) adds fixed-power nonnegative
means (684 tests). Frequency-severity composition and AFT are next; real A9
quality and complete application artifacts remain open.

[Sprint 033 / B10 composition](033-b10-frequency-severity.md) adds matched
paid-loss problems and persisted two-model inference (693 tests). AFT is next;
real A9 joins/quality and joint selection remain open.

[Sprint 034 / B10 AFT](034-b10-aft.md) adds explicit event/right-censored
targets and persisted fixed-scale survival inference (716 tests). Next audit
CPU/B11 coverage; full A6, extension tasks, real application results and CUDA
remain incomplete.
