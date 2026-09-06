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
