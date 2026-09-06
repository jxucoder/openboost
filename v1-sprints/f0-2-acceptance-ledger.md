# F0.2 independent reference acceptance mapping

As of Sprint 010: reference evidence only. **F0.2 closed; F0.3 not yet executed; F1 not started**
at this audit. Sources: [tasks](../planning/foundation-tasks.md), [plan](../planning/agent-boosting-foundation-plan.md).
Existing evidence here does not mean the corresponding A/R/C or E-gate passed.

## Application references

| Scope | Independent evidence | Remaining work |
|---|---|---|
| A1/R1 | Sprint 001 scalar math, three growth policies, two rounds | Public conformance/external differences in F1/F0.3 |
| A2/A3/R1 | Sprint 003 classification; 009 full native categorical growth/raw-transform two rounds | Public components/real quality in F1/F4 |
| A4/R3 | Sprint 004 pair/query normalization, lambda, two rounds, NDCG | Real query evaluation/sampling protocol; not a F0.2 speed task |
| A5/R2 | Sprint 004 three weighted quantiles/routed leaves/two rounds; 008 penalty; 010 composed predictions | Persistence/public components in F1 |
| A6/R8 | Sprint 004 stump; 009 three-policy multilevel vector/projection/K=1/two rounds | Public components/real quality in F1/F4; retain both structures |
| A7/A8/R4 | Sprint 005 Poisson/Gamma/exposure/weights/two rounds; 010 offset once/new exposure | Real adapters in F0.3 |
| A9/R4 | Sprint 005 Tweedie/paid-count joins; 010 two-round Poisson/Gamma fits and product | Real quality F4, persistence F1 |
| A10/R5 | Sprint 005 event/censoring, tails, units, two rounds | Output/persistence later; real IPCW fixed in F0.3 |
| A11/R6 | Sprint 006 Fisher/ordinary, NLL/CRPS, joint/ordered; 008 exact D4; 010 versioned acceptance/best | Public runtime/real quality F1/F4 |
| A12/R7 | Sprint 006 Jacobian/GGN, three directions, repeated Z, two rounds, misspecification/nonidentifiability | Real quality/formula artifacts F1/F4; synthetic parameters do not prove physical truth |
| A13/R9 | Sprint 007 K=1/2 isolation, M=1/8/32 sequential, best/failures/RNG | Sequential simulation does not prove batching, cost or multiple-recipe integration |

## Author changes and cross-case boundaries

| Scope | Evidence and next step |
|---|---|
| D1 expectile | Sprint 008 tau=.8, weighted base, two rounds, tau=.5 reduction, r=0 convention |
| D2 cohort split | Sprint 001 candidate feasibility, no legal split, independent information weights |
| D3 penalized quantile | Sprint 008 breakpoint/stationary enumeration, subgradient, anchor/lambda, two-round routed leaves |
| D4 ordered acceptance | Sprint 008 six alphas, reverse/NaN rejection; 010 two-round commits, best, logical-step keys |
| D5 scheduling | Sprint 007 independent/sequential/reordered/regrouped, retry, stop, ID seed/content changes |
| C1 identity/bind | Sprint 009 mixed transforms/fitted identity/raw prediction; production typed contracts F1 |
| C2/C3 tree/leaf | Scalar, D3, categories, multilevel vectors; not production conformance |
| C4 state/run | Sprint 007/009/010 isolation, mapping, offset/two-stage, versioned commit, best caches/terms; runtime F1 |
| C5 artifacts | Serialization round trips are F1 construction; memory snapshots are not persistence evidence |
| C6/C7 eval/workflow | F0.3 freeze/judge, F2 author evaluation, F5 installed workflows incomplete |

## Exit audit and next phase

1. Sprint 008 supplies exact D1/D3/D4 definitions/counterexamples, not E5 author-cost evaluation.
2. Sprint 009 supplies categories/multilevel vectors/mixed data; 010 supplies offset/dual-model/finite state compositions.
3. F0.2 preparation is complete: 288 tests pass and references run with openboost imports blocked.
   Do not mark F1 requirements passed early.
4. Begin real manifests, capability smoke, budgets/held-out tasks/judge freezing in F0.3.

R1–R9/C1–C7/A1–A13 all remain required. Optional GPU entries never make CPU applications optional.
Commands/counterexamples/reflection: [Sprint 010](010-reference-integration-exit.md).
Mathematics: `tests/v1/test_{scalar,tree,data,classification,extended,positive_aft,coupled,runs,author,mixed,integration}_reference.py`.
Isolation: `tests/v1/test_reference_independence.py`. All evidence is tiny CPU reference work,
not external-library equivalence, real quality, serialization, author-cost or CUDA correctness.
