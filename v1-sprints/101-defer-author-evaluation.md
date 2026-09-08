# Sprint 101: Defer agent evaluation and resume foundation engineering

Status: adopted from the user's request, "can we skip this agent friendliness for
now", following the Sprint 100 review. This is an execution-order amendment.
It supersedes the active authoring priority in 085 and 093. No model, worker or
GPU run is part of this planning change.

## Decision

Pause 069 author-pilot preparation, 077/F2/E5 comparative agent trials and the
094–100 verifier, isolation, accounting and cancellation work supporting them.
Do not request or execute Sprint 100's proposed live model test. Its packet stays
pending and unapproved; keep the source freeze and all prior evidence unchanged.
Do not resume this work automatically at a CUDA milestone. Revisit it only when
the user chooses to resume author evaluation.

OpenBoost remains a programmable boosting foundation. Readable public operations,
composability, installed extensions and mathematical correctness remain product
requirements. D1/D2 development fixtures can check component behavior without
running an AI author. Existing D2 CUDA coverage tests an installed component; it
does not require an independent agent attempt or establish easier authoring.
Training and ordinary correctness tests require no external language-model calls.

F2/E5 is deferred and unpassed. It no longer blocks foundation construction,
required CUDA recipes, train-many or scoped quality/cost evaluation. This expands
085's sequencing exception beyond the initial scalar feasibility slice. Preserve
the original author-evaluation thresholds for possible later use. Do not claim
validated author benefit, adoption or completion of every original v1 gate.
E7 independent adoption evidence also remains unestablished; no outreach or
author-study infrastructure enters the current engineering queue.

All R1–R9/C1–C7/A1–A13 implementation and application requirements remain. CPU
remains the semantic reference and a usable development backend. Wider CPU
searches and speculative CPU speed work stay paused. Independent mathematical
oracles, isolation between training runs and persistence checks remain active;
they are distinct from isolation of an AI author's evaluation environment.

## Plan and next engineering checkpoints

1. Record the deferral in canonical guidance, the main plan, evaluation protocol
   and sprint index. Check that frozen sources and evidence remain unchanged.
2. Resume the existing Normal CUDA correction at the
   [run-8 checkpoint](092-comparison-run8-request.md). Its local construction is
   already complete; actual hardware validation is the next unresolved question.
   The existing upload/compute allowance remains pending. Preserve its 385
   historical and 529 revised cases, source hashes and zero-retry limit.
3. After reviewing that device result, advance the existing required-recipe,
   train-many and cost cards below. Commit verified slices and reflect every
   three implementation commits or sooner after a correctness counterexample.

| Checkpoint | Acceptance and next decision |
| --- | --- |
| 092 / Normal CUDA correction | All 529 revised cases and declared artifacts pass on real hardware; reconcile the 385 historical cases with their exact preregistered disagreements. Verify numerical comparison, acceptance/best/patience, ownership, D2 and fresh CPU inference. Stop for retrospective after the bounded run, including on failure. |
| [080 / Required CUDA recipes](080-cuda-required-recipes.md) | Add binary, multiclass, Poisson/exposure, scoped AFT and multi-output coverage through shared public operations. Require gradients, statistics, splits, leaves, trajectories, task metrics and fresh CPU inference to agree with the declared CPU references. Preserve scalar and Normal regressions. |
| [081 / Train-many](081-cuda-train-many.md) | Verify independent device execution, then actual compatible batching at M=1/8/32. Preserve per-run RNG, rejection, stopping, failures and memory bounds. A loop over fits establishes only the reference. |
| [082 / Practical quality and cost](082-end-to-end-cost.md) | Use a frozen real workload to identify end-to-end cost once its required device path is verified; this can inform recipe/batching work before all ports finish. Include preparation, transfers, compilation policy, fit, prediction and failed work. Preserve formal matched-quality E4/P7 gates and individual A1–A13 evaluations. |

These checkpoints need engineering evidence, not an agent-friendliness score.
Actual hardware execution still uses a concrete approved resource packet; the
deferral does not spend or renew any earlier allowance. No new abstraction or
optimization should be added merely to fill time before a hardware result.

## Verification and reflection

The starting inconsistency is in the execution instructions: they still direct
the next model toward 069 accounting and the pending 100 cancellation probe.
The amendment must remove those as current priorities while keeping their
historical results and the original evaluation criteria readable.

This is a documentation change; no new runtime tests are needed. Verify local
Markdown links, build the documentation and check whitespace. Check run-8 and
100 source hashes, pending authorization and unchanged prior artifacts before
committing. Record exact outcomes in the
[learning entry](../learnings/2026-09-07-v1-defer-author-evaluation.md).

Recent accounting work established limited harness behavior, but it did not
improve boosting execution or measure author benefit. The remaining engineering
questions have direct evidence targets. Future work should first resolve those
questions rather than perfect infrastructure for a deferred study.
