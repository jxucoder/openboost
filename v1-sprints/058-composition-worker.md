# Sprint 058: Fit and replay matched frequency-severity composition

Parent: e603f1f. Status: complete for component-selected execution/replay.

## Plan and acceptance

1. Test public component parity and strict composition packet/job contracts.
2. Train paid-event Poisson and count-weighted Gamma through paid_loss_problems;
   retain separate stopping and final/best selection, persist FrequencySeverity.
3. Run all five hash-bound Sprint 057 packets within 90-second combined fit and
   30-second fresh replay caps. Verify all named outputs, products and units.
4. Regression/lint/docs, evidence, reflection and local commit.

Component selection is not joint aggregate selection. No test labels, source
population changes, compound distribution, quality/search or CUDA claims.

## Results and reflection

Nine focused tests pass, including direct public component identities, stopping,
fresh persistence, output products and rejected ambiguous/invalid packets.
Full CPU regression: 901 passed. All five hash-bound real packets pass combined
fit and fresh replay. Every named array is exactly reproduced; rate times severity
matches annualized mean and exposure conversion matches period mean. Source,
input and output hashes match the retained
[raw evidence](../benchmarks/v1/evidence/composition-058/README.md).

No foundation changes were needed. The source-binding prerequisite mattered:
this executes paid-event frequency, not A7 raw claim frequency. The result verifies
composition mechanics and persistence on real inputs; separate component selection
is recorded honestly and cannot substitute for joint A9 quality/search acceptance.
Next connect A10 survival evaluation, then A12/unresolved A4 and real searches/D5.
Joint composition selection remains in A9's unfinished quality workflow.
See [learning](../learnings/2026-09-06-v1-composition-worker.md).

Closure: Ruff, strict MkDocs and whitespace checks pass. No push or publication.
