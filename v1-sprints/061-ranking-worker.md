# Sprint 061: Current query-aware A4 adapter

Parent: ca73a2d. Status: adapter verified; real A4 binding remains open.

## Plan and acceptance

1. Bind contiguous disjoint query groups and explicit per-query weights through
   the public ranking recipe; preserve integer source row IDs for tie behavior.
2. Verify pairwise/lambda modes, final/best selection and fresh score replay;
   reject row weights, fragmented/overlapping groups and ambiguous row identities.
3. Run regression/lint/docs and record the missing MSLR real-data prerequisite.
   Synthetic tests do not close the required real A4 gate.
4. Commit locally and schedule CPU workflow/phase-gap review before more expansion.

## Results and reflection

Pairwise and lambda modes pass weighted direct parity with final/best selection
and exact fresh score/row-ID replay. The fixture uses unequal query weights and
reversed source row IDs to exercise stable tie identities. Seven counterexamples
reject fragmented/overlapping queries, ordinary row weights, overlapping/string
source rows, out-of-contract relevance and mis-shaped query weights.

No foundation production changes were needed. The configured build/v1-data
inventory has no MSLR files and no committed MSLR preprocessing freeze exists.
Existing source-access/agreement gaps remain unresolved; synthetic verification
does not close real A4 integration or establish ranking scalability. The current
public ranking implementation still enumerates pairs quadratically per query.

This completes the missing adapter implementation, not every application workflow.
Next audit the finite CPU exit list: A4 source/binding, A13 real searches and joint
A9 selection, remaining D5/independent author gates, and B11 acceptance. Then
record the exact B12 GPU starting slice or an explicit phase-overlap amendment.
Do not use more objective adapters or test counts as substitutes for that decision.
See [learning](../learnings/2026-09-06-v1-current-ranking-worker.md).

Closure: 923 CPU tests passed; Ruff, strict MkDocs and whitespace pass.
Nothing pushed.
