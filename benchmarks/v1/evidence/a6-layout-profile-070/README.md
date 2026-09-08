# Sprint 070: Vector layout reuse diagnostic

Clean source revision: `b642bd5`. All 32 uploaded source hashes match that revision;
all six returned artifacts verify. This uses the same approved Parkinsons packet,
shared fold-zero configuration and 60-second diagnostic as a6-profile-070.
The child exits 124 at the intended soft deadline and retains raw pstats.

## Observation

The immutable `_vector_layout` resolver runs once (raw pstats), while its
`_vector_indices` caller runs 1524608 times. The name-resolution cache is exercised
in the installed worker. `_vector_indices` takes 5.659 cumulative seconds in this
sample; the previous diagnostic records 8.836 seconds across 1299063 calls.

Candidate scores: 301590 in this diagnostic, versus 256753 in the earlier sample.
These are separate instrumented runs, with different sampled prefixes; they do
not establish a paired full-fit speedup. `choose` still takes 52.073 cumulative
seconds; vector_score takes 39.192, vector_leaf 29.092, newton_leaf 15.948 and
_nonnegative 9.111. Cumulative times overlap and must not be added.

## Correctness and limits

Fifteen focused checks include exact model bytes/predictions against the original
resolver across depthwise/best-first/symmetric and projected output paths, mutation
isolation and malformed/invalid leaf behavior. Full CPU regression: 1058 passed,
one Linux-only skip; lint/docs pass. Public arithmetic and callbacks are unchanged.
The cache stores only immutable schemas and is bounded to 128 layouts.

No full-fit paired cost result, new real quality result or formal gate is claimed.
Next assess prepared default scoring parameters and temporary leaf construction,
with exact conformance and unchanged custom callbacks. Parent-score reuse requires
correct candidate-set ownership. Keep these as separate changes and retain paired
full-fit evidence before claiming end-to-end benefit.
