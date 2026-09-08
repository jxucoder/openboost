# Sprint 081: Compatible CUDA train-many

Status: planned. Mapping: B13 / F3.3 / R9, A13 / C4 / E1 and E4 preparation.
Depends on: applicable cells in [080](080-cuda-required-recipes.md) and 065 isolation.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first failing check

Execute a compatible group with real device batching while preserving independent
run semantics. First compare M=1/8/32 same-ID outcomes against independent execution,
including different stop budgets and one forced failure. An ordinary loop over
GPU fits is a sequential reference, not evidence of batching.

## Work

- Freeze at least one compatible real R1/R4/R8 ensemble and its declared grouping
  constraints, memory caps and binning/encoding identity. Verify sequential device
  execution before adding batched operations.
- Reuse preparation and resident inputs, then batch compatible work with explicit
  per-run active masks, buffers, RNG and acceptance/stop state. Declare unsupported
  combinations; do not silently merge incompatible parameter counts or schemas.
- Check permutations, regrouping, same-ID retry, distinct-ID streams, early stop,
  rejected updates and fault isolation. Failed runs retain their own outcome.
- Record preparation-only savings separately from batching, and measure full-set
  latency, throughput, memory/cap behavior, compilation, transfers and failed work.

## Acceptance and reflection

M=1/8/32 preserves every run's independent predictions, selected quality, state,
RNG and stopping under E1 CPU/CUDA rules. Neither failed nor stopped runs can
contaminate neighbors. Compatible groups meet the frozen memory cap and execute
the declared batched path; unsupported groups have explicit status.

Exported models replay independently on CPU. Performance observations are preliminary
until 082 runs the frozen matched-quality E4 comparisons. Reflect on what batching
requires from public ownership/layout and whether the measured gain exceeds binning
reuse alone. No Ray, multi-GPU or general distributed scheduler enters this sprint.

## Results

Not run. Current M=1/8/32 evidence is sequential CPU execution only.
