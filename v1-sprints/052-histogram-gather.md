# Sprint 052: Reuse selected histogram statistics

Parent: 215d85f. Status: complete; all five bounded full-data replays pass.

## Plan and acceptance

1. Reproduce feature-by-feature allocation of identical selected statistic columns.
2. Gather selected row statistics once, retain contiguous columns for bincount,
   and preserve both original accumulation order and C-order parent reduction.
3. Check exact legacy-formula sums/counts/parent totals across numeric, missing,
   categorical, weighted/vector/independent fields and full/reordered/empty rows.
4. Run regression and the five existing full Covertype packets under unchanged
   90-second fit/30-second replay caps. Preserve failures and compare fold zero
   exactly with Sprint 051. Record memory tradeoff, reflect and commit locally.

No dtype, row order, scoring, objective or binning changes. A histogram-local
scratch buffer is not a persistent cache or shared-run fusion. Full A3 quality
and comparative speed gates remain open regardless of smoke outcome.

## Results and reflection

All three new tests failed the redundant-buffer check before the change and pass
afterward, with exact legacy-formula statistics. Full CPU regression: 851 passed,
including independent tree/routing and recipe checks. The parent total keeps its
original reduction layout; no numerical tolerance was introduced for conformance.

All five frozen full Covertype jobs now complete within the unchanged cap, in
70.3, 69.9, 68.1, 67.8 and 67.9 seconds for folds 0–4. Seven-class probability
shape, normalization, source IDs and exact fresh inference pass for every fold.
Fold zero's model bytes and all prediction arrays exactly match Sprint 051.
All recorded source and output hashes match. See the committed
[raw evidence](../benchmarks/v1/evidence/histogram-052/README.md) and
[learning](../learnings/2026-09-06-v1-histogram-gather.md).

The full-data timeout counterexample is resolved for this bounded configuration.
Three slices (profiling, row hashing, histogram gathering) support two local
implementation fixes without changing public contracts. This is evidence for
keeping the abstraction boundary, not evidence that all CPU cost work is done.
Scratch memory increases during transpose construction and peak RSS is unmeasured.
Repeated comparative timings and matched-quality searches were not performed.

Return to M3's missing real-data adapters, starting with A5 temporal quantile
integration, then remaining application rows and real selection searches; remaining
D5 author checks also stay open. Full A1–A13 quality, formal author evaluation,
CUDA parity/cost and adoption remain required. No push or publication.

Closure checks: focused tests 3 passed; Ruff, strict MkDocs and diff whitespace
checks passed. The full 851-test regression completed before the real-data run.
