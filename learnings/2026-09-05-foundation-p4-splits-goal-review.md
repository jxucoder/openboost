# 2026-09-05: Goal review and P4.2 numeric split/routing

## Context

User asked to review the larger goal, then continue. P4.1 established device
histograms, but not usable external GPU algorithms or adoption/value evidence.

## Decision or Result

The target remains useful calibration-first distributional/risk modeling plus
an easier route from Python research to a reproducible implementation. Treat
the GPU foundation as a bounded test of that goal, not a product success metric.
G3/G4/G5 remain open: independent GPU extensions, end-to-end engineering value,
and external author use. Continue the shortest path to the Normal objective,
bounded-leaf and channel-schedule examples; avoid new tree families/criteria.
The next product checkpoint is installed public-API extensions and recorded
implementation/installation costs, followed by real author attempts. No outreach
or publication is authorized by this review.

## Changes

- Record the goal review and next decision gates in the design document.
- New SplitBatch/find_splits/partition reuse HistogramBatch, return device arrays
  and fixed child indices. CPU compiled scan and CUDA kernel use float64 prefix
  sums/scoring, strict positive gain and inclusive min_gain/min_child_weight;
  exact ties resolve feature then threshold. Inactive/terminal/invalid slots are
  explicitly masked. Partition returns owned IDs from actual input rows.
- Numeric L2 only. Positive curvature required in each child even at minimum=0;
  histograms have no per-bin physical row counts to distinguish empty bins from
  zero-weight bins. This is a visible conservative split-admissibility boundary,
  sufficient for the first Normal/leaf experiment, not general Hessian support.
- Hist missing-bin mass is rejected; partition rejects all missing bins. Builder
  categorical/missing preflight remains required in P4.4. No default trainer,
  persistence or existing split backend was changed.
- The small CUDA split scan prioritizes deterministic reference semantics over
  optimized throughput. No speed claim; profiling and end-to-end comparison
  are still required before calling the GPU path valuable.

## Verification

- Initial collection failed because find_splits/partition did not exist.
- CPU expected split comes from exhaustive direct row masks, not histograms.
  Cases include weighted/zero-weight rows, positive/negative/zero gain, exact
  ties, gain/child boundaries, no legal split, terminal/inactive/empty nodes,
  bad parameters/IDs/children, missing bins and true routed child statistics.
- Final local and real T4 results follow below; no skipped job is a GPU pass.

## Failed Attempts

- Initial lint found compact multi-statement test lines; formatted new files.

## Risks and Follow-ups

- Numeric split/routing passed on real T4; whole experimental GPU training
  remains unverified.
- Positive-curvature children are narrower than arbitrary custom objectives.
- P4.3 leaf rule/reduction and P4.4 builder are next; no complete experimental
  GPU trainer, external adoption or matched-quality cost benefit is proved here.

## Commits

- `d3cebe6` — P4.1 frozen T4 evidence.

## Local implementation verification

- `OPENBOOST_BACKEND=cpu NUMBA_NUM_THREADS=1 uv run --no-sync pytest
  tests/test_batch_splits.py tests/test_batch_histograms.py
  tests/test_foundation_runner.py tests/test_experimental_objective.py
  tests/test_experimental_dispatch.py tests/test_experimental_persistence.py
  -n 0 -q`: **90 passed**.
- Production/changed-support lint passed; docs build passed with existing griffe
  warnings. No end-to-end GPU extension training was executed locally.

## Real-device verification and next value checkpoint

- Clean source `b75b95a`: **4 passed / 0 skipped**, real T4. 20.23 s pytest,
  24.79 s remote function. [Raw artifact](../benchmarks/results/foundation/20260905T151819Z-e5eb30b7/README.md).
- Feature/threshold/IDs matched exhaustive row oracle exactly; gains matched
  at rtol=atol=1e-10. Exact ties and gain/child equality, actual child aggregates,
  next-layer split topology, invalid routes and missing rejection passed.
- Source file hashes and embedded JUnit independently verified; offline runner
  accepted the saved evidence. No runtime/performance claim beyond test duration.
- P4.2 complete. After P4.3/P4.4, advance the CPU portions of P6's independent
  packages ahead of full P5 integration to expose usability/installation costs
  sooner. GPU gates stay mandatory. External author attempts remain G5, not
  something self-authored examples can satisfy.
