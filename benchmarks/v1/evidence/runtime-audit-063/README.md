# Runtime replay and trace-retention diagnostic

Review baseline: `47108db5d9fe3157e59621115860aa1dd06e9bf4` (merged PR #24).
`counts.json` records the dirty review state and hashes all production Python
files and the diagnostic. Production code was not changed. The dirty changes are
review documents and this diagnostic, not an optimized runtime.

Reproduce from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.runtime_cost_audit /tmp/openboost-runtime-counts.json
```

The script instruments `Tree.predict` during complete CPU fits of squared and
Normal recipes, with fixed steps, no patience, 32 training rows, 16 validation
rows, three features, eight bins and depth one. It checks final raw caches against
full model replay outside the counted interval. No timings or memory peaks are
measured. No external dataset, model-selection or GPU claim follows.

| Rounds | Squared tree-predict calls | Normal tree-predict calls | Squared trace array bytes | Normal trace array bytes |
|---|---:|---:|---:|---:|
| 4 | 60 | 120 | 2,304 | 8,704 |
| 8 | 216 | 432 | 4,352 | 16,896 |
| 16 | 816 | 1,632 | 8,448 | 33,280 |
| 32 | 3,168 | 6,336 | 16,640 | 66,048 |

For this always-accepted path, proposal validation, training-loss evaluation,
validation scoring and accepted-state construction jointly predict each new
candidate ensemble six times. Each round adds one tree for squared and two for
Normal. Consequently counts are `3*K*T*(T+1)` here, where K is the number of trees
added per round and T the completed rounds. This is not a formula for arbitrary
recipes, failed trials, optional stopping or arbitrary learner-to-parameter maps.
`Tree.predict` also transforms raw features with its fitted binning on every call.

Trace bytes count distinct ndarray objects directly referenced by step dataclass
fields, deduplicating shared object identities. They exclude object overhead,
other runtime/model buffers and peak allocations; they are not RSS. Source
inspection explains the linear growth in retained round arrays. This measure is
diagnostic and is not a general-purpose alias-aware memory profiler.

These costs were already acknowledged in the B05 learning record; the new evidence
quantifies their continued presence. Earlier Covertype profiling found histogram
and row hashing dominant on its bounded workload. This diagnostic does not replace
that result or establish which cost dominates 300/1000-round real jobs. Its next
use is to constrain a bounded profile and an incremental-state design.
