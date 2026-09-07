# Sprint 070: Real A6 candidate-scoring profile

Clean source revision: `dede27e`. The outer manifest pins all 32 source hashes,
the unchanged approved Parkinsons train/validation packet and configuration-00
resource plan. All six returned artifact hashes verify against the saved bytes.

## Result

The Modal diagnostic exits 124 at the intended 60-second soft deadline and retains
raw pstats, JSON and text timings. The 90-second hard child limit does not fire.
This is diagnostic completion, not a model fit or quality pass. Periodic stack
dumps are disabled explicitly in the recorded command.

| Call path | Cumulative seconds | Calls |
| --- | ---: | ---: |
| choose | 52.273 | 32 |
| vector_score | 39.420 | 256753 |
| vector_feasible | 11.562 | 271986 |
| vector_leaf | 28.933 | 770324 |
| _vector_indices | 8.836 | 1299063 |
| candidates | 7.017 | 32 |

Cumulative times overlap: vector_leaf and field lookup are descendants of scoring.
Do not add these rows or interpret them as end-to-end speedup estimates. The sample
includes three depthwise calls and 32 candidate-choice calls; it is a prefix of
the full workload. cProfile changes runtime and samples a single input/topology.

## Failed instrumentation attempts

The first Modal attempt at `6c608b2` exits -11 after 20.19 seconds while writing an
incomplete faulthandler stack dump; no profile is retained. The separate macOS
attempt retains timings but hits its 90-second hard limit with a truncated stack
dump. Both remain committed under a6-profile-modal-failed-070 and
 a6-profile-local-failed-070. The revised run succeeds with stack dumping disabled.
This supports isolating that instrumentation interaction; it does not establish
a universal explanation for native crashes or a fault in uninstrumented training.
The user explicitly approved the additional real-data Modal profile after review
initially restricted prior approval to the two completed resource probes.

## Next bounded implementation

Prepare invariant vector statistic indices and scoring parameters once per default
scorer, and investigate reusing the identical parent score within a candidate set.
Preserve public scalar operations/custom callbacks, finite/curvature validation,
leaf arithmetic order, strict-positive gain and lexicographic tie behavior. First
write conformance cases for malformed fields, zero/negative curvature, overflow,
regularization and near ties. Compare exact candidate choice, model bytes and
predictions before any performance claim. Re-profile this same packet/configuration
and retain full-fit paired evidence if the candidate change passes correctness.

Do not widen the search or start an unrelated optimization from this profile.

```bash
uv run --no-sync python -m benchmarks.v1.a6_resource_preflight /tmp/a6-profile /tmp/a6-packets --profile
```
