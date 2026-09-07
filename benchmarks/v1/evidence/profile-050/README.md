# Bounded full-input Covertype profile

This is a deliberately interrupted instrumented diagnostic, not a successful
training run or a speed comparison. The full Sprint 049 fold-zero job was run
for 60 seconds with a 90-second outer cap and one thread. Exit 124 is expected.
No core/recipe changes or reduced dataset were used.

`manifest.json` records exact command, job/input hashes, source hashes and outer
execution outcome. `profile.json` contains the top 100 cumulative call records;
`profile.txt` is a readable view and `profile.pstats` the trusted local binary
profile. `stacks.txt` contains periodic Python stack traces. Cumulative call times
overlap. cProfile overhead and an initial overlapping local regression run mean
these timings are diagnostic, not fair performance measurements.

To reproduce, regenerate the Sprint 049 packets if needed, then invoke the
recorded `profile_worker.py JOB --seconds 60` command in a fresh output directory
through `process_runner.execute` with timeout_s=90 and threads=1. The runner must
retain its hard cap because Python alarms can be delayed in native code. Source
export time is outside the timed worker. No test labels were scored.

The profile prioritizes histogram aggregation and repeated per-candidate row
hashing. Prediction-time transforms are present but are not the leading measured
path. See Sprint 050 for the next exact-semantics optimization.
