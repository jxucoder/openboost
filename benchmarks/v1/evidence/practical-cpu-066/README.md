# Sprint 066 practical CPU diagnostic

The protocol is frozen before execution. Use the original eight Housing numeric
features, fold-zero training prefixes and validation only. No test labels are
uploaded. Input and row-order hashes are in [protocol.json](protocol.json).

Reproduce with `uv run python -m benchmarks.v1.practical_cpu_profile freeze /tmp/openboost-practical-cpu-066`,
then `uv run python -m benchmarks.v1.practical_cpu_profile run /tmp/openboost-practical-cpu-066`.
The source archive must exist at `build/foundation_data/cal_housing.tgz` and match
the canonical Housing freeze. Use a fresh output directory.

Each fit has a 120-second deadline and a conservative 8-GiB process address-space
ceiling. This is distinct from measured guest peak RSS and requested container RAM.
Numerical pools must report two threads. Actual-image resource probes precede fits.
After the first failed fit, remaining fits are not run; only that case is profiled.
If all eight pass, both 8192-row, 128-round cases are profiled. Instrumentation has
its own 60-second soft deadline and is excluded from uninstrumented timing.

This is diagnostic evidence, not a formal quality or cost gate.

## Uninstrumented sweep

[Manifest](sweep/manifest.json) records clean revision `f5ca13b`, source and wheel
hashes, environment, exact arguments, actual-image preflight and per-case artifacts.
The [interruption record](sweep/interruption.json) accounts for all eight cases.
Modal preempted the first 128-round fit and announced infrastructure restart. The
app was explicitly stopped. This is neither a measured deadline failure nor a
memory failure. Timing and peak memory for that interrupted fit are unavailable.
The subsequent Normal 128-round fit was not run. Six completed fits are retained.

| Recipe | Training rows | Rounds | Fit seconds | Guest peak MiB | Trace arrays MiB |
| --- | ---: | ---: | ---: | ---: | ---: |

| squared | 8192 | 4 | 0.4516 | 84.87 | 0.562 |
| normal | 8192 | 4 | 0.8537 | 87.14 | 2.125 |
| squared | 2048 | 32 | 4.3923 | 86.80 | 1.016 |
| normal | 2048 | 32 | 8.8106 | 88.72 | 4.031 |
| squared | 8192 | 32 | 10.9033 | 90.76 | 4.062 |
| normal | 8192 | 32 | 16.0448 | 104.39 | 16.125 |

All six completed fits passed exact full-model raw-state replay and persisted model
replay. Validation metrics were independently recomputed from retained predictions
and frozen validation targets; source hashes were checked against the recorded Git
revision and all artifact hashes checked. Timings are one observation per case,
not repeated estimates or matched-quality comparisons. Trace bytes count retained
step arrays by identity, not process memory. The raw records separate setup, fit,
prediction, export and outer wall time.

The initial [launch failure](launch-error.json) is retained. It ran no remote work.
An explicit zero retry setting is unsupported for Modal generator functions;
omitting it does not prevent infrastructure preemption/recovery.

## Instrumented recovery profile

The [recovery manifest](profile/manifest.json) records clean revision `dff0590` and
identical input/protocol hashes. It reran resource checks and tiny rejection fixtures,
then only the interrupted squared-8192-128 diagnostic. No uninstrumented fits were
repeated. The worker exited 124 at its planned 60-second soft deadline, preserving
[raw profile](profile/squared-8192-128-profile/profile.json), text and pstats.
Manifest `complete` means acquisition finished; worker `error`/124 and profile
`deadline` mean the fit did not complete. There is no final metric or RSS result
for this partial profile.

In 60.001 instrumented fit seconds, Tree.predict accounts for 45.177 cumulative
seconds (75.3%), including binning transform at 40.476 seconds (67.5% of fit).
NumPy searchsorted alone uses 30.312 self seconds (50.5%). Tree construction uses
11.472 cumulative seconds (19.1%). These nested cumulative times overlap and must
not be added. The profile records 21,650 Tree.predict calls while the 85th trial
is in progress; 84 resolves completed. This supports the earlier quadratic replay
counterexample on practical input, without extrapolating an unobserved final time.

Decision: Sprint 067 should remove repeated accepted-ensemble replay and reuse
identity-bound fitted encodings while preserving independent full replay and exact
transaction semantics. Trace retention is measurable but is not the observed
primary CPU bottleneck. The sweep remains incomplete due to infrastructure
preemption, and Normal's long-run profile remains unmeasured. Retest the same
frozen cases after the change; retain these limitations in comparisons.
