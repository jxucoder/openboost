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

This is diagnostic evidence, not a formal quality or cost gate. Results pending.
