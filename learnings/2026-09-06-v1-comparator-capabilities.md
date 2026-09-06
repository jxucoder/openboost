# 2026-09-06: Real CPU and CUDA comparator capabilities

## Context

F0.3 requires installed comparator evidence before a fair quality protocol can be
frozen. Documentation alone cannot establish supported devices or persistence.

## Decision or Result

The isolated real-T4 matrix has 29 CPU and 28 CUDA passing built-in task cells,
with four/five explicit unsupported cells. These are synthetic fit/reload checks,
not OpenBoost capability, task quality, speed, or F0.3 completion.

## Changes

- [Capability probe](../benchmarks/v1/capability_smoke.py) and bounded
  [Modal harness](../benchmarks/v1/modal_preflight.py), hash-locked CUDA packages.
- Weighted mixed-data/base-offset and secondary NGBoost/GLM/AFT/formula probes.
- [Raw evidence and failure index](../benchmarks/v1/evidence/README.md).

## Verification

- `uv run --no-sync modal run benchmarks/v1/modal_preflight.py::main`: all
  supported CPU/CUDA cells passed in independent processes on a real T4.
- Pinned macOS CPU capability, adapter and secondary probes passed their declared
  cases. Raw artifacts record versions and source hashes. Ruff passed.
- The source/lock hashes identify the probe bytes. The isolated result does not
  record a harness digest; native build isolation dependencies are not fully pinned.

## Failed Attempts

The standard LightGBM wheel lacks CUDA; native compilation initially selected
missing Clang. GCC compilation succeeded. A subsequent single-process mixed-library
probe aborted in native CUDA code. Independent bounded processes preserved all
results and passed supported cells, without claiming the native root cause solved.
XGBoost reload must restore device before applying same-device tolerance. Original
cross-device deviations remain recorded; existing E1 tolerance was not changed.

## Risks and Follow-ups

Task semantics beyond these probes, real quality, full selection/test integration,
ranking data, unresolved licenses and independently frozen held-outs remain open.
Fresh-process isolation is required for evaluation. Public v1 training code remains
F1 work. See [Sprint 016](../v1-sprints/016-f0-3-completion.md).

## Commits

- This slice: `eval: verify isolated CPU and CUDA comparator capabilities`.

Py-Boost 0.5.2 subsequently passed weighted scalar/vector MSE fit and JSON reload
on the real T4 with zero prediction differences. The first run's `verbose=0`
callback failure is retained; the corrected interval is 10. Full suite after
support changes: 431 passed, no skips; Ruff and strict documentation build passed.
