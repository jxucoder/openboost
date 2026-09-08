# Run 12: Bounded binary/Poisson CUDA validation

The separately approved [108 packet](../../../../v1-sprints/108-glm-validation-request.md)
executes once at clean `fe12bebcb8cd1ee7928c270793762e29091df684`.
**All 571 cases pass on Tesla T4**, including all 153 new binary/Poisson cases.
All 85 uploaded files, 35 installed production Python files, eighteen pinned
packages and 77 mandatory JSON artifacts match. The literal [verdict](verdict.json),
[manifest](manifest.json), [JUnit](junit.xml) and [pytest output](pytest.log) remain
unchanged. The twelfth GPU allowance is consumed; no retry or further run occurs.

## Verified boundary

| Cohort | Passing cases |
| --- | ---: |
| Scalar storage, histogram, split, tree, runtime and score symmetry | 212 |
| Normal operations, comparison, runtime consumers and corrected recipe consumers | 167 |
| Parallel field validation and failure recovery | 39 |
| Binary/Poisson objective components | 38 |
| Binary/Poisson prescribed two-round composition | 6 |
| Binary/Poisson comparison, cleanup, transfer and actual PTX | 77 |
| Binary/Poisson recipe trajectories, three anchors, cleanup and persistence | 32 |

The GLM operations share existing fields, split/routing/leaf operations, growers
and transactions. They preserve explicit weights, offsets, Poisson exposure and
binary class metadata. All-row domain rejection, including zero-weight invalid
rows, ownership recovery and absence of host/reporting fallback pass their cases.
Prescribed depth-zero/one/two rounds match independent intermediate and final
mathematics. Sixteen default-grower recipe settings verify depths one/two, fixed
steps, bounded retries, full rejection and distinct accepted/best/patience state.

All 59 standalone numerical reports retain stored inputs, comparison bounds and
direct 220-digit likelihood differences. Binary tails and the stationary Poisson
rounded-loss ties pass their original sign requirements. Both retained PTX records
contain double addition, multiplication and division with each directed rounding
mode. These validate the observed T4 compilation of the bounded cases, not every
possible compiler, GPU architecture or input.

The sixteen recipe reports retain actual fixture dtype/shape/base64 bytes,
including missing-value patterns, final/best models, trial/stop histories and
187 actual comparison records. The offline audit recomputes all **246** retained
comparison differences and verifies their enclosures. All **32** final/best models
round-trip and reproduce the independent reference predictions and prefix/state
decisions from retained inputs. No cross-host input regeneration is necessary.
The remote sixteen final-model replays additionally use the separate NumPy/core-
only environment with CUDA/training imports denied.

The large-step Poisson depth-one/depth-two recipes finish with three accepted
terms but retain best prefixes two/one, respectively, and stop for patience.
Zero-step backtracking settings reject all trials, keep zero terms and stop after
two observations. These observations exercise the separation between current,
best and patience state; the controlled equal-reporting-score tests also pass.

## Environment and cost interpretation

The worker reports Tesla T4 with 15,360 MiB, driver 580.95.05, runtime/driver APIs
12090/13000, Python 3.12.1 and Linux 4.19.0/gVisor/x86_64/glibc 2.35. The image is
CUDA 12.6.3 / Ubuntu 22.04. Two CPU cores and 8,192 MiB are requested; eighteen
guest-visible CPUs do not represent the CPU entitlement. Package versions and
commands are in the manifest, including the separate CPU wheel build.

Pytest completes in **52.93 seconds**; worker time is **55.324 seconds**, below the
600/900-second caps. Total dispatch takes **382.977 seconds**, including image
construction and setup. There is one invocation and no retry. These are
instrumented correctness durations including compilation, Decimal audits,
diagnostic transfers and replay; they are not end-to-end fit-speed measurements.
No GLM speed ratio or formal E4 pass follows.

The 77 retained JSON artifacts total **1,013,949 bytes**, below the 64 MiB cap.
The audit verifies eighty raw artifact hashes plus the manifest binding. JUnit
contains no failed, skipped or duplicate cases. Pytest reports 1,118 warnings,
including the expected small-grid GPU-underutilization warnings; warnings are
retained and do not change numerical assertions or tolerances.

## Reproduction and limitations

The local audit runs on macOS 26.3 x86_64, Python 3.12.12 and NumPy 2.3.5.
It reads the retained array bytes rather than regenerating fixtures on this host.

With the executed production/oracle sources checked out and this evidence folder
available, run locally without CUDA:

```bash
OPENBOOST_BACKEND=cpu uv run --no-sync python benchmarks/v1/evidence/cuda-glm-108/analyze.py --check
```

[The derived analysis](analysis.json) rechecks clean Git provenance, embedded
protocol, installed packages/sources, exact artifact inventory, raw verdict,
numerical bounds, saved input bytes and model/state replay. The analyzer refuses
later production/oracle changes; replay historical evidence with the executed
sources in an isolated checkout. [The archive index](archive-index.json) binds
every retained file except itself. Local integrity controls reject changed
authorization, sources/packages, artifact indexes/hashes and invalid numerical
claims without modifying the raw results.

The combined local audit/freeze/retention controls pass 56 tests in 12.99 seconds.
Final CPU regression passes 2,277 tests with one Linux-only skip in 18.37 seconds;
production/changed-file lint and the documentation build pass.

This establishes bounded native binary/Poisson correctness alongside selected
regressions. It does not close full R1/R4 or Normal conformance, the remaining
required multiclass/AFT/vector-topology cells, compatible train-many, all real
application evaluations, formal E4 or author/adoption benefit. Earlier failures
and omitted regression cohorts keep their own immutable results.
