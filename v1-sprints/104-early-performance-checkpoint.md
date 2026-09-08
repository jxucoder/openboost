# Sprint 104: Early squared/Normal performance checkpoint

Status: closed after the single approved run; measurement acceptance failed
with one passing and five timed-out cases. The allowance is consumed. The user's
approval of an early GPU performance checkpoint after Sprint 103 moves this bounded measurement ahead
of further 080 recipe ports; it does not replace or relax formal 082/E4.

## Question and plan

Does the existing programmable implementation execute practically as row count
grows, and how much cost comes from compilation, dispatch, synchronization,
transfers, preparation and inference?

1. Construct deterministic 10,000/100,000-row squared and Normal workloads and an
   independent quality/measurement judge. First test that a faster result with
   worse quality or incomplete measurements cannot receive a speed ratio.
2. Verify the CPU path and freeze the exact sources, workload, case deadlines,
   quality thresholds, package versions and one bounded T4 execution before upload.
3. Measure first and repeated warm fits in fresh per-case processes on the same
   host. Run profiling separately. Preserve raw results and failures, commit them,
   and select the next optimization from the retrospective.

## Scope

Use synthetic data at realistic row counts; this is an execution checkpoint, not
real-data quality or adoption evidence. No external dataset or model API is needed.
Both backends use identical float32-origin inputs, training-derived binning,
fixed-step depthwise trees, seed and round budget. Normal uses joint Fisher updates.
CPU remains float64 internally and CUDA float32; measure prediction and task-metric
differences explicitly. A quality failure withholds a comparative speed ratio.

Measure dataset construction, host binning, fit/export, model load, single-row
and batch CPU prediction independently. CUDA-trained models use their supported
CPU inference path. First fit includes triggered compilation; first/warm differences
are not pure JIT time. A separate profiled invocation records compiler calls,
kernel dispatch, synchronization and blocking transfer call costs. Inclusive
profile categories overlap and must not be summed as an exclusive decomposition.

Use existing source/installation checks and the existing T4 runner limits:
one container, two CPUs, 8192 MiB, 900 function seconds, 600 test seconds and zero
retries. The concrete source freeze and child deadlines follow local calibration.
Preserve all previous consumed allowances and raw evidence. No production
optimization is bundled into the initial measurement.

## Acceptance and retrospective

All four primary workload/backend pairs and the additional control must complete with finite metrics, identical
inputs/settings, complete repetitions and explicit provenance. Validate exported
model replay, declared rounds and resource cleanup. Compare held-out predictions
and squared loss or Normal NLL/CRPS against the CPU run, and independently recompute
metrics from saved predictions. Report speed ratios only for comparable quality.
Publish timeouts, mismatches and compilation/cache limitations.

The purpose is to expose costs, not require a speed win. Formal external-library,
multi-seed real-data, train-many and E4 budgets remain in 082. No XGBoost/LightGBM/
CatBoost speed claim follows from an internal CPU/CUDA comparison. Keep the
author-study deferral active. Stop after this checkpoint to choose a measured
optimization or return to required CUDA recipe construction.

## Concrete freeze and execution bounds

[Run-10 protocol](104-performance-run10.json) freezes 46 uploaded files (about
358 kB), including 31 unchanged production files. The user approved this checkpoint
in response to the recommendation to measure the existing GPU paths at realistic
sizes. These are implementation bounds for that approved task: one Modal T4,
two CPUs, 8192 MiB, 900 function seconds, 600 test seconds and zero retries.
Image construction is outside the function cap. No external data is uploaded;
the remote children generate their own deterministic fixtures.

Use 16 numeric features with 1% missingness, nonuniform weights/zero-weight rows,
mean offsets, 32 bins, depth three, twenty fixed rounds and learning rate 0.1.
Normal uses joint Fisher updates; both backends export the selected best model.
CPU has one first and one warm fit; CUDA has one first and three warm fits.
All seeds are seven. This small repeat count is a checkpoint, not a confidence
interval or a substitute for 082's three-seed real-data matrix.

The local development calibration takes approximately 26 seconds per Normal fit
even at 1,000 rows. The CPU comparison loops over individual rows using interval
arithmetic. Preserve this observation as a reason to bound larger probes, not a
new CPU optimization or a qualified speed ratio. Add a 1,000-row Normal control
to the four required 10,000/100,000-row squared/Normal cases. CPU child limits are
45/60 seconds for squared and 90/30/30 seconds for Normal's three sizes; each CUDA
child has 50 seconds. A separate first/warm Normal 100,000-row profile has 60
seconds. Total child caps are 565 seconds within the shared 600-second test cap.
Timeouts remain failures, including partial completed repetitions. Never replace
the public CPU reference with a cheaper algorithm to make the comparison finish.

Both CPU and GPU children execute in fresh processes on the same host with one
BLAS/OpenMP thread. CPU uses an isolated NumPy/core installation. Each GPU child
gets a distinct CuPy/driver cache path; first-fit JIT remains part of first-fit
cost. Save every raw timing, model, held-out prediction, state count, buffer/transfer
counter and child outcome in sixteen declared JSON artifacts, capped at 32 MiB.

The independent metric formulas must reproduce saved scores. A pair is eligible
for a warm CPU/GPU fit ratio only if complete, unprofiled, source/input/config
identities match, exported models replay exactly, repetitions retain identical
models, all rounds/terms complete, CUDA cleanup reaches zero, each task metric
differs by at most 1%, and per-channel normalized prediction RMSE is at most 1%.
These thresholds qualify this timing comparison; they do not relax E1 conformance.

## Local verification and command

Six exact GPU test IDs collect from the isolated wheel/source snapshot; zero
device cases have executed locally. Sixteen local checks exercise real CPU model
export, the independent Normal formula oracle, invalid timing/quality cases and
an actual subprocess deadline that retains partial work. The source audit caught
and corrected an invalid combination of explicit binning and a separate bin-count
argument before freeze. No production source changes are needed.

After the clean freeze commit, execute once:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.cuda_performance_preflight benchmarks/v1/evidence/early-performance-104
```

The fixed output must be absent. This is the tenth GPU allowance; the first nine
remain consumed. Preserve its complete or failed result and stop for retrospective.

## Automatic approval review and pending upload

The concrete packet is committed at `8677bba`. Automatic approval review rejects
the dispatch before process creation: the performance study is approved in
substance, but uploading private repository sources to Modal requires explicit
authorization of the payload and destination. No output directory, upload or
remote invocation occurs, so no run-10 allowance is consumed. Do not retry or use
another transfer mechanism to bypass this decision.

The upload authorization is restored to pending; compute/study authorization
remains approved. All 45 prefrozen source hashes, six cases, sixteen artifact
paths and resource limits remain unchanged. The earlier collection report retains
its collection-time protocol hash and still means zero executed GPU cases.

The remaining approval is the concrete **46-file, approximately 358 kB source
upload to Modal** for this one T4 invocation: two CPUs, 8192 MiB, 900 function
seconds, 600 test seconds and zero retries. The protocol lists every file and hash.
Image construction is additional to the function cap. No other external action
or broader repository upload is included.

## Explicit upload authorization

The user replies "approve" to the concrete request above on September 8, 2026.
This authorizes the exact 46-file, approximately 358 kB source upload to Modal and
one T4 invocation with two CPUs, 8192 MiB, 900 function seconds, 600 test seconds
and zero retries. Change only the protocol's upload authorization to approved;
all 45 prefrozen source hashes, cases, thresholds and deadlines remain unchanged.
The original local collection report remains historical. Commit this authorization
before dispatch, execute once, preserve any failed measurements and stop for
retrospective. No additional invocation or upload is authorized.

## Run-10 result and retrospective

Execute once at clean `c8f7ebc57274399dbe120cd97d7933455e6559e4` on real T4.
The [raw evidence and offline audit](../benchmarks/v1/evidence/early-performance-104/README.md)
retain all sixteen declared JSON artifacts, JUnit, log and the false verdict.
All 46 uploaded hashes, 31 installed core sources and eighteen pinned packages
match. Worker time is 463.174 s; total dispatch time including image setup is
547.098 s. No retry occurs. The embedded dispatch protocol remains approved;
the active protocol separately records both allowances as consumed.

Only squared 10,000 rows completes both backends and qualifies: CPU first/warm
7.006/7.044 s, GPU first/three-warm median 9.394/2.941 s. The warm ratio is 2.395,
relative half-MSE difference 1.09e-7 and normalized prediction RMSE 1.95e-7.
This is an internal synthetic observation, not competitive speed evidence.

Squared 100,000 rows completes CPU first/warm fits in 16.744/16.928 s. GPU retains
20.486 s first and 13.757/13.748 s warm fits before the 50 s child timeout; the
required third warm fit and final quality artifact are missing. No ratio qualifies.
Normal GPU warm medians are 5.111 s at 1,000 rows and 6.858 s at 10,000 rows.
CPU Normal retains one 48.009 s fit at 1,000 rows before its 90 s timeout, and no
completed fits at 10,000/100,000 rows within their 30 s caps. Normal GPU 100,000
rows retains only its 49.503 s first fit before timeout. All these pairs fail.

The separate 100,000-row profile retains one 57.953 s fit before its 60 s timeout;
the warm profile is missing. Its inclusive export wait is 28.964 s for 5,553
blocking exports, while launch cost is 22.766 s including 15.694 s in compiler
calls. These overlap and cannot be summed or treated as pure transfer/compiler
times. Every retained finished GPU fit closes its owned buffers to zero.

Three conclusions govern the next work:

1. The GPU implementation works at these sizes, but has substantial fixed and
   row-dependent cost. Normal 10,000 rows requires 9,417 launches and 12,982
   synchronizations. Squared 10,000/100,000 rows keep 4,531 launches while warm
   fit time rises from 2.94 s to observed 13.75 s. Dispatch stays near 1.4 s.
   Source inspection shows serial validation/reductions and full-row histogram
   scans; the profile does not isolate which kernel dominates.
2. Local Normal calibration underestimated the remote CPU first-fit cost, and
   the GPU/profile deadlines cannot accommodate all required repetitions at the
   observed sizes. Partial fits are valuable diagnoses but never completed
   timing pairs. Future packets need measured deadline headroom and progress
   snapshots containing models and quality, without changing this run's verdict.
3. The synthetic generator is not an exact portable input artifact. On macOS
   x86_64/NumPy 2.3.5, regenerated Problem hashes differ from Linux even though all
   five fully saved models replay exactly and local metric differences are at
   most 4.74e-9. The exact cause remains unisolated; a double-sine substitution
   does not fix it. Same-host CPU/GPU identities and the remote independent judge
   pass for the qualifying pair. Retain this limitation and save exact inputs
   in the next packet; do not substitute local targets into the frozen judgment.

The [proposed next slice](105-parallel-validation-and-reproducible-cost.md) first
addresses parallel boolean/domain validation without changing numerical reduction
or acceptance policy, and repairs input/progress retention. Stop here for the
planned retrospective boundary. Further hardware requires a new concrete freeze
and allowance. Required 080 recipes, 081 train-many, 082/E4, all R/C/A families
and deferred author evidence remain open; none is replaced by this checkpoint.
