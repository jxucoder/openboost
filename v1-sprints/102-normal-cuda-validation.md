# Sprint 102: Normal CUDA validation and prefix-expectation retrospective

Status: the explicitly approved run executes once at `469ca0e`. All 409 declared
JSON artifacts return. Revised results are **528 passes, one failure**; historical
results are **359 passes, 26 failures**, exactly the preregistered disagreements.
The overall verdict remains false. The GPU/upload allowance is consumed without
retry. Agent evaluation and the Sprint 100 model test remain deferred.

## Plan and acceptance

1. Verify the source closure and local dispatch/judging checks, record the exact
   approval and commit before execution. Complete: all 37 local checks pass and
   all 85 prefrozen source hashes match the 86-file upload closure.
2. Execute the original [run-8 request](092-comparison-run8-request.md): one T4,
   two CPUs, 8192 MiB, 900 function seconds, a shared 600-second test deadline and
   zero retries. Complete: no resources, cases, tolerances or source hashes changed.
3. Preserve every raw verdict/artifact, verify provenance and complete the
   retrospective before further hardware. Complete: 416 raw artifact hashes and
   86 dispatch source hashes verify, including 409 declared JSON outputs.

Full acceptance still requires all 529 revised cases, the exact historical
outcomes and complete artifacts. Classifying a fixture error does not pass the
failed case or execute its later assertions. The known split near-tie, original
P7/E4, other required CUDA recipes and all application requirements remain open.

## Result

[Raw evidence and environment](../benchmarks/v1/evidence/cuda-comparison-092/README.md)
retain both JUnit/logs, source/package identity, comparison traces, PTX and cost.
The installed core/D2 sources and eighteen pinned package versions match. All
nineteen models per cohort replay on CPU without CuPy, Numba or the D2 extension.

All 383 revised original requirements, 117 comparison operation checks and twelve
transaction consumer cases pass. Fourteen of fifteen dedicated recipe cases pass;
both lowering/cost checks pass. Actual PTX contains the six required directed-double
operations. All 2,548 recorded trajectory comparisons pass their independent
stored-input audit, including 1,381 improvements, 946 worsenings and 221 unchanged
states. The two old false-improvement candidates are now correctly classified as
worsening, and the complete revised forward/reverse conflict trajectories pass.

The remaining failure is
`test_recipe_current_best_and_patience_have_distinct_validation_anchors[forward]`:
observed `best_n_terms=9`, expected `10`. Its stale-round sequence and budget stop
already pass. Assertions after this prefix check, including its final explicit
ownership check, do not execute and remain unverified for that particular case.

## Root cause and retrospective

The fixture changes only the mean and supplies zero-valued log-scale leaves.
In forward order, term nine creates the final improvement and term ten is a no-op.
Best-model selection requires strict improvement, so a no-op must retain the
existing best prefix. The exact-rational fixture derivation gives joint ten,
forward nine and reverse ten. Joint commits both terms atomically; reverse places
the last mean change at term ten. This matches the observed state and existing
consumer contract. The error is the unconditional ten-term test expectation.

Correct that expectation in a separate source change. Add a CPU semantic check
for ordered versus joint commits and retention of an earlier best prefix after
an accepted no-op. Keep the original raw failure and its frozen source revision;
do not rehash the consumed run-8 packet or replace its verdict. The corrected CUDA
case still needs actual execution under a new frozen allowance. No production
numerical policy or tolerance needs to change for this counterexample.

Uninstrumented tiny-fixture fits take 0.226–0.233 seconds for weighted data and
0.348–0.350 seconds for D2. Each performs fifteen comparisons with 480 comparison
bytes exported. Reference NLL/CRPS and repeated model identities match. Full
dispatch is 633.550 seconds; worker time is 205.335 seconds. Image construction
and instrumented correctness work are distinct from training cost. These figures
justify no matched-quality speed or real-workload claim.

The numerical comparison design now has substantial real-device support. Finish
the narrow fixture correction and its validation before adding more CUDA recipe
families. Continue with 080, compatible train-many and measured workload cost
thereafter; the deferred agent study does not re-enter this sequence.

## Verification and authorization history

The [offline analyzer](../benchmarks/v1/evidence/cuda-comparison-092/analyze.py)
verifies raw hashes, source identity, recorded enclosures, PTX hashes and CPU replay
bindings, and derives the prefix independently. Its analysis never changes the
literal failed verdict. Run it with `--check` to compare against the saved report.

The earlier `finish` instruction was initially interpreted as approval in
`b58b168`; automatic review blocked process creation, and `dffcfd5` restored the
pending packet. No upload or allowance was consumed then. The subsequent explicit
`approve` response authorized the exact 86-file upload and bounded invocation;
`469ca0e` is the clean execution revision. All eight GPU allowances are now consumed.

## Local fixture correction after the retrospective

The raw archive and analysis are committed at `c36f96a` before changing tests.
The CUDA assertion now checks ten accepted terms independently of the best
prefix, expecting nine only for forward order. All later ownership, prediction,
stopping and comparison assertions remain. No production code, tolerance or
original raw artifact changes.

Three new CPU cases exercise the public transaction path with the same controlled
mean sequence. They verify every committed best prefix, including joint atomic
boundaries, ordered zero terms and identical final current/best predictions.
All thirteen focused consumer checks pass in 0.26 seconds. All fifteen device
recipe cases collect with unchanged test IDs; none reruns on hardware in this
slice. The historical analyzer still verifies every archived source at `469ca0e`.
Full CPU regression passes 1949 tests with one Linux-only skip in 13.25 seconds.
Ruff and documentation checks pass; the existing 090 evidence-link warning remains.

The consumed run-8 protocol retains its original test-source hash. It intentionally
differs from the corrected active test; never regenerate that consumed freeze to
hide the difference. A subsequent hardware packet must identify the corrected
source and preserve the original failed result. The full revised gate remains open.
