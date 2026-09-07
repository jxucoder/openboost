# Sprint 091: Measure the Normal acceptance boundary

Status: local preparation complete; [run 7](091-acceptance-run7-request.md) is
frozen with upload/compute pending. Local investigation was authorized by the
user's approval after the run-6 retrospective. All six previous hardware allowances
are consumed. No new device execution or source upload is authorized by this card.

## Question and construction plan

Run 6 retains two failures in the original ordered, ordinary, depth-zero Normal
backtracking tests. Before changing the foundation's acceptance contract, determine
whether the device computes incorrect statistics, an incorrect loss, or a rounding
scale loss comparison that cannot agree with a different-precision trajectory.
This supports the programmable foundation's correctness; it adds no recipe scope
and does not close Normal, P7, E4, author benefit or the full R/C/A requirements.

1. Add an independent original-row high-precision loss-difference oracle. Preserve
   exact binary input values, offsets and weights. Record a reproducible study of
   adjacent float32 states around the saved D2 base. The smallest counterexample
   checks whether float64 full-loss subtraction gives the wrong improvement sign.
2. Add separate real-CUDA diagnostic cases that call the unchanged failing test
   with observation wrappers. Retain initialization, round/channel, raw/metric
   bits, gradients/Fisher, root fields/summaries, leaf, coefficient, acceptance,
   version and best prefix before its assertions. Verify observation ownership;
   do not use observed values to select a step. Preserve partial traces on error.
3. Prepare offline analysis at identical captured inputs/state. Distinguish
   original float64 trajectory expectations from mathematics on stored device
   values. Freeze all 383 old tests unchanged plus the diagnostics, exact source
   hashes, output limits and a single-run budget for a new concrete request.
4. After a separately authorized measurement, archive and reflect before changing
   acceptance semantics. A diagnostic-completeness result is not a conformance
   pass. The original failed run remains immutable.

## Acceptance and evaluation

- The CPU oracle has analytic improving/worsening/identity examples, weighted and
  offset cases, a tiny true improvement and invalid-input checks. Its result is
  compared at two Decimal precisions. Agreement is numerical evidence, not a
  universal interval-arithmetic proof.
- The saved-base study records source/input hashes and its CPU environment. It
  never executes or emulates CUDA and cannot reconstruct the missing failed trace.
- New observations call the original implementation exactly once per intercepted
  operation. Owned raw copies are released; no fields, state, RNG or acceptance
  decision are replaced. Unexpected exceptions remain failures. Actual device
  observation invariance and trace completeness require real hardware.
- All old cases, sources, tolerances and immutable artifacts remain unchanged.
  The shared all-pass judge remains strict. Expected old failures may persist in
  a diagnostic run; report them as failures and assess information completeness
  separately, without a phase-exit claim.
- Run focused CPU tests, collect GPU tests without execution, check the isolated
  upload closure, lint and documentation. Commit each verified slice with learning
  evidence. No production acceptance-policy change is authorized by a green
  diagnostic test or by a loss epsilon chosen after measurement.

## Initial counterexample and reflection

At the recorded passing D2 model's float32 base, moving the mean one float32 value
down or up makes float64 full-NLL subtraction negative by one ULP. Independent
80-digit original-row math instead gives positive differences around `5.90e-18`
and `9.73e-17`. This falsifies the assumption that a float64 total-loss comparison
always resolves adjacent float32 candidates correctly. It does not establish
which round/channel or candidate caused either failed GPU assertion. Preserve
that distinction while constructing the oracle and device observations.

## Local slices

- `ba46b3a`: independent difference oracle, fourteen CPU checks and study producer.
- The clean-revision [study artifact](../benchmarks/v1/evidence/normal-acceptance-local-091/README.md)
  is retained separately from device evidence. Added reproduction/trace-analysis
  checks bring the focused suite to 45 passes, including twelve run-6 archive
  checks. Two new real-CUDA observation cases collect without executing.
- The wrappers call the unchanged failing test, record the original assertion
  as a failure, and measure named fields and actual prepared inputs. Constructor
  geometry occurs before run initialization and is captured separately. Logging
  ownership guards and restoration require actual device verification.
- `e0a043b`: original-test observation harness and retained counterexample.
- [Run 7](091-acceptance-run7.json) freezes 70 files and 385 cases with 78 declared
  JSON artifacts, the same pinned dependencies/resource ceilings and no retry.
  Seven dispatch/freeze checks pass. Full CPU regression: **1474 passed**, one
  Linux-only skip. Isolated installed-core collection: all 385 cases, no execution.
  Production/changed-file Ruff and MkDocs pass. Upload and compute remain pending.

## Reflection at the request boundary

The second algorithm family exposed a numerical issue that belongs to the
foundation's acceptance boundary. A float64 full-loss reference is insufficient
as a universal arbiter for near-stationary float32 candidates. The new oracle
demonstrates both false improvements and hidden real improvements, which rules out
a blanket epsilon as an evidence-based fix. No source correction is justified
until the device's exact failing inputs, reductions and decisions are retained.

The next run intentionally measures the unresolved failure without changing its
contract. Its purpose is to choose a defensible comparison/precision policy, not
accumulate passing tests or claim another algorithm family complete. Preserve
the original failure and the split near-tie independently. After one separately
authorized result, reflect again before correcting semantics or expanding CUDA.
Original P7/E4, remaining required recipes and actual independent author benefit
stay on the plan; no author attempt, push or hardware retry occurred here.
