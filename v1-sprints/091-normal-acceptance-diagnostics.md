# Sprint 091: Measure the Normal acceptance boundary

Status: local investigation authorized by the user's approval after the run-6
retrospective. All six hardware allowances are consumed. No new device execution
or source upload is authorized by this card.

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
