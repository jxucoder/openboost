# Covertype fold zero after invariant row-hash hoisting

The unchanged four-round/depth-two/32-bin/seven-class job completes in 87.4 seconds
under the original 90-second cap. Fresh-process seven-class probabilities and row
IDs match exactly. This is one bounded rerun, not a comparative speed benchmark
or full five-fold A3 acceptance. The prior original run timed out at 90 seconds.

`manifest.json` records the exact job/command, input hash, source/revision identity,
process result and fresh-replay command/hash. Raw model, validation predictions,
replay, training metadata and process records/log are retained. Regenerate the
Sprint 049 packets if needed, then run the recorded command through process_runner
with timeout_s=90 and threads=1 in a fresh output directory. Replay has a 30-second
cap. Source export time is outside this worker measurement.

Only candidate row-digest hoisting changed. No profiling or concurrent regression
run was active during this fit. No memory cap or CUDA. A single result near the
cap does not establish reproducibility, quality parity or a stable speed ratio.
