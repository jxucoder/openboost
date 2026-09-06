# 2026-09-06: Counts, paid amounts and survival worker packets

## Context

Insurance application populations differ despite sharing policy partitions.
Poisson exposure offsets and Tweedie exposure weights cannot be interchanged.
Survival requires event state and a training-only censoring distribution.

## Decision or Result

Map policy partitions to retained positive claims or eligible policies before
using their frozen encoders. A7 receives period counts plus exposure offsets;
A8 receives individual payment amounts and unit weights; A9 receives annualized
paid totals and one exposure weight. A10 exports events and the frozen training G.

## Changes

- `worker_data.py`: application population binding, weights/offset contracts,
  period artifacts, event arrays and hashed censoring support JSON.
- `worker_data_smoke.py`: selectable applications, positive-mean and AFT output
  checks; all executed commands remain recorded.
- Four hand-worked tests cover shared policy splits, individual claim units,
  annualization/weighting, offset inputs and changed censoring support rejection.

## Verification

- `build/v1-env/bin/python -m benchmarks.v1.worker_data_smoke build/v1-worker-positive-survival-001 --applications A7 A8 A9 A10`: all 20 real-data CPU validation fits passed.
- [Raw summary](../benchmarks/v1/evidence/real-positive-survival-binding-cpu.json)
  retains actual command, source/data/packet hashes, environment and worker receipts.
- All 488 v1 tests passed, including 14 binding tests. Ruff across production,
  evaluation support and tests plus strict MkDocs passed.
- Each worker checks saved-model replay before emitting artifacts. Positive means
  and two-column fixed-scale AFT predictions passed; no test truth was scored.

## Failed Attempts

No failed fixture or real-data worker in this slice. No clipping, altered exclusions or new split rules
were introduced to make the worker inputs acceptable.

## Risks and Follow-ups

The numeric packets do not bind GLM/composed A9 controls yet. A4/A13, coupled
controls, full search/test release and restricted execution remain open.
Veteran's original-source license record remains unresolved. CPU smoke is not
survival quality, a CUDA check or full F0.3 acceptance.

## Commits

- This slice: `eval: bind counts paid amounts and survival worker inputs`.
