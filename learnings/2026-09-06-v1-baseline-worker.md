# 2026-09-06: Validation-only baseline worker

## Context

Installed comparator support does not provide the execution adapters needed by
F0.3. Count-model offsets are external to saved tree state in several libraries;
losing that state would silently change predictions after loading.

## Decision or Result

The numeric fixed-round worker supports the declared built-in task subset and
retains its external exposure state in a trusted local bundle. It rejects test
arrays, unknown job fields, unsupported early stopping and invalid target/weight/
exposure/row-ID inputs. It is not yet the complete 16-trial search pipeline.

## Changes

- [Worker](../benchmarks/v1/baseline_worker.py): validation predictions, model bundle,
  replay check before writing artifacts, explicit positive exposure at prediction.
- [Synthetic probe](../benchmarks/v1/worker_smoke.py) and
  [raw result](../benchmarks/v1/evidence/worker-cpu.json): 30 CPU task/library cells.
- [Adversarial checks](../tests/v1/test_baseline_worker.py): reject silent-input paths.

## Verification

- `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 build/v1-env/bin/python -m benchmarks.v1.worker_smoke`:
  all 30 cells passed fit/reload; all three count adapters passed exposure doubling.
- `uv run --no-sync pytest tests/v1/test_baseline_worker.py -n 0 -q`: nine passed.
- Full regression suite: 431 passed, no skips; Ruff and strict MkDocs passed.
  Actual comparator execution used the pinned isolated CPU environment.

## Failed Attempts

Review caught that unused arrays and unknown job options could be ignored.
The worker now rejects them; the synthetic fixture supplies only task-relevant
fields. NaN exposure and nonbinary censoring indicators fail before fitting.

## Risks and Follow-ups

The bundle uses pickle for trusted local comparator artifacts only; never load
an untrusted model with it. Replay is within the process, not the E1 independent
new-process production persistence gate. Inputs are finite pre-encoded numeric
arrays; category/missing preprocessing must be applied by the frozen caller.
Ranking, early stopping, validation-selection receipts, test unlocking, A9 composed
controls, A12 structural comparators and A13 scheduling remain incomplete. A12 here
is only its ordinary GBDT comparator. No OpenBoost model or real quality gate passed.

## Commits

- This slice: `eval: add validation-only baseline workers with offset replay`.
