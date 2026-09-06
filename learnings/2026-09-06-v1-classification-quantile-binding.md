# 2026-09-06: Classification and rolling quantile worker binding

## Context

The first exporter assumed each fold partitioned every source row. That fits
random/grouped partitions but contradicts the preregistered Bike rolling origins.
Adult additionally requires train-only categorical encoding and source identities.

## Decision or Result

A5 partitions must form a disjoint chronological source prefix with whole-date
boundaries, and exactly match the frozen row/encoding hashes. Later rows remain
unused in that origin. Other supported tasks retain complete partition coverage.
Adult preserves source/physical-line IDs; Bike preserves instant IDs. Positional
indices continue to verify preprocessing against the original freeze.

## Changes

- Added verified Adult, Covertype and Bike source-to-worker binding for all folds.
- Added train-only category vocabulary verification and source-ID uniqueness.
- Validate all folds before output, materializing one encoded fold at a time.
- Expanded the real worker smoke to seven application paths, with binary and
  seven-class probability checks and three-column quantile schema checks.

## Verification

- `build/v1-env/bin/python -m benchmarks.v1.worker_data_smoke build/v1-worker-classification-quantile-001`: all 35 real-data validation fits passed across seven applications and five folds.
- [Raw summary](../benchmarks/v1/evidence/real-classification-quantile-binding-cpu.json)
  retains CLI, code/data/packet hashes, environment, stopping and output identities.
- All 484 v1 tests passed, including ten binding tests.
- Ruff across production/evaluation/tests, strict MkDocs and diff whitespace checks passed.
- The worker checks model reload before emitting success. No test scores were read.

## Failed Attempts

The full-coverage assumption was identified before the new real-data run and
replaced with a task-specific chronological rule. No split or threshold changed.

## Risks and Follow-ups

A4/A7/A8/A9/A10/A13, coupled controls, complete search/test release and OS access
isolation remain open. These short CPU fits validate plumbing, not task quality
or GPU behavior. Dense intermediate packets are evaluation controls only.

## Commits

- This slice: `eval: bind classification and rolling quantile datasets`.
