# v1 evaluation preparation

Sprint 011 adds an **artifact integrity judge**, not the full F0.3 runner or E0–E7
judge. No real dataset manifest, installed baseline capability result, frozen
resource budget or held-out author task is delivered by this slice.

Run from a repository checkout with the project environment:

```bash
uv run --no-sync python -m benchmarks.v1.judge /absolute/path/to/run
```

The command reads `manifest.json` and `cases.jsonl`, prints a JSON report to stdout,
and exits 0 only for `integrity_pass: true`; otherwise it exits 1. It does not train,
execute artifact code or modify the run directory. `gate_results` is always empty.
A producer's `pass` is a status claim, never an independently verified quality score.

## Integrity schema: `openboost-integrity-v0`

Objects use exactly the documented fields; unknown versions/fields are rejected.
JSON duplicate keys, non-finite numbers and blank JSONL records are rejected.

Manifest fields:

- `schema`: the version above; `protocol_sha256`: lowercase SHA-256 of the declared protocol.
- `provenance`: `code_sha` (40 lowercase hex), `dirty` (must be false in this version),
  `environment` (nonempty JSON object). Dirty code needs a patch digest in a future
  version; a boolean alone cannot identify different uncommitted implementations.
- `expected`: nonempty list of cells. Each has `id`, `application` (`A1`–`A13`),
  `required` (boolean), `backend` (`cpu`/`cuda`), `model`, `fold` (nonempty strings),
  `seed` (nonnegative integer), `dataset_sha256`, `split_sha256`,
  `preprocessing_sha256`, and `config` (JSON object).

IDs must be unique; every A-ID needs a required CPU cell. This only checks the
**declared matrix**. The future frozen protocol must independently define actual
fold counts, methods, datasets, budget and device requirements; one cell per A-ID
is not a sufficient experimental design. This schema does not yet validate
package/wheel metadata, real data hashes, license, row IDs, target units, resource
limits or the authenticity of recorded provenance.

Each JSONL case has exactly:

- `id`, `status` (`not_run/pass/fail/unsupported/error/timeout`), `cache_key`;
- `backend` (`cpu/cuda/none`), `fallback` (boolean), `exit_code` (integer or null);
- `artifacts`, `metrics` (finite numeric values by metric name), `reason` (string).

`cache_key(manifest, cell)` hashes the canonical JSON of the **entire manifest and
cell**. Code, environment, protocol, matrix, data, split, preprocessing, config and
seed changes invalidate old keys. This conservative version has no cross-matrix
cache reuse. It cannot detect an input that a producer failed to declare.

All cells, including optional cells, must have a record and a hashed `log` artifact.
Required non-pass records fail integrity. Optional non-pass records remain visible
with a nonempty reason; they cannot establish GPU or quality success. A pass record
requires exit code 0, the expected backend without fallback, nonempty metrics, and
hashed `predictions`, `model`, `log` artifacts. Artifact entries have `path` (relative
inside the run directory) and `sha256`; path escapes and symlinks outside are rejected.
Model and log bytes are hash-checked but not interpreted. Prediction bytes must be
JSON containing a nonempty rectangular finite numeric array. Matching prediction
rows/output semantics to targets and recomputing metrics belong to the forthcoming
independent evaluators. No dataset data is confused with prediction validation:
legitimate censored-label infinity remains part of the AFT target contract.

## Verification and next steps

[Adversarial tests](../../tests/v1/test_artifact_judge.py) generate clearly synthetic
bundles in temporary directories. They exercise missing/duplicate/unknown cases,
input changes with stale cache keys, false pass claims, worker failures, unsupported
GPU, missing/corrupted artifacts, invalid prediction values and the CLI exit code.
Synthetic manifests are test inputs, not committed benchmark results.

Next F0.3 slices must acquire and hash real datasets, implement their split/target
adapters, smoke-test pinned baselines, freeze resources/configurations and held-out
verifiers, add the execution runner and independent metric/gate evaluation. Full
benchmark runs remain prohibited until actual hashes and budgets are frozen.
