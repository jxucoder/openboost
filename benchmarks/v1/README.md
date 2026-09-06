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

## A5 real data preparation

[Sprint 012](../../v1-sprints/012-bike-data-freeze.md) freezes the UCI Bike Sharing
hourly archive and five full-date rolling splits in [datasets/bike.json](datasets/bike.json).
Source: Hadi Fanaee-T (2013), [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset),
DOI 10.24432/C5W894, CC BY 4.0. The download has **17,379** hourly rows, versus
17,389 on the UCI page checked on 2026-09-06; archive and CSV hashes identify the
actual data used. Raw data is downloaded locally and is not vendored in Git.

```bash
curl --fail --location 'https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip' -o /tmp/openboost-v1-bike.zip
uv run --no-sync python -m benchmarks.v1.bike /tmp/openboost-v1-bike.zip --verify benchmarks/v1/datasets/bike.json
```

`load_archive` verifies both pinned archive and `hour.csv` hashes before parsing.
`parse_hour` exposes X (float64 calendar columns), y (hourly total count), original
integer row IDs, and ISO dates. No fitted preprocessing is performed. Calendar
values and chronological unique timestamps are checked. All records are retained;
missing hours are not synthesized. IDs/dates support alignment and splitting only.
Observed weather, temperature, humidity, wind and casual/registered counts never
enter X. Numeric/calendar encoding is explicit; later adapters may choose categorical
representations without learning from validation/test.

For D=731 dates, each origin uses floor(D*p/100), floor(D*(p+10)/100),
floor(D*(p+20)/100) endpoints for p=50,55,60,65,70. Each date stays whole.
Origins overlap; they are not independent trials. The unused final 10% of dates
is intentionally outside the predeclared windows, not an extra selection set.
The freeze contains per-part row counts, date boundaries and row-ID hashes, plus
feature/target array hashes and adapter source hash. `--verify` refuses altered
source or data/splits; replay provenance (revision/environment/argv) may differ.

This **data preparation record is not an integrity-v0 run manifest**. It honestly
records a dirty checkout and identifies the adapter bytes separately. It does not
freeze training budgets/configurations, run any quantile model, or pass E3/A5.
