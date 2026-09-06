# Sprint 012: A5 Bike Sharing data and date splits

Starting revision: `e2f2c6b`. Status: complete for A5 data preparation only.

## Plan and acceptance

1. Fetch the UCI ZIP and record license, archive/member hashes and actual rows. Read hour.csv only,
   allow calendar predictors, exclude measured weather, casual/registered, instant and date from X.
   Original instant is a row ID only.
2. Rolling origins train on the first50/55/60/65/70% of full dates, followed by10% validation and 10% test.
   Endpoints are floor(D*p/100); freeze dates/counts/row-ID hashes. Smallest failure: dates cannot
   cross partitions; weather/component counts cannot affect X.
3. Verify pinned archives with a reproducible CLI record; counterexamples, regression/lint, learning and commit.

No model training/evaluation or test-driven protocol choice. This is A5 only; all other required cases remain.

## Results and evidence

[Freeze](../benchmarks/v1/datasets/bike.json) records archive/member/X/y/row-ID hashes, five windows'
dates/counts/hashes and provenance. Actual data: 17,379 rows, 731 dates, 7 calendar features. UCI's page
says17,389; retain the discrepancy rather than overriding parsed bytes. adapter_sha256 identifies
source; parent revision and dirty=true are honest. This is not an integrity-v0 run manifest or a full training evaluation.

Window train/validation/test row counts: 8645/1744/1750,9529/1746/1752,10389/1750/1752,
11275/1752/1752,12139/1752/1752. Dates/IDs are disjoint within windows, overlapping across rolling
origins; do not interpret them as independent random folds. Use floor over731 dates, not row percentages;
do not synthesize missing hours.

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.bike /tmp/openboost-v1-bike.zip --verify benchmarks/v1/datasets/bike.json
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost benchmarks/v1 tests/v1 tests/conftest.py
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict
```

Real archive replay matched all data fields; corrupting a frozen test count made CLI exit nonzero.
Added24 tests; 360 passed, no skips. Cases cover full dates, uneven hours, leaking fields, invalid
calendar/count, duplicate IDs/timestamps, corrupted archives and adapter source identity.
macOS/Python 3.12.12/NumPy 2.3.5.

## Reflection

Source-page row counts differ from actual files, and hourly records per date vary. Use byte hashes
and parsed data, not webpage metadata or row percentages as split truth. Sandboxed curl DNS failed;
authorized download succeeded. Initial tests lacked the module; later parametrized cases needed
pytest reimported after earlier unused-import cleanup. Final regression passed.
Next: remaining datasets, starting with A1/A11 Housing archive hashes and seeds 3–4; then other
required data/baseline capabilities/budgets. A5 has no model result and F0.3 remains open.

## Commits

- This slice: `data: freeze A5 bike sharing inputs and rolling splits`.
