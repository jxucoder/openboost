# Sprint 070: Installed comparator bin-budget verification

Clean source `63b4859`; six synthetic CPU A6 fits using the existing locked local
interpreter. XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10; complete Python/OS,
NumPy version, exact CLI, source hashes and generated input are in manifest.json.

| Library | Requested bins | Observed native parameter | Fresh replay |
| --- | --- | --- | --- |
| XGBoost | 7 / 255 | max_bin 7 / 255 | Exact |
| LightGBM | 7 / 255 | max_bin 7 / 255 on all three target models | Exact |
| CatBoost | 7 / 255 | border_count 6 / 254 | Exact |

All six fits retain native stopping records with a three-round patience and
an eight-round maximum. This verifies stopping configuration/records, not that
every fit reaches patience. Each saved model is replayed in a fresh Python process
and predictions compare exactly. Targets include two varying outputs and one
constant output; scaling uses only the 120 training rows. Forty validation rows
and five numeric features complete the deterministic seed-70 fixture.

All three source hashes verify against the clean revision and all thirteen raw
artifact hashes verify. Pickle bundles are trusted local experiment artifacts.
The effective parameters were inspected on the fitted native models, rather than
inferred only from input dictionaries. CatBoost's border count is one less than
the finite-interval budget; quantization algorithms are not asserted equivalent.

```bash
build/v1-env/bin/python -m benchmarks.v1.bin_budget_smoke /tmp/bin-budget-check
```

This is a local synthetic parameter/persistence check, with one thread per fit.
It is not a real-data quality result, a 300/1000-round resource preflight, a Linux
isolation test or CUDA evidence. No data upload or remote jobs were required.
The implementation passed 1115 CPU tests (one Linux-only skip), lint and docs.
