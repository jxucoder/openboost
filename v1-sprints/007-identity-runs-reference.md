# Sprint 007: Identity, run isolation and model selection

Starting revision: `c0e0b2c`. Status: sprint complete; F0.2 ongoing.
B01/F0.2, independent A13/D5/C1/C4 references.

## Plan and acceptance

1. Typed content hashes and row-ID binding: equal shapes do not permit reuse across different
   contents/folds/schema/cuts/dictionaries. Target/weight/offset belong to Problem identity, not only prepared data.
2. Sequential runs fit actual small scalar/vector squared-loss trees with independent budgets,
   seeds, best and early stopping. Stable sampling keys make independent/sequential/reordered/regrouped
   results equal. Preserve failures without contaminating other runs.
3. Select completed runs by validation only; stored best terms reconstruct raw. Pin an RNG fixture,
   audit F0.2 gaps, run regression/lint, reflect and commit.

Hashes are semantic references, not production PreparedData, persistence or security signatures.
Regrouping simulates sequential scheduling, not batched GPU execution; no speed/memory claim.
Injected failures test handling; actual required failures cannot count as passes. E-gates remain incomplete.

## Results and verification

Added runs.py, 21 tests: **229 passed, no skips**, Ruff passed. Import-isolated subprocess executes
an actual run and a fixed-seed assertion.

- DataIdentity stores typed SHA256 plus immutable row IDs, covering values/order/schema/transformer
  version/cuts/dictionaries. NaNs normalize; None/string/int/float remain distinct; mapping key order
  is irrelevant. Binding checks prepared order and every named role; hashes do not validate target support.
- Review found digest-only identities cannot validate redeclared row order. Store row IDs too;
  reversing all fields together still cannot bind to the original prepared ordering.
- Real squared-loss trees have separate K=1/K=2 raw/budgets and joint output commits per round.
  M=1/8/32 preserve all records. Independent/sequential/reverse/regrouped results match exactly;
  regrouping is not parallelism.
- Best changes only on strict validation improvement; ties retain earlier rounds. Base is round0.
  Independent patience works; harmful training restores base. Stored terms/coefficients reconstruct best_raw.
- Failed runs keep errors/completed rounds while others remain unchanged. Same-ID retry uses the
  same first sample. Invalid config/no problem/all-failed cases are explicit; another available model
  cannot erase a required failure.
- RNG uses first64 bits of SHA256 over typed UTF-8 JSON key(seed, run_id, round, component, purpose),
  excluding attempt/scheduler position. `(7,'run-α',2,'tree','rows')` → `6175955064790668999`.
  PCG64 is a CPU reference, not a cross-backend/future-NumPy bitwise promise.
- Selection reads successful validation-best records only. Different Problem IDs cannot be mixed
  even when losses are numeric. Same-problem ties use run ID; heterogeneous execution does not imply comparable metrics.

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
```

Local macOS CPU/Python 3.12.12/NumPy 2.3.5/pytest9.0.2. Missing-module collection failed first;
first7 cases passed, then boundaries/fixed keys/M counts and lint were completed. No GPU,
production cache/runtime, real selection cost, persistence or E-gates ran. Run recipe is unit-weight
squared loss only; general binding of weight/offset roles does not mean other recipes were executed.

## Reflection

K versus M and prepared versus Problem identity affect correctness beyond shapes. Heterogeneous K
runs stay isolated; changed targets/validation change Problem IDs and reject incomparable selection;
uniformly reversed fields still violate prepared order. Public foundation must retain explicit
input/target/output contracts. Scheduling executes; it cannot guess metric comparability.
Next: [F0.2 ledger](f0-2-acceptance-ledger.md) tracks exact D1/D3/D4 fixtures, full categorical/vector
growth and finite compositions. Fill those before closing F0.2 and starting F0.3. The229 tests are
not v1 product completion, nor a reason to keep adding the same mathematical tests indefinitely.

## Commits

- This slice: `test: add identity and isolated run references for v1`.
