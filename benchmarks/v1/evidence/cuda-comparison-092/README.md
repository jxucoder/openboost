# Normal comparison run 8: numerical correction passes, one prefix assertion fails

The one approved T4 invocation at clean `469ca0ea4fa2ca0af3ea007bfa8c0f02e51e2d95`
returns **528/529 revised passes** and **359/385 historical passes**, with no
skips or errors. The historical cohort has exactly its 26 preregistered
disagreements. The overall [verdict](verdict.json) remains **false** because the
revised cohort has one failure. The allowance is consumed; no retry occurred.

The [manifest](manifest.json) binds all 86 uploaded source hashes to that revision,
the installed core and D2 sources, eighteen package versions, both test commands
and all 416 raw artifact hashes. All **409 declared JSON outputs** are present.
The [historical JUnit](historical/junit.xml), [revised JUnit](revised/junit.xml) and
their `pytest.log` files retain literal results. Derived analysis is separate from
this immutable dispatch evidence.

## What the device run establishes

- All 106 stored-input comparison cases pass, including both run-7 false
  improvements. The coefficient-4 training change is bounded between
  `5.904800678568961e-18` and `5.904937990550645e-18`, strictly worsening; the
  100-digit estimate is approximately `5.904866222924252e-18`. Both revised
  forward/reverse transaction cases pass instead of accepting this candidate.
- The 117-case comparison operation group and all twelve dedicated transaction
  consumer/ownership cases pass. Across 147 revised trajectory artifacts, all
  2,548 recorded comparisons pass the independent stored-input audit: 1,381
  improvements, 946 worsenings and 221 unchanged states. No audit errors occur.
- [Actual PTX](revised/diagnostics/lowering.json) contains all six required
  directed-double add/multiply/divide instructions. There is no host comparison
  fallback in the exercised path. The unsupported-range boundary remains explicit.
- All 383 bound original requirements pass in the revised cohort, including
  installed D2, trajectories and saved inference. Nineteen models in each cohort
  replay in the separate CPU environment without CuPy, Numba or the D2 extension.
- Fourteen of fifteen dedicated recipe comparison cases pass. The remaining
  failure is classified below; later assertions in that failing case were not run.

## Remaining failure and independent interpretation

`test_recipe_current_best_and_patience_have_distinct_validation_anchors[forward]`
expects ten best-model terms and observes nine. Its stopping observations already
match the expected stale-round sequence `[1, 2, 3, 4, 0]` and budget termination.

The fixture fixes target zero and log-scale zero. Each sweep changes the mean,
then commits a zero-valued log-scale term in forward order. The mean sequence is
`2 → 3 → 1.95 → 1.97 → 1.9 → 1.89`, using its declared float32 leaves. Term nine
establishes the final strictly better mean; term ten leaves predictions unchanged.
Strict best selection therefore retains nine. Joint updates commit both terms
atomically and reverse updates place the last mean change at term ten.

The independent exact-rational fixture derivation in [analysis.json](analysis.json)
gives final best prefixes **joint 10, forward 9, reverse 10**. This identifies an
incorrect fixture expectation, consistent with the existing strict-improvement
contract. It is not a new GPU execution and does not turn the original failure
into a pass. Correct the expectation in a separate source change, verify the
no-op prefix rule on CPU, then validate the changed CUDA test with a newly frozen
allowance. Preserve all old sources and raw verdicts.

## Environment and bounded cost

The worker reports Tesla T4, 15360 MiB GPU memory, driver 580.95.05, Python 3.12.1,
CUDA runtime API 12090 and driver API 13000. Requested capacity was two CPUs and
8192 MiB; eighteen visible CPUs are not an exclusive allocation. The full
dispatch took **633.550 seconds**, including image construction, and the worker
took **205.335 seconds**. These are distinct from training cost.

The four [uninstrumented fixture fits](revised/diagnostics/cost.json) include
fixture/context creation, preparation, training and export. Compilation may be
warm from earlier tests. Weighted fits take 0.226–0.233 seconds; D2 fits take
0.348–0.350 seconds. CPU prediction takes 0.396–0.453 milliseconds. Each fit makes
15 comparisons and exports 480 comparison bytes. Their final owned live bytes
are zero after closing. Reference NLL/CRPS and repeated model identities match.
These tiny fixtures establish no real-data quality or matched-quality speed claim.

## Offline verification and retrospective

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python benchmarks/v1/evidence/cuda-comparison-092/analyze.py --check
```

The analyzer verifies raw hashes and sources at the dispatch revision, checks
recorded high-precision estimates against every available enclosure, verifies
PTX hashes and the 38 CPU model replays, and derives the fixture prefix separately.
It preserves the failed verdict. The [sprint retrospective](../../../../v1-sprints/102-normal-cuda-validation.md)
records the next correction. Full revised acceptance, the known split near-tie,
other required CUDA recipes, train-many and formal P7/E4 remain open. Agent
evaluation remains deferred under Sprint 101.
