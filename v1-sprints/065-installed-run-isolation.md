# Sprint 065: Installed run isolation

Planning baseline: `df23796`; implementation parent `2871f0f`.
Status: complete. Mapping: N1 / B11 / R9 / C4–C7 / D5 / E1–E2 and E6 development checks.
Depends on: [Sprint 064](064-programmable-stopping-and-isolation.md).
Shared evidence and closure rules: [sprint roadmap](roadmap-after-063.md).

## Outcome and first failing check

An installed mixed schedule preserves custom stopping, independent RNG streams,
preparation identity and failures. This is the second slice of the earlier Sprint
064 plan, now an independently reviewable sprint. Do not count internal packages
or verifiers as E5 authors or E7 users.

First add installed checks for distinct stable run IDs, changed feature/row identity
and the external stopping loop from Sprint 064. Establish which checks are absent
and observe any semantic failure before editing the core. Existing preparation
behavior may already pass; do not invent a production defect to justify a test.

## Implementation: Installed stopping, RNG and preparation conformance

Extend examples/v1_extensions/scheduler_checks.py and its existing verify.py
workflow. Reuse current public wheels, mixed K=1/2 fixtures and M=1/8/32 schedules.
This sprint adds checks; edit the core only if a new failing case demonstrates a
separate defect, with its own narrow verification and commit.

### Required cases

| Case | Expected result |
|---|---|
| Same seed, distinct stable run IDs | Sampled keyed streams differ; do not infer different predictions from a deterministic recipe |
| Same run ID under retry/reorder/regroup | RNG draws, accepted state, stop state and predictions match independent execution exactly |
| Changed feature values, unchanged source row IDs | Old PreparedData is rejected even when shapes/names match |
| Changed source row IDs, unchanged feature values | Old PreparedData is rejected |
| Fresh preparation for either changed dataset | Shared-prepared execution matches fresh independent direct execution |
| Changed target/weights with identical features and row identity | Valid preparation reuse continues to work; do not reject safe sharing |
| External stopping policy mixed with built-in and ordered recipes | True reason/count/payload survives; other runs retain their independent outcomes |
| Failed preparation or malformed stopping status | Explicit retained failure; neighboring valid runs and same-ID retry are unaffected |

Compute independent direct results before forbidding Binning.fit during shared
execution. Restore any instrumentation after the check. Use newly constructed
owned data for mutation cases, not illegal in-place mutation of read-only input.
Retain existing different-patience, invalid-result and plugin-free inference checks.

### Installed acceptance and evidence

Run copied checks from outside the repository with Python -I and imports from
site-packages. Build/install the wheel and extension packages in the existing
isolated uv workflow, then remove training plugins and verify saved inference.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python examples/v1_extensions/verify.py /tmp/openboost-v1-sprint065
```

Use a fresh output directory. If cached dependencies are unavailable, report the
installation failure separately from semantic failures. Evidence belongs under
benchmarks/v1/evidence/scheduling-065/ after verification; record source/wheel hashes,
revision/dirty state, environment, exact commands and all expected error outcomes.
Expected fault-injection failures must be labeled as such, never hidden or treated
as successful training. Do not count this internal verifier as an independent author.

## Exit and reflection

- All declared installed M=1/8/32 cases pass with K=1/2, including retained expected
  faults and successful same-ID retry. Fresh inference passes after plugin removal.
- Record source/wheel hashes, import locations, exact commands and environment;
  validate committed artifacts independently and run lint/docs for changed files.
- Run relevant regression if a defect required a core fix; isolate that fix in its
  own verified commit. Inspect staged changes and commit the verifier/evidence locally.
- Reflect on real public-boundary failures versus missing installed coverage.
  N1 completion alone is not full E2/E5/E6 acceptance. Next: Sprint 066.

## Results

Installed verification passes with no core edits. Both the original patience suite
and new mixed custom-policy suite pass at M=1/8/32, including K=1/2, distinct-ID
streams, same-ID retry, reorder/regroup and no preparation refits. Changed features
or row IDs fail explicitly; changed targets/weights reuse safely; fresh preparation
matches direct execution and neighboring runs are unchanged.

The genuine public loop retains its threshold/loss tuple and terminates at two of
five rounds with independently checked raw predictions. Ten models replay exactly
after all training plugins and copied custom-policy source are removed. The
[raw evidence](../benchmarks/v1/evidence/scheduling-065/README.md) contains the full
17-artifact matrix and manifest with source/wheel/environment/command hashes.
See the [learning](../learnings/2026-09-06-v1-installed-isolation.md) for verification.

Reflection: the structural contract change survived the installed boundary;
remaining RNG/preparation cases needed evidence rather than another core abstraction.
N1 is complete as development conformance. It does not pass formal E5/E6 or establish
cost, CUDA or adoption. Next measure practical CPU execution in Sprint 066 before
changing runtime state/trace ownership.
