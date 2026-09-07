# Installed custom stopping and run isolation

Implementation parent `2871f0f` plus the dirty verifier sources hashed in manifest.json.
Production sources are unchanged from that parent and individually hashed. This
is internal CPU development conformance, not E5 authoring, E7 adoption, GPU or cost evidence.

Reproduce from the repository root with a fresh output directory:

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python examples/v1_extensions/verify.py /tmp/openboost-v1-sprint065
```

The verifier builds five wheels offline, runs copied checks outside the repository
under Python -I with core/plugin imports from site-packages, uninstalls all four
training plugins, removes the copied custom-policy source, and starts a fresh
interpreter for exact inference on ten models. macOS/Python/package versions,
thread count, commands and source/wheel/artifact hashes are in manifest.json.
Cached dependencies are needed. No environment or performance generalization is implied.

- scheduler-checks.json retains the original M=1/8/32 patience suite and adds
  M=1/8/32 custom-policy/built-in/ordered schedules. Keyed draws distinguish run
  IDs but replay exactly under retry/reorder/regroup. Shared execution forbids
  binning refits after independent direct results have been computed.
- The independent threshold oracle gives losses 0.125 and 0.03125 and predictions
  (-0.75, -0.75, -0.75, 0.75, 0.75, 0.75), stopping at two of five rounds. True
  reason/count and author-owned loss payload survive structural result validation.
- Malformed custom completion is a retained expected ValueError and the same ID
  succeeds on retry. Changed feature values or row IDs reject stale preparation;
  fresh preparation matches direct execution. Changed targets/weights with identical
  features/IDs safely reuse preparation. Neighboring runs remain unchanged.
- threshold-model.json and threshold-inference.json add custom-loop inference to
  the nine earlier mixed/missing, ordered and expectile models. All earlier D1–D4
  mathematical checks and plugin-free inference remain exercised.

All 17 manifest-listed artifact hashes and source hashes were checked independently.
The stored custom predictions were also checked against the exact recurrence;
the required scheduling counts and three preparation outcomes were verified.
Expected injected failures are not failed training hidden from the report.
No new core edit was necessary after Sprint 064's completion contract change.
