# Sprint 070: First real A6 resource probes

Clean original revision: `9387d9f` (full revision in manifest). Harness: `4a2960a`.
The user explicitly approved the 1494662-byte Parkinsons train/validation packet
and allowlisted source upload after the initial automatic approval rejection.
This run uploaded 31 public source/packaging files and the verified fold-zero
worker packet. No test features or test labels were uploaded or scored.

## Results

| Topology | Fit worker wall seconds | Completed rounds | Stop reason | Peak guest RSS bytes | Fresh validation replay |
| --- | ---: | ---: | --- | ---: | --- |
| Shared | 1077.1430751449998 | 59 | patience | 108097536 | Exact |
| Independent | 767.1319667130001 | 65 | patience | 136032256 | Exact |

Both use the unchanged configuration-00 300-round budget, patience 50, 255 bins,
and summary retention. Early stopping is part of the frozen protocol. Selected
models represent their best validation state, not necessarily the final round.

Each fit runs as UID/GID 65534 with a hard 8-GiB address ceiling, one worker thread
and 1800-second timeout. Modal requests two CPUs, 8192 MiB, a 1900-second function
limit and zero application retries. Jobs execute sequentially and stop on failure.
Linux ru_maxrss is recorded in bytes; it is distinct from address space and from
requested container capacity. Host cgroup enforcement is not independently observed.
Fresh inference runs in a separate evaluator process. No retry or preemption was
reported. Source/environment/image identities and exact child commands are retained.

## Verification

Local inspection verifies all 31 source hashes against the recorded clean revision,
all 20 returned artifact hashes, the frozen plan hash and original worker packet
hash. Both execution records match their manifest copies. Returned replay arrays
match predictions exactly, including row IDs. Persisted target scales match the
training-only fold freeze exactly. The data has 3487 training rows, 1151 validation
rows, 38 encoded features and two original-unit UPDRS targets.

The original input packet is not duplicated here. Rebuild it with worker_data's
verified A6 exporter and check the recorded hash. The returned validation feature
packets, predictions, model bytes, training records and resource records are retained.

```bash
uv run --no-sync python -m benchmarks.v1.worker_data A6 /tmp/a6-packets
uv run --no-sync python -m benchmarks.v1.a6_resource_preflight /tmp/a6-probes /tmp/a6-packets
```

## Reflection and next step

These two real configurations meet their resource and replay checks. They do not
qualify the other 158 OpenBoost configurations, comparators, 1000-round fits or a
full selection/test-release workflow. No real quality or comparative speed gate
is passed. Guest RSS is modest in these observations, but fit time is substantial
even with early stopping. Review a bounded profile on this exact practical input
before expanding expensive searches; do not optimize based on an assumed bottleneck.
The full coverage ledger and author-accounting packet remain open.
