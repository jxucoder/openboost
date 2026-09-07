# Sprint 070 worker permission and resource probe

Clean source revision: `26a6797db75195cb406ede5f6f6cca14a13b1603`.
The unmodified manifest records all three uploaded source hashes, exact commands,
Modal image/version, Linux/Python/NumPy details and 17 raw artifact hashes.

Reproduce on an authorized Modal account:

```bash
uv run --no-sync python -m benchmarks.v1.access_preflight /tmp/openboost-access-070
```

## Results

All **13 checks pass** across seven real subprocess cases: one numerical success
and six expected failures. This is a mechanism probe, not a trained-model benchmark.

| Case | Observed result |
| --- | --- |
| Numerical | Success in 0.472 seconds; UID/GID 65534, empty groups, no_new_privs; root restoration and hard-limit raising rejected; evaluator canary absent |
| Synthetic test-label read | PermissionError on the known protected file |
| Synthetic verifier write | PermissionError on the known protected file; evaluator file hashes unchanged |
| Parent environment read | PermissionError on `/proc/2/environ` |
| 9-GiB mapping | ENOMEM under the 8-GiB hard address limit |
| Intentional worker error | Nonzero exit retained |
| 0.25-second deadline | Exit -9 in 0.252 seconds, partial log retained |

Loaded OpenBLAS reports two threads. Touching 64 MiB increases ru_maxrss from
84,393,984 to 150,982,656 bytes; the high-water value remains after freeing it.
The proc high-water counter is unavailable and is not counted as verified.

Local artifact inspection verifies all 17 file hashes, all three source hashes
against the recorded Git revision, seven execution-record copies, and the actual
permission/memory/deadline logs. Protected fixtures contain synthetic values only;
no real labels or sealed verifier content was uploaded or returned.

## Limits and next work

The container requests two CPUs, 8192 MiB and a 120-second function timeout with no
application retries. RLIMIT_AS bounds process address space, not aggregate RSS or
container RAM. The successful worker configures the 1800-second search deadline but
finishes immediately; only the short forced deadline is actually observed expiring.

This establishes a UID/file permission mechanism. Network, new sessions and separate
same-UID attempts require additional isolation; it is not a complete hostile-code
sandbox. Full search-worker/container integration, 300/1000-round qualification,
protocol-derived coverage and independent author accounting remain open. The stub
model and tiny prediction array establish no application quality or cost gate.
