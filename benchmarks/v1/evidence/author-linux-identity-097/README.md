# Corrected Linux worker identity smoke

At clean `518eccf07eba9398b66e48910bec185ba4e76151`, the one approved CPU smoke
passes all nineteen original checks, both identity guards and actual provider
timeout. The [original 096 failure](../author-linux-isolation-096/README.md) remains
unchanged. This is known-code worker validation, not independent author evidence.

## Observed result

| Check | Actual observation |
| --- | --- |
| Identity before/after fresh exec | UID/GID triples `[1000, 1000, 1000]`, no supplementary groups, `no_new_privs=1` |
| Six installed public examples | All exit 0; writable author workspace also works |
| Core/material writes | Both denied with `EACCES`; installed core hashes unchanged |
| Root restoration | `setresuid(0, 0, 0)` denied with `EPERM` |
| Five private paths and child/symlink reads | Unavailable; all four controller evaluator hashes unchanged |
| Network observation | Loopback works; external TCP attempt returns `ENETUNREACH`, with provider `block_network=True` |
| Lifetime | Separate-session child PID/session 19 is recorded before actual Sandbox timeout; provider return code 124 |

Python 3.12.10 runs on `Linux-4.19.0-gvisor-x86_64-with-glibc2.36`; the worker reports
two CPUs. Packages: OpenBoost 1.0.0.dev0, NumPy 2.3.5 and uv 0.12.1. Controller SDK:
Modal 1.3.0.post1. Requested resources are two CPUs and 2048 MiB, with no GPU,
secrets or volumes. Requested capacity is not measured RSS or a memory-exhaustion
test. Creation, including image work, takes 22.5403 seconds; the wait phase takes
90.9490 seconds. Neither is a boosting performance measurement.

The provider reports actual expiry after all checks and the child marker. The
captured `KeyboardInterrupt` occurs in the parent's deliberate final sleep;
it is retained, not removed as noise. `terminate()` completes without a cleanup
error. The harness does not independently query the child's PID after expiry;
the observation is Sandbox timeout and completed output streams, not a general
descendant-liveness audit or enforcement of the full 1800-second author budget.

## Reproduction and provenance

The original command, with exit 0 and no controller output, is retained in
`controller-command.json`:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.authoring.modal_worker --execute /tmp/openboost-author-linux-097
```

The allowance is consumed; this command is historical and normal reuse is blocked.
There was one Sandbox and no application retry. `freeze.json` retains the exact
approved payload; the active freeze changes only authorization to `consumed`.
Its 24 input hashes match the execution revision. The original thirteen uploaded
files are byte-identical to 096; only `/opt/launcher.py` is added. No evaluator,
solution, repository directory or model runner is delivered.

`manifest.json` is the untouched raw controller report. Its eighteen artifact
hashes cover the exact freeze, build log, stdout, stderr and fourteen delivered
files. `analysis.json` derives the case comparison and identity/integrity summary.
`archive.json` indexes 21 raw/derived files, including the original manifest,
command record and analysis. The README and index itself are not indexed.

All five originally failing checks now pass with the original probe and cases.
Local replay requires the real timeout; substituting a normal exit fails. The
old 096 result still fails, even if falsely supplied a timeout status.
See the [097 retrospective](../../../../v1-sprints/097-worker-identity-result.md).
Generated-token enforcement, model/settings, fair arms and author attempts remain
open; no E5, adoption or GPU claim follows from this result.
