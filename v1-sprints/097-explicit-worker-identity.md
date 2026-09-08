# Sprint 097: Explicit worker process identity

Status: construction and bounded corrected validation approved by the user's
"sure" following [096's result](096-linux-worker-result.md). Preserve the original
failed run and consumed allowance. This is one new CPU smoke under the same
90-second / 2-CPU / 2048-MiB bound, image build if needed, zero application retries,
no GPU and no model generation. Freeze exact inputs before service access.

## Plan and first failing check

1. Add a small trusted Linux launcher. Set `no_new_privs`, clear supplementary
   groups, set real/effective/saved GID and UID to 1000, and verify the resulting
   values. Execute a fresh launcher process and verify inheritance before it
   executes the worker. Any failed operation or wrong identity stops execution.
2. Use that launcher on the only current worker command path. Remove the ignored
   Dockerfile `USER` instruction. Keep all nineteen original probe cases and the
   twelve author files byte-for-byte unchanged; add only the launcher to delivery.
   Require both identity records as well as the original cases and timeout marker.
3. Test failure ordering, actual unsupported-host rejection, historical failure
   replay and installed SDK construction. Commit a separate freeze with fourteen
   uploaded files, including the original generic probe and the new launcher.
4. Run the corrected smoke once, archive every raw outcome and stop for reflection.
   CPU program usability, evaluator exclusion, write protection and actual timeout
   must all pass. This is still not a complete author runner or token accounting.

First failing check: the current worker command launches Python directly, without
dropping privileges. Add a failing command-path check before implementing the
launcher. A mocked syscall sequence tests control flow only; the real Linux run
must establish actual privilege reduction and inherited identity.

## Contract and verification

Only a trusted controller may initiate the root launcher. It receives an argv list,
not shell text, and executes no candidate code until identity checks pass. A future
author tool must use this path for every command; direct root `Sandbox.exec` access
is not part of this smoke and must not be exposed. The evaluator, credentials and
future token ledger remain on the trusted side, outside this image.

Two additional records precede the unchanged original probe: `before_exec` and
`after_exec`, each requiring UID/GID triples `[1000, 1000, 1000]`, empty supplementary
groups and `no_new_privs=1`. The normal final probe checks must remain green, and
the provider must actually time out after the separate-session child starts.
Missing, duplicate or invalid records, a normal exit, or any original failed case
fails the gate. Keep the 096 classifier available to replay its immutable result.

The implementation follows [Modal's process-identity guidance](https://modal.com/docs/guide/existing-images#user),
the [Linux setresuid/setresgid contract](https://man7.org/linux/man-pages/man2/setresuid.2.html),
and the [inherited no_new_privs contract](https://man7.org/linux/man-pages/man2/PR_SET_NO_NEW_PRIVS.2const.html).
The explicit drop sequence in `benchmarks/v1/process_runner.py` is a local
precedent, not evidence for this new Sandbox path. This scope does not expand
CPU searches, the numerical fixture catalog, independent author trials or GPU run 8.

## Local construction result

The command-path test failed first with a missing `worker_command`. The corrected
launcher and controller pass all 52 focused checks, including real unsupported-host
rejection and historical failed-run replay. The full CPU suite passes 1880 tests
with one Linux-only skip; Ruff passes. Mocked syscall checks verify ordering and
fail-closed control flow, not Linux privilege enforcement. The separate
`097-worker-identity-smoke.json` freezes the exact corrected source and fourteen
uploads; the original thirteen uploaded files and all nineteen cases are unchanged.
Remote acceptance is still pending until the one approved smoke executes.
