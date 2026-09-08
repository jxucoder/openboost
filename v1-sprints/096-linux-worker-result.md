# Sprint 096 result and retrospective

Status: the single authorized CPU smoke is consumed and the isolation gate failed.
No retry. Source: clean `7985645`; [raw evidence](../benchmarks/v1/evidence/author-linux-isolation-096/README.md).
The construction plan, original approved payload and old failed probes remain
historical evidence. This record supersedes their pending execution status.

## What happened

All six public examples execute in the installed Linux environment. Eight further
checks pass, including unavailable evaluator paths, child/symlink read denial and
an unavailable outbound connection. Five checks fail: worker identity, core writes,
material writes, root restoration and core integrity. The process runs as root.
The deliberate write replaces `openboost/__init__.py` with `b"tamper"` inside the
worker; its recorded digest confirms those exact bytes. The four controller-side
evaluator files remain unchanged. The process exits 1 before the timeout marker,
so neither container timeout nor separate-session child cleanup is verified.

The builder explicitly skips `USER 1000:1000`. Modal's
[Dockerfile compatibility documentation](https://modal.com/docs/guide/existing-images#user)
also explains that `USER` is ignored and recommends reducing program privileges
through OS facilities such as `setuid`. This was a mistake in our harness and
an incomplete provider-contract review, not an unexplained platform failure.
Local SDK construction and 25 green tests did not establish runtime identity.

## What the evidence changes

Keep the explicit image-file boundary: the worker is usable, receives the intended
core, and does not receive the evaluator. Fix how worker commands acquire their
identity. Do not discard the write-denial requirements because their removal would
turn the current partial result green. Do not modify expected numerical results,
the nineteen original case definitions, or the old failed artifacts.

The current freeze is marked `consumed`, preventing normal `--execute` reuse.
Its original approved version is archived with the result. No file/case/resource
payload changes accompany that status update. All 22 original source hashes and
all 85 GPU run-8 source hashes still match. No production source was edited.

## Next local correction and acceptance

1. Replace reliance on image `USER` with a trusted launcher that drops supplementary
   groups, real/effective/saved GID and UID, and sets `no_new_privs` before `exec`.
   The existing Linux drop sequence in `benchmarks/v1/process_runner.py` is a
   starting point; its older evidence is not proof for this actual Sandbox path.
2. Verify the post-drop identity before executing any author code. Every future
   remote command must pass through this launcher; a direct root `Sandbox.exec`
   tool must not be exposed to the author. Keep evaluator/controller credentials
   and usage accounting outside the worker. Fail closed if privilege reduction
   is unavailable. Retain the original positive examples and nineteen checks.
3. Freeze the corrected command, added identity observations and exact upload/source
   changes as a separate proposed run. Only after a concrete allowance, execute
   one bounded smoke and require all original checks plus actual provider timeout
   after the separate-session child starts. Until then, isolation stays failed.

This is the planned retrospective boundary, not an independent author result.
Avoid expanding the sandbox framework or starting another native-policy search.
After isolation passes, return to real generated-token exhaustion and the frozen
fair-arm/model/settings packet. No synthetic token counter substitutes for that
gate. The pending GPU run-8 request and all R/C/A/E obligations remain unchanged.
