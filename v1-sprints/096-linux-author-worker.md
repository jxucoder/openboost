# Sprint 096: Linux author-worker preparation

Status: local construction verified; remote smoke pending authorization.
Continues [095's failed native-isolation gate](095-author-packet-and-local-isolation.md)
within [069](069-authoring-pilot.md). The run-8 device freeze remains unchanged.

## Decision and plan

Use one Modal CPU Sandbox for the next file-isolation smoke. The installed client
is `modal==1.3.0.post1`; its public interface supports `block_network`, finite
Sandbox lifetime, explicit image files and process completion status. Docker,
Podman, Colima, Lima and Orb were not found in the inspected PATH. Do not keep
widening the unsuccessful macOS profile.

1. Construct a worker image from the twelve archived 095 author files and one
   generic probe. Never add the repository directory, evaluator, known extensions,
   host credentials, conversation history or shared volumes to that image.
2. Run as a non-root user with a writable work directory and root-owned installed
   core/materials. Require all six public examples to execute before interpreting
   read/write denials. Check child and symlink access, core integrity, unavailable
   evaluator paths, no injected model/service credentials and outbound denial.
3. Finish the probe by starting a child in a new process session and deliberately
   exhausting the Sandbox lifetime. Retain the provider's timeout status, raw
   output, loaded file hashes and every earlier outcome. This tests container
   termination, separately from a process-group timeout or author wall accounting.
4. Verify delivery closure and failure handling locally, freeze the exact CPU
   smoke, then obtain any missing upload/run authorization. One 90-second CPU
   Sandbox, 2 CPUs, 2048 MiB, no GPU, no retries; image build is separate and must
   be reported. No author/model generation is part of this request.

First failing check: a directory upload can copy an unlisted expected-answer file
even if the declared packet hashes match. The new image builder must accept only
the explicit file closure and reject symlink/traversal/changed/missing inputs.
Provider APIs must be constructed against the installed SDK without starting a job.

## Acceptance and limits

Local tests establish file selection and result classification only. The remote
gate requires usable positive execution, every declared denial/integrity check,
the exact case set, and an actual provider timeout after the final marker. A
startup crash, missing result, ordinary nonzero exit or locally fabricated timeout
cannot pass. The evaluator remains on the controller host and its hashes must be
unchanged. There is no candidate execution in the trusted judge/controller.

The generic probe is designer-authored and contains no task solution. The image
is a prospective worker environment, not a complete author runner. No interface
allows an author to request local controller commands. A future model controller
must keep credentials and token records outside this worker and expose only
explicit remote tools. Independent task adapters, model/settings, fair incumbent
paths, edit accounting and actual token-budget exhaustion remain open.

Modal documents isolated Sandboxes and outbound blocking in its
[networking/security guide](https://modal.com/docs/guide/sandbox-networking).
Use only the parameters present in the pinned installed client; newer documented
domain-allowlist or VM features are unnecessary. These are provider guarantees to
test within the declared scope, not measured results from this sprint.

## Actual token interface decision

The [094 installed Codex audit](../benchmarks/v1/evidence/author-runner-audit-094/README.md)
remains blocked; no new simulated usage interface is introduced. The official
[Responses create reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
defines `max_output_tokens` as a bound including reasoning and visible output,
with reasoning usage reported as a breakdown of output tokens. This supplies a
concrete alternative to test: sequential provider requests capped by the remaining
20,000-token allowance, subtracting `usage.output_tokens` once, never adding its
reasoning subset again. This is a proposed controller rule, not a measured pass.

Unknown or interrupted usage must stop dispatch and remain unknown, with no retry
that could conceal consumed tokens. A real bounded generation/timeout smoke,
model/settings/access freeze and independent-run authorization are still needed.
No API credential was read and no model request was sent during this inspection.

## Local result and next falsifiable gate

All 59 focused author-preparation checks pass, including 25 new Linux-delivery/
classification tests. The full CPU regression passes 1853 tests with one Linux-only
skip. Ruff passes; MkDocs builds with its existing external-evidence link warning.
These outcomes verify local preparation only. No Linux probe or provider timeout
has run, and neither real token accounting nor independent author benefit is passed.

The next action is exactly the pending CPU smoke in
[`096-linux-worker-smoke.json`](096-linux-worker-smoke.json). Obtain its explicit
upload/run allowance, commit that authorization without changing the frozen
payload, execute once and retain the result. Do not add further native-policy
experiments or a broader fixture catalog. GPU run 8 remains a separate request.
