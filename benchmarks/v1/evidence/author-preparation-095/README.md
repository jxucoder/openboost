# Author preparation and native-isolation evidence

Clean source revision **`862c4046375ad498f0bb15a4407db8609f4fe18c`**. The
[top-level manifest](manifest.json) hashes eighty retained files and separates
three outcomes; there is no combined preparation pass or independent author attempt.

| Work | Result | Boundary |
| --- | --- | --- |
| Current author view | Prepared; twelve files, thirty-one core Python modules, source-verified typing marker, eleven explicitly omitted links | Model/tools/incumbent arms and budget enforcement remain unfrozen |
| Installed verifier v2 | Two D1 and nine D2 cases pass, including the added invalid-input/identity observations; deliberate changed model fails | Known designer implementations; no author-cost claim |
| Strong native isolation | Fails before the first positive example with interpreter exit -6 | No usable isolated worker or successful access-denial gate |

The [packet manifest](packet/manifest.json) records original and delivered file
hashes, omitted links, evaluator-tool hashes and the offline build command.
The [verifier manifest](verifiers/manifest.json) retains its complete runtime/input
closure, installed versions and file hashes, built wheels, observations and models.
These clean runs share the same revision; the verifier records a clean checkout.
The packet builder rejects dirty checkouts before export.

The [clean native failure](isolation/clean/manifest.json) and its
[exact policy](isolation/clean/worker.sb) are retained. Setup installs independent
copies of NumPy 2.3.5 and OpenBoost 1.0.0.dev0. The sandboxed x86_64 interpreter
aborts on macOS 26.3 before the first public CPU example. That startup failure
cannot count as a successful denial. No root cause, including Rosetta, is established.

Earlier dirty development results are separate. The
[narrow profile](isolation/development-narrow/manifest.json) passes six installed
public examples and its thirteen declared checks. The subsequent
[unlisted-answer counterexample](isolation/development-narrow/unlisted-counterexample.json)
exits zero and confirms that another copy of the actual expected answers is readable.
Its bytes are hashed, never printed. This invalidates broader isolation even though
the original checks passed. The initial stricter startup failure, selected system
runtime-path diagnostics and empty PID-specific log result are also preserved under
`isolation/development-allowlist/`. Their dirty input snapshots are separate from
the clean packet and must not be interpreted as independent attempts.

Reproduce the passing preparation from the recorded revision with new destinations:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.prepare_author_packet /tmp/openboost-author-packet-095-clean
UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync python -m benchmarks.v1.authoring.verify /tmp/openboost-author-verifiers-095-clean
```

The native failure was reproduced with the following local command outside the
existing nested tool sandbox. It changes no host policy; only its child processes
receive the declared policy. Expect failure on the recorded environment:

```bash
.venv/bin/python -m benchmarks.v1.authoring.isolation /tmp/openboost-author-packet-095-clean /tmp/openboost-author-verifiers-095-clean/evaluator /tmp/openboost-author-isolation-095-clean
```

Interpreter environments are not vendored. Their setup commands, retained core/
extension wheels, evaluator inputs and profile bytes make the declared probes
reviewable; exact NumPy binaries remain platform-specific dependencies. No model
tokens, author time, GPU work or competitive cost were measured. `dispatch_ready`
remains false. Stop native-policy expansion and decide a supported worker
environment plus real generated-token accounting before further author execution.
