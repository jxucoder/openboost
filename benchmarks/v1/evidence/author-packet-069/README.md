# Sprint 069: Prepared OpenBoost author view

Source revision `cfca092`. This is prepared material, not an independent attempt
or authoring-cost result. dispatch_ready remains false and attempts is empty.

The author directory contains three task/instruction documents, four current v1
API documents, the built OpenBoost wheel and uv's generated wheel .gitignore.
All nine file hashes verify. All twenty production Python modules in the wheel
match the committed source exactly. Wheel members contain only the public package
and distribution metadata; no existing example solutions, tests or evaluator
oracles are included. Five evaluator input hashes are recorded separately, not
copied into the author directory. They still need a standalone invocation and
complete dependency closure before an attempt can be dispatched.

The first build failed because sandbox DNS could not fetch hatchling from PyPI.
The failure is retained in failed-build.log. A network-enabled build succeeded,
recorded in build.log. Neither operation is an author attempt. The initial file
inventory check expected eight files; inspection identified uv's .gitignore as
the ninth and verified it explicitly. No input was removed to satisfy the check.

```bash
uv run --no-sync python -m benchmarks.v1.prepare_author_packet /tmp/author-packet
```

The preparer rejects a dirty repository before output creation. The exported
materials are a narrow view, not an OS sandbox: actual evaluator access denial,
token/time enforcement, model/runner settings and incumbent comparison arms are
still missing. Some ordinary documentation links point beyond this subset; final
attempt navigation must be reviewed when the execution environment is frozen.
No held-out task was inspected and no independent agent was launched.
