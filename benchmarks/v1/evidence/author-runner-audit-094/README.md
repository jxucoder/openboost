# Installed author-runner accounting audit

Read-only inspection on macOS with `codex-cli 0.153.2`. No model generation,
app-server startup, author attempt or resource allowance was used. The
[manifest](manifest.json) retains exact commands, availability checks and hashes
of the CLI help/version and three locally generated public JSON schemas.

The installed CLI advertises JSON events. The app-server usage schema contains
thread/turn identifiers and output/reasoning-output token counters; its goal API
also accepts a token budget. These are observable interfaces, not evidence that
our fixed **20,000 generated tokens / 1,800 seconds** rule is enforced. The schema
does not establish whether reasoning tokens are included in the output count,
when updates arrive, how interrupted usage is finalized, or how an in-flight
request stops at the exact limit. No synthetic zero or invented usage was recorded.

Official documentation corroborates the event interfaces: noninteractive runs
emit JSONL including turn completion usage, while the app server exposes token
usage notifications and goal accounting. Neither inspected interface, without a
real enforcement smoke, closes this project's frozen accounting requirement.
See [noninteractive execution](https://learn.chatgpt.com/docs/non-interactive-mode)
and [app-server protocol](https://learn.chatgpt.com/docs/app-server).

`docker` and `podman` were absent from the inspected PATH; that is not a claim
about every possible host facility. The existing Linux process/access preflight
requires a different environment and has not tested the actual author bundle.
macOS process separation and read-only workspace settings alone do not establish
denial of evaluator reads. No local authentication/configuration/history files
were read or copied into this evidence.

**Dispatch remains blocked.** Next prerequisite: a concrete runner/environment
with defined token accounting, forced token/time exhaustion and actual evaluator
read/write denial. Model/settings/tools and appropriate incumbent paths must also
be frozen before an authorized independent attempt. Until then, useful local work
is standalone verifier and packet preparation, not another accounting simulation.
