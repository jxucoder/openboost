# CLAUDE.md

The canonical repository guidance is [`AGENTS.md`](./AGENTS.md). Read it in full
before changing code, benchmarks, documentation, CI, packaging, or release
state.

Claude-specific compatibility notes:

- Use `uv` for environments and commands.
- Store durable decisions, failed experiments, and user corrections in
  `learnings/`, not the ignored `tasks/` directory.
- Keep commits small and verified. Do not push or publish unless requested.
- Treat GPU, distributed, out-of-core, and performance claims according to the
  evidence gates in `AGENTS.md`; comments and green-but-skipped CI are not proof.
