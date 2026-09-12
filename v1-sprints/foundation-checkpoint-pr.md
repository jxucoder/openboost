# Foundation checkpoint PR

The user requested a PR on 2026-09-11. This branch curates the foundation source
at development checkpoint `91a519d` onto public `main` at `31303e32`. The original
development checkout, its complete raw evidence and later unfinished changes
remain intact. This is a draft review checkpoint; OpenBoost v1 remains incomplete.

## Plan and scope

1. Preserve the existing checkout and construct a separate branch from `main`.
2. Copy the checkpoint production source, self-contained tests and component
   documentation; record exact source hashes and omitted test obligations.
3. Verify the file/import/documentation closure and publish a draft PR. Resolve
   candidate runtime and evidence gaps before marking it ready for merge.

The [manifest](../planning/foundation-checkpoint-pr-manifest.json) identifies
every production file and all 104 changed test/helper files selected from the
checkpoint. Existing main tests and public history remain present. Forty-seven
changed benchmark/archive-linked test files are omitted from this PR, including
mixed files containing useful behavioral controls. Two derived files retain
303 self-contained Normal/multiclass comparison cases from those mixed files;
their original function bodies are unchanged. Archive replay obligations are not
waived. One existing GLM audit wrapper and its snapshot helper reconstruct
original sources from history already public on main, preserving all eleven
original controls without requiring a new archive. Later full-budget Housing and classification construction is excluded.

The [public checkpoint guide](../docs/v1/checkpoint.md) states the implemented
boundary. A compact, unchanged historical audit report is included, while the
large underlying archives remain in the original development history. That report
does not make the original experiments independently replayable from this branch.

## Verification boundary

The smallest check is exact identity of every copied source with the checkpoint.
Source/import/JSON/link checks and lint are metadata checks. No installed CPU
regression, CUDA test, training, wheel build or documentation build is run locally.
The PR's draft guard prevents automatic heavy CPU/docs jobs from starting;
skipped jobs provide no passing evidence. The guard does not change their test
commands or waive the need for validation.

Before readiness, validate the actual curated installed package and its CPU,
CUDA, serialization and documentation consumers in bounded Modal execution.
Reconcile omitted-test coverage and the retained tests' historical source/artifact
dependencies. Reconcile source-specific historical observations with the final
candidate. Full-budget train-many, formal E4, remaining applications and deferred
E5 remain open. No merge, release or artifact archive publication is included.
