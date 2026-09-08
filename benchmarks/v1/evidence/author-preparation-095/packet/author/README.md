# OpenBoost D1/D2 CPU author view

Start with [D1](v1-sprints/069-author-packet/D1.md) and [D2](v1-sprints/069-author-packet/D2.md), then [state](docs/v1/cpu-state.md), [operations](docs/v1/numeric-ops.md), [trees](docs/v1/trees.md), [squared training](docs/v1/squared.md), [preparation](docs/v1/preparation.md) and [stopping](docs/v1/stopping.md).

Only selected public CPU documentation is bundled. Links to other pages are explicitly rendered as plain labels; they do not expose evaluator or example-solution files. The core wheel contains its public Python implementation, including experimental CUDA modules. This CPU packet does not establish their current hardware validation or supply CUDA dependencies.

Install with uv in a fresh environment using NumPy 2.3.5 and the wheel in `wheels/`. Existing extension solutions and mathematical judges are not part of this author view. Model, tools, incumbent arms and budget enforcement are not frozen; no independent attempt is dispatched by this export.
