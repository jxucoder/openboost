# Sprint 042: Shared recipe result contract

Parent: b99376d. Status: complete for result interoperability. Mapping: Sprint 038 M2, D5/C4/C6.

## Plan and acceptance

1. Reproduce OrderedResult rejection with an external recipe, then replace the
   scheduler's concrete built-in result dependency with a public structural contract.
2. Validate accepted-state identity, completed stop metadata and outer-round trace
   count while leaving per-round diagnostic content owned by the recipe.
3. Verify heterogeneous built-in/ordered M=1/8/32 shared-preparation runs under
   permutation, regrouping, retry, differing stop rounds and isolated malformed results.
4. Rebuild installed extension evidence, run regression/lint/docs and commit locally.

No objective/type-name exceptions or conversion to a built-in result class.
Accepted parameter commits need not equal outer rounds. A successful result must
be complete; interrupted work must not masquerade as a successful outcome.
This closes the Sprint 041 interoperability counterexample, not complete E2/E5.

## Results and reflection

RecipeResult is a public structural protocol with accepted state, per-outer-round
steps and completed StopState. Runtime validation checks actual field types,
requested context/problem identities and trace length. The scheduler no longer
imports concrete recipe result classes or interprets their diagnostic payloads.
The existing external OrderedResult is returned unchanged, without an adapter,
subclass requirement or extension source modification.

Eleven focused tests pass, including the formerly rejected external result,
seven malformed/unfinished/foreign cases and M=1/8/32 mixed independent executions.
Ordered recipes legitimately have two model commits per outer round. Shared input
does not refit; reversed/regrouped/retried runs preserve state, predictions, stopping
and scoped RNG. Failures remain visible without contaminating later runs.

Installed four-wheel verification reruns D2/D3/D4 and adds mixed scheduling checks
at M=1/8/32. All six ordered cases match direct execution, and eight persisted
models retain exact predictions after all plugins are removed. Raw evidence is
in [recipe-results-042](../benchmarks/v1/evidence/recipe-results-042/README.md).
CPU regression: 794 passed. Ruff, strict MkDocs and offline sdist/wheel pass on
macOS/Python 3.12.12/NumPy 2.3.5.

Observation: algorithm-owned trace records exposed unnecessary dependence on a
built-in result class. Decision: share only the state/completion/outer-round
contract, not a catalog of permissible diagnostics. This is a foundation revision
prompted by an exploratory failure; it is not a frozen zero-core-edit author win.
Earlier failed artifacts remain unchanged. Next D1 and remaining D5 author probes,
alongside current real-data integration. Full E2/E5/E6 and real task gates remain open.

See [learning record](../learnings/2026-09-06-v1-result-contract.md).
