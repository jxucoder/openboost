# Sprint 015: English repository prose

Starting revision: `594519f`. Status: complete.
Mapping: cross-cutting repository documentation; no F0–F5 phase advancement.

## Plan and acceptance

1. Inventory repository text and translate existing non-English prose, including
   historical research, plans, application contracts, evaluation, and sprint records.
2. Preserve requirements, formulas, thresholds, raw artifacts, historical outcomes,
   source identities, and links. Add the English requirement to canonical guidance.
3. Check remaining text, Markdown links/fences, tests, lint, and documentation;
   record the result and commit independently from evaluation data preparation.

Smallest counterexample: an existing Chinese heading in a historical plan violates
this requirement even if every new file is English. Mathematical symbols, source
identifiers, and literal data values are not prose to translate.

## Results and verification

All existing Chinese prose in repository files has been translated. AGENTS.md and
the sprint execution rules now require English for future prose, comments, and
instructions. Historical recommendations remain historical; the current foundation
mission, full R1–R9/C1–C7/A1–A13 scope, and evaluation-first order are unchanged.

Validation:

- Scanned every Git-tracked and nonignored untracked UTF-8 text file for CJK
  ideographs; none remain. Reviewed other non-ASCII letters: mathematical Greek,
  transpose/subscript symbols, and real-number notation remain intentionally.
- Compared source URLs and long hexadecimal identities against the previous
  revision. Three retired implementation link targets now use their verified
  historical Git revision; no evidence source was removed.
- Rendered Markdown links with fenced-code/table handling and checked local file
  targets in changed documents and incoming anchors to changed documents: pass.
- Code fences are balanced; `git diff --check` passes.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q`:
  396 passed, no skips.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost benchmarks/v1 tests/v1 tests/conftest.py`:
  pass.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict`:
  pass. The separate link audit covers planning/sprint files outside MkDocs navigation.

No raw benchmark artifact or production implementation changed in this slice.
The Adult data preparation is the separate preceding commit `594519f`.

## Reflection

Observation: requirements and historical evidence were split across languages.
Evidence: the inventory found non-English prose in active specifications and old
strategy/execution records, not in runtime code or frozen data.
Decision: translate the entire prose surface while preserving the distinction
between superseded recommendations and active v1 requirements.
Next: continue B02/F0.3 data, baseline capability, budget, runner, and quality-judge
preparation. English documentation does not establish implementation, GPU performance,
model quality, or external adoption. F0.3 and F1–F5 remain incomplete.

## Commits

- This slice: `docs: use English throughout repository prose`.
