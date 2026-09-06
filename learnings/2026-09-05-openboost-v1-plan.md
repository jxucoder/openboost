# 2026-09-05: OpenBoost v1 scope, current releases and evaluation

## Context

The user designates this planning round as the real OpenBoost v1 and asks for
clear execution, acceptance and evaluation criteria. Insurance and survival/AFT
were examples of real applications, explicitly not a closed application list.
The user also requires review of the latest XGBoost, CatBoost and LightGBM
releases and public plans. Foundation-first and no backward compatibility remain
the governing decisions.
The user then rejected describing insurance/AFT as special important cases:
all listed use cases are needed. This entry and the active protocol include that
correction; the initial planning state is preserved in commit `a9fde60`.

## Decision or Result

Define v1 around the cost of a verified algorithm change. Ordinary Python
recipes own algorithm decisions; composable bulk operations and explicit state
support CPU reference execution and a declared CUDA subset. R1–R9 and C1–C7
specify required recipes and shared capabilities; A1–A13 make every application
an individually required implementation and evaluation obligation. No special
priority attaches to insurance/AFT. Datasets can be selected, but use cases
cannot be replaced by a count of representative successes.

The reviewed latest releases are XGBoost 3.4.1, CatBoost 1.2.10 and LightGBM
4.7.0. XGBoost's expanded experimental vector-leaf hist support, CatBoost's
existing GPU custom objectives and LightGBM's new GPU/data interoperability
capabilities invalidate a differentiation story based only on those features.
Distinguish shipped capabilities, roadmaps, maintainer intentions and requests;
no unified, dated CatBoost product roadmap was established by this review.

E0–E6 define engineering completion; E7 separately tests outside adoption.
Numerical criteria are proposed requirements, not measured results: independent
correctness, real-task evidence for every A1–A13 item from at least six sources, fair tuned quality,
end-to-end GPU/train-many/inference cost, five Agent modification tasks plus two
held-out tasks, and clean-environment delivery. Failed or missing required cases
cannot pass. Freeze resource/data manifests before full evaluation; retain old
protocols and failures if requirements are later revised. Protocol `v1-plan-r2`
replaces the previous eight-task coverage gate. Poisson, Gamma and Tweedie have
separate application checks; distributional prediction, a real Formula task and
the train-many model-selection workflow are all required.

## Changes

- [v1 plan](../planning/agent-boosting-foundation-plan.md): required scope,
  architecture decisions, F0–F5 dependencies, independently reviewable slices
  and executor handoff. F0/F1/F4/F5 trace every A-ID; discovery trials can start
  after the small CPU path. Implementation order follows component dependencies.
- [Eval protocol](../planning/openboost-v1-evaluation.md): explicit gates,
  baselines, budgets, tolerances, failure accounting and result artifacts.
- [Application contracts](../planning/foundation-application-contracts.md):
  individually required A1–A13 matrix with recipe mappings, per-use-case checks
  and dataset choices; separate conditional means from predictive distributions.
- [Release and plan review](../planning/boosting-release-review-2026-09-05.md):
  primary-source snapshot, status distinctions and concrete design implications.
- [Agent guide](../AGENTS.md) and learning index: durable v1 direction and links.

## Verification

- Read official release pages and relevant roadmap/issues, objective/device
  documentation and primary dataset references. Sources are linked in the
  planning artifacts. New competitor versions were not installed or benchmarked.
- Read current WeibullAFT fit/objective, censoring tests and distributional
  exposure call sites/tests. Current event-indicator survival behavior does not
  establish interval-label support or fixed-noise AFT semantics when shape varies.
- Documentation validation with
  `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python`:
  initial slice and scope correction both passed all 7 Markdown files, 36 local
  links and balanced code fences. Scope-correction checks also passed: A1–A13
  are complete and unique, all R1–R9 recipes are mapped, required coverage
  references agree and the protocol is r2. `git diff --check` passed; the staged
  diff is inspected before commit.
- No runtime implementation, data download, Modal job, external contact or
  publication occurred. Model regression tests do not validate a planning-only
  change and were not run.

## Failed Attempts

- An insurance/survival-only amendment would overfit the latest examples. The
  user's clarification led to separate structural and application coverage axes.
- The first broad plan still allowed eight representative task results and
  synthetic-only Formula evidence to satisfy engineering coverage. The user
  rejected that narrowing. Every named use case now has a required evidence
  obligation; a missing real Formula task or omitted Gamma/Tweedie evaluation
  leaves v1 incomplete.
- GitHub API and some filtered pages were unavailable; successfully read release
  and specific issue pages support the snapshot, not exhaustive roadmap coverage.
- No new algorithm experiments. The prior P7 T4 ratio of 12.888 versus its 1.2
  budget remains a failure, not rewritten as a pass or a universal disproof of
  programmable boosting. See the main plan's original artifact link.

## Risks and Follow-ups

- Scope is a full v1 target, not permission to build every subsystem at once.
  Start F0.1 task cards, F0.2 independent oracles and F0.3 runnable/frozen protocol;
  then use small CPU paths and discovery trials to challenge the abstractions.
- Candidate data, licensing, preprocessing, actual build support and all new
  quality/cost/adoption claims remain unverified. Numeric gates are engineering
  choices, not statistically demonstrated universal parity guarantees.
- A new algorithm can be mathematically correct and predict worse. Internal
  plugins and Agent success do not demonstrate outside authors or adoption.

## Commits

- `f30c2ed` — preceding clean foundation design and execution plan.
- `a9fde60` — initial v1 scope and evaluation gates.
- This correction: `docs: require every v1 use case individually`.
