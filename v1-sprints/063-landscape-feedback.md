# Sprint 063 addendum: Evaluate the supplied landscape survey

Baseline: `961b6f9`, after the retrospective. Status: selective source/code review
complete; no production changes or new formal evaluation tasks frozen.

## Review scope and judgment

The user supplied another AI's GBM survey as optional input. Read it as a source
of hypotheses, verify the consequential additions, compare them with actual public
call paths, and incorporate only findings that change an execution decision.
This review checked GBNet, NGBoost releases, XGBoost 3.4 and three proposed research
probes. It did not verify every version, paper, benchmark or release claim in the
survey. Unchecked projects do not become baseline requirements by appearing in it.

The central conclusion agrees with Sprint 063: the value hypothesis is lower
cost for verified algorithm changes. The useful new contribution is a sharper
set of comparator paths and research examples that expose interface assumptions.
The resulting changes are small: add a stopping-result counterexample to the
next development checks, sharpen the coupling claims and strengthen author-task
selection. Keep N1–N5 and all required applications; do not start a new catalog.

## Verified additions and their implications

| Input | What the primary source supports | Decision for OpenBoost |
|---|---|---|
| [GBNet](https://github.com/mthorrell/gbnet) | XGBoost/LightGBM modules use PyTorch gradients/Hessians and explicit boosting steps; trees cannot propagate gradients, so modules belong at the first layer and training data stay fixed | Include it when comparing Formula/composition authoring. This does not establish editable internal split search; inspect the actual task path before judging |
| [NGBoost releases](https://github.com/stanfordmlgroup/ngboost/releases) | 0.5.9 adds a symbolic distribution factory; 0.5.11 supports distinct base learners for distribution parameters | Include these existing tools in appropriate distribution/learner controls; manually deriving a new distribution is not the only opponent path |
| [XGBoost 3.4 notes](https://xgboost.readthedocs.io/en/stable/changes/v3.4.0.html) | Expanded histogram vector-leaf features retain experimental status; quantile/absolute-error leaf estimation changed | Preserve version/algorithm semantics in comparisons. Shared vector outputs do not by themselves certify a coupled Hessian solve |
| [Coupled vector-leaf preprint](https://arxiv.org/html/2606.29326v1) | A full-Hessian leaf/split formulation; examples are narrow, and full boosting comparisons remain future work | A useful development probe for matrix statistics and solves, not evidence of a superior algorithm |
| [ScoreStop preprint](https://arxiv.org/html/2606.02740v1) | Stopping uses validation gradients and update directions through a functional score test, with assumptions/calibration beyond patience | Test whether a recipe-owned stopping policy can report its true terminal reason through public scheduling |
| [Parallel distributional boosting preprint](https://arxiv.org/html/2607.13550v1) | A common descent direction permits one base learner per iteration across multiple targets | Probe learned learner-to-parameter mappings. The paper's term "parallel" is not M independent runs or multi-GPU execution |

These are documentation/paper findings, not locally reproduced competitor
capabilities or quality results. AlphaXiv overview endpoints returned 404; its
full-text fallbacks and arXiv primary HTML were available. Full paper replication,
statistical guarantees and publication claims beyond the checked versions were
not evaluated. Future comparisons pin software versions and environments before
execution; this review does not alter frozen search configurations.

## A concrete current restriction: stopping-result interoperability

Public recipes can own their loop. However, [RecipeResult/validate_result](../src/openboost/results.py)
requires a concrete `StopState`, and [StopState](../src/openboost/stopping.py)
encodes scalar-score patience/budget termination. The scheduler accepts structurally
different result classes but still prescribes this particular stopping record.
This is narrower than independently programmable stopping.

A tiny source-level probe produced the following on the review baseline:

```text
external reason: score_test
run_many outcome: ValueError
message: recipe result requires AcceptedState and StopState
```

Reproduce with the installed development environment:

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python - <<'PY'
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from openboost import NumericData, Problem, RunContext
from openboost.recipes import squared
from openboost.runs import RunSpec, run_many

@dataclass(frozen=True)
class ExternalStop:
    completed_rounds: int
    reason: str = "score_test"

x = NumericData(np.arange(4)[:, None], np.arange(4), ("x",))
p = Problem(x, [[0], [1], [2], [4]], x.row_ids)
ctx = RunContext("external-stop-review", 63)
fit = squared(p, p, context=ctx, rounds=1, bins=4)
result = SimpleNamespace(state=fit.state, steps=fit.steps, stop=ExternalStop(1))
outcome = run_many([RunSpec(ctx, p, p, lambda *a, **kw: result)])[0]
assert outcome.error_type == "ValueError" and outcome.result is None
print(outcome.error_message)
PY
```

This is an expected rejection under today's concrete contract, not a numerical
failure or an implementation of ScoreStop. A custom loop can already stop; an
adapter or subclass could encode that fact using the existing type. Those
workarounds are not evidence of natural public interoperability, and relabeling
an external decision as exhausted patience/budget would misreport the algorithm.

**Next development probe:** alongside the existing N1 RNG/preparation checks,
introduce an external completed stopping result, retaining true reason and round
count through mixed scheduling. If it fails, separate common completion metadata
from policy-specific diagnostics with the smallest contract change. Preserve
unfinished-result rejection, immutable run identity, one outer-round observation,
failure isolation and existing patience behavior. Model acceptance, best-model
selection and stopping remain distinct decisions.

Do not implement the full statistical method merely to test this boundary.
Actual ScoreStop reproduction would require its statistic, calibration, validation
independence and dependence assumptions to be judged separately; an arbitrary
threshold callback is not that reproduction.

## Distinguish three kinds of coupling

1. **Shared topology/vector payload:** one tree emits multiple values. Current
   CPU growth and persistence support this with diagonal vector Newton helpers.
2. **Coupled per-row direction:** solve a parameter-space system for each row,
   then fit learners to those directions. Current Formula uses this route.
3. **Coupled leaf optimization:** aggregate matrix curvature and gradients over
   each routed leaf, then solve the aggregate problem; score splits consistently.
   This is not established by either preceding capability.

For an independently chosen two-row convex quadratic fixture, let conventional
loss gradients be `g1=(1,-1)`, `g2=(2,1)` and Hessians
`H1=[[2,1],[1,2]]`, `H2=[[4,0],[0,1]]`. The average per-row Newton direction is
`(-0.75,0)`, while the aggregate leaf optimum is `(-9/17,3/17)`. Direct NumPy
solves reproduced those different values in this review. This demonstrates the
semantic distinction, not a quality comparison between complete algorithms.

Current [RowFields](../src/openboost/stats.py) can carry named additive channels,
and [growers](../src/openboost/tree.py) accept custom scoring, legality and additive
leaf callbacks. Flattening the symmetric matrix entries into named channels may
already express a small coupled leaf extension. Test that path before adding a
new core tensor abstraction. A dedicated full-Hessian tree helper is not currently
implemented; the extension's sufficiency remains unverified.

A bounded development task would use K=2/3, nonzero off-diagonal terms, nonunit/zero
weights, explicit regularization and singular-system policy. Independently check
both split gains and leaf stationarity, then at least two rounds and fresh-process
inference. Changing only leaf values while leaving diagonal split scoring must be
labeled a different method. CPU success is not CUDA support; quadratic statistic
storage and small-system solves need their own cost accounting.

## Learner mappings: existing boundary, new algorithmic test

[TreeTerm](../src/openboost/artifacts.py) already accepts an L-by-K mapping, and
[vector tests](../tests/v1/test_public_vector_multiclass.py) exercise L differing
from K. Thus the survey does not reveal a missing output-mapping representation.
What is unverified is a recipe that learns a common update direction and uses it
across rounds. Test that program through existing terms/transactions; do not
claim the cited parallel-distributional algorithm from a fixed projection example.

Keep this as a later development option within N3. The stopping counterexample
is concrete now; full coupled leaves are the next potentially revealing probe.
Implementing all three papers is not a prerequisite for the first CUDA experiment.

## Plan changes and limits

- Add the stopping-result probe to N1 before declaring the scheduling contract
  ready for interface freeze. Record any necessary core change as development.
- In N3, choose opponents by task: Py-Boost for editable GPU execution, GBNet for
  differentiable model composition, and NGBoost's current tools for distribution/
  parameter-learner authoring. XGBoost/LightGBM/CatBoost remain quality controls
  where their semantics match. Do not build a Cartesian product of every library.
- Use the coupled-leaf and adaptive-mapping ideas as development task candidates,
  keeping mathematics, package authoring and real quality as separate outcomes.
  They do not replace the five required D types or either sealed H task.
- All tasks discussed here are visible to the designer and are development tasks.
  Do not label them unseen or silently insert them into a previously frozen cohort.
- Retain the method/foundation scorecards already separated in Sprint 063. Generative,
  graph, RL and general autograd expansion are not added to v1 by this review.
- Keep N2 practical state/trace work and N4 real selection closure. The proposed
  GPU overlap is still not adopted. No new GPU job, participant contact, agent
  delegation or remote publication is implied by this feedback.

Verification: selective primary-source reads, current call-path inspection, the
expected stopping rejection and independent quadratic example above. Documentation
checks are recorded in the [learning entry](../learnings/2026-09-06-v1-landscape-feedback.md).
