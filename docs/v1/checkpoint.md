# Foundation checkpoint

OpenBoost v1 remains under construction. This PR brings the implementation from
development snapshot `91a519dd5227344266a3eb1c86ce21b45acf32c6` onto public main,
with selected public/reference tests and component guides. The snapshot identifier
records provenance; its full development ancestry is not included in this branch.

The repository selection manifest at
`planning/foundation-checkpoint-pr-manifest.json` records source hashes and omitted
tests. Existing public history remains present.
Later full-budget Housing and classification construction is outside this snapshot.

## Implemented boundary

The guides cover explicit CPU and CUDA components, including multiclass and AFT
device recipes, vector/mapped updates, exact Normal split/leaf policies, resident
feature sharing, and compatible scalar squared scheduling. CPU multiclass uses a
shared vector tree; the device recipe uses independent scalar trees with a joint
update. These are different recipe choices, not an equivalence claim.

Training uses the explicit [device interfaces](execution.md); CPU recipes do not
automatically dispatch to CUDA. Device categorical growth, arbitrary grouped
recipes, and complete feature, quality or speed parity are not established.

## Historical observations and available evidence

The unchanged [train-many audit report](evidence/train-many-audit.json) records a
passing audit of the original two-round Housing diagnostic: 17 cases at M=1/8/32,
440 final/best model replays, and 416 ownership observations across 11 grouped
cases. The ownership total comprises 139 constructor and 277 completed-round
observations; sequential cases have no grouped-phase observations. The audit also
records model/oracle and ownership corruption controls, original source hashes
and the archive hash.

This is a historical observation tied to its frozen sources and inputs. The
large underlying input/model arrays, original archives and full replay closure
are retained in the original development history and omitted from this PR. The
compact report alone cannot reproduce or independently revalidate that experiment.
Earlier failed attempts remain in that history; a later passing audit does not
turn those original verdicts into passes.

Other guides describe historical component work on multiclass, AFT, vector trees,
exact Normal policies and scheduling. Their newer raw execution archives are also
omitted here, so those descriptions supply interface context rather than fresh
validation of this candidate. Earlier evidence already present on public main
continues to describe its recorded source revision.

## Candidate validation and remaining work

Exact source, syntax, import, lint and link checks are metadata checks. Test
collection does not execute the tests. The current validation status and original
receipts are recorded in [the Modal result index](evidence/pr27-modal-validation/report.json).
Read its status and source binding before claiming a pass. The merge gate requires
installed CPU regression on Python 3.10 and 3.12, wheel and source builds, strict
documentation, and 441 selected current CUDA cases on real T4 hardware, including
fresh CPU serialization consumers and eighteen installed D2 extension cases.

The D2 extension cases compare against the same run's ninety independently checked
current Normal trajectories. Thirty-six older extension trajectories retain their
original pre-exact-policy expectations and are explicitly historical; their two
nontrajectory guards remain current. The manifest lists the archive-linked tests
omitted from this checkpoint. They remain source-specific obligations in the
development history and are not counted as passing candidate checks.

GitHub performs a lightweight check of committed Modal receipts, JUnit results,
raw artifact hashes and the candidate's tracked-file inventory. Any tracked input
change requires refreshed validation; only the result directory is excluded from
the binding. This check performs no training, numerical tests or builds itself.

The two-round diagnostic does not establish full-budget train-many feasibility,
an end-to-end speed improvement, formal E4 cost acceptance, or complete v1.
Every required [application family](https://github.com/jxucoder/openboost/blob/main/planning/foundation-application-contracts.md)
and [recipe/component/evaluation obligation](https://github.com/jxucoder/openboost/blob/main/planning/openboost-v1-evaluation.md)
remains in scope. Complete real-data quality and execution cost, final-source
conformance and installed extensions still need their corresponding evidence.
The author study (E5) remains deferred and unpassed; adoption (E7) is separate.
This checkpoint is a review artifact, not a release or a claim of completed v1.
