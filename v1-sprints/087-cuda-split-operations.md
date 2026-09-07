# Sprint 087: Public CUDA candidate and feasibility composition

Status: local construction after the user's continuation at the 078-A retrospective.
Baseline `4251a36`. Implements 086's 078-B; it is not a new phase or GPU allowance.
Both 085 device runs are consumed. No further remote dispatch is authorized yet.

## Retrospective decision

Real T4 statistics preserve identity, objective weights and independent information.
The remaining product risk is expressing an algorithm change through public device
operations. Continue with independently composable candidate scores and masks,
then routing and leaves. Keep CPU searches paused and author accounting gaps visible.
Do not turn this work into a private GPU trainer or a generic compiler project.

## Public construction contract

- `candidates(histogram)` returns a device batch with left/right named sums,
  integer child counts, parent totals and an active-candidate mask. Its padded
  slots are lexicographic feature/threshold/missing-direction positions. A slot
  is active only for a bin observed anywhere in the prepared numeric data, matching
  the CPU candidate universe even when a node uses a subset of rows.
- `newton_scores(batch, reg_lambda, split_penalty)` and `feasible(batch,
  min_child_h)` are separate operations. Scores for structurally illegal candidates
  are zero; legal gains must be finite. Standard scalar operations require named,
  once-weighted gradient/curvature and nonnegative row curvature.
- `child_minimum(batch, name, minimum)` checks a declared independent information
  column on both children. It validates that original information is nonnegative.
  `mask_and(a, b)` composes predicates on exactly the same batch. D2 uses these
  public operations for arbitrary cohort names/minima, without task-name dispatch.
- `scores(batch, buffer)` / `mask(batch, buffer)` bind explicitly supplied resident
  float32/bool vectors. Borrowed lifetime and exact batch identity are validated.
  These are explicit bindings, not a claim of external custom-kernel support.
- `choose(batch, scores, mask)` returns one registered split or None, with strictly
  positive gain and exact lexicographic ties. It always excludes inactive padding,
  even if a supplied mask enables it. Only a compact winner index returns to host.
- `partition(rows, split)` produces two original-position device row views in the
  input order, bound to the selected batch. Export only compact child sizes for
  allocation. `leaf(histogram, reg_lambda)` returns an owned scalar Newton value.
  Child histograms are recomputed from those actual routed rows.

All kernel arrays/scratch remain in the owned CuPy pool/stream. Explicit validation
flags, winner/size exports and sync counts remain visible. No host candidate loop,
bulk round-trip or writable authoritative state is exposed. Unsupported scalar
schemas, wrong contexts/batches, stale/released buffers and invalid parameters fail.
No training or accepted/proposal integration is included in this slice.

## Frozen development cases before implementation

The independent oracle enumerates conditions and original rows in float64; it
does not import histogram or device production operations. Compare its candidate
universe, child sums/counts, gains, masks, winner, routes and leaves with public CPU
operations locally before adding kernels. Cases use lambda=1 and minimum=1 unless
specified, with fixed integer bin codes and original row IDs distinct from positions.

| Case | Inputs and deciding property |
| --- | --- |
| S1 | Six distinct bins, g=[-6,1,1,1,1,2], h=ones, alternating A/B; unconstrained winner differs from D2's best feasible winner |
| S2 | Same gradients, one predictor with codes [0,0,0,1,1,1], A only in first three rows, B only in last three; no D2-feasible split |
| S3 | 086's eight-row F1, including missing values, zero weights, independent information and unsorted subset [6,0,3,7] |
| S4 | Duplicate columns [0,0,1,1], g=[-2,-2,2,2], h=ones; exact feature/missing-direction ties resolve lexicographically |
| S5 | S4 with zero gradients, then with zero curvature/positive lambda; no positive legal split; zero-curvature leaf with lambda=1 remains defined |
| S6 | One column [0,0,missing,missing], g=[-2,-2,2,2], h=ones; last observed bin separates missing rows |
| S7 | Codes [0,0,2,2] and a second all-missing feature; inactive bins/padding never become eligible, including under an explicitly supplied all-true mask |
| S8 | S1 with cohort names/order changed and minima 0/1/2; predicates compose and bind to the same batch without fixed cohort names |
| S9 | Empty/all-one-bin nodes, invalid denominators and invalid/overflowing parameters, nonfinite gains, negative information/curvature, wrong-batch buffers, stale rows/splits and allocation failure |

Integer counts/routes and active universe are exact. Float32 comparisons use
unchanged E1 rtol=1e-4/atol=1e-5. Exact tie fixtures have no additional near-tie band;
other winning gains are separated. Similar final scores cannot excuse a wrong
non-tied split. No two-round training, quality or speed claim follows these cases.

## Work and acceptance

1. Commit the independent fixtures/oracle and CPU agreement; first failing check
   is the missing public candidate operation.
2. Implement public records/operations and Python CUDA kernels, with real-device
   tests kept explicitly not_run. Preserve all earlier 078-A/storage cases.
3. Freeze a reviewable installed-wheel validation package with exact cases,
   dependencies, source identities and proposed resource bounds. Verify its local
   guards/judge and CPU regression, lint and documentation.
4. Only after the package is concrete request a new one-run T4 allowance: maximum
   900-second function, 600-second tests, 16-MiB private pools, zero automatic
   retries. Include previous 33 cases and the new fixed small split cases. This
   proposal does not itself authorize a run. A failure consumes an approved run.

Hardware acceptance requires every preregistered cell to pass on an installed
wheel, with independent intermediate checks and installed source/version integrity.
Until then report implemented/unverified. No simulator or local collection counts
as CUDA correctness. No new author agent, held-out access or external contact.

Reflect after each verified slice/three implementation commits, or an ownership
counterexample. Next, after verified 078-B, is 078-C resident two-round training
under atomic state and CPU-readable inference contracts.
