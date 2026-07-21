"""GPU-accelerated Generalized Additive Model (GAM).

This implements an EBM-style model that's fully GPU-parallelized:
- Shape functions learned via gradient boosting
- All features updated in parallel each round
- Inherently interpretable (each feature has a 1D lookup table)

Unlike InterpretML's EBM which trains features sequentially (CPU-bound),
this trains all feature shape functions simultaneously on GPU.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .._array import BinnedArray, array
from .._backends import is_cuda
from .._callbacks import (
    Callback,
    CallbackManager,
    EarlyStopping,
    TrainingState,
    warn_if_early_stopping_without_eval_set,
)
from .._loss import LossFunction, compute_loss_value, get_loss_function
from .._persistence import PersistenceMixin
from .._validation import validate_eval_set

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class OpenBoostGAM(PersistenceMixin):
    """GPU-accelerated Generalized Additive Model.
    
    An interpretable model where:
        prediction = sum(shape_function[i](feature[i]) for all features)
    
    Each shape function is a lookup table mapping binned feature values
    to contribution scores. Trained via parallel gradient boosting.
    
    Args:
        n_rounds: Number of boosting rounds.
        learning_rate: Shrinkage factor (smaller = more stable, needs more rounds).
        reg_lambda: L2 regularization on leaf values.
        loss: Loss function ('mse', 'logloss', or callable).
        n_bins: Number of bins for histogram building (2-256). Binned data is
            uint8 with bin 255 reserved for missing values, so at most 254
            usable bins; 255/256 are clamped to 254 by ``ob.array``. Shape
            function tables are always allocated 256-wide so every uint8 bin
            index (including the missing-value bin) stays in bounds.
        interactions: Number of pairwise interaction terms (GA2M-style) to
            learn after main-effects training. 0 (default) disables
            interactions and preserves the exact pre-existing behavior.
            Candidate pairs are ranked FAST-style: a one-shot 2D histogram
            Newton step is scored on the main-effects residual gradients for
            every feature pair (rows are subsampled for ranking on large
            datasets), the top-``interactions`` pairs are selected, and 2D
            shape tables on the (bin_i, bin_j) grid are then boosted with the
            same Newton update and ``reg_lambda``.
        interaction_rounds: Boosting rounds for the interaction stage.
            None (default) uses ``n_rounds``.
        smoothing: Fused-ridge smoothing strength for 1D shape functions
            (default 0.0 = off, exact pre-existing behavior). Each round's
            per-feature Newton update ``u`` is replaced by the solution of the
            tridiagonal system ``(W + smoothing * D^T D) s = W u`` where ``D``
            is the first-difference matrix and ``W`` is a 0/1 diagonal marking
            bins that contain data. Occupied bins anchor the solution while
            empty bins interpolate between their neighbors, damping
            sparse-bin noise. Applies to ordinal (numeric) bins only:
            categorical features, the missing-value bin (255), and 2D
            interaction tables are not smoothed.
        monotone: Optional dict mapping feature index -> +1 (non-decreasing)
            or -1 (non-increasing). After every boosting round the feature's
            accumulated 1D shape function is projected onto the constraint
            with weighted isotonic regression (PAVA, weighted by per-bin
            sample counts). Applies to ordinal bins only; the missing-value
            bin and 2D interaction tables are unconstrained. Default None
            (no constraints, exact pre-existing behavior).

    Note:
        The interaction stage, smoothing, monotone projection, and the
        callbacks/eval_set machinery all run on CPU. With a CUDA backend,
        plain fits keep the GPU main-effects path (interaction boosting then
        runs on CPU afterwards); fits that request smoothing, monotone
        constraints, callbacks, or eval_set fall back to CPU training with a
        warning.

    Example:
        ```python
        import openboost as ob
        
        gam = ob.OpenBoostGAM(n_rounds=1000, learning_rate=0.01)
        gam.fit(X_train, y_train)
        predictions = gam.predict(X_test)
        
        # Interpret: plot shape function for feature 0
        gam.plot_shape_function(0, feature_name="age")
        ```
    """
    
    n_rounds: int = 1000
    learning_rate: float = 0.01
    reg_lambda: float = 1.0
    loss: str | LossFunction = 'mse'
    n_bins: int = 254
    n_trees: int | None = field(default=None, repr=False)
    interactions: int = 0
    interaction_rounds: int | None = None
    smoothing: float = 0.0
    monotone: dict[int, int] | None = None

    def __post_init__(self) -> None:
        # uint8 binning contract: at most 254 usable bins, bin 255 reserved for
        # missing values. 256 stays accepted as "maximum resolution" (the
        # historical default; ob.array clamps it to 254); anything outside
        # [2, 256] cannot be represented and fails fast here.
        if not 2 <= self.n_bins <= 256:
            raise ValueError(
                f"n_bins must be in [2, 256] (uint8 binning: at most 254 usable bins, "
                f"bin 255 reserved for missing values); got {self.n_bins}"
            )
        if self.interactions < 0:
            raise ValueError(f"interactions must be >= 0; got {self.interactions}")
        if self.interaction_rounds is not None and self.interaction_rounds < 0:
            raise ValueError(
                f"interaction_rounds must be None or >= 0; got {self.interaction_rounds}"
            )
        if self.smoothing < 0:
            raise ValueError(f"smoothing must be >= 0; got {self.smoothing}")
        if self.monotone:
            for f_idx, direction in self.monotone.items():
                if direction not in (-1, 1):
                    raise ValueError(
                        f"monotone[{f_idx}] must be +1 (non-decreasing) or "
                        f"-1 (non-increasing); got {direction}"
                    )
        if self.n_trees is not None:
            if self.n_rounds != 1000:
                warnings.warn(
                    "Both n_trees and n_rounds specified. Using n_trees.",
                    UserWarning,
                    stacklevel=2,
                )
            self.n_rounds = self.n_trees

    # Fitted attributes
    shape_values_: NDArray | None = field(default=None, init=False, repr=False)
    pair_shape_values_: dict[tuple[int, int], NDArray] = field(
        default_factory=dict, init=False, repr=False
    )
    interaction_pairs_: list[tuple[int, int]] = field(
        default_factory=list, init=False, repr=False
    )
    evals_result_: dict[str, dict[str, list[float]]] = field(
        default_factory=dict, init=False, repr=False
    )
    X_binned_: BinnedArray | None = field(default=None, init=False, repr=False)
    _loss_fn: LossFunction | None = field(default=None, init=False, repr=False)

    @property
    def trees_(self) -> dict[str, object]:
        """Snapshot adapter for the shared callback machinery.

        ``EarlyStopping(restore_best=True)`` (see ``_callbacks.py``, which is
        shared across all models and must not be edited here) snapshots
        ``model.trees_`` with ``copy.deepcopy`` whenever the validation metric
        improves and assigns the best snapshot back at train end. A GAM has no
        tree list; its learnable state is the 1D/2D lookup tables. Exposing
        that state through a ``trees_`` property (with a setter that restores
        it) makes restore-to-best work for GAM without touching the callback
        code. This property is not a dataclass field and never appears in
        ``vars(self)``, so persistence ignores it.
        """
        return {
            'shape_values': self.shape_values_,
            'pair_shape_values': self.pair_shape_values_,
            'interaction_pairs': self.interaction_pairs_,
        }

    @trees_.setter
    def trees_(self, snapshot: dict[str, object]) -> None:
        self.shape_values_ = snapshot['shape_values']
        self.pair_shape_values_ = snapshot['pair_shape_values']
        self.interaction_pairs_ = snapshot['interaction_pairs']

    def _post_load(self) -> None:
        """Backfill attributes missing from models saved before interactions."""
        if not hasattr(self, 'pair_shape_values_') or self.pair_shape_values_ is None:
            self.pair_shape_values_ = {}
        if not hasattr(self, 'interaction_pairs_') or self.interaction_pairs_ is None:
            self.interaction_pairs_ = []
        if not hasattr(self, 'evals_result_') or self.evals_result_ is None:
            self.evals_result_ = {}
        # Restore tuple keys: joblib/pickle keeps them intact, but be robust
        # to states where keys were stored as lists.
        self.pair_shape_values_ = {
            tuple(k): v for k, v in self.pair_shape_values_.items()
        }
        self.interaction_pairs_ = [tuple(p) for p in self.interaction_pairs_]

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        callbacks: list[Callback] | None = None,
        eval_set: list[tuple[NDArray, NDArray]] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> OpenBoostGAM:
        """Fit the GAM model.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            callbacks: List of Callback instances for training hooks
                (e.g. ``EarlyStopping``, ``Logger``, ``HistoryCallback``),
                mirroring ``GradientBoosting.fit``.
            eval_set: Validation set(s) as a list of ``(X_val, y_val)``
                tuples (a single tuple is auto-wrapped). Every eval set is
                evaluated each round with the training loss as the metric and
                the per-round history is stored in ``evals_result_`` as
                ``{'eval_0': {'<loss>': [...]}, ...}``. Early stopping
                monitors the LAST eval set.
            early_stopping_rounds: Convenience for
                ``EarlyStopping(patience=early_stopping_rounds,
                restore_best=True)``. Requires ``eval_set``. When training
                stops (or ends), the model is restored to the best iteration
                and ``best_iteration_`` / ``best_score_`` are set. With
                ``interactions > 0`` the main-effect and interaction rounds
                form one monitored sequence; stopping during the main-effect
                phase skips the interaction phase.

        Returns:
            self: The fitted model.
        """
        y = np.asarray(y, dtype=np.float32).ravel()

        # LOW-12: Initialize base score
        loss_name = self.loss if isinstance(self.loss, str) else ''
        if loss_name in ('logloss', 'binary_crossentropy'):
            p = np.clip(np.mean(y), 1e-7, 1 - 1e-7)
            self.base_score_ = np.float32(np.log(p / (1 - p)))
        else:
            self.base_score_ = np.float32(np.mean(y))

        # Get loss function
        self._loss_fn = get_loss_function(self.loss)

        # Bin the data
        if isinstance(X, BinnedArray):
            self.X_binned_ = X
        else:
            self.X_binned_ = array(X, n_bins=self.n_bins)

        # Reset per-fit state
        self.pair_shape_values_ = {}
        self.interaction_pairs_ = []
        self.evals_result_ = {}

        self._validate_monotone_features()

        cb_list = list(callbacks) if callbacks else []
        if early_stopping_rounds is not None:
            cb_list.append(
                EarlyStopping(patience=early_stopping_rounds, restore_best=True)
            )
        eval_set = validate_eval_set(eval_set, self.X_binned_.n_features)
        warn_if_early_stopping_without_eval_set(cb_list, eval_set)

        # Smoothing, monotone projection, and per-round callback/eval work all
        # need host-side histograms, so they require the CPU loop.
        needs_cpu_loop = (
            bool(cb_list)
            or bool(eval_set)
            or self.smoothing > 0
            or bool(self.monotone)
        )

        # Choose training path
        if is_cuda() and not needs_cpu_loop:
            self._fit_gpu(y)
            if self.interactions > 0:
                # Interaction boosting is CPU-only (documented); the GPU
                # main-effects path is kept and the 2D stage runs on host.
                data = self.X_binned_.data
                binned = (
                    data.copy_to_host()
                    if hasattr(data, 'copy_to_host')
                    else np.asarray(data)
                )
                pred = self._predict_from_shape(binned, self.shape_values_)
                n_inter = (
                    self.n_rounds
                    if self.interaction_rounds is None
                    else self.interaction_rounds
                )
                cb_manager = CallbackManager([])
                state = TrainingState(model=self, n_rounds=n_inter)
                metric = self.loss if isinstance(self.loss, str) else 'loss'
                self._boost_interactions(
                    binned, y, pred, cb_manager, state, [], metric, n_inter, 0
                )
        else:
            if is_cuda():
                warnings.warn(
                    "OpenBoostGAM: smoothing/monotone/callbacks/eval_set "
                    "require the CPU training path; falling back to CPU "
                    "training (the GPU path is only used for plain fits).",
                    UserWarning,
                    stacklevel=2,
                )
            self._fit_cpu(y, cb_list, eval_set)

        return self

    def _validate_monotone_features(self) -> None:
        """Validate monotone feature indices against the binned data."""
        if not self.monotone:
            return
        n_features = self.X_binned_.n_features
        is_cat = self.X_binned_.is_categorical
        for f_idx in self.monotone:
            if not 0 <= f_idx < n_features:
                raise ValueError(
                    f"monotone feature index {f_idx} is out of range for "
                    f"data with {n_features} features"
                )
            if len(is_cat) > f_idx and is_cat[f_idx]:
                raise ValueError(
                    f"monotone constraint on feature {f_idx} is invalid: "
                    "the feature is categorical (bins are unordered)"
                )

    def _fit_gpu(self, y: NDArray):
        """GPU training path - all features in parallel."""
        from numba import cuda

        from .._backends._cuda import build_histogram_cuda
        
        n_features = self.X_binned_.n_features
        n_samples = self.X_binned_.n_samples
        binned_gpu = self.X_binned_.data  # (n_features, n_samples) on GPU
        
        # Initialize shape functions to zero.
        # Always 256-wide regardless of n_bins: histogram kernels emit
        # (n_features, 256) and uint8 bin indices (incl. MISSING_BIN=255) must
        # never index out of bounds.
        shape_values_gpu = cuda.device_array((n_features, 256), dtype=np.float32)
        _fill_zeros_2d_gpu(shape_values_gpu)
        
        # Initialize predictions with base score
        base = getattr(self, 'base_score_', np.float32(0.0))
        pred_host = np.full(n_samples, base, dtype=np.float32)
        pred_gpu = cuda.to_device(pred_host)
        
        y_gpu = cuda.to_device(y)
        
        # Boosting loop
        for _ in range(self.n_rounds):
            # Compute gradients on GPU
            grad_gpu, hess_gpu = self._loss_fn(pred_gpu, y_gpu)
            
            # Build histograms for ALL features (your existing kernel!)
            hist_grad, hist_hess = build_histogram_cuda(binned_gpu, grad_gpu, hess_gpu)
            
            # Update ALL shape functions in parallel
            threads = 256
            blocks = (n_features * 256 + threads - 1) // threads
            _update_shape_functions_kernel[blocks, threads](
                hist_grad, hist_hess, shape_values_gpu,
                np.float32(self.learning_rate), np.float32(self.reg_lambda)
            )
            
            # Update predictions using shape functions
            threads = 256
            blocks = (n_samples + threads - 1) // threads
            _predict_gam_kernel[blocks, threads](binned_gpu, shape_values_gpu, pred_gpu)
        
        self.shape_values_ = shape_values_gpu.copy_to_host()
    
    def _fit_cpu(
        self,
        y: NDArray,
        callbacks: list[Callback] | None = None,
        eval_set: list[tuple[NDArray, NDArray]] | None = None,
    ):
        """CPU training path: main effects, then optional 2D interactions."""
        from .._backends._cpu import build_histogram_cpu

        n_features = self.X_binned_.n_features
        n_samples = self.X_binned_.n_samples

        # Get binned data on CPU
        if hasattr(self.X_binned_.data, 'copy_to_host'):
            binned = self.X_binned_.data.copy_to_host()
        else:
            binned = np.asarray(self.X_binned_.data)

        # Always 256-wide regardless of n_bins: build_histogram_cpu returns
        # (n_features, 256) and uint8 bin indices (incl. MISSING_BIN=255) must
        # never index out of bounds.
        self.shape_values_ = np.zeros((n_features, 256), dtype=np.float32)
        shape_values = self.shape_values_  # live alias, mutated in place
        base = getattr(self, 'base_score_', np.float32(0.0))
        pred = np.full(n_samples, base, dtype=np.float32)

        n_inter_rounds = 0
        if self.interactions > 0:
            n_inter_rounds = (
                self.n_rounds
                if self.interaction_rounds is None
                else self.interaction_rounds
            )

        metric = self.loss if isinstance(self.loss, str) else 'loss'

        # Bin eval sets once with the training bin edges
        eval_data: list[tuple[NDArray, NDArray]] = []
        if eval_set:
            for X_e, y_e in eval_set:
                X_e_binned = (
                    X_e if isinstance(X_e, BinnedArray)
                    else self.X_binned_.transform(X_e)
                )
                data_e = X_e_binned.data
                if hasattr(data_e, 'copy_to_host'):
                    data_e = data_e.copy_to_host()
                eval_data.append(
                    (np.asarray(data_e), np.asarray(y_e, dtype=np.float32).ravel())
                )
        self.evals_result_ = {
            f'eval_{i}': {metric: []} for i in range(len(eval_data))
        }

        # Per-bin sample counts anchor the monotone projection weights
        bin_counts = None
        if self.monotone:
            bin_counts = np.zeros((n_features, 256), dtype=np.int64)
            for f in range(n_features):
                bin_counts[f] = np.bincount(binned[f], minlength=256)

        cb_manager = CallbackManager(callbacks)
        state = TrainingState(model=self, n_rounds=self.n_rounds + n_inter_rounds)
        cb_manager.on_train_begin(state)

        stopped = False
        for round_idx in range(self.n_rounds):
            state.round_idx = round_idx
            cb_manager.on_round_begin(state)

            # Compute gradients
            grad, hess = self._loss_fn(pred, y)
            grad = grad.astype(np.float32)
            hess = hess.astype(np.float32)

            # Build histograms
            hist_grad, hist_hess = build_histogram_cpu(binned, grad, hess)

            # Update shape functions
            mask = hist_hess > 0
            updates = np.zeros_like(hist_grad)
            updates[mask] = -hist_grad[mask] / (hist_hess[mask] + self.reg_lambda)

            if self.smoothing > 0:
                updates = self._smooth_updates(updates, hist_hess)

            shape_values += self.learning_rate * updates

            # Project onto the monotone cone AFTER each round (not once at
            # fit end): subsequent rounds then compute gradients from the
            # constrained model and boosting self-corrects for the
            # projection. Projecting only at fit end would train against
            # unconstrained predictions, so the final projection could
            # arbitrarily distort the fit and the interaction stage would
            # select pairs from residuals of a model that no longer exists.
            if self.monotone:
                self._apply_monotone(bin_counts)

            # Update predictions
            pred = self._predict_from_shape(binned, shape_values)

            if not self._record_round(cb_manager, state, pred, y, eval_data, metric):
                stopped = True
                break

        if not stopped and self.interactions > 0:
            self._boost_interactions(
                binned, y, pred, cb_manager, state, eval_data, metric,
                n_inter_rounds, self.n_rounds,
            )

        cb_manager.on_train_end(state)

    def _record_round(
        self,
        cb_manager: CallbackManager,
        state: TrainingState,
        pred: NDArray,
        y: NDArray,
        eval_data: list[tuple[NDArray, NDArray]],
        metric: str,
    ) -> bool:
        """Record eval history and run round-end callbacks.

        Returns:
            True to continue training, False to stop early.
        """
        last_val = None
        for i, (binned_e, y_e) in enumerate(eval_data):
            pred_e = self._predict_host(binned_e)
            val = compute_loss_value(self.loss, pred_e, y_e)
            self.evals_result_[f'eval_{i}'][metric].append(val)
            last_val = val

        if cb_manager.callbacks:
            state.train_loss = compute_loss_value(self.loss, pred, y)
            if last_val is not None:
                # Early stopping monitors the LAST eval set's metric
                state.val_loss = last_val
            return cb_manager.on_round_end(state)
        return True

    def _smooth_updates(self, updates: NDArray, hist_hess: NDArray) -> NDArray:
        """Fused-ridge smoothing of one round's per-feature Newton updates.

        Solves ``(W + smoothing * D^T D) s = W u`` per feature, where ``D`` is
        the first-difference matrix over the feature's ordinal bins and ``W``
        is the 0/1 occupancy diagonal (``hist_hess > 0``). Occupied bins
        anchor the solution at the raw Newton update; empty bins carry no data
        term and interpolate between their neighbors. The system is symmetric
        tridiagonal and positive definite whenever at least one bin is
        occupied, so it is solved with a banded Cholesky solve.

        Only ordinal bins ``0..len(bin_edges)-1`` participate: categorical
        features (no numeric edges) and the missing-value bin (255) keep
        their raw updates, since adjacency is meaningless for them.
        """
        from scipy.linalg import solveh_banded

        s = float(self.smoothing)
        smoothed = updates.copy()
        edges_list = self.X_binned_.bin_edges
        for f in range(updates.shape[0]):
            edges = edges_list[f] if f < len(edges_list) else ()
            n_used = len(edges)
            if n_used < 2:
                continue  # constant or categorical feature: nothing to smooth
            w = (hist_hess[f, :n_used] > 0).astype(np.float64)
            if not np.any(w):
                continue
            u = updates[f, :n_used].astype(np.float64)
            # D^T D is tridiagonal: diag [1, 2, ..., 2, 1], off-diag -1
            diag = np.full(n_used, 2.0)
            diag[0] = diag[-1] = 1.0
            ab = np.zeros((2, n_used))
            ab[0, 1:] = -s
            ab[1, :] = w + s * diag
            smoothed[f, :n_used] = solveh_banded(ab, w * u, lower=False).astype(
                np.float32
            )
        return smoothed

    def _apply_monotone(self, bin_counts: NDArray) -> None:
        """Project constrained 1D shape functions onto the monotone cone.

        Weighted isotonic regression (PAVA) over the feature's ordinal bins,
        weighted by per-bin sample counts so empty bins (which may still be
        hit by test data) conform to the constraint without influencing the
        anchored values. The missing-value bin (255) is unconstrained.
        """
        edges_list = self.X_binned_.bin_edges
        for f_idx, direction in self.monotone.items():
            edges = edges_list[f_idx] if f_idx < len(edges_list) else ()
            n_used = len(edges)
            if n_used < 2:
                continue
            w = np.maximum(bin_counts[f_idx, :n_used].astype(np.float64), 1e-9)
            self.shape_values_[f_idx, :n_used] = _pava(
                self.shape_values_[f_idx, :n_used], w, increasing=direction > 0
            )

    def _select_interaction_pairs(
        self, binned: NDArray, grad: NDArray, hess: NDArray
    ) -> list[tuple[int, int]]:
        """Rank feature pairs FAST-style on main-effects residuals.

        For every pair (i, j) the potential is the Newton gain of a one-shot
        2D histogram step, ``sum_cells g^2 / (h + reg_lambda)``, minus both
        features' 1D gains on the same residuals — i.e. the extra structure a
        2D grid captures beyond what either 1D refit could. Rows are
        subsampled (deterministically) for ranking on large datasets.
        """
        from .._backends._cpu import build_histogram_cpu

        n_features, n_samples = binned.shape
        max_rank_rows = 50_000
        if n_samples > max_rank_rows:
            idx = np.random.default_rng(0).choice(
                n_samples, max_rank_rows, replace=False
            )
            idx.sort()
            b = np.ascontiguousarray(binned[:, idx])
            g, h = grad[idx], hess[idx]
        else:
            b, g, h = binned, grad, hess

        lam = float(self.reg_lambda)
        hist_g, hist_h = build_histogram_cpu(b, g, h)
        hist_g = hist_g.astype(np.float64)
        hist_h = hist_h.astype(np.float64)
        gain1 = np.where(hist_h > 0, hist_g**2 / (hist_h + lam), 0.0).sum(axis=1)

        g64 = g.astype(np.float64)
        h64 = h.astype(np.float64)
        scored: list[tuple[float, int, int]] = []
        for i in range(n_features):
            row_i = b[i].astype(np.intp) * 256
            for j in range(i + 1, n_features):
                combined = row_i + b[j]
                g2 = np.bincount(combined, weights=g64, minlength=65536)
                h2 = np.bincount(combined, weights=h64, minlength=65536)
                m = h2 > 0
                gain2 = float((g2[m] ** 2 / (h2[m] + lam)).sum())
                scored.append((gain2 - float(gain1[i]) - float(gain1[j]), i, j))

        # Deterministic ranking: score desc, then (i, j) asc as tie-break
        scored.sort(key=lambda t: (-t[0], t[1], t[2]))
        k = min(self.interactions, len(scored))
        return [(i, j) for _, i, j in scored[:k]]

    def _boost_interactions(
        self,
        binned: NDArray,
        y: NDArray,
        pred: NDArray,
        cb_manager: CallbackManager,
        state: TrainingState,
        eval_data: list[tuple[NDArray, NDArray]],
        metric: str,
        n_inter_rounds: int,
        round_offset: int,
    ) -> None:
        """Select top-k feature pairs and boost their 2D shape tables (CPU)."""
        # Residual gradients of the fitted main-effects model drive ranking
        grad, hess = self._loss_fn(pred, y)
        grad = np.asarray(grad, dtype=np.float32)
        hess = np.asarray(hess, dtype=np.float32)

        pairs = self._select_interaction_pairs(binned, grad, hess)
        self.interaction_pairs_ = pairs
        if not pairs or n_inter_rounds == 0:
            return

        # Precompute flattened (bin_i * 256 + bin_j) indices per pair
        combined_idx: dict[tuple[int, int], NDArray] = {}
        for i, j in pairs:
            combined_idx[(i, j)] = binned[i].astype(np.intp) * 256 + binned[j]
            self.pair_shape_values_[(i, j)] = np.zeros((256, 256), dtype=np.float32)

        lam = float(self.reg_lambda)
        for r in range(n_inter_rounds):
            state.round_idx = round_offset + r
            cb_manager.on_round_begin(state)

            grad, hess = self._loss_fn(pred, y)
            g64 = np.asarray(grad, dtype=np.float64)
            h64 = np.asarray(hess, dtype=np.float64)

            # All selected pairs take one parallel Newton step per round,
            # matching the parallel per-feature updates of the main loop.
            for pair in pairs:
                comb = combined_idx[pair]
                g2 = np.bincount(comb, weights=g64, minlength=65536)
                h2 = np.bincount(comb, weights=h64, minlength=65536)
                upd = np.zeros(65536, dtype=np.float32)
                m = h2 > 0
                upd[m] = (-g2[m] / (h2[m] + lam)).astype(np.float32)
                self.pair_shape_values_[pair] += (
                    self.learning_rate * upd.reshape(256, 256)
                )
                # Incremental prediction update is exact: this round only
                # adds `lr * upd` to this pair's lookup table.
                pred += self.learning_rate * upd[comb]

            if not self._record_round(cb_manager, state, pred, y, eval_data, metric):
                break

    def _predict_from_shape(self, binned: NDArray, shape_values: NDArray) -> NDArray:
        """CPU prediction using 1D shape functions only."""
        n_samples = binned.shape[1]
        n_features = binned.shape[0]
        base = getattr(self, 'base_score_', np.float32(0.0))
        pred = np.full(n_samples, base, dtype=np.float32)
        for f in range(n_features):
            pred += shape_values[f, binned[f, :]]
        return pred

    def _predict_host(self, binned: NDArray) -> NDArray:
        """Full CPU prediction: base + 1D shape lookups + 2D pair lookups."""
        pred = self._predict_from_shape(binned, self.shape_values_)
        pairs = getattr(self, 'pair_shape_values_', None)
        if pairs:
            for (i, j), table in pairs.items():
                pred += table[binned[i, :], binned[j, :]]
        return pred

    def predict(self, X: NDArray | BinnedArray) -> NDArray:
        """Generate predictions.
        
        Args:
            X: Features, shape (n_samples, n_features).
            
        Returns:
            predictions: Shape (n_samples,).
        """
        if self.shape_values_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Bin the data if needed, using training bin edges for consistency
        if isinstance(X, BinnedArray):
            X_binned = X
        elif self.X_binned_ is not None:
            X_binned = self.X_binned_.transform(X)
        else:
            X_binned = array(X, n_bins=self.n_bins)
        
        n_samples = X_binned.n_samples

        # The GPU kernel only handles 1D lookups; models with interaction
        # tables predict on host (documented CPU-only interaction support).
        if is_cuda() and not getattr(self, 'pair_shape_values_', None):
            from numba import cuda

            binned_gpu = X_binned.data
            shape_gpu = cuda.to_device(self.shape_values_)
            pred_gpu = cuda.device_array(n_samples, dtype=np.float32)

            threads = 256
            blocks = (n_samples + threads - 1) // threads
            _predict_gam_kernel[blocks, threads](binned_gpu, shape_gpu, pred_gpu)

            result = pred_gpu.copy_to_host()
            base = getattr(self, 'base_score_', np.float32(0.0))
            if base != 0.0:
                result += base
            return result
        else:
            if hasattr(X_binned.data, 'copy_to_host'):
                binned = X_binned.data.copy_to_host()
            else:
                binned = np.asarray(X_binned.data)
            return self._predict_host(binned)

    def predict_proba(self, X: NDArray | BinnedArray) -> NDArray:
        """Predict class probabilities (for classification).
        
        Only valid when loss='logloss'.
        """
        if self.loss not in ('logloss', 'binary_crossentropy'):
            raise ValueError("predict_proba only available for classification losses")
        
        raw_pred = self.predict(X)
        prob_1 = 1 / (1 + np.exp(-raw_pred))
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])
    
    def get_feature_importance(self) -> NDArray:
        """Get feature importance based on shape function variance.
        
        Returns:
            importance: Shape (n_features,), higher = more important.
        """
        if self.shape_values_ is None:
            raise RuntimeError("Model not fitted.")
        
        # Importance = variance of shape function (how much it varies)
        return np.var(self.shape_values_, axis=1)
    
    def get_shape_function(self, feature_idx: int) -> NDArray:
        """Get the shape function values for a feature.
        
        Args:
            feature_idx: Index of the feature.
            
        Returns:
            values: Shape (256,), contribution for each bin.
        """
        if self.shape_values_ is None:
            raise RuntimeError("Model not fitted.")
        return self.shape_values_[feature_idx].copy()

    def get_pair_shape_function(self, i: int, j: int) -> NDArray:
        """Get the 2D interaction shape table for feature pair (i, j).

        Args:
            i: First feature index (order-insensitive).
            j: Second feature index.

        Returns:
            values: Shape (256, 256), contribution for each (bin_i, bin_j).

        Raises:
            RuntimeError: If the model is not fitted.
            KeyError: If the pair was not selected during fitting.
        """
        if self.shape_values_ is None:
            raise RuntimeError("Model not fitted.")
        pairs = getattr(self, 'pair_shape_values_', {}) or {}
        key = (i, j) if (i, j) in pairs else (j, i)
        if key not in pairs:
            raise KeyError(
                f"No interaction table for feature pair ({i}, {j}). "
                f"Selected pairs: {sorted(pairs)}"
            )
        return pairs[key].copy()

    def plot_shape_function(self, feature_idx: int, feature_name: str | None = None, ax=None):
        """Plot the shape function for a feature.

        The x-axis shows original feature values, recovered from the bin
        edges learned at fit time. Categorical or constant features (which
        have no numeric bin edges) fall back to raw bin indices. The
        contribution learned for the missing-value bin (255) is not shown.

        Args:
            feature_idx: Index of the feature to plot.
            feature_name: Optional name for the x-axis label.
            ax: Optional matplotlib Axes to draw on. When None, a new
                figure and axes are created.

        Returns:
            The matplotlib Axes containing the plot.

        Raises:
            RuntimeError: If the model is not fitted.
            ValueError: If feature_idx is out of range.
        """
        if self.shape_values_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_features = self.shape_values_.shape[0]
        if not 0 <= feature_idx < n_features:
            raise ValueError(
                f"feature_idx={feature_idx} is out of range for a model "
                f"fitted with {n_features} features"
            )

        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib") from err

        values = self.shape_values_[feature_idx]
        label = feature_name if feature_name is not None else f"Feature {feature_idx}"

        edges = np.array([], dtype=np.float64)
        if self.X_binned_ is not None and feature_idx < len(self.X_binned_.bin_edges):
            edges = np.asarray(self.X_binned_.bin_edges[feature_idx], dtype=np.float64)

        if edges.size > 0:
            # ob.array bins via searchsorted + clip, so occupied bins are
            # 0..len(edges)-1: bin 0 lies below edges[0], bin b between
            # edges[b-1] and edges[b], and the last bin extends past
            # edges[-1]. Use edge midpoints as representative x positions,
            # anchored at the outermost edges.
            n_used = edges.size
            x = np.empty(n_used, dtype=np.float64)
            x[0] = edges[0]
            x[-1] = edges[-1]
            if n_used > 2:
                x[1:-1] = 0.5 * (edges[:-2] + edges[1:-1])
            y_vals = values[:n_used]
            xlabel = label
        else:
            x = np.arange(values.shape[0])
            y_vals = values
            xlabel = f"{label} (binned value)"

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=(10, 4))

        ax.step(x, y_vals, where='mid')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Contribution to prediction")
        ax.set_title(f"Shape Function: {label}")
        if created_fig:
            fig.tight_layout()
        return ax


# =============================================================================
# Helpers
# =============================================================================


def _pava(values: NDArray, weights: NDArray, increasing: bool = True) -> NDArray:
    """Weighted isotonic regression via pool-adjacent-violators (PAVA).

    Returns the monotone (non-decreasing when ``increasing``, else
    non-increasing) sequence minimizing the weighted squared error to
    ``values``. Weights must be strictly positive.
    """
    v = np.asarray(values, dtype=np.float64)
    if not increasing:
        v = -v
    w = np.asarray(weights, dtype=np.float64)

    means: list[float] = []
    wsums: list[float] = []
    counts: list[int] = []
    for k in range(v.shape[0]):
        m, ww, c = float(v[k]), float(w[k]), 1
        # Pool with the previous block while it violates monotonicity
        while means and means[-1] > m:
            m_prev, w_prev, c_prev = means.pop(), wsums.pop(), counts.pop()
            total = w_prev + ww
            m = (m_prev * w_prev + m * ww) / total
            ww = total
            c += c_prev
        means.append(m)
        wsums.append(ww)
        counts.append(c)

    out = np.repeat(np.asarray(means), np.asarray(counts))
    return out if increasing else -out


# =============================================================================
# CUDA Kernels
# =============================================================================

_update_shape_functions_kernel = None
_predict_gam_kernel = None
_fill_zeros_2d_kernel = None
_fill_zeros_1d_kernel = None


def _init_cuda_kernels():
    """Initialize CUDA kernels lazily."""
    global _update_shape_functions_kernel, _predict_gam_kernel
    global _fill_zeros_2d_kernel, _fill_zeros_1d_kernel
    
    if _update_shape_functions_kernel is not None:
        return
    
    from numba import cuda, float32, int32
    
    @cuda.jit
    def update_shape_kernel(hist_grad, hist_hess, shape_values, learning_rate, reg_lambda):
        """Update all shape functions in parallel from histograms."""
        idx = cuda.grid(1)
        n_features = hist_grad.shape[0]
        total_elements = n_features * 256
        
        if idx < total_elements:
            feature = idx // 256
            bin_idx = idx % 256
            
            g = hist_grad[feature, bin_idx]
            h = hist_hess[feature, bin_idx]
            
            if h > float32(0.0):
                update = -g / (h + reg_lambda)
                shape_values[feature, bin_idx] += learning_rate * update
    
    @cuda.jit
    def predict_kernel(binned, shape_values, predictions):
        """Predict by summing shape function lookups."""
        sample_idx = cuda.grid(1)
        n_features = binned.shape[0]
        n_samples = binned.shape[1]
        
        if sample_idx < n_samples:
            total = float32(0.0)
            for f in range(n_features):
                bin_idx = binned[f, sample_idx]
                total += shape_values[f, int32(bin_idx)]
            predictions[sample_idx] = total
    
    @cuda.jit
    def fill_zeros_2d(arr):
        """Fill 2D array with zeros."""
        idx = cuda.grid(1)
        rows, cols = arr.shape
        total = rows * cols
        if idx < total:
            r = idx // cols
            c = idx % cols
            arr[r, c] = float32(0.0)
    
    @cuda.jit
    def fill_zeros_1d(arr):
        """Fill 1D array with zeros."""
        idx = cuda.grid(1)
        if idx < arr.shape[0]:
            arr[idx] = float32(0.0)
    
    _update_shape_functions_kernel = update_shape_kernel
    _predict_gam_kernel = predict_kernel
    _fill_zeros_2d_kernel = fill_zeros_2d
    _fill_zeros_1d_kernel = fill_zeros_1d


def _fill_zeros_2d_gpu(arr):
    """Fill 2D GPU array with zeros."""
    _init_cuda_kernels()
    rows, cols = arr.shape
    total = rows * cols
    threads = 256
    blocks = (total + threads - 1) // threads
    _fill_zeros_2d_kernel[blocks, threads](arr)


def _fill_zeros_1d_gpu(arr):
    """Fill 1D GPU array with zeros."""
    _init_cuda_kernels()
    n = arr.shape[0]
    threads = 256
    blocks = (n + threads - 1) // threads
    _fill_zeros_1d_kernel[blocks, threads](arr)


# Initialize kernels if CUDA available
if is_cuda():
    try:
        _init_cuda_kernels()
    except Exception:
        warnings.warn("Failed to compile GAM CUDA kernels; will retry on first use", stacklevel=1)

