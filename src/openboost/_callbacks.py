"""Callback system for OpenBoost training.

Phase 13: Pluggable hooks for training (early stopping, logging, checkpoints).

Inspired by Keras/PyTorch Lightning - allows customizable behavior without
modifying core training loops. Works with any model (GradientBoosting, DART,
OpenBoostGAM, etc.).

Example:
    >>> from openboost import GradientBoosting, EarlyStopping, Logger
    >>> 
    >>> model = GradientBoosting(n_trees=1000)
    >>> model.fit(X, y, 
    ...     callbacks=[
    ...         EarlyStopping(patience=50),
    ...         Logger(period=10),
    ...     ],
    ...     eval_set=[(X_val, y_val)]
    ... )
    >>> print(f"Stopped at iteration {model.best_iteration_}")
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class TrainingState:
    """Shared state passed to callbacks during training.
    
    This object is passed to all callbacks at each training event,
    allowing them to inspect and modify training behavior.
    
    Attributes:
        model: The model being trained (modified in place).
        round_idx: Current boosting round (0-indexed).
        n_rounds: Total number of rounds requested.
        train_loss: Training loss for current round (if computed).
        val_loss: Validation loss for current round (if eval_set provided).
        extra: Dict for custom data (research callbacks can use this).
    """
    model: Any
    round_idx: int = 0
    n_rounds: int = 0
    train_loss: float | None = None
    val_loss: float | None = None
    extra: dict = field(default_factory=dict)


class Callback(ABC):
    """Base class for training callbacks.
    
    Subclass this to create custom callbacks for training hooks.
    All methods are optional - override only what you need.
    
    Example (custom callback):
        >>> class GradientTracker(Callback):
        ...     def __init__(self):
        ...         self.grad_norms = []
        ...     
        ...     def on_round_end(self, state):
        ...         if 'grad_norm' in state.extra:
        ...             self.grad_norms.append(state.extra['grad_norm'])
        ...         return True
        >>> 
        >>> tracker = GradientTracker()
        >>> model.fit(X, y, callbacks=[tracker])
        >>> plt.plot(tracker.grad_norms)
    """
    
    def on_train_begin(self, state: TrainingState) -> None:
        """Called at the start of training.
        
        Args:
            state: Current training state.
        """
        pass
    
    def on_round_begin(self, state: TrainingState) -> None:
        """Called at the start of each boosting round.
        
        Args:
            state: Current training state.
        """
        pass
    
    def on_round_end(self, state: TrainingState) -> bool:
        """Called at the end of each boosting round.
        
        Args:
            state: Current training state.
            
        Returns:
            True to continue training, False to stop early.
        """
        return True
    
    def on_train_end(self, state: TrainingState) -> None:
        """Called at the end of training.
        
        Args:
            state: Current training state.
        """
        pass


class EarlyStopping(Callback):
    """Stop training when validation metric stops improving.
    
    Works with ANY model that provides val_loss in TrainingState.
    Requires `eval_set` to be passed to `fit()`.
    
    Args:
        patience: Number of rounds without improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        restore_best: If True, restore model to best iteration after stopping.
        verbose: If True, print message when stopping.
        
    Attributes (after training):
        best_score: Best validation score achieved.
        best_round: Round at which best score was achieved.
        stopped_round: Round at which training was stopped (or None).
        
    Example:
        >>> callback = EarlyStopping(patience=50, min_delta=1e-4)
        >>> model.fit(X, y, callbacks=[callback], eval_set=[(X_val, y_val)])
        >>> print(f"Best round: {model.best_iteration_}")
    """
    
    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.0,
        restore_best: bool = True,
        verbose: bool = False,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.verbose = verbose
        
        # State
        self.best_score: float = float('inf')
        self.best_round: int = 0
        self.wait: int = 0
        self.stopped_round: int | None = None
        self._best_trees: list | None = None
        self._best_weights: list | None = None  # For DART
    
    def on_train_begin(self, state: TrainingState) -> None:
        """Reset state at start of training."""
        self.best_score = float('inf')
        self.best_round = 0
        self.wait = 0
        self.stopped_round = None
        self._best_trees = None
        self._best_weights = None
    
    def on_round_end(self, state: TrainingState) -> bool:
        """Check if we should stop training."""
        if state.val_loss is None:
            return True  # No validation set, continue
        
        current_score = state.val_loss
        
        if current_score < self.best_score - self.min_delta:
            # Improvement found
            self.best_score = current_score
            self.best_round = state.round_idx
            self.wait = 0
            
            if self.restore_best:
                # Snapshot current model state
                self._best_trees = [t for t in state.model.trees_]
                # Handle DART tree weights
                if hasattr(state.model, 'tree_weights_'):
                    self._best_weights = list(state.model.tree_weights_)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_round = state.round_idx
                if self.verbose:
                    print(f"Early stopping at round {state.round_idx}. "
                          f"Best was round {self.best_round} with score {self.best_score:.6f}")
                return False  # Stop training
        
        return True
    
    def on_train_end(self, state: TrainingState) -> None:
        """Restore best model if requested."""
        if self.restore_best and self._best_trees is not None:
            state.model.trees_ = self._best_trees
            if self._best_weights is not None and hasattr(state.model, 'tree_weights_'):
                state.model.tree_weights_ = self._best_weights
        
        # Set attributes on model
        state.model.best_iteration_ = self.best_round
        state.model.best_score_ = self.best_score


class Logger(Callback):
    """Log training progress to stdout.
    
    Args:
        period: Print every N rounds (default: 1).
        show_train: Include training loss in output.
        show_val: Include validation loss in output.
        
    Example:
        >>> callback = Logger(period=10)  # Log every 10 rounds
        >>> model.fit(X, y, callbacks=[callback], eval_set=[(X_val, y_val)])
        [0]   train: 0.5234  valid: 0.5456
        [10]  train: 0.2134  valid: 0.2345
        [20]  train: 0.1234  valid: 0.1456
        ...
    """
    
    def __init__(
        self,
        period: int = 1,
        show_train: bool = True,
        show_val: bool = True,
    ):
        self.period = max(1, period)
        self.show_train = show_train
        self.show_val = show_val
    
    def on_round_end(self, state: TrainingState) -> bool:
        """Print progress if at logging period."""
        if state.round_idx % self.period == 0:
            parts = [f"[{state.round_idx}]"]
            
            if self.show_train and state.train_loss is not None:
                parts.append(f"train: {state.train_loss:.6f}")
            
            if self.show_val and state.val_loss is not None:
                parts.append(f"valid: {state.val_loss:.6f}")
            
            if len(parts) > 1:  # Have something to print
                print("  ".join(parts))
        
        return True


class ModelCheckpoint(Callback):
    """Save model periodically or when validation score improves.
    
    Args:
        filepath: Path to save model (use .pkl extension).
        save_best_only: If True, only save when validation improves.
        verbose: If True, print message when saving.
        
    Example:
        >>> callback = ModelCheckpoint('best_model.pkl', save_best_only=True)
        >>> model.fit(X, y, callbacks=[callback], eval_set=[(X_val, y_val)])
    """
    
    def __init__(
        self,
        filepath: str,
        save_best_only: bool = True,
        verbose: bool = False,
    ):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_score: float = float('inf')
    
    def on_round_end(self, state: TrainingState) -> bool:
        """Save model if conditions are met."""
        should_save = False
        
        if not self.save_best_only:
            should_save = True
        elif state.val_loss is not None and state.val_loss < self.best_score:
            self.best_score = state.val_loss
            should_save = True
        
        if should_save:
            self._save_model(state.model)
            if self.verbose:
                print(f"Model saved to {self.filepath}")
        
        return True
    
    def _save_model(self, model) -> None:
        """Save model using pickle."""
        import pickle
        with open(self.filepath, 'wb') as f:
            pickle.dump(model, f)


class LearningRateScheduler(Callback):
    """Adjust learning rate during training.
    
    Args:
        schedule: Function (round_idx) -> learning_rate_multiplier
        
    Example:
        >>> # Decay learning rate by 0.95 each round
        >>> scheduler = LearningRateScheduler(lambda r: 0.95 ** r)
        >>> model.fit(X, y, callbacks=[scheduler])
        
        >>> # Step decay: halve LR at round 50 and 100
        >>> def step_decay(r):
        ...     if r < 50: return 1.0
        ...     elif r < 100: return 0.5
        ...     else: return 0.25
        >>> scheduler = LearningRateScheduler(step_decay)
    """
    
    def __init__(self, schedule):
        self.schedule = schedule
        self._initial_lr: float | None = None
    
    def on_train_begin(self, state: TrainingState) -> None:
        """Store initial learning rate."""
        self._initial_lr = state.model.learning_rate
    
    def on_round_begin(self, state: TrainingState) -> None:
        """Update learning rate for this round."""
        if self._initial_lr is not None:
            multiplier = self.schedule(state.round_idx)
            state.model.learning_rate = self._initial_lr * multiplier


class HistoryCallback(Callback):
    """Record training history (losses per round).
    
    Attributes (after training):
        history: Dict with 'train_loss' and 'val_loss' lists.
        
    Example:
        >>> history = HistoryCallback()
        >>> model.fit(X, y, callbacks=[history], eval_set=[(X_val, y_val)])
        >>> plt.plot(history.history['train_loss'], label='train')
        >>> plt.plot(history.history['val_loss'], label='valid')
        >>> plt.legend()
    """
    
    def __init__(self):
        self.history: dict[str, list[float]] = {
            'train_loss': [],
            'val_loss': [],
        }
    
    def on_train_begin(self, state: TrainingState) -> None:
        """Reset history."""
        self.history = {'train_loss': [], 'val_loss': []}
    
    def on_round_end(self, state: TrainingState) -> bool:
        """Record losses."""
        if state.train_loss is not None:
            self.history['train_loss'].append(state.train_loss)
        if state.val_loss is not None:
            self.history['val_loss'].append(state.val_loss)
        return True


class CallbackManager:
    """Orchestrates multiple callbacks.
    
    Used internally by training loops to manage callback execution.
    
    Args:
        callbacks: List of Callback instances.
    """
    
    def __init__(self, callbacks: list[Callback] | None = None):
        self.callbacks = list(callbacks) if callbacks else []
    
    def on_train_begin(self, state: TrainingState) -> None:
        """Call on_train_begin for all callbacks."""
        for cb in self.callbacks:
            cb.on_train_begin(state)
    
    def on_round_begin(self, state: TrainingState) -> None:
        """Call on_round_begin for all callbacks."""
        for cb in self.callbacks:
            cb.on_round_begin(state)
    
    def on_round_end(self, state: TrainingState) -> bool:
        """Call on_round_end for all callbacks.
        
        Returns:
            True if training should continue, False if any callback wants to stop.
        """
        for cb in self.callbacks:
            if not cb.on_round_end(state):
                return False
        return True
    
    def on_train_end(self, state: TrainingState) -> None:
        """Call on_train_end for all callbacks."""
        for cb in self.callbacks:
            cb.on_train_end(state)
