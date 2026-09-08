"""Development loss-threshold policy composed from public CPU operations.

This deliberately simple training-loss rule demonstrates completion metadata;
it is not a statistical stopping test or evidence of generalization quality.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np

from openboost.artifacts import TreeTerm
from openboost.binning import prepare_training
from openboost.objectives import Squared
from openboost.ops import newton_leaf
from openboost.runtime import AcceptedState, initialize, propose_terms, resolve
from openboost.tree import depthwise


@dataclass(frozen=True)
class LossThresholdStatus:
    rounds: int
    completed_rounds: int
    reason: str
    threshold: float
    losses: tuple[float, ...]


@dataclass(frozen=True)
class ThresholdResult:
    state: AcceptedState
    steps: tuple[float, ...]
    stop: LossThresholdStatus


def squared_until_loss(
    train, validation, *, context, rounds=5, threshold=0.05, bins=6, prepared=None
):
    """Fit half-step unregularized trees until measured training loss is small.

    Check after each completed round. Zero budget performs no update. Validation
    still chooses best_model through public transactions, separately from this rule.
    """
    if type(rounds) is not int or rounds < 0:
        raise ValueError("nonnegative integer rounds required")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("finite nonnegative loss threshold required")
    Squared.validate(train)
    Squared.validate(validation)
    binned = prepare_training(train.data, bins=bins, prepared=prepared)
    state = initialize(context, train, validation, Squared.base(train), score=Squared.loss)
    losses = []
    reason = "budget"
    for _ in range(rounds):
        tree = depthwise(
            binned,
            Squared.fields(train, state.train_raw),
            max_depth=1,
            leaf=partial(newton_leaf, reg_lambda=0.0),
        )
        proposal = propose_terms(state, (TreeTerm(tree, [[1]], coefficient=0.5),))
        state = resolve(state, proposal, accept=True, score=Squared.loss)
        losses.append(Squared.loss(train, state.train_raw))
        if losses[-1] <= threshold:
            reason = "loss_threshold"
            break
    steps = tuple(losses)
    return ThresholdResult(
        state, steps, LossThresholdStatus(rounds, len(steps), reason, float(threshold), steps)
    )
