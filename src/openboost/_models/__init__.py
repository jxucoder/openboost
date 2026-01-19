"""High-level models built on the core infrastructure.

These models provide scikit-learn-like APIs and use fit_tree()
from the core module to build trees.

Phase 13: Added sklearn-compatible wrappers.
"""

from ._boosting import GradientBoosting, MultiClassGradientBoosting
from ._dart import DART
from ._gam import OpenBoostGAM
from ._batch import ConfigBatch, BatchTrainingState
from ._sklearn import OpenBoostRegressor, OpenBoostClassifier

__all__ = [
    "GradientBoosting",
    "MultiClassGradientBoosting",
    "DART",
    "OpenBoostGAM",
    "ConfigBatch",
    "BatchTrainingState",
    # Phase 13: sklearn-compatible wrappers
    "OpenBoostRegressor",
    "OpenBoostClassifier",
]
