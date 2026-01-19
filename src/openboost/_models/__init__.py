"""High-level models built on the core infrastructure.

These models provide scikit-learn-like APIs and use fit_tree()
from the core module to build trees.

Phase 13: Added sklearn-compatible wrappers.
Phase 15: Added distributional GBDT, NaturalBoost, and linear leaf GBDT.
Phase 16: Renamed NGBoost -> NaturalBoost for clarity.
"""

from ._boosting import GradientBoosting, MultiClassGradientBoosting
from ._dart import DART
from ._gam import OpenBoostGAM
from ._batch import ConfigBatch, BatchTrainingState
from ._sklearn import (
    OpenBoostRegressor,
    OpenBoostClassifier,
    OpenBoostDistributionalRegressor,
    OpenBoostLinearLeafRegressor,
)

# Phase 15/16: Distributional GBDT and NaturalBoost
from ._distributional import (
    DistributionalGBDT,
    # Primary names (Phase 16)
    NaturalBoost,
    NaturalBoostNormal,
    NaturalBoostLogNormal,
    NaturalBoostGamma,
    NaturalBoostPoisson,
    NaturalBoostStudentT,
    NaturalBoostTweedie,
    NaturalBoostNegBin,
    # Backward compatibility aliases
    NGBoost,
    NGBoostNormal,
    NGBoostLogNormal,
    NGBoostGamma,
    NGBoostPoisson,
    NGBoostStudentT,
    NGBoostTweedie,
    NGBoostNegBin,
)

# Phase 15: Linear Leaf GBDT
from ._linear_leaf import LinearLeafGBDT, LinearLeafTree

__all__ = [
    # Standard GBDT
    "GradientBoosting",
    "MultiClassGradientBoosting",
    "DART",
    "OpenBoostGAM",
    "ConfigBatch",
    "BatchTrainingState",
    # Phase 13: sklearn-compatible wrappers
    "OpenBoostRegressor",
    "OpenBoostClassifier",
    # Phase 15: sklearn wrappers for new models
    "OpenBoostDistributionalRegressor",
    "OpenBoostLinearLeafRegressor",
    # Phase 15/16: Distributional GBDT - Primary names
    "DistributionalGBDT",
    "NaturalBoost",
    "NaturalBoostNormal",
    "NaturalBoostLogNormal",
    "NaturalBoostGamma",
    "NaturalBoostPoisson",
    "NaturalBoostStudentT",
    "NaturalBoostTweedie",
    "NaturalBoostNegBin",
    # Backward compatibility (deprecated)
    "NGBoost",
    "NGBoostNormal",
    "NGBoostLogNormal",
    "NGBoostGamma",
    "NGBoostPoisson",
    "NGBoostStudentT",
    "NGBoostTweedie",
    "NGBoostNegBin",
    # Phase 15: Linear Leaf GBDT
    "LinearLeafGBDT",
    "LinearLeafTree",
]
