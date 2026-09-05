"""Public CPU A+B+C example, also executed by verify_wheels.py."""

import hashlib
import json

import numpy as np
from bounded_leaves import BoundedNewton
from normal_fisher import ChannelDecay, NormalFisher

from openboost.experimental import Booster, LevelWiseBuilder, TrainerConfig

rng = np.random.default_rng(7)
X = rng.normal(size=(64, 3)).astype(np.float32)
y = (1.2 * X[:, 0] + 0.4 * rng.normal(size=64)).astype(np.float32)
weights = rng.choice(np.array([0, 0.5, 1, 2], np.float32), 64)
objective = NormalFisher()
model = Booster(
    objective=objective,
    tree_builder=LevelWiseBuilder(leaf_rule=BoundedNewton(0.5)),
    step_schedule=ChannelDecay(tau=1),
    config=TrainerConfig(n_trees=2, max_depth=2, learning_rate=0.2, random_state=7),
).fit(X, y, sample_weight=weights)
params = objective.constrain(model.predict_raw(X))
assert params["mu"].shape == (64,) and np.all(params["sigma"] > 0)
model.save("demo.ob")
np.savez("demo.npz", X=X, **model.predict_raw(X))
print(
    json.dumps(
        {
            "data_sha256": hashlib.sha256(
                X.tobytes() + y.tobytes() + weights.tobytes()
            ).hexdigest(),
            "samples": 64,
            "seed": 7,
            "coefficients": model.coefficients_,
            "finite_positive_scale": bool(np.isfinite(params["sigma"]).all()),
        }
    )
)
