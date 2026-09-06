"""Independent bounded Newton rule using only the public extension API."""

import numpy as np

from openboost.experimental import NewtonLeafRule


class BoundedNewton:
    """Clip Newton leaf outputs; retain the builder's original split criterion."""

    supported_devices = frozenset({"cpu", "cuda"})

    def __init__(self, bound=0.5):
        if not np.isfinite(bound) or bound <= 0:
            raise ValueError("bound must be finite and positive")
        self.bound = float(bound)

    def values(self, G, H, *, config, context):
        if context.device not in self.supported_devices:
            raise ValueError("Unsupported execution device")
        values = NewtonLeafRule().values(G, H, config=config, context=context)
        return context.xp.clip(values, -self.bound, self.bound)
