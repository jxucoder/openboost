"""D3: externally implemented weighted pinball plus anchored quadratic leaves."""

import numpy as np

from openboost.tree import depthwise


class PenalizedLeaves:
    """Replace the public routed leaf callback while retaining recipe state/growth.

    Bisection on the monotone right derivative is independent of OpenBoost's
    built-in breakpoint scan. The strictly positive penalty gives a unique leaf.
    """

    def __init__(self, *, q=0.5, penalty=1.0, anchor=0.0, grower=depthwise):
        if (
            not all(np.isscalar(v) and np.isfinite(v) for v in (q, penalty, anchor))
            or not 0 < q < 1
            or penalty <= 0
        ):
            raise ValueError("q in (0,1), positive penalty and finite anchor required")
        self.q, self.penalty, self.anchor, self.grower = q, penalty, anchor, grower

    def solve(self, view, total=None, names=None):
        mass = float(view.weight.sum())
        if not np.isfinite(mass):
            raise ValueError("finite total weight required")
        if mass == 0:
            return float(self.anchor)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            low = float(self.anchor - (1 - self.q) * mass / self.penalty)
            high = float(self.anchor + self.q * mass / self.penalty)
            if not np.isfinite(low) or not np.isfinite(high):
                raise ValueError("finite solver bracket required")
            for _ in range(100):
                middle = low / 2 + high / 2
                derivative = (
                    self.penalty * (middle - self.anchor)
                    + view.weight[view.residual <= middle].sum()
                    - self.q * mass
                )
                if derivative >= 0:
                    high = middle
                else:
                    low = middle
        return float(low / 2 + high / 2)

    def __call__(self, data, fields, *, row_leaf, **options):
        # The external solver deliberately replaces the recipe's default leaf.
        # q must also be supplied to the quantile recipe for matching split/loss semantics.
        return self.grower(data, fields, row_leaf=self.solve, **options)
