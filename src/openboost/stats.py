"""Named additive CPU row fields with explicit training-weight ownership."""

from dataclasses import dataclass, replace

import numpy as np

from .data import Problem, _owned


@dataclass(frozen=True, eq=False)
class RowFields:
    problem_identity: str
    data_identity: str
    names: tuple[str, ...]
    values: np.ndarray
    roles: tuple[str, ...]

    def __post_init__(self):
        names, roles = tuple(self.names), tuple(self.roles)
        values = _owned(self.values, ndim=2)
        if (
            len(names) != values.shape[1]
            or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n for n in names)
        ):
            raise ValueError("unique names must match fields")
        if len(roles) != len(names) or any(
            r not in ["unweighted", "training", "independent"] for r in roles
        ):
            raise ValueError("explicit field weight roles required")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "values", values)

    def add_independent(self, name, values):
        a = np.asarray(values, dtype=float)
        if a.shape != (len(self.values),):
            raise ValueError("independent field must align with rows")
        return RowFields(
            self.problem_identity,
            self.data_identity,
            (*self.names, name),
            np.column_stack([self.values, a]),
            (*self.roles, "independent"),
        )


def apply_weight(fields, problem):
    if (
        not isinstance(fields, RowFields)
        or not isinstance(problem, Problem)
        or fields.problem_identity != problem.identity
        or fields.data_identity != problem.data.identity
        or len(fields.values) != len(problem.weight)
    ):
        raise ValueError("fields must belong to this problem")
    if "training" in fields.roles or "unweighted" not in fields.roles:
        raise ValueError("training weight already applied or no unweighted fields")
    values = fields.values.copy()
    with np.errstate(over="raise", invalid="raise"):
        values[:, np.array(fields.roles) == "unweighted"] *= problem.weight[:, None]
    return replace(
        fields,
        values=values,
        roles=tuple("training" if r == "unweighted" else r for r in fields.roles),
    )


def newton(problem, gradient, curvature):
    """Scalar unweighted derivatives -> named, once-weighted G/H fields."""
    g, h = np.asarray(gradient, dtype=float), np.asarray(curvature, dtype=float)
    if g.shape != (len(problem.target),) or h.shape != g.shape or np.any(h < 0):
        raise ValueError("aligned scalar gradient and nonnegative curvature required")
    fields = RowFields(
        problem.identity,
        problem.data.identity,
        ("gradient", "curvature"),
        np.column_stack([g, h]),
        ("unweighted", "unweighted"),
    )
    return apply_weight(fields, problem)
