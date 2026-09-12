"""Normal trajectories with exact original-row trees; no production imports.

Geometry uses the independent scalar Normal formulas. Trees enumerate original
row partitions with rational reductions. Comparison bounds and three separate
snapshot anchors are independent of the production recipe/runtime loop.
"""

import numpy as np

from .device_normal import base, direction, geometry
from .exact_growth import fit, predict
from .normal_comparison import compare


def run(
    f,
    *,
    policy="depthwise",
    update="joint",
    mode="natural",
    damping=0.0,
    count=3,
    depth=2,
    leaves=None,
    rate=0.1,
    fixed=False,
    regularization=1.0,
    penalty=0.0,
    patience=None,
    min_delta=0.0,
    minimum=None,
):
    initial = base(f["target"], f["offset"], f["weight"])
    raw = np.broadcast_to(initial, f["offset"].shape).copy()
    validation = np.broadcast_to(initial, f["validation_offset"].shape).copy()
    best = validation.copy()
    patience_raw = validation.copy()
    version = nterms = best_terms = stale = completed = 0
    history = []
    groups = {"joint": ((0, 1),), "forward": ((0,), (1,)), "reverse": ((1,), (0,))}[update]

    def validation_compare(before, after):
        return compare(
            before, after, f["validation_target"], f["validation_offset"], f["validation_weight"]
        )

    for outer in range(count):
        for channels in groups:
            loss, g, h = geometry(raw, f["target"], f["offset"], f["weight"])
            z = direction(g, h, mode=mode, damping=damping)
            nodes = []
            fields = []
            delta = np.zeros_like(raw)
            valid_delta = np.zeros_like(validation)
            for k in channels:
                values = np.column_stack((-f["weight"] * z[:, k], f["weight"]))
                if minimum is not None:
                    values = np.column_stack((values, f["information"]))
                tree = fit(
                    f["x"],
                    values,
                    policy=policy,
                    depth=depth,
                    leaves=leaves,
                    regularization=regularization,
                    penalty=penalty,
                    information_minima={2: minimum, 3: minimum} if minimum is not None else None,
                )
                fields.append(values)
                nodes.append(tree)
                delta[:, k] = predict(tree, f["x"])
                valid_delta[:, k] = predict(tree, f["validation_x"])
            attempts = []
            before = raw
            before_version = version
            accepted = False
            for j in range(1 if fixed else 6):
                alpha = rate * 0.5**j
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        candidate = raw + alpha * delta
                        valid_candidate = validation + alpha * valid_delta
                    geometry(candidate, f["target"], f["offset"], f["weight"])
                    geometry(
                        valid_candidate,
                        f["validation_target"],
                        f["validation_offset"],
                        f["validation_weight"],
                    )
                except (ValueError, OverflowError, FloatingPointError):
                    if fixed:
                        raise
                    attempts.append((alpha, "invalid"))
                    continue
                change = compare(raw, candidate, f["target"], f["offset"], f["weight"])
                accepted = fixed or change.improves()
                attempts.append((alpha, "accepted" if accepted else "rejected"))
                if accepted:
                    raw = candidate
                    validation = valid_candidate
                    version += 1
                    nterms += len(channels)
                    if validation_compare(best, validation).improves():
                        best = validation.copy()
                        best_terms = nterms
                    break
            history.append(
                dict(
                    round=outer,
                    channels=channels,
                    before=before,
                    raw=raw,
                    loss_before=loss,
                    loss=geometry(raw, f["target"], f["offset"], f["weight"])[0],
                    validation=validation.copy(),
                    gradient=g,
                    fisher=h,
                    direction=z,
                    fields=fields,
                    nodes=nodes,
                    attempts=attempts,
                    accepted=accepted,
                    before_version=before_version,
                    version=version,
                    best_terms=best_terms,
                    best=best.copy(),
                    validation_change=None,
                )
            )
        change = validation_compare(patience_raw, validation)
        if change.improves(min_delta):
            patience_raw = validation.copy()
            stale = 0
        else:
            stale += 1
        completed += 1
        history[-1]["validation_change"] = change
        if patience is not None and stale >= patience:
            break
    reason = "patience" if patience is not None and stale >= patience else "budget"
    return dict(
        initial=initial,
        steps=history,
        raw=raw,
        validation=validation,
        best=best,
        best_terms=best_terms,
        completed=completed,
        stale=stale,
        reason=reason,
    )
