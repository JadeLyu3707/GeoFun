#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: yl12783, Nov. 2026, New York
#
# Volume metrics between the predicted geometry {phi > 0} and the true geometry Omega, on voxel centres.
# All volumes are reported as fractions of |Omega|.

from typing import Dict
import numpy as np

from .grid import GroundTruth


def _stats(d: np.ndarray, prefix: str) -> Dict[str, float]:
    if d.size == 0:
        return {f"{prefix}_med": 0.0, f"{prefix}_p95": 0.0, f"{prefix}_max": 0.0}
    return {f"{prefix}_med": float(np.median(d)), f"{prefix}_p95": float(np.percentile(d, 95)),
            f"{prefix}_max": float(d.max())}


def geometry_metrics(pred: np.ndarray, gt: GroundTruth, dv: float) -> Dict[str, float]:
    """pred: bool grid of phi(g) > 0.

    extra  = |FP| dv / |Omega|,  FP = {phi > 0, outside}
    missed = |FN| dv / |Omega|,  FN = {phi <= 0, inside}
    loss   = extra + missed = |pred (sym. diff.) Omega| / |Omega|
    out_*  = stats of dist(g, Omega) over FP   (thickness of the outer error layer)
    in_max = max dist(g, boundary) over FN     (depth of the deepest missed voxel)
    """
    fp = pred & ~gt.inside
    fn = gt.inside & ~pred
    m = {
        "extra": float(fp.sum() * dv / gt.volume),
        "missed": float(fn.sum() * dv / gt.volume),
        "iou": float((pred & gt.inside).sum() / max((pred | gt.inside).sum(), 1)),
        **_stats(gt.d_out[fp], "out"),
        "in_max": float(gt.d_in[fn].max()) if fn.any() else 0.0,
    }
    m["loss"] = m["extra"] + m["missed"]
    for name, body, near in [("plate", gt.plate, ~gt.near_obstacle), ("obstacle", gt.obstacle, gt.near_obstacle)]:
        if body.any():
            m[f"{name}_extra"] = float((fp & near).sum() / body.sum())
            m[f"{name}_missed"] = float((fn & body).sum() / body.sum())
            m.update({f"{name}_{k}": v for k, v in _stats(gt.d_out[fp & near], "out").items()})
    return m
