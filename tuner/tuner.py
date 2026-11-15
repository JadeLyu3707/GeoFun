#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: yl12783, Nov. 2026, New York
#
# There is no training: phi has two scalars. sigma is grid-searched, and for each sigma tau is chosen in
# closed form from the distribution of S(g) = sum_i exp(-|g - x_i|^2 / 2 sigma^2) over the inside voxels.

import logging
from typing import Dict, List, Tuple

import numpy as np

from models import GaussianPhi
from .grid import GroundTruth, VoxelGrid
from .metrics import geometry_metrics

logger = logging.getLogger(__name__)


class Tuner:
    def __init__(
        self,
        sigmas: List[float],
        max_miss: float = 0.001,
        n_tau_candidates: int = 201,
        tau_quantile_max: float = 0.5,
        truncate: float = 4.0,
    ):
        params = locals()
        for k, v in params.items():
            if k != 'self' and not k.startswith('_'):
                setattr(self, k, v)

    def tune(self, points: np.ndarray, grid: VoxelGrid, gt: GroundTruth) -> Tuple[List[Dict], Dict]:
        """For every sigma, two choices of tau:
            solid     : the largest tau with missed <= max_miss  (= max_miss-quantile of S over inside voxels)
            min-error : argmin over candidate taus of loss = extra + missed
        The recommended setting is the "solid" row with the smallest loss."""
        rows = []
        for s in self.sigmas:
            model = GaussianPhi(points, s, truncate=self.truncate)
            S = model.kernel_sum_on_grid(grid.axes)
            S_in = S[gt.inside]
            cands = np.quantile(S_in, np.linspace(0, self.tau_quantile_max, self.n_tau_candidates))
            n_in = gt.inside.sum()
            errs = [((S > c) & ~gt.inside).sum() + (n_in - (S_in > c).sum()) for c in cands]
            for kind, thr in [("solid", np.quantile(S_in, self.max_miss)), ("min-error", cands[int(np.argmin(errs))])]:
                m = geometry_metrics(S > thr, gt, grid.dv)
                m.update(sigma=s, kind=kind, tau=float(model.scale * thr), s_threshold=float(thr))
                rows.append(m)
                logger.info(f"sigma={s} {kind}: tau={m['tau']:.5f} loss={m['loss']:.3f} iou={m['iou']:.3f}")
            del S
        best = min([r for r in rows if r["kind"] == "solid"], key=lambda r: r["loss"])
        return rows, best


def format_table(rows: List[Dict]) -> str:
    head = (f"{'sigma':>6s} {'choice':>9s} {'tau':>8s} | {'loss':>5s} {'IoU':>5s} {'extra':>6s} {'missed':>6s} | "
            f"outer layer [mm] med/p95/max | missed depth max [mm]")
    lines = [head]
    for m in rows:
        lines.append(f"{m['sigma']:6.3f} {m['kind']:>9s} {m['tau']:8.5f} | {m['loss']:5.3f} {m['iou']:5.3f} "
                     f"{100*m['extra']:5.1f}% {100*m['missed']:5.2f}% | {1e3*m['out_med']:4.1f} / {1e3*m['out_p95']:4.1f} / "
                     f"{1e3*m['out_max']:4.1f}        | {1e3*m['in_max']:4.1f}")
    return "\n".join(lines)
