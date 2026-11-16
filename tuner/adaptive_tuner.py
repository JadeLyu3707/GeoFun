#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: yl12783, Nov. 2026, New York
#
# Grid search for the adaptive field: (d, k, c) are enumerated; for each, the field F = phi + tau is computed once
# and tau is chosen on it (fixed values + "solid" + "min-error"), so no field is recomputed per tau.

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

from models import AdaptiveGaussianPhi, separable_gaussian_sum
from .grid import GroundTruth, VoxelGrid
from .metrics import geometry_metrics

logger = logging.getLogger(__name__)


def choose_taus(F: np.ndarray, inside: np.ndarray, max_miss: float, n_cand: int, q_max: float,
                fixed: Sequence[float]) -> List[Tuple[str, float]]:
    F_in = np.sort(F[inside]); F_out = np.sort(F[~inside])
    def errors(t):   # |FN| + |FP| for threshold t (inside <=> F > t)
        return np.searchsorted(F_in, t, "right") + (len(F_out) - np.searchsorted(F_out, t, "right"))
    cands = np.quantile(F_in, np.linspace(0, q_max, n_cand))
    taus = [("solid", float(np.quantile(F_in, max_miss))),
            ("min-error", float(cands[int(np.argmin(errors(cands)))]))]
    taus += [(f"tau={t}", float(t)) for t in fixed]
    return taus


class AdaptiveTuner:
    def __init__(
        self,
        ks: List[int],
        cs: List[float],
        ds: List[int],
        fixed_taus: List[float],
        clamp: Tuple[float, float] = (0.25, 4.0),
        max_miss: float = 0.001,
        n_tau_candidates: int = 201,
        tau_quantile_max: float = 0.5,
        baseline_sigma: float = 0.010,
    ):
        params = locals()
        for k, v in params.items():
            if k != 'self' and not k.startswith('_'):
                setattr(self, k, v)

    def _rows(self, F, gt, grid, info) -> List[Dict]:
        rows = []
        f_in = F[gt.inside]
        for kind, tau in choose_taus(F, gt.inside, self.max_miss, self.n_tau_candidates,
                                     self.tau_quantile_max, self.fixed_taus):
            m = geometry_metrics(F > tau, gt, grid.dv)
            m.update(info, kind=kind, tau=tau, field_in_median=float(np.median(f_in)),
                     field_in_p01=float(np.quantile(f_in, 0.001)))
            rows.append(m)
        return rows

    def tune(self, points: np.ndarray, grid: VoxelGrid, gt: GroundTruth) -> Tuple[List[Dict], Dict]:
        rows = []
        # baseline: the earlier fixed-width field, -tau + 1/n sum 1/(sqrt(2 pi) sigma) exp(...)
        s = self.baseline_sigma
        F = separable_gaussian_sum(points, 1.0 / (len(points) * np.sqrt(2 * np.pi) * s), s, grid.axes)
        rows += [r for r in self._rows(F, gt, grid, dict(method="baseline", d=0, k=0, c=0.0, sigma_med=s))
                 if r["kind"] in ("solid", "min-error")]
        for d in self.ds:
            for k in self.ks:
                for c in self.cs:
                    model = AdaptiveGaussianPhi(points, k=k, c=c, d=d, clamp=self.clamp)
                    F = model.field_on_grid(grid.axes)
                    info = dict(method="adaptive", d=d, k=k, c=c, sigma_med=float(np.median(model.sigma)))
                    new = self._rows(F, gt, grid, info); rows += new
                    sol = new[0]
                    logger.info(f"d={d} k={k} c={c}: sigma_med={1e3*info['sigma_med']:.1f}mm, "
                                f"field inside median {sol['field_in_median']:.3f}, solid tau={sol['tau']:.4f} "
                                f"loss={sol['loss']:.3f}")
                    del F
        primary = [r for r in rows if r["method"] == "adaptive" and r["d"] == self.ds[0] and r["kind"] == "solid"]
        best = min(primary, key=lambda r: r["loss"])
        return rows, best


def format_adaptive_table(rows: List[Dict]) -> str:
    """One line per (method, d, k, c): solid / min-error / best fixed tau."""
    keyf = lambda r: (r["method"], r["d"], r["k"], r["c"])
    groups = {}
    for r in rows:
        groups.setdefault(keyf(r), []).append(r)
    head = (f"{'method':>8s} {'d':>1s} {'k':>3s} {'c':>5s} {'sig_med':>7s} {'F_in':>6s} | "
            f"{'solid tau':>9s} {'loss':>5s} {'IoU':>5s} {'layer':>5s} | {'min-err tau':>11s} {'loss':>5s} {'miss':>5s} | "
            f"{'best fixed':>10s} {'loss':>5s} {'miss':>6s}")
    lines = [head, "-" * len(head)]
    for (method, d, k, c), g in groups.items():
        sol = next(r for r in g if r["kind"] == "solid"); mer = next(r for r in g if r["kind"] == "min-error")
        fixed = [r for r in g if r["kind"].startswith("tau=")]
        bf = min(fixed, key=lambda r: r["loss"]) if fixed else None
        line = (f"{method:>8s} {d:1d} {k:3d} {c:5.2f} {1e3*sol['sigma_med']:6.1f}m {sol['field_in_median']:6.3f} | "
                f"{sol['tau']:9.4f} {sol['loss']:5.3f} {sol['iou']:5.3f} {1e3*sol.get('out_mean', 0):4.1f}m | "
                f"{mer['tau']:11.4f} {mer['loss']:5.3f} {100*mer['missed']:4.1f}% | ")
        line += (f"{bf['kind']:>10s} {bf['loss']:5.3f} {100*bf['missed']:5.1f}%" if bf else "")
        lines.append(line)
    return "\n".join(lines)
