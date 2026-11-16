#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: yl12783, Nov. 2026, New York

import logging
import os
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import PlateSnapshot, OBSTACLE
from models import GaussianPhi
from .grid import GroundTruth, VoxelGrid
from .metrics import geometry_metrics

logger = logging.getLogger(__name__)


class Evaler:
    """Metrics of a fixed field + slice figures.

    Slices are for display only; all metrics are computed on the full 3D grid. On each slice:
      background = phi at the voxel centres of that plane
      green      = phi = 0, drawn with matplotlib `contour` (marching squares, linear interpolation along cell edges)
      black      = chi = 0.5, the true boundary
      dots       = x_i within `near_tol` of the plane
    """

    def __init__(self, out_folder: str, truncate: float = 4.0, dpi: int = 95):
        self.out_folder = out_folder
        self.truncate = truncate
        self.dpi = dpi

    def Eval(self, snapshot: PlateSnapshot, grid: VoxelGrid, gt: GroundTruth, sigma: float, tau: float) -> Dict:
        """Fixed-width model (models.GaussianPhi)."""
        model = GaussianPhi(snapshot.points, sigma, tau, truncate=self.truncate)
        phi = model.phi_from_sum(model.kernel_sum_on_grid(grid.axes))
        m = geometry_metrics(phi > 0, gt, grid.dv)
        m.update(sigma=sigma, tau=tau)
        self.plot_slices(snapshot, grid, gt, phi, m, f"sigma={sigma}, tau={tau:.5f}",
                         f"slices_sigma{sigma}_tau{tau:.5f}.png", near_tol=sigma)
        return m

    def EvalField(self, snapshot, grid, gt, field: np.ndarray, tau: float, label: str, fname: str,
                  near_tol: float) -> Dict:
        """Any precomputed field F = phi + tau on the grid."""
        phi = field - tau
        m = geometry_metrics(phi > 0, gt, grid.dv)
        self.plot_slices(snapshot, grid, gt, phi, m, label, fname, near_tol)
        return m

    def plot_slices(self, snapshot, grid, gt, phi, m, label, fname, near_tol):
        xi, ax_c = snapshot.points, grid.axes
        centre = snapshot.obstacle_points.mean(0) if len(snapshot.obstacle_points) else xi.mean(0)
        z_mid = snapshot.plate_points[:, 2].mean()
        views = [
            (f"top view z={z_mid:.4f} (plate mid-thickness)", 2, z_mid, 0, 1, None),
            (f"x-z section y={centre[1]:.3f}", 1, centre[1], 0, 2, None),
            (f"y-z section x={centre[0]:.3f}", 0, centre[0], 1, 2, None),
            (f"zoom x-z section y={centre[1]:.3f}", 1, centre[1], 0, 2,
             ((centre[0] - 0.08, centre[0] + 0.08), (z_mid - 0.05, z_mid + 0.08))),
        ]
        vm = np.percentile(np.abs(phi), 99.5)
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        for ax, (title, axis, val, iu, iv, zoom) in zip(axs.flat, views):
            k = int(np.argmin(np.abs(ax_c[axis] - val)))
            F, M = np.take(phi, k, axis=axis), np.take(gt.inside, k, axis=axis)
            u, v = ax_c[iu], ax_c[iv]
            ax.imshow(F.T, origin="lower", extent=[u[0], u[-1], v[0], v[-1]], cmap="RdBu_r", vmin=-vm, vmax=vm)
            ax.contour(u, v, M.T.astype(float), levels=[0.5], colors="k", linewidths=1)
            ax.contour(u, v, F.T, levels=[0], colors="lime", linewidths=1.2)
            near = np.abs(xi[:, axis] - ax_c[axis][k]) < near_tol
            ax.scatter(xi[near, iu], xi[near, iv], s=5, c="k")
            ax.set_title(title); ax.set_aspect("equal"); ax.set_xlabel("xyz"[iu]); ax.set_ylabel("xyz"[iv])
            if zoom:
                ax.set_xlim(*zoom[0]); ax.set_ylim(*zoom[1])
        md = snapshot.metadata
        thick = f", mean outer layer {1e3*m['out_mean']:.1f} mm" if "out_mean" in m else ""
        fig.suptitle(f"{md.split_name}/{md.traj} t={md.t}: x_i = {md.n_points} nodes, {label}   "
                     f"black: true geometry, green: phi=0\nloss {m['loss']:.3f} (extra {100*m['extra']:.1f}% + missed "
                     f"{100*m['missed']:.2f}%), IoU {m['iou']:.3f}, outer layer median {1e3*m['out_med']:.1f} / "
                     f"p95 {1e3*m['out_p95']:.1f} mm{thick}")
        fig.tight_layout()
        self._save(fig, fname)

    def plot_point_stats(self, snapshot, model, fname: str):
        """Per-point r_k, sigma_i and w_i of an AdaptiveGaussianPhi."""
        xi = snapshot.points; ob = snapshot.node_type == OBSTACLE
        fig, axs = plt.subplots(2, 3, figsize=(18, 11))
        for ax, val, name in [(axs[0, 0], model.sigma * 1e3, "sigma_i [mm]"), (axs[0, 1], model.w, "w_i")]:
            order = np.argsort(xi[:, 2])
            sc = ax.scatter(xi[order, 0], xi[order, 1], c=val[order], s=10, cmap="viridis",
                            marker="o", edgecolors="none")
            ax.scatter(xi[ob, 0], xi[ob, 1], s=12, facecolors="none", edgecolors="r", linewidths=0.4)
            ax.set_aspect("equal"); ax.set_title(f"top view coloured by {name} (red rings: obstacle)")
            ax.set_xlabel("x"); ax.set_ylabel("y"); fig.colorbar(sc, ax=ax, shrink=0.8)
        ax = axs[0, 2]
        ax.scatter(model.r_k[~ob] * 1e3, model.r_k_smooth[~ob] * 1e3, s=4, label="plate")
        ax.scatter(model.r_k[ob] * 1e3, model.r_k_smooth[ob] * 1e3, s=4, c="r", label="obstacle")
        lim = [0, max(model.r_k.max(), model.r_k_smooth.max()) * 1e3 * 1.05]
        ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlabel("r_k [mm]"); ax.set_ylabel("smoothed r_k [mm]")
        ax.set_title("kNN radius before / after smoothing"); ax.legend()
        for ax, val, name in [(axs[1, 0], model.sigma * 1e3, "sigma_i [mm]"), (axs[1, 1], model.w, "w_i"),
                              (axs[1, 2], model.r_k * 1e3, "r_k [mm]")]:
            ax.hist(val[~ob], bins=60, alpha=0.7, label="plate")
            ax.hist(val[ob], bins=60, alpha=0.7, color="r", label="obstacle")
            ax.set_title(f"distribution of {name}"); ax.legend()
        fig.suptitle(f"k={model.k}, c={model.c}, d={model.d}: sigma_i median {1e3*np.median(model.sigma):.1f} mm, "
                     f"clamped to [{model.clamp[0]}, {model.clamp[1]}] x median; sum w_i = {model.w.sum():.3e}")
        fig.tight_layout()
        self._save(fig, fname)

    def _save(self, fig, fname):
        os.makedirs(self.out_folder, exist_ok=True)
        f = os.path.join(self.out_folder, fname)
        fig.savefig(f, dpi=self.dpi); plt.close(fig)
        logger.info(f"saved {f}")
