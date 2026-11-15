#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: yl12783, Nov. 2026, New York

import logging
import os
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import PlateSnapshot
from models import GaussianPhi
from .grid import GroundTruth, VoxelGrid
from .metrics import geometry_metrics

logger = logging.getLogger(__name__)


class Evaler:
    """Metrics of a fixed (sigma, tau) + slice figures.

    Slices are for display only; all metrics are computed on the full 3D grid. On each slice:
      background = phi at the voxel centres of that plane
      green      = phi = 0, drawn with matplotlib `contour` (marching squares, linear interpolation along cell edges)
      black      = chi = 0.5, the true boundary
      dots       = x_i within sigma of the plane
    """

    def __init__(self, out_folder: str, truncate: float = 4.0, dpi: int = 95):
        self.out_folder = out_folder
        self.truncate = truncate
        self.dpi = dpi

    def Eval(self, snapshot: PlateSnapshot, grid: VoxelGrid, gt: GroundTruth, sigma: float, tau: float) -> Dict:
        model = GaussianPhi(snapshot.points, sigma, tau, truncate=self.truncate)
        phi = model.phi_from_sum(model.kernel_sum_on_grid(grid.axes))
        m = geometry_metrics(phi > 0, gt, grid.dv)
        m.update(sigma=sigma, tau=tau)
        self.plot(snapshot, grid, gt, model, phi, m)
        return m

    def plot(self, snapshot, grid, gt, model, phi, m):
        xi, ax_c, s = snapshot.points, grid.axes, model.sigma
        centre = snapshot.obstacle_points.mean(0) if len(snapshot.obstacle_points) else xi.mean(0)
        z_mid = snapshot.plate_points[:, 2].mean()
        views = [
            (f"top view z={z_mid:.4f} (plate mid-thickness)", 2, z_mid, 0, 1, None),
            (f"x-z section y={centre[1]:.3f}", 1, centre[1], 0, 2, None),
            (f"y-z section x={centre[0]:.3f}", 0, centre[0], 1, 2, None),
            (f"zoom x-z section y={centre[1]:.3f}", 1, centre[1], 0, 2,
             ((centre[0] - 0.08, centre[0] + 0.08), (z_mid - 0.04, z_mid + 0.07))),
        ]
        vm = np.abs(phi).max()
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        for ax, (title, axis, val, iu, iv, zoom) in zip(axs.flat, views):
            k = int(np.argmin(np.abs(ax_c[axis] - val)))
            F, M = np.take(phi, k, axis=axis), np.take(gt.inside, k, axis=axis)
            u, v = ax_c[iu], ax_c[iv]
            ax.imshow(F.T, origin="lower", extent=[u[0], u[-1], v[0], v[-1]], cmap="RdBu_r", vmin=-vm, vmax=vm)
            ax.contour(u, v, M.T.astype(float), levels=[0.5], colors="k", linewidths=1)
            ax.contour(u, v, F.T, levels=[0], colors="lime", linewidths=1.2)
            near = np.abs(xi[:, axis] - ax_c[axis][k]) < s
            ax.scatter(xi[near, iu], xi[near, iv], s=5, c="k")
            ax.set_title(title); ax.set_aspect("equal"); ax.set_xlabel("xyz"[iu]); ax.set_ylabel("xyz"[iv])
            if zoom:
                ax.set_xlim(*zoom[0]); ax.set_ylim(*zoom[1])
        md = snapshot.metadata
        fig.suptitle(f"{md.split_name}/{md.traj} t={md.t}: x_i = {md.n_points} nodes, sigma={s}, tau={model.tau:.5f}"
                     f"   black: true geometry, green: phi=0\nloss {m['loss']:.3f} (extra {100*m['extra']:.1f}% + missed "
                     f"{100*m['missed']:.2f}%), IoU {m['iou']:.3f}, outer layer median {1e3*m['out_med']:.1f} / "
                     f"p95 {1e3*m['out_p95']:.1f} mm")
        fig.tight_layout()
        os.makedirs(self.out_folder, exist_ok=True)
        f = os.path.join(self.out_folder, f"slices_sigma{s}_tau{model.tau:.5f}.png")
        fig.savefig(f, dpi=self.dpi); plt.close(fig)
        logger.info(f"saved {f}")
