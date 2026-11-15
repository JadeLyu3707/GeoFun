#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: yl12783, Nov. 2026, New York

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import ndimage


class VoxelGrid:
    """Voxel-centred grid g = lo + (k + 1/2) h covering the points' bounding box padded by `pad`."""

    def __init__(self, res: float = 0.001, pad: float = 0.05):
        self.res = res
        self.pad = pad
        self.axes: List[np.ndarray] = []

    def build(self, points: np.ndarray):
        lo, hi = points.min(0) - self.pad, points.max(0) + self.pad
        self.axes = [np.arange(lo[d] + self.res / 2, hi[d], self.res) for d in range(3)]
        return self

    @property
    def shape(self):
        return tuple(len(c) for c in self.axes)

    @property
    def dv(self) -> float:
        return self.res ** 3

    def rasterize(self, tets: np.ndarray) -> np.ndarray:
        """Indicator of the union of tetrahedra (K, 4, 3) at the voxel centres (barycentric test)."""
        m = np.zeros(self.shape, bool)
        for T in tets:
            r = [slice(np.searchsorted(self.axes[d], T[:, d].min()),
                       np.searchsorted(self.axes[d], T[:, d].max(), "right")) for d in range(3)]
            if any(s.start >= s.stop for s in r):
                continue
            G = np.stack(np.meshgrid(*[self.axes[d][r[d]] for d in range(3)], indexing="ij"), -1)
            lam = np.linalg.solve((T[1:] - T[0]).T, (G.reshape(-1, 3) - T[0]).T).T
            m[tuple(r)] |= ((lam >= -1e-9).all(1) & (lam.sum(1) <= 1 + 1e-9)).reshape(G.shape[:3])
        return m

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.shape} voxels at {self.res * 1e3:.2f} mm>"


@dataclass
class GroundTruth:
    inside: np.ndarray       # chi(g)
    plate: np.ndarray        # chi restricted to the plate's tetrahedra
    obstacle: np.ndarray     # chi restricted to the obstacle's tetrahedra
    d_out: np.ndarray        # dist(g, Omega) for g outside, 0 inside
    d_in: np.ndarray         # dist(g, outside) for g inside, 0 outside
    near_obstacle: np.ndarray  # outside voxels closer to the obstacle than to the plate
    volume: float            # voxelized |Omega|


def build_ground_truth(grid: VoxelGrid, tets: np.ndarray, is_obstacle_tet: np.ndarray) -> GroundTruth:
    plate = grid.rasterize(tets[~is_obstacle_tet])
    obstacle = grid.rasterize(tets[is_obstacle_tet]) if is_obstacle_tet.any() else np.zeros(grid.shape, bool)
    inside = plate | obstacle
    d_out = ndimage.distance_transform_edt(~inside).astype(np.float32) * grid.res
    d_in = ndimage.distance_transform_edt(inside).astype(np.float32) * grid.res
    if obstacle.any():
        near_obstacle = ndimage.distance_transform_edt(~obstacle) < ndimage.distance_transform_edt(~plate)
    else:
        near_obstacle = np.zeros(grid.shape, bool)
    return GroundTruth(inside, plate, obstacle, d_out, d_in, near_obstacle, float(inside.sum() * grid.dv))
