# @author: yl12783, Nov. 2026, New York
#
# GINO-style SDF on a global regular grid, from the dataset's own tetrahedral mesh (plan A).
#   1. surface mesh per body = boundary faces of its tetrahedra (vertices = original nodes)
#   2. global bbox over all samples, margin on every side, R^3 grid via linspace + meshgrid(indexing="ij")
#   3. SDF per body with open3d RaycastingScene.compute_signed_distance (negative inside); union = min over bodies
#      (Open3D requires non-intersecting meshes, and the plate and actuator touch)
#   4. normalization: coordinates mapped isotropically to [-1, 1] (longest padded axis), SDF divided by the
#      train-set std of the raw SDF (keeps the sign and the zero level set); raw values are stored too

import logging
import os
import time
from typing import Dict, List, Tuple

import h5py as h5
import numpy as np
import open3d as o3d

from .surface import body_surfaces

logger = logging.getLogger(__name__)


class SDFBuilder:

    def __init__(
        self,
        base_path: str,
        out_path: str,
        dataset_name: str = "deforming_plate",
        splits: Tuple[str, ...] = ("train", "valid", "test"),
        t: int = 0,
        R: int = 64,
        margin: float = 0.05,
        nsamples: int = 5,
        max_trajectories: int = -1,
    ):
        params = locals()
        for k, v in params.items():
            if k != 'self' and not k.startswith('_'):
                setattr(self, k, v)
        self.files = {s: os.path.join(base_path, dataset_name, "data", s, f"{dataset_name}_{s}.h5") for s in splits}

    def _keys(self, f):
        keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        return keys if self.max_trajectories < 0 else keys[: self.max_trajectories]

    def global_bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for s, path in self.files.items():
            with h5.File(path, "r") as f:
                for k in self._keys(f):
                    X = f[k]["world_pos"][self.t]
                    lo, hi = np.minimum(lo, X.min(0)), np.maximum(hi, X.max(0))
        ext = hi - lo
        return lo - self.margin * ext, hi + self.margin * ext

    def make_grid(self, lo, hi) -> np.ndarray:
        axes = [np.linspace(lo[d], hi[d], self.R) for d in range(3)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), -1)            # (R, R, R, 3)

    def sample_sdf(self, X, cells, node_type, grid: np.ndarray) -> Tuple[np.ndarray, Dict]:
        surfs = body_surfaces(X, cells, node_type)
        q = o3d.core.Tensor(grid.reshape(-1, 3).astype(np.float32))
        sdf = np.full(len(grid.reshape(-1, 3)), np.inf, np.float32)
        info = {}
        for name, s in surfs.items():
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(s.vertices),
                                             o3d.utility.Vector3iVector(s.triangles))
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            sdf = np.minimum(sdf, scene.compute_signed_distance(q, nsamples=self.nsamples).numpy())
            info[name] = dict(n_vertices=len(s.vertices), n_triangles=len(s.triangles),
                              n_boundary_edges=s.n_boundary_edges, is_closed=s.is_closed,
                              o3d_watertight=bool(mesh.is_watertight()),
                              vertices_match=bool(np.array_equal(np.asarray(mesh.vertices), X[s.vertex_ids])))
        info["surface_vertex_ids"] = np.unique(np.concatenate([s.vertex_ids for s in surfs.values()]))
        info["surfaces"] = surfs
        return sdf.reshape(grid.shape[:3]), info

    def build(self):
        os.makedirs(self.out_path, exist_ok=True)
        lo, hi = self.global_bbox()
        grid = self.make_grid(lo, hi)
        center = (lo + hi) / 2
        scale = float((hi - lo).max() / 2)                                # isotropic: longest axis -> [-1, 1]
        spacing = (hi - lo) / (self.R - 1)
        logger.info(f"global bbox {lo} -> {hi}, grid {self.R}^3, spacing {1e3*spacing} mm")

        records: List[Dict] = []
        raw: Dict[Tuple[str, str], np.ndarray] = {}
        train_sum, train_sq, train_n = 0.0, 0.0, 0
        for s, path in self.files.items():
            with h5.File(path, "r") as f:
                for k in self._keys(f):
                    t0 = time.time()
                    g = f[k]
                    X = g["world_pos"][self.t].astype(np.float64)
                    cells, nt = g["cells"][:], g["node_type"][:]
                    sdf, info = self.sample_sdf(X, cells, nt, grid)
                    raw[(s, k)] = sdf
                    if s == "train":
                        train_sum += float(sdf.astype(np.float64).sum()); train_sq += float((sdf.astype(np.float64) ** 2).sum())
                        train_n += sdf.size
                    bodies = {b: v for b, v in info.items() if b in ("plate", "obstacle")}
                    records.append(dict(split=s, traj=k, mesh_method="tet_boundary",
                                        is_watertight=all(v["is_closed"] and v["o3d_watertight"] for v in bodies.values()),
                                        vertices_match=all(v["vertices_match"] for v in bodies.values()),
                                        n_points=len(X), n_surface_points=len(info["surface_vertex_ids"]),
                                        n_boundary_edges=sum(v["n_boundary_edges"] for v in bodies.values()),
                                        seconds=time.time() - t0))
                    if len(records) % 100 == 0:
                        logger.info(f"{len(records)} samples done")

        mean = train_sum / max(train_n, 1)
        std = float(np.sqrt(max(train_sq / max(train_n, 1) - mean ** 2, 0.0)))
        np.savez(os.path.join(self.out_path, "meta.npz"),
                 grid_raw=grid.astype(np.float32), grid=((grid - center) / scale).astype(np.float32),
                 bbox_lo=lo, bbox_hi=hi, center=center, scale=scale, sdf_mean=mean, sdf_std=std,
                 R=self.R, t=self.t, margin=self.margin,
                 sdf_convention="negative inside; sdf = sdf_raw / sdf_std; coords = (x - center) / scale")
        for s, path in self.files.items():
            os.makedirs(os.path.join(self.out_path, s), exist_ok=True)
            with h5.File(path, "r") as f:
                for k in self._keys(f):
                    g = f[k]
                    X = g["world_pos"][self.t].astype(np.float32)
                    rec = next(r for r in records if r["split"] == s and r["traj"] == k)
                    np.savez(os.path.join(self.out_path, s, f"{k}.npz"),
                             points=((X - center) / scale).astype(np.float32), points_raw=X,
                             node_type=g["node_type"][:],
                             sdf=(raw[(s, k)] / std)[..., None].astype(np.float32),
                             sdf_raw=raw[(s, k)][..., None].astype(np.float32),
                             mesh_method=rec["mesh_method"], is_watertight=rec["is_watertight"])
        return records, raw, dict(grid=grid, lo=lo, hi=hi, center=center, scale=scale, mean=mean, std=std)
