# @author: yl12783, Nov. 2026, New York
#
# Surface triangle mesh of a tetrahedral mesh: the faces that belong to exactly one tetrahedron.
# Vertices are the dataset's own nodes (no new points); faces are oriented outward.

from dataclasses import dataclass
from typing import Dict

import numpy as np

TET_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])   # face k is opposite vertex k


@dataclass
class SurfaceMesh:
    vertices: np.ndarray       # (V, 3) coordinates, = points[vertex_ids]
    triangles: np.ndarray      # (F, 3) indices into vertices, outward oriented
    vertex_ids: np.ndarray     # (V,) indices into the snapshot's point array
    n_boundary_edges: int      # edges used by != 2 triangles (0 for a closed surface)
    is_closed: bool            # every edge shared by exactly two triangles


def tet_boundary_surface(X: np.ndarray, cells: np.ndarray) -> SurfaceMesh:
    """X: (N, 3) node coordinates; cells: (K, 4) tetrahedra (node indices)."""
    faces = cells[:, TET_FACES].reshape(-1, 3)                       # (4K, 3)
    opposite = cells[np.repeat(np.arange(len(cells)), 4), np.tile(np.arange(4), len(cells))]
    key = np.sort(faces, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    keep = counts[inv.ravel()] == 1
    tri, opp = faces[keep], opposite[keep]

    # orient outward: normal must point away from the tetrahedron's 4th vertex
    a, b, c = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), X[opp] - a) > 0
    tri[flip] = tri[flip][:, [0, 2, 1]]

    vertex_ids, local = np.unique(tri, return_inverse=True)
    tri_local = local.reshape(-1, 3)

    edges = np.sort(np.concatenate([tri_local[:, [0, 1]], tri_local[:, [1, 2]], tri_local[:, [2, 0]]]), axis=1)
    _, ecount = np.unique(edges, axis=0, return_counts=True)
    n_bad = int((ecount != 2).sum())
    return SurfaceMesh(X[vertex_ids], tri_local, vertex_ids, n_bad, n_bad == 0)


def body_surfaces(X: np.ndarray, cells: np.ndarray, node_type: np.ndarray, obstacle_type: int = 1
                  ) -> Dict[str, SurfaceMesh]:
    """One closed surface per body. The plate and the actuator share no tetrahedra."""
    is_obst = (node_type[cells] == obstacle_type).all(1)
    out = {"plate": tet_boundary_surface(X, cells[~is_obst])}
    if is_obst.any():
        out["obstacle"] = tet_boundary_surface(X, cells[is_obst])
    return out
