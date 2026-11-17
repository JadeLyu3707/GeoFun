# @author: yl12783, Nov. 2026, New York
#
# Independent checks of the SDF: exact point-in-tetrahedron sign and brute-force point-to-triangle distance.

import numpy as np


def inside_tets(q: np.ndarray, X: np.ndarray, cells: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Exact union-of-tetrahedra membership for query points q: (M, 3), via barycentric coordinates."""
    P = X[cells]                                                  # (K, 4, 3)
    Tinv = np.linalg.inv(np.transpose(P[:, 1:] - P[:, :1], (0, 2, 1)))  # (K, 3, 3)
    lo, hi = P.min(1), P.max(1)
    out = np.zeros(len(q), bool)
    for s in range(0, len(q), chunk):
        Q = q[s:s + chunk]
        box = ((Q[:, None] >= lo[None] - 1e-12) & (Q[:, None] <= hi[None] + 1e-12)).all(-1)   # (m, K)
        qi, ki = np.nonzero(box)
        if len(qi) == 0:
            continue
        lam = np.einsum("nij,nj->ni", Tinv[ki], Q[qi] - P[ki, 0])
        ok = (lam >= -1e-9).all(1) & (lam.sum(1) <= 1 + 1e-9)
        hit = np.zeros(len(Q), bool); hit[qi[ok]] = True
        out[s:s + chunk] = hit
    return out


def inside_tets_grid(axes, X: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Exact union-of-tetrahedra membership on a regular grid axes[0] x axes[1] x axes[2] (linspace axes).
    Each tetrahedron only tests the grid points inside its own bounding box."""
    shape = tuple(len(a) for a in axes)
    lo = np.array([a[0] for a in axes]); h = np.array([a[1] - a[0] for a in axes])
    P = X[cells]
    Tinv = np.linalg.inv(np.transpose(P[:, 1:] - P[:, :1], (0, 2, 1)))
    i0 = np.clip(np.ceil((P.min(1) - lo) / h - 1e-9).astype(int), 0, None)
    i1 = np.minimum(np.floor((P.max(1) - lo) / h + 1e-9).astype(int), np.array(shape) - 1)
    out = np.zeros(shape, bool)
    for k in np.flatnonzero((i1 >= i0).all(1)):
        r = [np.arange(i0[k, d], i1[k, d] + 1) for d in range(3)]
        I = np.stack(np.meshgrid(*r, indexing="ij"), -1).reshape(-1, 3)
        Q = lo + I * h
        lam = (Q - P[k, 0]) @ Tinv[k].T
        ok = (lam >= -1e-9).all(1) & (lam.sum(1) <= 1 + 1e-9)
        out[tuple(I[ok].T)] = True
    return out


def point_triangle_distance(p: np.ndarray, V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """min over triangles of the Euclidean distance from each p (M, 3) to triangle (V[F]); Ericson's
    closest-point-on-triangle, fully vectorized over triangles."""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    out = np.empty(len(p))
    for i, x in enumerate(p):
        ab, ac, ap = B - A, C - A, x - A
        d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
        bp = x - B; d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
        cp = x - C; d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        denom = va + vb + vc
        with np.errstate(divide="ignore", invalid="ignore"):
            v = vb / denom; w = vc / denom
            closest = A + ab * v[:, None] + ac * w[:, None]                       # interior
            # vertex regions
            m = (d1 <= 0) & (d2 <= 0); closest[m] = A[m]
            m = (d3 >= 0) & (d4 <= d3); closest[m] = B[m]
            m = (d6 >= 0) & (d5 <= d6); closest[m] = C[m]
            # edge regions
            m = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~((d1 <= 0) & (d2 <= 0)) & ~((d3 >= 0) & (d4 <= d3))
            t = d1 / (d1 - d3); closest[m] = A[m] + ab[m] * t[m, None]
            m = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~((d1 <= 0) & (d2 <= 0)) & ~((d6 >= 0) & (d5 <= d6))
            t = d2 / (d2 - d6); closest[m] = A[m] + ac[m] * t[m, None]
            m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~((d3 >= 0) & (d4 <= d3)) & ~((d6 >= 0) & (d5 <= d6))
            t = (d4 - d3) / ((d4 - d3) + (d5 - d6)); closest[m] = B[m] + (C[m] - B[m]) * t[m, None]
        out[i] = np.sqrt(((closest - x) ** 2).sum(1).min())
    return out
