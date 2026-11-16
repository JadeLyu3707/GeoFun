# @author: yl12783, Nov. 2026, New York

from typing import List
import numpy as np


def separable_gaussian_sum(points: np.ndarray, coef: np.ndarray, sigmas: np.ndarray,
                           axes: List[np.ndarray], dtype=np.float32) -> np.ndarray:
    """F(g) = sum_i coef_i * exp(-|g - x_i|^2 / (2 sigma_i^2)) on the grid axes[0] x axes[1] x axes[2], exactly
    (no truncation). Uses exp(-|g - x|^2 / 2s^2) = prod_d exp(-(g_d - x_d)^2 / 2s^2), so the whole grid is one
    matrix product (nx, n) @ (n, ny*nz)."""
    X = np.asarray(points, np.float64)
    n = len(X)
    s2 = 2.0 * np.broadcast_to(np.asarray(sigmas, np.float64), (n,)) ** 2
    E = [np.exp(-(axes[d][:, None] - X[None, :, d]) ** 2 / s2[None]).astype(dtype) for d in range(3)]
    Ex = E[0] * np.broadcast_to(np.asarray(coef, np.float64), (n,)).astype(dtype)[None]   # (nx, n)
    Eyz = (E[1][:, None, :] * E[2][None, :, :]).reshape(-1, n)                            # (ny*nz, n)
    F = Ex @ Eyz.T
    return F.reshape(len(axes[0]), len(axes[1]), len(axes[2]))
