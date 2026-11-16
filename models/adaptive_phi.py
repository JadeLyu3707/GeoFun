# @author: yl12783, Nov. 2026, New York
#
# Density-weighted, adaptive-width Gaussian geometry field
#
#   phi(x) = -tau + sum_i (1/n) (1/rho_i) K_{sigma_i}(x, x_i)
#
#   r_k(x_i)  distance from x_i to its k-th nearest neighbour (excluding x_i itself)
#   rho_i     = (k / n) / V_d(r_k(x_i))                       local probability density
#   w_i       = (1/n) (1/rho_i)   (= V_d(r_k) / k)            quadrature weight; n appears nowhere else
#   sigma_i   = c * mean_{j in {i} u N_k(i)} r_k(x_j)         smoothed kNN radius, clamped to [lo, hi] * median
#   K         = (2 pi sigma_i^2)^(-d/2) exp(-|x - x_i|^2 / (2 sigma_i^2))
#
# phi > 0 inside, phi = 0 boundary, phi < 0 outside.

from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree

from .kernels import separable_gaussian_sum

BALL_VOLUME = {
    1: lambda r: 2.0 * r,
    2: lambda r: np.pi * r ** 2,
    3: lambda r: 4.0 / 3.0 * np.pi * r ** 3,
}


class AdaptiveGaussianPhi:

    def __init__(
        self,
        points: np.ndarray,
        k: int = 16,
        c: float = 1.0,
        d: int = 3,
        tau: Optional[float] = None,
        clamp: Tuple[float, float] = (0.25, 4.0),
    ):
        self.points = np.asarray(points, np.float64)
        self.n = n = len(self.points)
        self.k, self.c, self.d, self.tau, self.clamp = k, c, d, tau, tuple(clamp)

        dist, idx = cKDTree(self.points).query(self.points, k=k + 1)   # column 0 is the point itself
        self.r_k = dist[:, k]
        with np.errstate(divide="ignore"):
            self.rho = (k / n) / BALL_VOLUME[d](self.r_k)
            self.w = (1.0 / n) * (1.0 / self.rho)
        self.r_k_smooth = self.r_k[idx].mean(1)                        # over x_i and its k neighbours
        sigma = c * self.r_k_smooth
        med = np.median(sigma)
        self.sigma = np.clip(sigma, clamp[0] * med, clamp[1] * med)
        self.coef = self.w * (2 * np.pi * self.sigma ** 2) ** (-d / 2)  # w_i * normalization

    def field_on_grid(self, axes: List[np.ndarray]) -> np.ndarray:
        """phi + tau = sum_i w_i K_{sigma_i}(g, x_i) on the grid, exactly."""
        return separable_gaussian_sum(self.points, self.coef, self.sigma, axes)

    def __call__(self, x: np.ndarray, chunk: int = 8192) -> np.ndarray:
        assert self.tau is not None, "tau is not set"
        x = np.asarray(x, np.float64); out = np.empty(len(x))
        for s in range(0, len(x), chunk):
            d2 = ((x[s:s + chunk, None, :] - self.points[None]) ** 2).sum(-1)
            out[s:s + chunk] = (self.coef[None] * np.exp(-d2 / (2 * self.sigma[None] ** 2))).sum(1)
        return out - self.tau

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__}: n={self.n}, k={self.k}, c={self.c}, d={self.d}, "
                f"sigma median={np.median(self.sigma):.4f}, tau={self.tau}>")
