# @author: yl12783, Nov. 2026, New York
#
# Gaussian-kernel geometry field
#   phi(x) = -tau + 1/n sum_i C exp(-|x - x_i|^2 / (2 sigma^2)),   C = 1 / (sqrt(2 pi) sigma)
# predicted geometry = {x : phi(x) > 0}.

from typing import List, Optional
import numpy as np


class GaussianPhi:

    def __init__(
        self,
        points: np.ndarray,
        sigma: float,
        tau: Optional[float] = None,
        truncate: float = 4.0,
    ):
        self.points = np.asarray(points, np.float64)
        self.n = len(self.points)
        self.sigma = float(sigma)
        self.tau = None if tau is None else float(tau)
        self.truncate = truncate

    @property
    def C(self) -> float:
        return 1.0 / (np.sqrt(2 * np.pi) * self.sigma)

    @property
    def scale(self) -> float:
        """phi = -tau + scale * S, with S(x) = sum_i exp(-|x - x_i|^2 / 2 sigma^2)."""
        return self.C / self.n

    def kernel_sum_on_grid(self, axes: List[np.ndarray]) -> np.ndarray:
        """S on the tensor grid axes[0] x axes[1] x axes[2]. Exact up to dropping contributions from
        points farther than `truncate * sigma` (each <= exp(-truncate^2 / 2) of one point's peak)."""
        shape = tuple(len(c) for c in axes)
        S = np.zeros(shape, np.float32)
        r = [int(np.ceil(self.truncate * self.sigma / (c[1] - c[0]))) for c in axes]
        for x in self.points:
            idx = [np.searchsorted(axes[d], x[d]) for d in range(3)]
            sl = [slice(max(i - r[d], 0), min(i + r[d] + 1, shape[d])) for d, i in enumerate(idx)]
            e = [np.exp(-(axes[d][sl[d]] - x[d]) ** 2 / (2 * self.sigma ** 2)).astype(np.float32) for d in range(3)]
            S[tuple(sl)] += e[0][:, None, None] * e[1][None, :, None] * e[2][None, None, :]
        return S

    def phi_from_sum(self, S: np.ndarray) -> np.ndarray:
        assert self.tau is not None, "tau is not set"
        return self.scale * S - self.tau

    def __call__(self, x: np.ndarray, chunk: int = 8192) -> np.ndarray:
        """Exact phi at arbitrary query points x: (m, 3)."""
        assert self.tau is not None, "tau is not set"
        x = np.asarray(x, np.float64); out = np.empty(len(x))
        for k in range(0, len(x), chunk):
            d2 = ((x[k:k + chunk, None, :] - self.points[None]) ** 2).sum(-1)
            out[k:k + chunk] = np.exp(-d2 / (2 * self.sigma ** 2)).sum(1)
        return self.scale * out - self.tau

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: n={self.n}, sigma={self.sigma}, tau={self.tau}>"
