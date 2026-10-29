"""Exact discretization of small LTI systems for piecewise-constant inputs.

For dx/dt = A x + B u with u constant over an interval h:

    x(t + h) = Phi(h) x(t) + Gamma(h) u,
    Phi = exp(A h),  Gamma = int_0^h exp(A s) ds B.

Both are computed from an eigen-decomposition of A (computed once), and
cached per interval length. Interval lengths are quantized to an integer
number of time quanta so the cache is effective for switching simulations.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


class DiscreteLTI:
    def __init__(self, a: np.ndarray, b: np.ndarray, quantum: float = 1e-9):
        self.a = np.asarray(a, float)
        self.b = np.asarray(b, float)
        self.n = self.a.shape[0]
        self.quantum = quantum
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        lam, v = np.linalg.eig(self.a)
        self._lam = lam
        self._v = v
        self._vinv = np.linalg.inv(v)
        self._vinv_b = self._vinv @ self.b
        cond = np.linalg.cond(v)
        self._use_eig = bool(np.isfinite(cond) and cond < 1e8 and np.all(np.abs(lam) > 0))

    def mats(self, steps: int) -> tuple[np.ndarray, np.ndarray]:
        """(Phi, Gamma) for an interval of ``steps`` quanta."""
        hit = self._cache.get(steps)
        if hit is not None:
            return hit
        h = steps * self.quantum
        if self._use_eig:
            e = np.exp(self._lam * h)
            phi = (self._v * e) @ self._vinv
            g = (e - 1.0) / self._lam
            gam = (self._v * g) @ self._vinv_b
            phi, gam = phi.real, gam.real
        else:
            m = np.zeros((self.n + self.b.shape[1],) * 2)
            m[: self.n, : self.n] = self.a * h
            m[: self.n, self.n:] = self.b * h
            em = expm(m)
            phi, gam = em[: self.n, : self.n], em[: self.n, self.n:]
        if len(self._cache) < 200000:
            self._cache[steps] = (phi, gam)
        return phi, gam
