"""Hawkes process self-exciting memory kernel definitions."""

from abc import ABC, abstractmethod

import numpy as np


class Kernel(ABC):
    """Abstract Base Class for Hawkes process memory kernels phi(dt)."""

    @abstractmethod
    def eval(self, delta_t: float | np.ndarray) -> float | np.ndarray:
        """Compute the kernel decay value phi(dt) for dt >= 0."""

    @abstractmethod
    def integrate(self, delta_t: float | np.ndarray) -> float | np.ndarray:
        """Compute integrated kernel value integral_0^{dt} phi(s) ds for dt >= 0."""


class ExponentialKernel(Kernel):
    """Exponential decay kernel: phi(dt) = alpha * exp(-beta * dt).

    Parameters:
        alpha (float): Excitation magnitude (alpha >= 0).
        beta (float): Exponential decay rate (beta > 0).
    """

    def __init__(self, alpha: float, beta: float):
        if alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {alpha}")
        if beta <= 0:
            raise ValueError(f"Beta must be strictly positive, got {beta}")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def eval(self, delta_t: float | np.ndarray) -> float | np.ndarray:
        dt = np.maximum(0.0, np.asarray(delta_t, dtype=np.float64))
        val = self.alpha * np.exp(-self.beta * dt)
        return float(val) if np.ndim(delta_t) == 0 else val

    def integrate(self, delta_t: float | np.ndarray) -> float | np.ndarray:
        dt = np.maximum(0.0, np.asarray(delta_t, dtype=np.float64))
        val = (self.alpha / self.beta) * (1.0 - np.exp(-self.beta * dt))
        return float(val) if np.ndim(delta_t) == 0 else val


class PowerLawKernel(Kernel):
    """Power-law decay kernel: phi(dt) = alpha * (delta + dt)^(-gamma).

    Parameters:
        alpha (float): Excitation magnitude (alpha >= 0).
        gamma (float): Power-law decay exponent (gamma > 1 for finite total mass).
        delta (float): Softening offset to avoid singularity at dt=0 (delta > 0, default=1.0).
    """

    def __init__(self, alpha: float, gamma: float, delta: float = 1.0):
        if alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {alpha}")
        if gamma <= 1.0:
            raise ValueError(f"Gamma must be strictly greater than 1.0, got {gamma}")
        if delta <= 0:
            raise ValueError(f"Delta must be strictly positive, got {delta}")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.delta = float(delta)

    def eval(self, delta_t: float | np.ndarray) -> float | np.ndarray:
        dt = np.maximum(0.0, np.asarray(delta_t, dtype=np.float64))
        val = self.alpha * np.power(self.delta + dt, -self.gamma)
        return float(val) if np.ndim(delta_t) == 0 else val

    def integrate(self, delta_t: float | np.ndarray) -> float | np.ndarray:
        dt = np.maximum(0.0, np.asarray(delta_t, dtype=np.float64))
        val = (self.alpha / (self.gamma - 1.0)) * (
            np.power(self.delta, 1.0 - self.gamma)
            - np.power(self.delta + dt, 1.0 - self.gamma)
        )
        return float(val) if np.ndim(delta_t) == 0 else val
