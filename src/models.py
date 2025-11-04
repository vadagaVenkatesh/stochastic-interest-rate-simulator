"""Stochastic Interest Rate Models with Ultra-Low Latency.

Implements Vasicek, CIR, and Hull-White models using vectorized NumPy
operations for high-performance Monte Carlo simulations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class ShortRateModel(ABC):
    """Abstract base class for short rate models."""

    def __init__(self, params: dict, name: str):
        """Initialize model with parameters.

        Args:
            params: Dictionary of model-specific parameters
            name: Model name
        """
        self.params = params
        self.name = name
        self._validate_params()

    @abstractmethod
    def _validate_params(self) -> None:
        """Validate model parameters."""
        pass

    @abstractmethod
    def _drift(self, r: np.ndarray, t: float) -> np.ndarray:
        """Compute drift component."""
        pass

    @abstractmethod
    def _diffusion(self, r: np.ndarray, t: float) -> np.ndarray:
        """Compute diffusion component."""
        pass

    def simulate_euler(self, r0: float, T: float, dt: float, paths: int,
                       seed: Optional[int] = None) -> np.ndarray:
        """Euler-Maruyama scheme simulation.

        Args:
            r0: Initial short rate
            T: Time horizon
            dt: Time step
            paths: Number of paths
            seed: Random seed

        Returns:
            Array of shape (steps, paths)
        """
        if seed is not None:
            np.random.seed(seed)

        steps = int(T / dt) + 1
        t_grid = np.linspace(0, T, steps)
        rates = np.full((steps, paths), r0, dtype=np.float64)

        dW = np.random.normal(0, np.sqrt(dt), (steps - 1, paths))

        for i in range(steps - 1):
            r_curr = rates[i]
            drift = self._drift(r_curr, t_grid[i])
            diffusion = self._diffusion(r_curr, t_grid[i])
            rates[i + 1] = r_curr + drift * dt + diffusion * dW[i]

        return rates

    def simulate_milstein(self, r0: float, T: float, dt: float, paths: int,
                          seed: Optional[int] = None) -> np.ndarray:
        """Milstein scheme simulation.

        Args:
            r0: Initial short rate
            T: Time horizon
            dt: Time step
            paths: Number of paths
            seed: Random seed

        Returns:
            Array of shape (steps, paths)
        """
        if seed is not None:
            np.random.seed(seed)

        steps = int(T / dt) + 1
        t_grid = np.linspace(0, T, steps)
        rates = np.full((steps, paths), r0, dtype=np.float64)

        dW = np.random.normal(0, np.sqrt(dt), (steps - 1, paths))
        dW_sq = dW ** 2

        for i in range(steps - 1):
            r_curr = rates[i]
            drift = self._drift(r_curr, t_grid[i])
            diffusion = self._diffusion(r_curr, t_grid[i])

            eps = 1e-8
            diffusion_perturb = self._diffusion(r_curr + eps, t_grid[i])
            diffusion_deriv = (diffusion_perturb - diffusion) / eps

            rates[i + 1] = (r_curr + drift * dt + diffusion * dW[i] +
                           0.5 * diffusion_deriv * diffusion * (dW_sq[i] - dt))

        return rates


class VasicekModel(ShortRateModel):
    """Vasicek model: dr = a(b-r)dt + sigma*dW."""

    def _validate_params(self) -> None:
        """Validate parameters."""
        required = ['a', 'b', 'sigma']
        if not all(k in self.params for k in required):
            raise ValueError(f"Vasicek requires: {required}")
        if self.params['a'] <= 0 or self.params['sigma'] <= 0:
            raise ValueError("a and sigma must be positive")

    def _drift(self, r: np.ndarray, t: float) -> np.ndarray:
        """Drift: a(b-r)."""
        a = self.params['a']
        b = self.params['b']
        return a * (b - r)

    def _diffusion(self, r: np.ndarray, t: float) -> np.ndarray:
        """Diffusion: sigma."""
        return np.full_like(r, self.params['sigma'], dtype=np.float64)


class CIRModel(ShortRateModel):
    """Cox-Ingersoll-Ross model: dr = a(b-r)dt + sigma*sqrt(r)*dW."""

    def _validate_params(self) -> None:
        """Validate parameters."""
        required = ['a', 'b', 'sigma']
        if not all(k in self.params for k in required):
            raise ValueError(f"CIR requires: {required}")
        if self.params['a'] <= 0 or self.params['b'] <= 0 or self.params['sigma'] <= 0:
            raise ValueError("All CIR parameters must be positive")

    def _drift(self, r: np.ndarray, t: float) -> np.ndarray:
        """Drift: a(b-r)."""
        a = self.params['a']
        b = self.params['b']
        return a * (b - r)

    def _diffusion(self, r: np.ndarray, t: float) -> np.ndarray:
        """Diffusion: sigma*sqrt(r)."""
        sigma = self.params['sigma']
        r_clipped = np.maximum(r, 1e-10)
        return sigma * np.sqrt(r_clipped)


class HullWhiteModel(ShortRateModel):
    """Hull-White model: dr = (theta(t) - a*r)dt + sigma*dW."""

    def _validate_params(self) -> None:
        """Validate parameters."""
        required = ['a', 'sigma']
        if not all(k in self.params for k in required):
            raise ValueError(f"Hull-White requires: {required}")
        if self.params['a'] <= 0 or self.params['sigma'] <= 0:
            raise ValueError("a and sigma must be positive")

    def _theta(self, t: float) -> float:
        """Time-dependent drift parameter."""
        if 'theta_func' in self.params:
            return self.params['theta_func'](t)
        return self.params.get('theta_0', 0.05)

    def _drift(self, r: np.ndarray, t: float) -> np.ndarray:
        """Drift: theta(t) - a*r."""
        a = self.params['a']
        theta_t = self._theta(t)
        return theta_t - a * r

    def _diffusion(self, r: np.ndarray, t: float) -> np.ndarray:
        """Diffusion: sigma."""
        return np.full_like(r, self.params['sigma'], dtype=np.float64)
