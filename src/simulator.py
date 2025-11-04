"""Unified Monte Carlo Simulator for Interest Rate Models.

Provides high-performance path simulation interface for any short-rate model.
"""

import numpy as np
from typing import Optional, Tuple, List
from multiprocessing import Pool, cpu_count
from functools import partial


class MonteCarlo Simulator:
    """Monte Carlo path simulator for short-rate models."""

    def __init__(self, model, num_processes: Optional[int] = None):
        """Initialize simulator.

        Args:
            model: Short-rate model instance
            num_processes: Number of processes for parallelization
        """
        self.model = model
        self.num_processes = num_processes or cpu_count()

    def simulate(self, r0: float, T: float, dt: float, paths: int,
                 scheme: str = 'euler', seed: Optional[int] = None,
                 use_parallel: bool = False) -> np.ndarray:
        """Run Monte Carlo simulation.

        Args:
            r0: Initial short rate
            T: Time horizon
            dt: Time step
            paths: Number of paths
            scheme: 'euler' or 'milstein'
            seed: Random seed
            use_parallel: Enable parallelization

        Returns:
            Simulated paths of shape (steps, paths)
        """
        if scheme == 'euler':
            sim_func = self.model.simulate_euler
        elif scheme == 'milstein':
            sim_func = self.model.simulate_milstein
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        if seed is not None:
            np.random.seed(seed)

        if not use_parallel or paths < 1000:
            return sim_func(r0, T, dt, paths, seed)

        # Parallel simulation
        paths_per_process = paths // self.num_processes
        remaining = paths % self.num_processes

        def run_simulation(seed_val):
            np.random.seed(seed_val)
            return sim_func(r0, T, dt, paths_per_process, seed_val)

        seeds = [seed + i if seed else i for i in range(self.num_processes)]

        with Pool(self.num_processes) as pool:
            results = pool.map(run_simulation, seeds)

        paths_array = np.hstack([r[:, :paths_per_process] for r in results])

        if remaining > 0:
            np.random.seed(seed + self.num_processes if seed else self.num_processes)
            extra = sim_func(r0, T, dt, remaining, None)
            paths_array = np.hstack([paths_array, extra])

        return paths_array

    def compute_statistics(self, paths: np.ndarray,
                          dt: float) -> dict:
        """Compute path statistics.

        Args:
            paths: Simulated paths
            dt: Time step

        Returns:
            Dictionary with statistics
        """
        times = np.arange(paths.shape[0]) * dt

        stats = {
            'times': times,
            'mean': np.mean(paths, axis=1),
            'std': np.std(paths, axis=1),
            'min': np.min(paths, axis=1),
            'max': np.max(paths, axis=1),
            'q05': np.percentile(paths, 5, axis=1),
            'q50': np.percentile(paths, 50, axis=1),
            'q95': np.percentile(paths, 95, axis=1),
        }
        return stats

    def compute_acf(self, paths: np.ndarray, lags: int = 20) -> np.ndarray:
        """Compute autocorrelation function.

        Args:
            paths: Simulated paths
            lags: Number of lags

        Returns:
            ACF values
        """
        final_path = paths[:, 0]
        acf_vals = np.zeros(lags)

        for lag in range(lags):
            if lag == 0:
                acf_vals[lag] = 1.0
            else:
                acf_vals[lag] = np.corrcoef(final_path[:-lag],
                                            final_path[lag:])[0, 1]
        return acf_vals
