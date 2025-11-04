"""Model Calibration to Market Data.

Calibrates stochastic interest rate models to market data using
optimization routines and maximum likelihood estimation.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Callable, Tuple, Optional


class ModelCalibrator:
    """Calibrate models to market yields and bond prices."""

    def __init__(self, model_class, market_data: Dict):
        """Initialize calibrator.

        Args:
            model_class: Short-rate model class
            market_data: Market bond prices or yields
        """
        self.model_class = model_class
        self.market_data = market_data
        self.params_history = []

    def objective_function(self, params: np.ndarray) -> float:
        """Compute objective (error) for given parameters.

        Args:
            params: Model parameters

        Returns:
            MSE between simulated and market prices
        """
        try:
            model = self.model_class(dict(zip(['a', 'b', 'sigma'], params)))
        except:
            return 1e10

        error = 0.0
        for bond_data in self.market_data.get('bonds', []):
            # Placeholder: compute simulated price
            simulated_price = 0.95  # Replace with actual simulation
            market_price = bond_data['price']
            error += (simulated_price - market_price) ** 2

        return error

    def calibrate_mle(self, historical_rates: np.ndarray,
                     initial_params: Optional[np.ndarray] = None) -> Dict:
        """Calibrate using MLE on historical rate changes.

        Args:
            historical_rates: Historical short rate data
            initial_params: Initial guess for parameters

        Returns:
            Dictionary with calibrated parameters
        """
        rate_changes = np.diff(historical_rates)
        dt = 1 / 252  # Daily data

        def neg_log_likelihood(params):
            a, b, sigma = params
            if a <= 0 or sigma <= 0:
                return 1e10

            expected_change = a * (b - historical_rates[:-1]) * dt
            residuals = rate_changes - expected_change
            variance = sigma ** 2 * dt

            nll = np.sum(residuals ** 2 / variance) + len(residuals) * np.log(variance)
            return nll

        if initial_params is None:
            initial_params = np.array([0.1, np.mean(historical_rates), 0.01])

        result = minimize(neg_log_likelihood, initial_params,
                         method='Nelder-Mead',
                         options={'maxiter': 1000})

        self.params_history.append(result.x)

        return {
            'params': result.x,
            'nll': result.fun,
            'success': result.success
        }

    def calibrate_global(self, bounds: list) -> Dict:
        """Global optimization using differential evolution.

        Args:
            bounds: Parameter bounds [(a_min, a_max), ...]

        Returns:
            Calibration results
        """
        result = differential_evolution(self.objective_function, bounds,
                                       maxiter=300, atol=1e-6, tol=1e-6,
                                       workers=1, updating='deferred')

        self.params_history.append(result.x)

        return {
            'params': result.x,
            'error': result.fun,
            'success': result.success
        }

    def calibrate_local(self, initial_params: np.ndarray,
                       bounds: Optional[list] = None) -> Dict:
        """Local optimization.

        Args:
            initial_params: Starting parameters
            bounds: Parameter bounds

        Returns:
            Calibration results
        """
        result = minimize(self.objective_function, initial_params,
                         method='L-BFGS-B', bounds=bounds,
                         options={'ftol': 1e-8})

        self.params_history.append(result.x)

        return {
            'params': result.x,
            'error': result.fun,
            'success': result.success
        }

    def parameter_sensitivity(self, params: np.ndarray,
                             perturbation: float = 0.01) -> Dict:
        """Compute parameter sensitivities.

        Args:
            params: Current parameters
            perturbation: Perturbation size

        Returns:
            Sensitivity matrix
        """
        base_error = self.objective_function(params)
        sensitivities = np.zeros(len(params))

        for i in range(len(params)):
            params_perturb = params.copy()
            params_perturb[i] += perturbation * params[i]
            perturb_error = self.objective_function(params_perturb)
            sensitivities[i] = (perturb_error - base_error) / (perturbation * params[i])

        return {'sensitivities': sensitivities}
