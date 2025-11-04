"""Unit tests for model calibration."""

import unittest
import numpy as np
import sys
sys.path.insert(0, '../src')

from models import VasicekModel
from calibration import ModelCalibrator


class TestModelCalibrator(unittest.TestCase):
    """Test model calibration."""

    def setUp(self):
        """Set up calibrator."""
        self.market_data = {'bonds': []}
        self.calibrator = ModelCalibrator(
            VasicekModel, self.market_data)

    def test_calibrator_initialization(self):
        """Test calibrator initialization."""
        self.assertEqual(self.calibrator.model_class, VasicekModel)
        self.assertEqual(self.calibrator.market_data, self.market_data)

    def test_objective_function(self):
        """Test objective function computation."""
        params = np.array([0.1, 0.05, 0.01])
        error = self.calibrator.objective_function(params)
        self.assertIsInstance(error, (float, np.floating))
        self.assertGreaterEqual(error, 0)

    def test_mle_calibration(self):
        """Test MLE calibration."""
        np.random.seed(42)
        historical_rates = np.random.normal(0.03, 0.005, 1000)
        
        result = self.calibrator.calibrate_mle(historical_rates)
        
        self.assertIn('params', result)
        self.assertIn('nll', result)
        self.assertIn('success', result)
        self.assertEqual(len(result['params']), 3)
        self.assertGreater(result['nll'], 0)

    def test_local_optimization(self):
        """Test local optimization calibration."""
        initial_params = np.array([0.1, 0.05, 0.01])
        bounds = [(0.01, 0.5), (0.01, 0.1), (0.001, 0.1)]
        
        result = self.calibrator.calibrate_local(initial_params, bounds)
        
        self.assertIn('params', result)
        self.assertIn('error', result)
        self.assertEqual(len(result['params']), 3)

    def test_parameter_sensitivity(self):
        """Test parameter sensitivity analysis."""
        params = np.array([0.1, 0.05, 0.01])
        sensitivity_result = self.calibrator.parameter_sensitivity(params)
        
        self.assertIn('sensitivities', sensitivity_result)
        sensitivities = sensitivity_result['sensitivities']
        self.assertEqual(len(sensitivities), 3)

    def test_calibration_history(self):
        """Test calibration history tracking."""
        initial_params = np.array([0.1, 0.05, 0.01])
        self.calibrator.calibrate_local(initial_params)
        
        self.assertEqual(len(self.calibrator.params_history), 1)
        np.testing.assert_array_almost_equal(
            self.calibrator.params_history[0], initial_params, decimal=1)


if __name__ == '__main__':
    unittest.main()
