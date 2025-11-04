"""Unit tests for stochastic interest rate models."""

import unittest
import numpy as np
import sys
sys.path.insert(0, '../src')

from models import VasicekModel, CIRModel, HullWhiteModel


class TestVasicekModel(unittest.TestCase):
    """Test Vasicek model."""

    def setUp(self):
        """Initialize model."""
        self.params = {'a': 0.1, 'b': 0.05, 'sigma': 0.01}
        self.model = VasicekModel(self.params)

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.params['a'], 0.1)
        self.assertEqual(self.model.params['b'], 0.05)
        self.assertEqual(self.model.params['sigma'], 0.01)

    def test_drift(self):
        """Test drift calculation."""
        r = np.array([0.03, 0.04, 0.05])
        drift = self.model._drift(r, 0)
        expected = 0.1 * (0.05 - r)
        np.testing.assert_array_almost_equal(drift, expected)

    def test_diffusion(self):
        """Test diffusion calculation."""
        r = np.array([0.03, 0.04, 0.05])
        diffusion = self.model._diffusion(r, 0)
        expected = np.array([0.01, 0.01, 0.01])
        np.testing.assert_array_almost_equal(diffusion, expected)

    def test_euler_simulation(self):
        """Test Euler-Maruyama simulation."""
        paths = self.model.simulate_euler(r0=0.03, T=1.0, dt=0.01,
                                         paths=100, seed=42)
        self.assertEqual(paths.shape, (101, 100))
        self.assertAlmostEqual(paths[0, 0], 0.03)

    def test_milstein_simulation(self):
        """Test Milstein simulation."""
        paths = self.model.simulate_milstein(r0=0.03, T=1.0, dt=0.01,
                                            paths=100, seed=42)
        self.assertEqual(paths.shape, (101, 100))
        self.assertAlmostEqual(paths[0, 0], 0.03)


class TestCIRModel(unittest.TestCase):
    """Test CIR model."""

    def setUp(self):
        """Initialize model."""
        self.params = {'a': 0.1, 'b': 0.05, 'sigma': 0.01}
        self.model = CIRModel(self.params)

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.params['a'], 0.1)

    def test_diffusion_positivity(self):
        """Test that diffusion is always positive."""
        r = np.array([0.0, 0.001, 0.01])
        diffusion = self.model._diffusion(r, 0)
        self.assertTrue(np.all(diffusion >= 0))

    def test_euler_simulation(self):
        """Test Euler-Maruyama simulation."""
        paths = self.model.simulate_euler(r0=0.03, T=1.0, dt=0.01,
                                         paths=100, seed=42)
        self.assertEqual(paths.shape, (101, 100))


class TestHullWhiteModel(unittest.TestCase):
    """Test Hull-White model."""

    def setUp(self):
        """Initialize model."""
        self.params = {'a': 0.1, 'sigma': 0.01, 'theta_0': 0.05}
        self.model = HullWhiteModel(self.params)

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.params['a'], 0.1)
        self.assertEqual(self.model.params['sigma'], 0.01)

    def test_theta_parameter(self):
        """Test theta parameter retrieval."""
        theta = self.model._theta(0.0)
        self.assertEqual(theta, 0.05)

    def test_euler_simulation(self):
        """Test Euler-Maruyama simulation."""
        paths = self.model.simulate_euler(r0=0.03, T=1.0, dt=0.01,
                                         paths=100, seed=42)
        self.assertEqual(paths.shape, (101, 100))


class TestModelValidation(unittest.TestCase):
    """Test model parameter validation."""

    def test_vasicek_invalid_a(self):
        """Test Vasicek with invalid a parameter."""
        params = {'a': -0.1, 'b': 0.05, 'sigma': 0.01}
        with self.assertRaises(ValueError):
            VasicekModel(params)

    def test_cir_invalid_sigma(self):
        """Test CIR with invalid sigma parameter."""
        params = {'a': 0.1, 'b': 0.05, 'sigma': -0.01}
        with self.assertRaises(ValueError):
            CIRModel(params)

    def test_hull_white_missing_param(self):
        """Test Hull-White with missing parameters."""
        params = {'a': 0.1}  # Missing sigma
        with self.assertRaises(ValueError):
            HullWhiteModel(params)


if __name__ == '__main__':
    unittest.main()
