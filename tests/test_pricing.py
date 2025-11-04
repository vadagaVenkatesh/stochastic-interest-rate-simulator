"""Unit tests for bond pricing and derivative valuation."""

import unittest
import numpy as np
import sys
sys.path.insert(0, '../src')

from pricing import BondPricer, YieldCurveBootstrapper, DerivativePricer


class TestBondPricer(unittest.TestCase):
    """Test bond pricing functions."""

    def setUp(self):
        """Set up test paths."""
        np.random.seed(42)
        self.rates = np.random.normal(0.03, 0.01, (101, 100))
        self.rates[0] = 0.03
        self.dt = 0.01

    def test_zero_coupon_bond_price(self):
        """Test zero-coupon bond pricing."""
        price = BondPricer.price_zero_coupon_bond(self.rates, self.dt, 1.0)
        self.assertGreater(price, 0)
        self.assertLess(price, 1.0)

    def test_zero_coupon_bond_bounds(self):
        """Test that ZCB price is between 0 and 1."""
        for T in [0.5, 1.0, 5.0, 10.0]:
            price = BondPricer.price_zero_coupon_bond(self.rates, self.dt, T)
            self.assertGreaterEqual(price, 0)
            self.assertLessEqual(price, 1.0)

    def test_coupon_bond_price(self):
        """Test coupon bond pricing."""
        price = BondPricer.price_coupon_bond(self.rates, self.dt, 5.0,
                                            coupon_rate=0.04, frequency=2)
        self.assertGreater(price, 0)
        self.assertLess(price, 2.0)

    def test_ytm_calculation(self):
        """Test yield to maturity calculation."""
        bond_price = 0.95
        ytm = BondPricer.ytm_from_price(bond_price, T=5.0,
                                       coupon_rate=0.05, frequency=2)
        self.assertGreater(ytm, 0)
        self.assertLess(ytm, 0.1)


class TestYieldCurveBootstrapper(unittest.TestCase):
    """Test yield curve bootstrapping."""

    def setUp(self):
        """Set up test data."""
        self.bonds_data = [
            {'maturity': 1.0, 'coupon_rate': 0.02, 'price': 0.98},
            {'maturity': 2.0, 'coupon_rate': 0.03, 'price': 0.96},
            {'maturity': 5.0, 'coupon_rate': 0.04, 'price': 0.95},
        ]

    def test_bootstrap_curve(self):
        """Test yield curve bootstrap."""
        maturities, spot_rates = YieldCurveBootstrapper.bootstrap_curve(
            self.bonds_data)
        self.assertEqual(len(maturities), 3)
        self.assertEqual(len(spot_rates), 3)
        self.assertTrue(np.all(spot_rates > 0))

    def test_interpolation(self):
        """Test curve interpolation."""
        maturities = np.array([1.0, 5.0, 10.0])
        spot_rates = np.array([0.02, 0.03, 0.035])
        
        interp_rate = YieldCurveBootstrapper.interpolate_curve(
            maturities, spot_rates, T=2.5)
        
        self.assertGreater(interp_rate, 0.02)
        self.assertLess(interp_rate, 0.03)


class TestDerivativePricer(unittest.TestCase):
    """Test derivative pricing."""

    def setUp(self):
        """Set up test paths."""
        np.random.seed(42)
        self.rates = np.random.normal(0.03, 0.01, (101, 100))
        self.dt = 0.01

    def test_swaption_price(self):
        """Test swaption pricing."""
        price = DerivativePricer.price_swaption(
            self.rates, self.dt, swap_tenor=5.0,
            swap_rate=0.03, strike=0.03)
        self.assertGreater(price, 0)

    def test_cap_price(self):
        """Test cap pricing."""
        price = DerivativePricer.price_cap(
            self.rates, self.dt, maturity=5.0,
            strike=0.04)
        self.assertGreaterEqual(price, 0)


if __name__ == '__main__':
    unittest.main()
