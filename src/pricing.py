"""Bond Pricing and Derivative Valuation.

Computes bond prices, yield curves, and interest rate derivative values
using Monte Carlo simulations and analytical formulas.
"""

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.interpolate import interp1d
from typing import Tuple, Optional, List


class BondPricer:
    """Bond pricing calculations."""

    @staticmethod
    def price_zero_coupon_bond(rates: np.ndarray, dt: float,
                              T: float) -> float:
        """Price zero-coupon bond using discount factor.

        Args:
            rates: Path of short rates (steps, paths)
            dt: Time step
            T: Bond maturity

        Returns:
            Bond price
        """
        # Discount factor = exp(-integral of rates)
        integral = np.cumsum(rates, axis=0) * dt
        discount_factors = np.exp(-integral)
        
        final_discount = discount_factors[-1]
        bond_price = np.mean(final_discount)
        
        return bond_price

    @staticmethod
    def price_coupon_bond(rates: np.ndarray, dt: float, T: float,
                         coupon_rate: float, frequency: int = 2) -> float:
        """Price coupon bond.

        Args:
            rates: Short rate paths
            dt: Time step
            T: Bond maturity
            coupon_rate: Annual coupon rate
            frequency: Coupon frequency (2 = semi-annual)

        Returns:
            Bond price
        """
        integral = np.cumsum(rates, axis=0) * dt
        discount_factors = np.exp(-integral)
        
        coupon_amount = coupon_rate / frequency
        coupon_times = np.arange(frequency, int(frequency * T) + 1) / frequency
        coupon_steps = (coupon_times / dt).astype(int)
        coupon_steps = coupon_steps[coupon_steps < rates.shape[0]]
        
        bond_price = 0.0
        for step in coupon_steps:
            bond_price += coupon_amount * np.mean(discount_factors[step])
        
        bond_price += np.mean(discount_factors[-1])
        
        return bond_price

    @staticmethod
    def ytm_from_price(price: float, T: float, coupon_rate: float,
                       frequency: int = 2) -> float:
        """Compute yield to maturity from bond price.

        Args:
            price: Bond price
            T: Maturity
            coupon_rate: Annual coupon rate
            frequency: Coupon frequency

        Returns:
            Yield to maturity
        """
        coupon_amount = coupon_rate / frequency
        num_coupons = int(frequency * T)
        
        def bond_price_func(y):
            y_period = y / frequency
            pv = 0.0
            for t in range(1, num_coupons + 1):
                pv += coupon_amount / ((1 + y_period) ** t)
            pv += 1.0 / ((1 + y_period) ** num_coupons)
            return pv - price
        
        ytm = brentq(bond_price_func, -0.5, 0.5)
        return ytm


class YieldCurveBootstrapper:
    """Bootstrap yield curve from bond prices."""

    @staticmethod
    def bootstrap_curve(bonds_data: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Bootstrap yield curve from bond data.

        Args:
            bonds_data: List of {maturity, coupon_rate, price}

        Returns:
            (maturities, spot_rates)
        """
        bonds_data = sorted(bonds_data, key=lambda x: x['maturity'])
        
        maturities = np.array([b['maturity'] for b in bonds_data])
        spot_rates = np.zeros_like(maturities)
        
        for i, bond in enumerate(bonds_data):
            T = bond['maturity']
            coupon = bond['coupon_rate']
            price = bond['price']
            
            def pv_func(y):
                pv = 0.0
                for j in range(1, int(2 * T) + 1):
                    t = j / 2
                    rate = spot_rates[i-1] if j == int(2*T) else y
                    pv += (coupon / 2) / ((1 + rate / 2) ** j)
                pv += 1.0 / ((1 + y / 2) ** int(2*T))
                return pv - price
            
            spot_rates[i] = brentq(pv_func, -0.5, 0.5)
        
        return maturities, spot_rates

    @staticmethod
    def interpolate_curve(maturities: np.ndarray, spot_rates: np.ndarray,
                          T: float) -> float:
        """Interpolate spot rate for arbitrary maturity.

        Args:
            maturities: Bootstrap maturities
            spot_rates: Bootstrap spot rates
            T: Target maturity

        Returns:
            Interpolated spot rate
        """
        f = interp1d(maturities, spot_rates, kind='cubic',
                     fill_value='extrapolate')
        return float(f(T))


class DerivativePricer:
    """Price interest rate derivatives."""

    @staticmethod
    def price_swaption(rates: np.ndarray, dt: float, swap_tenor: float,
                      swap_rate: float, strike: float,
                      notional: float = 1e6) -> float:
        """Price European swaption via Monte Carlo.

        Args:
            rates: Short rate paths
            dt: Time step
            swap_tenor: Swap tenor
            swap_rate: Forward swap rate
            strike: Swaption strike
            notional: Notional amount

        Returns:
            Swaption price
        """
        intrinsic = max(swap_rate - strike, 0) * notional
        
        # Time value approximation
        volatility = np.std(rates[-1] - rates[0])
        time_value = 0.25 * volatility * notional * np.sqrt(swap_tenor)
        
        return intrinsic + time_value

    @staticmethod
    def price_cap(rates: np.ndarray, dt: float, maturity: float,
                 strike: float, notional: float = 1e6) -> float:
        """Price interest rate cap.

        Args:
            rates: Short rate paths
            dt: Time step
            maturity: Cap maturity
            strike: Strike rate
            notional: Notional

        Returns:
            Cap price
        """
        steps = int(maturity / dt)
        dt_accrual = dt  # Simplified; typically quarterly
        
        intrinsic_values = np.maximum(rates[1:steps+1] - strike, 0)
        discount = np.exp(-np.cumsum(rates[:-1], axis=0)[:steps] * dt)
        
        payoffs = intrinsic_values * discount * dt_accrual * notional
        cap_price = np.mean(np.sum(payoffs, axis=0))
        
        return cap_price
