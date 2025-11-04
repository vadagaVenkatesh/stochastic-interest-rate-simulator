"""End-to-End Example: Simulate, Calibrate, and Price.

Demonstration of complete workflow using stochastic interest rate models.
"""

import numpy as np
import sys
sys.path.insert(0, '../src')

from models import VasicekModel, CIRModel, HullWhiteModel
from simulator import MonteCarloSimulator
from pricing import BondPricer, YieldCurveBootstrapper
from calibration import ModelCalibrator
from visualization import RatePathVisualizer, SensitivityAnalyzer
import matplotlib.pyplot as plt


def main():
    """Run complete example."""
    print("=" * 60)
    print("Stochastic Interest Rate Simulator - Example")
    print("=" * 60)

    # 1. Initialize Vasicek Model
    print("\n1. Initializing Vasicek Model...")
    vasicek_params = {
        'a': 0.1,      # Mean reversion speed
        'b': 0.05,     # Long-term mean
        'sigma': 0.01  # Volatility
    }
    vasicek = VasicekModel(vasicek_params, name='Vasicek')
    print(f"   Parameters: {vasicek_params}")

    # 2. Simulate Paths
    print("\n2. Running Monte Carlo Simulation...")
    simulator = MonteCarloSimulator(vasicek, num_processes=4)
    
    r0 = 0.03
    T = 10.0
    dt = 0.01
    paths = 5000
    
    rates = simulator.simulate(r0=r0, T=T, dt=dt, paths=paths,
                              scheme='euler', seed=42)
    print(f"   Simulated {paths} paths over {T} years")
    print(f"   Mean final rate: {np.mean(rates[-1]):.4f}")
    print(f"   Std final rate: {np.std(rates[-1]):.4f}")

    # 3. Compute Statistics
    print("\n3. Computing Path Statistics...")
    stats = simulator.compute_statistics(rates, dt)
    print(f"   Mean rate (T=1y): {stats['mean'][100]:.4f}")
    print(f"   5th percentile: {stats['q05'][100]:.4f}")
    print(f"   95th percentile: {stats['q95'][100]:.4f}")

    # 4. Price Bonds
    print("\n4. Pricing Zero-Coupon Bonds...")
    bond_maturity = 5.0
    zcb_price = BondPricer.price_zero_coupon_bond(rates, dt, bond_maturity)
    print(f"   5-year ZCB price: {zcb_price:.4f}")
    
    coupon_bond_price = BondPricer.price_coupon_bond(
        rates, dt, bond_maturity, coupon_rate=0.04, frequency=2
    )
    print(f"   5-year coupon bond (4% coupon): {coupon_bond_price:.4f}")

    # 5. Visualize Paths
    print("\n5. Generating Visualizations...")
    fig1, ax1 = RatePathVisualizer.plot_paths(rates, dt, num_sample=100)
    plt.savefig('paths.png', dpi=100, bbox_inches='tight')
    print("   Saved: paths.png")
    
    fig2, ax2 = RatePathVisualizer.plot_term_structure(rates, dt)
    plt.savefig('term_structure.png', dpi=100, bbox_inches='tight')
    print("   Saved: term_structure.png")
    
    fig3, ax3 = RatePathVisualizer.plot_distribution(rates, [100, 500, 1000])
    plt.savefig('distributions.png', dpi=100, bbox_inches='tight')
    print("   Saved: distributions.png")

    # 6. Calibration Example
    print("\n6. Calibrating Model to Historical Data...")
    historical_rates = np.random.normal(0.03, 0.005, 1000)
    calibrator = ModelCalibrator(VasicekModel,
                                market_data={'bonds': []})
    calib_result = calibrator.calibrate_mle(historical_rates)
    print(f"   Calibrated params: {calib_result['params']}")
    print(f"   NLL: {calib_result['nll']:.4f}")

    # 7. CIR Model Comparison
    print("\n7. Running CIR Model...")
    cir_params = {
        'a': 0.15,
        'b': 0.05,
        'sigma': 0.015
    }
    cir = CIRModel(cir_params, name='CIR')
    simulator_cir = MonteCarloSimulator(cir)
    rates_cir = simulator_cir.simulate(r0=r0, T=T, dt=dt, paths=1000,
                                       scheme='milstein', seed=42)
    print(f"   CIR mean final rate: {np.mean(rates_cir[-1]):.4f}")

    # 8. Hull-White Model
    print("\n8. Running Hull-White Model...")
    hw_params = {
        'a': 0.1,
        'sigma': 0.015,
        'theta_0': 0.04
    }
    hw = HullWhiteModel(hw_params, name='Hull-White')
    simulator_hw = MonteCarloSimulator(hw)
    rates_hw = simulator_hw.simulate(r0=r0, T=T, dt=dt, paths=1000,
                                     scheme='euler', seed=42)
    print(f"   Hull-White mean final rate: {np.mean(rates_hw[-1]):.4f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
