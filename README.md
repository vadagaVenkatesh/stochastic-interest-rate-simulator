# Stochastic Interest Rate Simulator

A comprehensive Python simulation framework for modeling short-rate interest rate models with advanced numerical schemes and financial applications.

## Overview

This repository provides a production-ready framework for simulating and analyzing short-rate models used in quantitative finance. It implements industry-standard discretization schemes and applies them to critical fixed-income applications including bond pricing, yield curve construction, and interest rate derivative valuation.

**Key Capabilities:**
- Multiple short-rate models (Vasicek, CIR, Hull-White)
- Advanced numerical discretization (Euler-Maruyama, Milstein)
- Bond pricing and yield curve analysis
- Interest rate derivative valuation
- Market data calibration
- Comprehensive visualization and sensitivity analysis

## Features

### 1. Short-Rate Models

The framework implements three fundamental short-rate models widely used in the financial industry:

- **Vasicek Model**: A mean-reverting Gaussian model for interest rate dynamics
  - Parameters: mean reversion rate (κ), long-term mean (θ), volatility (σ)
  - Analytical solutions available for bond prices

- **CIR Model**: Cox-Ingersoll-Ross model with non-negative rate guarantee
  - Parameters: mean reversion rate (κ), long-term mean (θ), volatility (σ)
  - Particularly suited for empirical fit and derivative pricing

- **Hull-White Model**: Extended Vasicek model with time-dependent parameters
  - Parameters: mean reversion rate (a), volatility (σ), time-dependent drift
  - Calibrates exactly to the initial term structure of interest rates

### 2. Discretization Schemes

Two industry-standard numerical schemes for solving stochastic differential equations:

- **Euler-Maruyama Scheme**: First-order weak convergence
  - Simple and computationally efficient
  - Suitable for Monte Carlo simulations with large sample sizes
  - Implementation: `r_{n+1} = r_n + μ(r_n)Δt + σ(r_n)√Δt * Z`

- **Milstein Scheme**: Higher-order accuracy with second-order weak convergence
  - Reduces discretization bias compared to Euler-Maruyama
  - Recommended for derivative pricing and accurate yield curves
  - Implementation includes volatility derivative term for improved accuracy

### 3. Bond Pricing and Valuation

Core fixed-income applications:

- **Zero-Coupon Bond Pricing**: Direct computation using model-specific analytical or numerical methods
- **Coupon Bond Pricing**: Aggregation of zero-coupon bond prices
- **Yield Curve Construction**: Bootstrapping and fitting techniques from market bond prices
- **Term Structure Evolution**: Tracking how the yield curve evolves over time under different scenarios

### 4. Interest Rate Derivative Valuation

Pricing tools for common interest rate derivatives:

- **Interest Rate Swaps**: Fixed vs. floating leg valuation
- **Swaptions**: Options on interest rate swaps
- **Caps and Floors**: Protection strategies for floating-rate borrowers/lenders
- **Bond Options**: Call and put options on bonds

### 5. Market Data Calibration

Calibrate model parameters to observed market data:

- **Maximum Likelihood Estimation (MLE)**: Parameter fitting using historical rate series
- **Bootstrap Calibration**: Fitting to initial term structure of bond prices
- **Optimization Routines**: Scipy-based optimization for least-squares calibration
- **Validation Metrics**: Root mean square error and other goodness-of-fit measures

### 6. Visualization and Sensitivity Analysis

Comprehensive tools for understanding model behavior:

- **Term Structure Evolution**: 3D surface plots showing yield curve dynamics over time
- **Rate Path Simulations**: Individual and ensemble visualizations of interest rate paths
- **Sensitivity Analysis**: Greeks computation (delta, gamma, vega, rho)
- **Parameter Impact Studies**: Tornado diagrams and surface plots showing model parameter effects
- **Distribution Analysis**: Histogram and Q-Q plots of simulated rate distributions

## Project Structure

```
stochastic-interest-rate-simulator/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vasicek.py          # Vasicek model implementation
│   │   ├── cir.py              # CIR model implementation
│   │   └── hull_white.py       # Hull-White model implementation
│   ├── simulators/
│   │   ├── __init__.py
│   │   ├── euler_maruyama.py   # Euler-Maruyama discretization scheme
│   │   └── milstein.py         # Milstein discretization scheme
│   ├── pricing/
│   │   ├── __init__.py
│   │   ├── bonds.py            # Bond pricing functions
│   │   ├── derivatives.py      # Interest rate derivative valuation
│   │   └── yield_curve.py      # Yield curve construction
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── mle.py              # Maximum likelihood estimation
│   │   ├── bootstrap.py        # Bootstrap calibration methods
│   │   └── optimizer.py        # Optimization utilities
│   └── visualization/
│       ├── __init__.py
│       ├── plots.py            # Core plotting functions
│       ├── term_structure.py   # Term structure visualizations
│       └── sensitivity.py      # Sensitivity analysis plots
├── examples/
│   ├── basic_simulation.py     # Simple example: simulate interest rates
│   ├── bond_pricing_demo.py    # Bond pricing application
│   ├── yield_curve_demo.py     # Yield curve construction
│   ├── calibration_example.py  # Parameter calibration example
│   └── sensitivity_analysis.py # Sensitivity analysis example
├── tests/
│   ├── __init__.py
│   ├── test_models.py          # Unit tests for models
│   ├── test_simulators.py      # Unit tests for discretization schemes
│   ├── test_pricing.py         # Unit tests for pricing functions
│   └── test_calibration.py     # Unit tests for calibration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                  # Python-specific gitignore
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/vadagaVenkatesh/stochastic-interest-rate-simulator.git
cd stochastic-interest-rate-simulator
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Interest Rate Simulation

```python
from src.models.vasicek import VasicekModel
from src.simulators.euler_maruyama import EulerMaruyamaSimulator

# Initialize Vasicek model
model = VasicekModel(kappa=0.15, theta=0.05, sigma=0.015)

# Create simulator
simulator = EulerMaruyamaSimulator(model, dt=0.01)

# Simulate paths
initial_rate = 0.04
n_paths = 1000
n_steps = 252  # One year, daily
rates = simulator.simulate(initial_rate, n_paths, n_steps)
```

### Bond Pricing

```python
from src.pricing.bonds import price_zero_coupon_bond

# Price a zero-coupon bond
maturity = 5  # 5 years
current_rate = 0.04
bond_price = price_zero_coupon_bond(model, current_rate, maturity)
```

### Yield Curve Construction

```python
from src.pricing.yield_curve import construct_yield_curve

# Construct yield curve from market bond prices
market_prices = {...}  # Market data
yields = construct_yield_curve(market_prices, model)
```

### Sensitivity Analysis

```python
from src.visualization.sensitivity import plot_rate_sensitivity

# Analyze sensitivity to parameter changes
plot_rate_sensitivity(model, 'kappa', values=[0.1, 0.15, 0.2])
```

## Dependencies

Core dependencies listed in `requirements.txt`:

- **numpy**: Numerical computations and array operations
- **scipy**: Optimization and statistical functions
- **pandas**: Data manipulation and analysis
- **matplotlib**: Static 2D visualizations
- **seaborn**: Statistical visualization and enhanced matplotlib styling
- **pytest**: Unit testing framework

## Agent-Driven Automation

This repository was scaffolded and initialized using GitHub Actions and AI agent automation (Comet Assistant), which:

1. **Repository Creation**: Automated setup of GitHub repository with proper naming and description
2. **Documentation**: Generated this comprehensive README with feature descriptions and usage examples
3. **Configuration**: Set up with Python .gitignore template
4. **Best Practices**: Follows industry standards for quantitative finance projects

This automation ensures consistent project structure and reduces setup time.

## Examples

The `examples/` directory contains ready-to-run scripts demonstrating key features:

- `basic_simulation.py`: Simulate interest rates using all three models
- `bond_pricing_demo.py`: Price bonds and analyze pricing sensitivity
- `yield_curve_demo.py`: Construct and visualize yield curves
- `calibration_example.py`: Calibrate models to market data
- `sensitivity_analysis.py`: Perform comprehensive sensitivity studies

## Testing

Run the test suite using pytest:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Applications

### Fixed-Income Portfolio Management
- Value bond holdings under different rate scenarios
- Calculate portfolio Greeks and risk metrics
- Perform stress testing on portfolio

### Derivative Pricing
- Price interest rate swaps, swaptions, caps, and floors
- Monte Carlo valuation for exotic derivatives
- Compare model predictions with market prices

### Risk Management
- Estimate value-at-risk (VaR) from simulated rate paths
- Conduct scenario analysis for ALM (Asset-Liability Management)
- Measure interest rate duration and convexity

### Academic Research
- Empirically test model assumptions
- Calibrate to historical data for parameter estimation
- Investigate model performance across different market regimes

## References

1. Vasicek, O. (1977). "An equilibrium characterization of the term structure." Journal of Financial Economics.
2. Cox, J. C., Ingersoll, J. E., & Ross, S. A. (1985). "A theory of the term structure of interest rates." Econometrica.
3. Hull, J., & White, A. (1990). "Pricing interest-rate derivative securities." Review of Financial Studies.
4. Glasserman, P. (2004). "Monte Carlo Methods in Financial Engineering." Springer.
5. Brigo, D., & Mercurio, F. (2006). "Interest Rate Models - Theory and Practice." Springer.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Submit a pull request

## Acknowledgments

This framework draws inspiration from academic literature in quantitative finance and best practices in scientific Python development. Special acknowledgment to the NumPy, SciPy, Matplotlib, and Seaborn communities.
