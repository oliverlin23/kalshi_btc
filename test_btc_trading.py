"""
Test script for Bitcoin trading models.

This script demonstrates how to use the btc_trading module and tests the functions
with synthetic and real-world scenarios.
"""

import numpy as np
from app.btc_trading import (
    price_binary_contract,
    estimate_volatility_per_second,
    estimate_recent_volatility,
    black_scholes_binary_option_price,
    monte_carlo_binary_option_price,
)


def generate_synthetic_price_data(
    initial_price: float = 50000,
    num_points: int = 86400,
    volatility_per_second: float = 0.001,
    drift: float = 0.0,
    random_seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic Bitcoin price data for testing.
    
    Args:
        initial_price: Starting price
        num_points: Number of price points (default: 86400 = 24 hours of 1-second data)
        volatility_per_second: Volatility per second (default: 0.001 = 0.1% per second)
        drift: Drift per second (default: 0.0)
        random_seed: Random seed for reproducibility
    
    Returns:
        Array of synthetic prices
    """
    np.random.seed(random_seed)
    
    prices = [initial_price]
    current_price = initial_price
    
    for _ in range(num_points - 1):
        # Geometric Brownian motion: dS = S * (mu*dt + sigma*dW)
        # Using log returns: d(log S) = (mu - 0.5*sigma^2)*dt + sigma*dW
        log_return = (drift - 0.5 * volatility_per_second ** 2) + volatility_per_second * np.random.standard_normal()
        current_price *= np.exp(log_return)
        prices.append(current_price)
    
    return np.array(prices)


def test_basic_pricing():
    """Test basic binary contract pricing."""
    print("=" * 60)
    print("TEST 1: Basic Binary Contract Pricing")
    print("=" * 60)
    
    current_price = 50000
    strike_price = 51000
    expiry_seconds = 3600  # 1 hour
    
    # Test with annualized volatility
    print(f"\nCurrent price: ${current_price:,.2f}")
    print(f"Strike price: ${strike_price:,.2f}")
    print(f"Expiry: {expiry_seconds} seconds ({expiry_seconds/3600:.1f} hours)")
    
    # Test Black-Scholes with annualized volatility
    result_bs = price_binary_contract(
        current_price=current_price,
        strike_price=strike_price,
        expiry_time_seconds=expiry_seconds,
        volatility=0.6,  # 60% annualized
        method="black_scholes",
        volatility_is_annualized=True
    )
    
    print(f"\nBlack-Scholes (60% annualized vol):")
    print(f"  Fair price: ${result_bs['fair_price']:.4f}")
    print(f"  Probability: {result_bs['probability']:.2%}")
    print(f"  Volatility: {result_bs['volatility']:.2%} (annualized)")
    
    # Test Monte Carlo
    result_mc = price_binary_contract(
        current_price=current_price,
        strike_price=strike_price,
        expiry_time_seconds=expiry_seconds,
        volatility=0.6,
        method="monte_carlo",
        num_simulations=100000,
        random_seed=42,
        volatility_is_annualized=True
    )
    
    print(f"\nMonte Carlo (60% annualized vol, 100k simulations):")
    print(f"  Fair price: ${result_mc['fair_price']:.4f}")
    print(f"  Probability: {result_mc['probability']:.2%}")
    print(f"  Standard error: {result_mc.get('standard_error', 0):.6f}")
    
    # Compare results
    diff = abs(result_bs['fair_price'] - result_mc['fair_price'])
    print(f"\nDifference between methods: {diff:.6f}")
    print(f"Methods agree within standard error: {diff < result_mc.get('standard_error', 0) * 2}")


def test_with_synthetic_data():
    """Test pricing using synthetic historical price data."""
    print("\n" + "=" * 60)
    print("TEST 2: Pricing with Synthetic Historical Data")
    print("=" * 60)
    
    # Generate 24 hours of synthetic price data (1-second intervals)
    print("\nGenerating 24 hours of synthetic Bitcoin price data...")
    historical_prices = generate_synthetic_price_data(
        initial_price=50000,
        num_points=86400,  # 24 hours * 3600 seconds
        volatility_per_second=0.001,  # 0.1% per second
        drift=0.0,
        random_seed=42
    )
    
    print(f"Generated {len(historical_prices):,} price points")
    print(f"Price range: ${historical_prices.min():,.2f} - ${historical_prices.max():,.2f}")
    print(f"Final price: ${historical_prices[-1]:,.2f}")
    
    # Estimate volatility from the data
    vol_per_second, vol_annualized = estimate_recent_volatility(
        historical_prices,
        lookback_window_seconds=86400  # Use all 24 hours
    )
    
    print(f"\nEstimated volatility:")
    print(f"  Per second: {vol_per_second:.6f} ({vol_per_second*100:.4f}%)")
    print(f"  Annualized: {vol_annualized:.2%}")
    
    # Price a contract using recent volatility
    current_price = historical_prices[-1]
    strike_price = current_price * 1.02  # 2% above current price
    expiry_seconds = 3600  # 1 hour
    
    print(f"\nPricing binary contract:")
    print(f"  Current price: ${current_price:,.2f}")
    print(f"  Strike price: ${strike_price:,.2f} ({((strike_price/current_price - 1)*100):.2f}% above)")
    print(f"  Expiry: {expiry_seconds} seconds ({expiry_seconds/3600:.1f} hours)")
    
    # Using per-second volatility (more appropriate for Bitcoin)
    result_recent = price_binary_contract(
        current_price=current_price,
        strike_price=strike_price,
        expiry_time_seconds=expiry_seconds,
        historical_prices=historical_prices,
        use_recent_volatility=True,
        lookback_window_seconds=86400,
        method="black_scholes"
    )
    
    print(f"\nUsing recent volatility (per-second):")
    print(f"  Fair price: ${result_recent['fair_price']:.4f}")
    print(f"  Probability: {result_recent['probability']:.2%}")
    if 'volatility_per_second' in result_recent:
        print(f"  Volatility per second: {result_recent['volatility_per_second']:.6f}")
    
    # Compare with annualized volatility
    result_annualized = price_binary_contract(
        current_price=current_price,
        strike_price=strike_price,
        expiry_time_seconds=expiry_seconds,
        historical_prices=historical_prices,
        use_recent_volatility=False,  # Use all historical data, annualized
        method="black_scholes"
    )
    
    print(f"\nUsing all historical data (annualized):")
    print(f"  Fair price: ${result_annualized['fair_price']:.4f}")
    print(f"  Probability: {result_annualized['probability']:.2%}")
    print(f"  Volatility: {result_annualized['volatility']:.2%} (annualized)")


def test_different_expiry_times():
    """Test how pricing changes with different expiry times."""
    print("\n" + "=" * 60)
    print("TEST 3: Pricing for Different Expiry Times")
    print("=" * 60)
    
    current_price = 50000
    strike_price = 51000
    volatility_per_second = 0.001  # 0.1% per second
    
    expiry_times = [
        (60, "1 minute"),
        (300, "5 minutes"),
        (3600, "1 hour"),
        (14400, "4 hours"),
        (86400, "24 hours"),
    ]
    
    print(f"\nCurrent price: ${current_price:,.2f}")
    print(f"Strike price: ${strike_price:,.2f}")
    print(f"Volatility: {volatility_per_second:.4f} per second ({volatility_per_second * np.sqrt(365.25*24*3600):.2%} annualized)")
    
    print(f"\n{'Expiry':<15} {'Fair Price':<15} {'Probability':<15}")
    print("-" * 45)
    
    for expiry_seconds, label in expiry_times:
        result = price_binary_contract(
            current_price=current_price,
            strike_price=strike_price,
            expiry_time_seconds=expiry_seconds,
            volatility=volatility_per_second,
            method="black_scholes",
            volatility_is_annualized=False
        )
        
        print(f"{label:<15} ${result['fair_price']:<14.4f} {result['probability']:<14.2%}")


def test_monte_carlo_convergence():
    """Test Monte Carlo convergence with different numbers of simulations."""
    print("\n" + "=" * 60)
    print("TEST 4: Monte Carlo Convergence")
    print("=" * 60)
    
    current_price = 50000
    strike_price = 51000
    expiry_seconds = 3600
    volatility = 0.001  # per second
    
    # Get Black-Scholes price as benchmark
    bs_price = black_scholes_binary_option_price(
        current_price, strike_price, expiry_seconds, volatility,
        volatility_is_annualized=False
    )
    
    print(f"\nBlack-Scholes benchmark: ${bs_price:.6f}")
    print(f"\n{'Simulations':<15} {'MC Price':<15} {'Error':<15} {'Std Error':<15}")
    print("-" * 60)
    
    num_sims_list = [1000, 10000, 100000, 1000000]
    
    for num_sims in num_sims_list:
        result = monte_carlo_binary_option_price(
            current_price, strike_price, expiry_seconds, volatility,
            num_simulations=num_sims,
            random_seed=42,
            volatility_is_annualized=False
        )
        
        mc_price, std_err = result
        error = abs(mc_price - bs_price)
        
        print(f"{num_sims:<15,} ${mc_price:<14.6f} ${error:<14.6f} ${std_err:<14.6f}")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("TEST 5: Edge Cases")
    print("=" * 60)
    
    # Test at expiry (time = 0)
    print("\n1. At expiry (time = 0):")
    result = price_binary_contract(
        current_price=50000,
        strike_price=51000,
        expiry_time_seconds=0,
        volatility=0.6,
        method="black_scholes"
    )
    print(f"   Price: ${result['fair_price']:.4f} (should be $0.0000 - price below strike)")
    
    result = price_binary_contract(
        current_price=51000,
        strike_price=50000,
        expiry_time_seconds=0,
        volatility=0.6,
        method="black_scholes"
    )
    print(f"   Price: ${result['fair_price']:.4f} (should be $1.0000 - price above strike)")
    
    # Test zero volatility
    print("\n2. Zero volatility:")
    result = price_binary_contract(
        current_price=50000,
        strike_price=51000,
        expiry_time_seconds=3600,
        volatility=0.0,
        method="black_scholes"
    )
    print(f"   Price: ${result['fair_price']:.4f} (should be $0.0000 - deterministic, won't reach strike)")
    
    # Test very high volatility
    print("\n3. Very high volatility:")
    result = price_binary_contract(
        current_price=50000,
        strike_price=51000,
        expiry_time_seconds=3600,
        volatility=0.01,  # 1% per second - very high!
        method="black_scholes",
        volatility_is_annualized=False
    )
    print(f"   Price: ${result['fair_price']:.4f} (high volatility increases probability)")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("BITCOIN TRADING MODEL TESTS")
    print("=" * 60)
    
    try:
        test_basic_pricing()
        test_with_synthetic_data()
        test_different_expiry_times()
        test_monte_carlo_convergence()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

