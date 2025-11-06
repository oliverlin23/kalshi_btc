"""
Bitcoin Trading Models for Binary Contract Pricing

This module provides functions for modeling Bitcoin price/volatility and pricing
binary contracts that resolve to $1 if Bitcoin is above a strike price at expiry,
or $0 otherwise.
"""

import numpy as np
from scipy.stats import norm
from typing import Optional, List, Tuple
import warnings


def estimate_volatility_from_returns(returns: np.ndarray, annualization_factor: float = 86400) -> float:
    """
    Estimate annualized volatility from a series of returns.
    
    Args:
        returns: Array of log returns (price changes)
        annualization_factor: Factor to annualize volatility (default: 86400 for 1-second data)
                             Use 86400 for per-second data, 3600 for per-hour, etc.
    
    Returns:
        Annualized volatility (as a decimal, e.g., 0.5 for 50%)
    """
    if len(returns) < 2:
        return 0.5  # Default to 50% volatility if insufficient data
    
    # Calculate standard deviation of returns
    std_dev = np.std(returns)
    
    # Annualize: multiply by sqrt(annualization_factor)
    annualized_vol = std_dev * np.sqrt(annualization_factor)
    
    return annualized_vol


def estimate_volatility_per_second(historical_prices: np.ndarray, time_period_seconds: float = 1.0) -> float:
    """
    Estimate volatility per second from historical prices.
    This is more appropriate for Bitcoin which has rapidly changing volatility.
    
    Returns volatility as a standard deviation per second (not annualized).
    This can be directly used in Monte Carlo simulations or scaled to the expiry time.
    
    Args:
        historical_prices: Array of historical Bitcoin prices
        time_period_seconds: Time between each price observation in seconds (default: 1.0)
    
    Returns:
        Volatility per second (standard deviation of log returns per second)
    """
    if len(historical_prices) < 2:
        # Default: assume 1% price movement per second (very high volatility)
        return 0.01
    
    # Calculate log returns
    log_returns = np.diff(np.log(historical_prices))
    
    # Calculate standard deviation of returns
    std_dev = np.std(log_returns)
    
    # Scale to per-second if needed
    if time_period_seconds != 1.0:
        # If data is sampled every N seconds, scale by sqrt(N)
        std_dev = std_dev / np.sqrt(time_period_seconds)
    
    return std_dev


def estimate_recent_volatility(
    historical_prices: np.ndarray,
    price_timestamps_seconds: Optional[np.ndarray] = None,
    lookback_window_seconds: float = 86400,  # 24 hours default
    time_period_seconds: float = 1.0
) -> Tuple[float, float]:
    """
    Estimate volatility from recent price data (e.g., last 24 hours).
    Returns both per-second volatility and annualized volatility.
    
    This is useful for Bitcoin where volatility changes quickly and recent
    volatility is more relevant than long-term annualized volatility.
    
    Args:
        historical_prices: Array of historical Bitcoin prices
        price_timestamps_seconds: Optional array of timestamps in seconds.
                                 If None, assumes uniform spacing.
        lookback_window_seconds: How far back to look (default: 86400 = 24 hours)
        time_period_seconds: Time between price observations if timestamps not provided
    
    Returns:
        Tuple of (volatility_per_second, annualized_volatility)
    """
    if len(historical_prices) < 2:
        return (0.01, 0.5)  # Default values
    
    # Filter to recent data if timestamps provided
    if price_timestamps_seconds is not None:
        if len(price_timestamps_seconds) != len(historical_prices):
            raise ValueError("price_timestamps_seconds must have same length as historical_prices")
        
        # Find most recent timestamp
        most_recent = np.max(price_timestamps_seconds)
        cutoff_time = most_recent - lookback_window_seconds
        
        # Filter to recent data
        mask = price_timestamps_seconds >= cutoff_time
        recent_prices = historical_prices[mask]
        
        if len(recent_prices) < 2:
            # Not enough recent data, use all available
            recent_prices = historical_prices
    else:
        # No timestamps - assume uniform spacing and take last N observations
        # Estimate number of observations in lookback window
        num_observations = int(lookback_window_seconds / time_period_seconds)
        recent_prices = historical_prices[-num_observations:] if len(historical_prices) > num_observations else historical_prices
    
    # Calculate per-second volatility
    vol_per_second = estimate_volatility_per_second(recent_prices, time_period_seconds)
    
    # Also calculate annualized for compatibility
    annualized_vol = vol_per_second * np.sqrt(365.25 * 24 * 3600)
    
    return (vol_per_second, annualized_vol)


def estimate_volatility_from_prices(prices: np.ndarray, time_period_seconds: float = 1.0) -> float:
    """
    Estimate annualized volatility from a series of prices.
    
    Args:
        prices: Array of Bitcoin prices over time
        time_period_seconds: Time between each price observation in seconds (default: 1.0)
    
    Returns:
        Annualized volatility (as a decimal)
    """
    if len(prices) < 2:
        return 0.5  # Default to 50% volatility if insufficient data
    
    # Calculate log returns
    log_returns = np.diff(np.log(prices))
    
    # Annualization factor: seconds per year / time_period_seconds
    seconds_per_year = 365.25 * 24 * 3600  # Account for leap years
    annualization_factor = seconds_per_year / time_period_seconds
    
    return estimate_volatility_from_returns(log_returns, annualization_factor)


def black_scholes_binary_option_price(
    current_price: float,
    strike_price: float,
    time_to_expiry_seconds: float,
    volatility: float,
    risk_free_rate: float = 0.0,
    volatility_is_annualized: bool = True
) -> float:
    """
    Price a binary option using the Black-Scholes formula.
    
    A binary option pays $1 if the underlying price is above the strike at expiry,
    and $0 otherwise.
    
    Args:
        current_price: Current Bitcoin price
        strike_price: Strike price (the threshold)
        time_to_expiry_seconds: Time until expiry in seconds
        volatility: Volatility (annualized if volatility_is_annualized=True, per-second otherwise)
        risk_free_rate: Risk-free interest rate (as a decimal, default: 0.0)
        volatility_is_annualized: If True, volatility is annualized. If False, volatility is per-second.
    
    Returns:
        Fair price of the binary option (0 to 1)
    """
    if time_to_expiry_seconds <= 0:
        # At expiry, price is 1 if current_price > strike_price, else 0
        return 1.0 if current_price > strike_price else 0.0
    
    if volatility <= 0:
        # If no volatility, deterministic outcome
        return 1.0 if current_price > strike_price else 0.0
    
    S = current_price
    K = strike_price
    r = risk_free_rate
    
    if S <= 0 or K <= 0:
        raise ValueError("Prices must be positive")
    
    # Convert volatility and time appropriately
    if volatility_is_annualized:
        # Convert time to years
        time_to_expiry_years = time_to_expiry_seconds / (365.25 * 24 * 3600)
        T = time_to_expiry_years
        sigma = volatility
    else:
        # Volatility is per-second, time is in seconds
        # sigma * sqrt(T) gives us the total volatility over the period
        T_seconds = time_to_expiry_seconds
        sigma_per_second = volatility
        # For the formula, we need sigma * sqrt(T) where T is in years
        # So: sigma_per_second * sqrt(T_seconds) = sigma_annualized * sqrt(T_years)
        # Therefore: sigma_annualized * sqrt(T_years) = sigma_per_second * sqrt(T_seconds)
        # We can use this directly in the formula
        T = time_to_expiry_seconds / (365.25 * 24 * 3600)  # Convert to years for discounting
        # sigma * sqrt(T) = sigma_per_second * sqrt(T_seconds)
        sigma = sigma_per_second * np.sqrt(365.25 * 24 * 3600)  # Annualize for formula
    
    if T <= 0:
        return 1.0 if current_price > strike_price else 0.0
    
    # Calculate d2 (simplified for binary option)
    # For a binary call: P(S_T > K) = N(d2)
    # where d2 = (ln(S/K) + (r - 0.5*sigma^2)*T) / (sigma * sqrt(T))
    
    # Calculate d2
    d2_numerator = np.log(S / K) + (r - 0.5 * sigma ** 2) * T
    d2_denominator = sigma * np.sqrt(T)
    
    if d2_denominator == 0:
        return 1.0 if current_price > strike_price else 0.0
    
    d2 = d2_numerator / d2_denominator
    
    # Binary call option price = e^(-r*T) * N(d2)
    # For risk-free rate = 0, this simplifies to N(d2)
    price = np.exp(-r * T) * norm.cdf(d2)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, price))


def monte_carlo_binary_option_price(
    current_price: float,
    strike_price: float,
    time_to_expiry_seconds: float,
    volatility: float,
    risk_free_rate: float = 0.0,
    num_simulations: int = 100000,
    random_seed: Optional[int] = None,
    volatility_is_annualized: bool = True
) -> Tuple[float, float]:
    """
    Price a binary option using Monte Carlo simulation.
    
    Args:
        current_price: Current Bitcoin price
        strike_price: Strike price (the threshold)
        time_to_expiry_seconds: Time until expiry in seconds
        volatility: Volatility (annualized if volatility_is_annualized=True, per-second otherwise)
        risk_free_rate: Risk-free interest rate (as a decimal, default: 0.0)
        num_simulations: Number of Monte Carlo simulations (default: 100000)
        random_seed: Random seed for reproducibility (default: None)
        volatility_is_annualized: If True, volatility is annualized. If False, volatility is per-second.
    
    Returns:
        Tuple of (estimated_price, standard_error)
    """
    if time_to_expiry_seconds <= 0:
        price = 1.0 if current_price > strike_price else 0.0
        return (price, 0.0)
    
    if volatility <= 0:
        price = 1.0 if current_price > strike_price else 0.0
        return (price, 0.0)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    S = current_price
    K = strike_price
    r = risk_free_rate
    
    # Convert volatility and time appropriately
    if volatility_is_annualized:
        # Convert time to years
        T = time_to_expiry_seconds / (365.25 * 24 * 3600)
        sigma = volatility
    else:
        # Volatility is per-second, time is in seconds
        T_seconds = time_to_expiry_seconds
        sigma_per_second = volatility
        # For Monte Carlo: S_T = S_0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        # With per-second volatility: sigma * sqrt(T) = sigma_per_second * sqrt(T_seconds)
        T = T_seconds / (365.25 * 24 * 3600)  # Convert to years for discounting
        sigma = sigma_per_second * np.sqrt(365.25 * 24 * 3600)  # Annualize for formula
    
    if T <= 0:
        price = 1.0 if current_price > strike_price else 0.0
        return (price, 0.0)
    
    # Generate random standard normal variables
    Z = np.random.standard_normal(num_simulations)
    
    # Calculate future prices using geometric Brownian motion
    # S_T = S_0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    drift = (r - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z
    future_prices = S * np.exp(drift + diffusion)
    
    # Binary option: 1 if price > strike, 0 otherwise
    payoffs = (future_prices > K).astype(float)
    
    # Discount to present value
    discounted_payoffs = payoffs * np.exp(-r * T)
    
    # Calculate mean and standard error
    price = np.mean(discounted_payoffs)
    standard_error = np.std(discounted_payoffs) / np.sqrt(num_simulations)
    
    # Clamp to [0, 1]
    price = max(0.0, min(1.0, price))
    
    return (price, standard_error)


def estimate_volatility_from_historical_data(
    historical_prices: np.ndarray,
    price_timestamps_seconds: Optional[np.ndarray] = None
) -> float:
    """
    Estimate volatility from historical price data.
    
    Args:
        historical_prices: Array of historical Bitcoin prices
        price_timestamps_seconds: Optional array of timestamps in seconds.
                                 If None, assumes uniform spacing of 1 second.
    
    Returns:
        Annualized volatility estimate
    """
    if len(historical_prices) < 2:
        return 0.5  # Default to 50% volatility
    
    if price_timestamps_seconds is None:
        # Assume uniform 1-second spacing
        return estimate_volatility_from_prices(historical_prices, time_period_seconds=1.0)
    else:
        # Calculate time differences
        time_diffs = np.diff(price_timestamps_seconds)
        
        # If all time differences are the same, use simple method
        if np.allclose(time_diffs, time_diffs[0]):
            return estimate_volatility_from_prices(historical_prices, time_period_seconds=time_diffs[0])
        else:
            # Variable time spacing - use weighted approach or median
            median_time_diff = np.median(time_diffs)
            return estimate_volatility_from_prices(historical_prices, time_period_seconds=median_time_diff)


def calculate_probability_above_strike(
    current_price: float,
    strike_price: float,
    time_to_expiry_seconds: float,
    volatility: float,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate the probability that Bitcoin price will be above strike at expiry.
    
    This is equivalent to the fair price of a binary option.
    
    Args:
        current_price: Current Bitcoin price
        strike_price: Strike price (the threshold)
        time_to_expiry_seconds: Time until expiry in seconds
        volatility: Annualized volatility (as a decimal)
        risk_free_rate: Risk-free interest rate (as a decimal, default: 0.0)
    
    Returns:
        Probability (0 to 1)
    """
    return black_scholes_binary_option_price(
        current_price, strike_price, time_to_expiry_seconds, volatility, risk_free_rate
    )


def price_binary_contract(
    current_price: float,
    strike_price: float,
    expiry_time_seconds: float,
    volatility: Optional[float] = None,
    historical_prices: Optional[np.ndarray] = None,
    historical_timestamps: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0,
    method: str = "black_scholes",
    num_simulations: int = 100000,
    random_seed: Optional[int] = None,
    use_recent_volatility: bool = True,
    lookback_window_seconds: float = 86400,
    volatility_is_annualized: Optional[bool] = None
) -> dict:
    """
    Determine the fair price of a binary contract on Bitcoin.
    
    The contract resolves to $1 if Bitcoin price is above strike_price at expiry,
    or $0 otherwise.
    
    Args:
        current_price: Current Bitcoin price
        strike_price: Strike price (the threshold)
        expiry_time_seconds: Time until expiry in seconds
        volatility: Volatility. If None, will be estimated from historical_prices.
                   Interpretation depends on volatility_is_annualized.
        historical_prices: Optional array of historical Bitcoin prices for volatility estimation.
                          Should be prices per second (or other time period) for recent period
                          (e.g., last 24 hours).
        historical_timestamps: Optional array of timestamps in seconds corresponding to
                              historical_prices. If None, assumes uniform 1-second spacing.
        risk_free_rate: Risk-free interest rate (as a decimal, default: 0.0)
        method: Pricing method - "black_scholes" (default) or "monte_carlo"
        num_simulations: Number of simulations for Monte Carlo method (default: 100000)
        random_seed: Random seed for Monte Carlo reproducibility (default: None)
        use_recent_volatility: If True and historical_prices provided, uses recent volatility
                              (last lookback_window_seconds) instead of all historical data.
        lookback_window_seconds: How far back to look for volatility estimation (default: 86400 = 24 hours)
        volatility_is_annualized: If True, volatility is annualized. If False, volatility is per-second.
                                 If None, auto-detects based on use_recent_volatility.
    
    Returns:
        Dictionary with:
        - 'fair_price': Fair price of the contract (0 to 1)
        - 'probability': Probability that price will be above strike at expiry
        - 'volatility': Volatility used in calculation
        - 'volatility_per_second': Volatility per second (if estimated from recent data)
        - 'method': Method used for pricing
        - 'standard_error': Standard error (only for Monte Carlo method)
    
    Example:
        >>> price_binary_contract(
        ...     current_price=50000,
        ...     strike_price=51000,
        ...     expiry_time_seconds=3600,  # 1 hour
        ...     volatility=0.6,  # 60% annualized volatility
        ...     method="black_scholes"
        ... )
    """
    # Estimate volatility if not provided
    vol_per_second = None
    if volatility is None:
        if historical_prices is not None and len(historical_prices) > 1:
            if use_recent_volatility:
                vol_per_second, volatility = estimate_recent_volatility(
                    historical_prices, historical_timestamps, lookback_window_seconds
                )
                # Default to per-second if using recent volatility
                if volatility_is_annualized is None:
                    volatility_is_annualized = False
            else:
                volatility = estimate_volatility_from_historical_data(
                    historical_prices, historical_timestamps
                )
                if volatility_is_annualized is None:
                    volatility_is_annualized = True
        else:
            # Default to 50% annualized volatility if no historical data
            volatility = 0.5
            if volatility_is_annualized is None:
                volatility_is_annualized = True
            warnings.warn(
                "No volatility provided and insufficient historical data. "
                "Using default volatility of 50% (annualized).",
                UserWarning
            )
    else:
        # Volatility provided - determine if annualized or per-second
        if volatility_is_annualized is None:
            # Default: assume annualized unless use_recent_volatility is True
            volatility_is_annualized = not use_recent_volatility
    
    # Validate inputs
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    if strike_price <= 0:
        raise ValueError("strike_price must be positive")
    if expiry_time_seconds < 0:
        raise ValueError("expiry_time_seconds must be non-negative")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    
    # Calculate fair price using selected method
    if method.lower() == "black_scholes":
        fair_price = black_scholes_binary_option_price(
            current_price, strike_price, expiry_time_seconds, volatility,
            risk_free_rate, volatility_is_annualized
        )
        standard_error = None
    elif method.lower() == "monte_carlo":
        fair_price, standard_error = monte_carlo_binary_option_price(
            current_price, strike_price, expiry_time_seconds, volatility,
            risk_free_rate, num_simulations, random_seed, volatility_is_annualized
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'black_scholes' or 'monte_carlo'")
    
    result = {
        'fair_price': fair_price,
        'probability': fair_price,  # For binary options, price = probability
        'volatility': volatility,
        'method': method.lower(),
        'volatility_is_annualized': volatility_is_annualized,
    }
    
    if vol_per_second is not None:
        result['volatility_per_second'] = vol_per_second
    
    if standard_error is not None:
        result['standard_error'] = standard_error
    
    return result

