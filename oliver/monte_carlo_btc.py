"""
Monte Carlo modeling for Bitcoin price prediction.

This module implements GBM (Geometric Brownian Motion) and Merton jump-diffusion
models to simulate possible future paths of BTC prices based on historical 1-minute data.
"""

import os
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------- Utility functions ----------

def log_returns(prices):
    """Compute log returns from price series."""
    prices = np.asarray(prices, dtype=float)
    return np.diff(np.log(prices))


def mad(x):
    """Compute Median Absolute Deviation (MAD)."""
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-12


# ---------- Parameter estimation ----------

def estimate_gbm_params(min_prices, risk_neutral=False, risk_free_rate=0.0, drift_factor=1.0, time_period_seconds=60):
    """
    Estimate GBM parameters from price data.
    
    Args:
        min_prices: Array of closing prices (can be per-second, per-5-second, or per-minute)
        risk_neutral: If True, use risk-neutral drift (risk_free_rate), else use historical mean
        risk_free_rate: Risk-free rate per time period (default 0.0, i.e., no drift)
        drift_factor: Factor to scale historical drift (0.0 = risk-neutral, 1.0 = full historical)
                     Overrides risk_neutral if provided
        time_period_seconds: Time interval between data points in seconds (default: 60 for 1-minute)
        
    Returns:
        Tuple of (mu_dt, sigma_dt) - per-minute drift and volatility (always converted to per-minute)
    """
    r = log_returns(min_prices)
    historical_mu = r.mean()
    
    # Calculate per-period parameters
    if drift_factor is not None and drift_factor != 1.0:
        # Interpolate between risk-neutral and historical
        mu_dt_period = risk_free_rate + drift_factor * (historical_mu - risk_free_rate)
    elif risk_neutral:
        mu_dt_period = risk_free_rate  # Risk-neutral: drift = risk-free rate (typically 0 for BTC)
    else:
        mu_dt_period = historical_mu   # P-measure: use historical sample mean
    
    sigma_dt_period = r.std(ddof=1)    # Volatility always estimated from data
    
    # Convert to per-minute equivalents
    # For volatility: sigma scales with sqrt(time)
    # For drift: mu scales linearly with time
    seconds_per_minute = 60.0
    conversion_factor = seconds_per_minute / time_period_seconds
    
    mu_dt = mu_dt_period * conversion_factor
    sigma_dt = sigma_dt_period * np.sqrt(conversion_factor)
    
    return mu_dt, sigma_dt


def estimate_merton_params(min_prices, c=3.5, risk_neutral=False, risk_free_rate=0.0, drift_factor=1.0, time_period_seconds=60):
    """
    Estimate Merton jump-diffusion parameters from price data.
    
    Args:
        min_prices: Array of closing prices (can be per-second, per-5-second, or per-minute)
        c: Multiplier for jump detection threshold (default 3.5)
        risk_neutral: If True, use risk-neutral drift (risk_free_rate), else use historical mean
        risk_free_rate: Risk-free rate per time period (default 0.0, i.e., no drift)
        drift_factor: Factor to scale historical drift (0.0 = risk-neutral, 1.0 = full historical)
                     Overrides risk_neutral if provided
        time_period_seconds: Time interval between data points in seconds (default: 60 for 1-minute)
        
    Returns:
        Dictionary with parameters: mu_dt, sigma_dt, lam, mu_J, delta (all per-minute)
    """
    r = log_returns(min_prices)
    historical_mu = r.mean()
    
    # Jump detection via robust threshold
    thresh = c * 1.4826 * mad(r)     # 1.4826 * MAD ~ sigma for Normal
    is_jump = np.abs(r) > max(thresh, 1e-8)

    r_jump = r[is_jump]
    r_diff = r[~is_jump] if (~is_jump).any() else r * 0 + 1e-8

    lam_per_period = is_jump.mean()     # expected jumps per time period
    # Jump size distribution (log-multipliers)
    if len(r_jump) >= 2:
        mu_J = r_jump.mean()
        delta = r_jump.std(ddof=1)
    elif len(r_jump) == 1:
        mu_J = r_jump.mean()
        delta = np.std(r_diff, ddof=1)  # fallback
    else:
        mu_J, delta = 0.0, 1e-6        # no jumps observed

    # Diffusive params from non-jump periods
    sigma_dt_period = max(r_diff.std(ddof=1), 1e-8)
    
    # Calculate per-period drift
    if drift_factor is not None and drift_factor != 1.0:
        mu_dt_period = risk_free_rate + drift_factor * (historical_mu - risk_free_rate)
    elif risk_neutral:
        mu_dt_period = risk_free_rate  # Risk-neutral: drift = risk-free rate (typically 0 for BTC)
    else:
        mu_dt_period = historical_mu   # P-measure: use historical sample mean
    
    # Convert to per-minute equivalents
    # For volatility: sigma scales with sqrt(time)
    # For drift: mu scales linearly with time
    # For jump intensity: lambda scales linearly with time
    seconds_per_minute = 60.0
    conversion_factor = seconds_per_minute / time_period_seconds
    
    mu_dt = mu_dt_period * conversion_factor
    sigma_dt = sigma_dt_period * np.sqrt(conversion_factor)
    lam_per_min = lam_per_period * conversion_factor

    return dict(mu_dt=mu_dt, sigma_dt=sigma_dt,
                lam=lam_per_min, mu_J=mu_J, delta=delta)


# ---------- Simulators ----------

def simulate_gbm_paths(S0, minutes_ahead, mu_dt, sigma_dt, n_paths=10000, seed=42):
    """
    Simulate GBM paths using Euler-exact method.
    
    Args:
        S0: Initial price
        minutes_ahead: Number of minutes to simulate
        mu_dt: Per-minute drift
        sigma_dt: Per-minute volatility
        n_paths: Number of Monte Carlo paths
        seed: Random seed
        
    Returns:
        Array of shape (minutes_ahead+1, n_paths) with price paths
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(minutes_ahead, n_paths))
    drift = (mu_dt - 0.5 * sigma_dt**2)
    increments = drift + sigma_dt * Z
    log_S = np.log(S0) + np.cumsum(increments, axis=0)
    S = np.vstack([np.full(n_paths, S0), np.exp(log_S)])
    return S


def simulate_merton_paths(S0, minutes_ahead, mu_dt, sigma_dt, lam, mu_J, delta, 
                          n_paths=10000, seed=123):
    """
    Simulate Merton jump-diffusion paths.
    
    Args:
        S0: Initial price
        minutes_ahead: Number of minutes to simulate
        mu_dt: Per-minute drift
        sigma_dt: Per-minute volatility
        lam: Jump intensity (expected jumps per minute)
        mu_J: Mean of jump size distribution
        delta: Std dev of jump size distribution
        n_paths: Number of Monte Carlo paths
        seed: Random seed
        
    Returns:
        Array of shape (minutes_ahead+1, n_paths) with price paths
    """
    rng = np.random.default_rng(seed)
    # Pre-draw diffusion
    Z = rng.standard_normal(size=(minutes_ahead, n_paths))
    # Poisson jump counts per minute
    N = rng.poisson(lam, size=(minutes_ahead, n_paths))
    # For efficiency, use CLT approximation: sum of N normals -> Normal(N*mu_J, N*delta^2)
    jump_mu = N * mu_J
    jump_sd = np.sqrt(np.maximum(N, 0) * (delta**2))

    # Drift includes compensator so the process is correctly centered
    kappa = np.exp(mu_J + 0.5 * delta**2) - 1.0
    drift = (mu_dt - 0.5 * sigma_dt**2 - lam * kappa)

    # Evolve
    increments = drift + sigma_dt * Z
    # Add jump terms ~ Normal(jump_mu, jump_sd^2)
    J = rng.normal(loc=jump_mu, scale=jump_sd)
    increments += J

    log_S = np.log(S0) + np.cumsum(increments, axis=0)
    S = np.vstack([np.full(n_paths, S0), np.exp(log_S)])
    return S


def estimate_heston_params(min_prices, risk_neutral=False, risk_free_rate=0.0, drift_factor=1.0):
    """
    Estimate Heston stochastic volatility parameters from 1-minute price data.
    
    Heston model: dS = μS dt + √V S dW1
                  dV = κ(θ - V) dt + σ_v √V dW2
                  where V is variance, κ is mean reversion, θ is long-term variance
    
    Args:
        min_prices: Array of 1-minute closing prices
        risk_neutral: If True, use risk-neutral drift
        risk_free_rate: Risk-free rate per minute
        drift_factor: Factor to scale historical drift
        
    Returns:
        Dictionary with parameters: mu_dt, v0, kappa, theta, sigma_v, rho
    """
    r = log_returns(min_prices)
    historical_mu = r.mean()
    
    # Estimate drift
    if drift_factor is not None and drift_factor != 1.0:
        mu_dt = risk_free_rate + drift_factor * (historical_mu - risk_free_rate)
    elif risk_neutral:
        mu_dt = risk_free_rate
    else:
        mu_dt = historical_mu
    
    # Estimate variance process parameters using realized variance
    # Use rolling window to estimate time-varying variance
    window = min(60, len(r) // 4)  # 1-hour window or 1/4 of data
    realized_var = []
    
    for i in range(window, len(r)):
        window_returns = r[i-window:i]
        realized_var.append(np.var(window_returns))
    
    realized_var = np.array(realized_var)
    
    # Initial variance (current level)
    v0 = max(np.var(r[-window:]), 1e-8)
    
    # Long-term variance (mean of realized variance)
    theta = max(np.mean(realized_var), 1e-8)
    
    # Mean reversion speed (from autocorrelation of variance)
    if len(realized_var) > 1:
        var_changes = np.diff(realized_var)
        var_lag = realized_var[:-1]
        # Simple regression: ΔV ≈ κ(θ - V) + noise
        # κ ≈ -cov(ΔV, V) / var(V) / (θ - mean(V))
        if np.var(var_lag) > 1e-10:
            kappa_est = -np.cov(var_changes, var_lag)[0,1] / np.var(var_lag) / max(theta - np.mean(var_lag), 1e-8)
            kappa = max(min(kappa_est, 1.0), 0.01)  # Clamp between 0.01 and 1.0
        else:
            kappa = 0.1
    else:
        kappa = 0.1
    
    # Volatility of volatility (from variance of variance changes)
    if len(var_changes) > 1:
        sigma_v = max(np.std(var_changes) / np.sqrt(np.mean(realized_var)), 1e-6)
        sigma_v = min(sigma_v, 2.0)  # Cap at reasonable level
    else:
        sigma_v = 0.3
    
    # Correlation between price and variance (leverage effect)
    # Typically negative for equities, can be positive/negative for crypto
    if len(r) > window:
        price_changes = r[-len(realized_var):]
        if len(price_changes) == len(realized_var) and np.std(price_changes) > 1e-10:
            rho = np.corrcoef(price_changes, realized_var)[0,1]
            rho = max(min(rho, 0.9), -0.9)  # Clamp between -0.9 and 0.9
        else:
            rho = -0.3  # Default negative correlation (leverage effect)
    else:
        rho = -0.3
    
    return dict(mu_dt=mu_dt, v0=v0, kappa=kappa, theta=theta, 
                sigma_v=sigma_v, rho=rho)


def simulate_heston_paths(S0, minutes_ahead, mu_dt, v0, kappa, theta, sigma_v, rho,
                          n_paths=10000, seed=456):
    """
    Simulate Heston stochastic volatility paths using Euler scheme.
    
    Args:
        S0: Initial price
        minutes_ahead: Number of minutes to simulate
        mu_dt: Per-minute drift
        v0: Initial variance
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma_v: Volatility of volatility
        rho: Correlation between price and variance
        n_paths: Number of Monte Carlo paths
        seed: Random seed
        
    Returns:
        Array of shape (minutes_ahead+1, n_paths) with price paths
    """
    rng = np.random.default_rng(seed)
    
    # Generate correlated Brownian motions
    Z1 = rng.standard_normal(size=(minutes_ahead, n_paths))
    Z2 = rng.standard_normal(size=(minutes_ahead, n_paths))
    Z_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    # Initialize
    log_S = np.log(S0) * np.ones((minutes_ahead + 1, n_paths))
    V = v0 * np.ones((minutes_ahead + 1, n_paths))
    
    # Euler scheme
    for t in range(minutes_ahead):
        # Ensure variance stays positive (Feller condition: 2*κ*θ > σ_v^2)
        # Use reflection or truncation
        V_curr = np.maximum(V[t], 1e-8)
        sqrt_V = np.sqrt(V_curr)
        
        # Update variance: dV = κ(θ - V) dt + σ_v √V dW2
        V[t+1] = V_curr + kappa * (theta - V_curr) + sigma_v * sqrt_V * Z_v[t]
        V[t+1] = np.maximum(V[t+1], 1e-8)  # Ensure non-negative
        
        # Update price: dS = μS dt + √V S dW1
        sqrt_V_next = np.sqrt(np.maximum(V[t+1], 1e-8))
        log_S[t+1] = log_S[t] + (mu_dt - 0.5 * V_curr) + sqrt_V * Z1[t]
    
    S = np.exp(log_S)
    S[0] = S0  # Ensure initial price is correct
    return S


def estimate_regime_switching_params(min_prices, n_regimes=2, risk_neutral=False, 
                                      risk_free_rate=0.0, drift_factor=1.0):
    """
    Estimate regime-switching model parameters.
    
    Assumes two regimes: low volatility and high volatility.
    Uses simple threshold method to identify regimes.
    
    Args:
        min_prices: Array of 1-minute closing prices
        n_regimes: Number of regimes (default: 2)
        risk_neutral: If True, use risk-neutral drift
        risk_free_rate: Risk-free rate per minute
        drift_factor: Factor to scale historical drift
        
    Returns:
        Dictionary with parameters for each regime and transition probabilities
    """
    r = log_returns(min_prices)
    historical_mu = r.mean()
    
    # Estimate drift
    if drift_factor is not None and drift_factor != 1.0:
        mu_dt = risk_free_rate + drift_factor * (historical_mu - risk_free_rate)
    elif risk_neutral:
        mu_dt = risk_free_rate
    else:
        mu_dt = historical_mu
    
    # Identify regimes using rolling volatility
    window = min(60, len(r) // 4)
    rolling_vol = []
    
    for i in range(window, len(r)):
        window_returns = r[i-window:i]
        rolling_vol.append(np.std(window_returns))
    
    rolling_vol = np.array(rolling_vol)
    
    # Simple threshold: median split for 2 regimes
    if n_regimes == 2:
        vol_threshold = np.median(rolling_vol)
        is_high_vol = rolling_vol > vol_threshold
        
        # Regime 0: Low volatility
        low_vol_returns = r[window:][~is_high_vol] if len(is_high_vol) > 0 else r[window:]
        sigma_low = max(np.std(low_vol_returns) if len(low_vol_returns) > 1 else np.std(r), 1e-8)
        
        # Regime 1: High volatility
        high_vol_returns = r[window:][is_high_vol] if len(is_high_vol) > 0 else r[window:]
        sigma_high = max(np.std(high_vol_returns) if len(high_vol_returns) > 1 else np.std(r) * 1.5, 1e-8)
        
        # Transition probabilities (simple: based on persistence)
        # P(stay in low vol) and P(stay in high vol)
        if len(is_high_vol) > 1:
            transitions = np.diff(is_high_vol.astype(int))
            # Count regime persistence
            low_to_low = np.sum((~is_high_vol[:-1]) & (~is_high_vol[1:]))
            low_to_high = np.sum((~is_high_vol[:-1]) & is_high_vol[1:])
            high_to_high = np.sum(is_high_vol[:-1] & is_high_vol[1:])
            high_to_low = np.sum(is_high_vol[:-1] & (~is_high_vol[1:]))
            
            p00 = low_to_low / max(low_to_low + low_to_high, 1)
            p11 = high_to_high / max(high_to_high + high_to_low, 1)
            p01 = 1 - p00
            p10 = 1 - p11
        else:
            # Default: high persistence
            p00 = 0.95
            p11 = 0.90
            p01 = 0.05
            p10 = 0.10
        
        # Current regime (based on recent volatility)
        current_vol = np.std(r[-window:]) if len(r) >= window else np.std(r)
        current_regime = 1 if current_vol > vol_threshold else 0
        
        return dict(
            mu_dt=mu_dt,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            p00=p00, p01=p01, p10=p10, p11=p11,
            current_regime=current_regime,
            vol_threshold=vol_threshold
        )
    else:
        # For simplicity, only implement 2-regime case
        raise ValueError("Only 2 regimes supported currently")


def simulate_regime_switching_paths(S0, minutes_ahead, mu_dt, sigma_low, sigma_high,
                                    p00, p01, p10, p11, current_regime,
                                    n_paths=10000, seed=789):
    """
    Simulate regime-switching paths.
    
    Args:
        S0: Initial price
        minutes_ahead: Number of minutes to simulate
        mu_dt: Per-minute drift
        sigma_low: Volatility in low regime
        sigma_high: Volatility in high regime
        p00, p01, p10, p11: Transition probabilities
        current_regime: Starting regime (0=low, 1=high)
        n_paths: Number of Monte Carlo paths
        seed: Random seed
        
    Returns:
        Array of shape (minutes_ahead+1, n_paths) with price paths
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(minutes_ahead, n_paths))
    U = rng.uniform(size=(minutes_ahead, n_paths))  # For regime transitions
    
    # Initialize
    log_S = np.log(S0) * np.ones((minutes_ahead + 1, n_paths))
    regime = current_regime * np.ones((minutes_ahead + 1, n_paths), dtype=int)
    
    # Simulate
    for t in range(minutes_ahead):
        # Determine volatility based on current regime
        sigma = np.where(regime[t] == 0, sigma_low, sigma_high)
        
        # Update price
        drift = (mu_dt - 0.5 * sigma**2)
        log_S[t+1] = log_S[t] + drift + sigma * Z[t]
        
        # Transition to next regime
        # For each path, check if it transitions
        transition_prob = np.where(regime[t] == 0, p01, p10)
        transitions = U[t] < transition_prob
        regime[t+1] = np.where(transitions, 1 - regime[t], regime[t])
    
    S = np.exp(log_S)
    S[0] = S0
    return S


# ---------- Data loading ----------

def load_last_24h_prices(csv_path, price_column='Close'):
    """
    Load the last 24 hours of 1-minute price data from CSV.
    
    Args:
        csv_path: Path to the CSV file
        price_column: Column name for prices (default 'Close')
        
    Returns:
        Tuple of (prices_array, last_timestamp, last_price)
    """
    print(f"Loading data from {csv_path}...")
    
    # For large files, read only the last portion
    # Estimate: need ~2000 rows for 24h + buffer, read last 5000 to be safe
    try:
        # Count total lines first (approximate)
        with open(csv_path, 'r') as f:
            # Skip header
            next(f)
            # Count remaining lines (rough estimate)
            line_count = sum(1 for _ in f)
        
        # If file is large, skip most rows
        if line_count > 10000:
            skip_rows = max(0, line_count - 5000)
            print(f"Reading last ~5000 rows from {line_count:,} total rows...")
            df = pd.read_csv(csv_path, usecols=['Timestamp', price_column], 
                            skiprows=range(1, skip_rows + 1))  # +1 to skip header
        else:
            df = pd.read_csv(csv_path, usecols=['Timestamp', price_column])
    except Exception as e:
        # Fallback: read entire file
        print(f"Warning: Using fallback reading method ({e})")
        df = pd.read_csv(csv_path, usecols=['Timestamp', price_column])
    
    # Convert timestamp to numeric
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df = df.dropna()
    df = df.sort_values('Timestamp')
    
    # Get last 24 hours (1440 minutes + buffer)
    last_timestamp = df['Timestamp'].max()
    cutoff_timestamp = last_timestamp - (24 * 60 * 60)  # 24 hours in seconds
    
    df_24h = df[df['Timestamp'] >= cutoff_timestamp].copy()
    
    if len(df_24h) < 100:
        print(f"Warning: Only {len(df_24h)} data points found in last 24h. Using last 1440 points.")
        df_24h = df.tail(1440).copy()
    
    prices = df_24h[price_column].values
    last_price = float(prices[-1])
    last_ts = df_24h['Timestamp'].iloc[-1]
    
    print(f"Loaded {len(prices)} data points from last 24 hours")
    print(f"Last timestamp: {datetime.fromtimestamp(last_ts, tz=timezone.utc)}")
    print(f"Last price: ${last_price:,.2f}")
    
    return prices, last_ts, last_price


# ---------- Analysis and reporting ----------

def analyze_paths(S_paths, S0, horizon_hours, model_name="Model"):
    """
    Analyze Monte Carlo simulation results.
    
    Args:
        S_paths: Array of shape (minutes+1, n_paths) with price paths
        S0: Initial price
        horizon_hours: Forecast horizon in hours
        model_name: Name of the model for reporting
    """
    terminal_prices = S_paths[-1]
    
    # Basic statistics
    mean_terminal = np.mean(terminal_prices)
    median_terminal = np.median(terminal_prices)
    std_terminal = np.std(terminal_prices)
    
    # Quantiles
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    quants = np.quantile(terminal_prices, q)
    
    # Expected return
    expected_return = (mean_terminal / S0 - 1) * 100
    
    print(f"\n{'='*60}")
    print(f"{model_name} - {horizon_hours}h Forecast Results")
    print(f"{'='*60}")
    print(f"Initial Price: ${S0:,.2f}")
    print(f"\nTerminal Price Statistics:")
    print(f"  Mean:     ${mean_terminal:,.2f} ({expected_return:+.2f}%)")
    print(f"  Median:   ${median_terminal:,.2f}")
    print(f"  Std Dev:  ${std_terminal:,.2f}")
    print(f"\nQuantiles:")
    for q_val, price in zip(q, quants):
        pct_change = (price / S0 - 1) * 100
        print(f"  {q_val*100:4.1f}%: ${price:,.2f} ({pct_change:+.2f}%)")
    
    return {
        'mean': mean_terminal,
        'median': median_terminal,
        'std': std_terminal,
        'quantiles': dict(zip(q, quants)),
        'expected_return_pct': expected_return
    }


def compute_probabilities(S_paths, S0, strikes):
    """
    Compute probability of ending above various strike levels.
    
    Args:
        S_paths: Array of shape (minutes+1, n_paths) with price paths
        S0: Initial price
        strikes: List of strike prices or percentage changes
        
    Returns:
        Dictionary mapping strikes to probabilities
    """
    terminal_prices = S_paths[-1]
    results = {}
    
    for strike in strikes:
        if isinstance(strike, str) and strike.endswith('%'):
            # Percentage-based strike
            pct = float(strike.rstrip('%'))
            K = S0 * (1 + pct / 100)
        else:
            K = float(strike)
            pct = (K / S0 - 1) * 100
        
        prob = (terminal_prices > K).mean()
        results[strike] = {
            'strike_price': K,
            'strike_pct': pct,
            'probability': prob
        }
    
    return results


# ---------- Analytical probability calculations (no Monte Carlo) ----------

def compute_gbm_probability_analytical(S0, K, minutes_ahead, mu_dt, sigma_dt):
    """
    Compute probability P(S_T > K) analytically for GBM without Monte Carlo.
    
    For GBM: log(S_T) ~ Normal(log(S_0) + (μ - 0.5*σ²)*T, σ²*T)
    
    Args:
        S0: Initial price
        K: Strike/target price
        minutes_ahead: Number of minutes to forecast
        mu_dt: Per-minute drift
        sigma_dt: Per-minute volatility
        
    Returns:
        Probability P(S_T > K)
    """
    T = minutes_ahead
    # Mean and variance of log(S_T)
    log_mean = np.log(S0) + (mu_dt - 0.5 * sigma_dt**2) * T
    log_var = sigma_dt**2 * T
    log_std = np.sqrt(log_var)
    
    # P(S_T > K) = P(log(S_T) > log(K))
    # = 1 - Φ((log(K) - log_mean) / log_std)
    if log_std < 1e-10:
        return 1.0 if S0 > K else 0.0
    
    z_score = (np.log(K) - log_mean) / log_std
    prob_above = 1.0 - norm.cdf(z_score)
    
    return prob_above


def compute_gbm_statistics_analytical(S0, minutes_ahead, mu_dt, sigma_dt):
    """
    Compute terminal price statistics analytically for GBM.
    
    Args:
        S0: Initial price
        minutes_ahead: Number of minutes to forecast
        mu_dt: Per-minute drift
        sigma_dt: Per-minute volatility
        
    Returns:
        Dictionary with mean, median, std, and quantiles
    """
    T = minutes_ahead
    log_mean = np.log(S0) + (mu_dt - 0.5 * sigma_dt**2) * T
    log_var = sigma_dt**2 * T
    log_std = np.sqrt(log_var)
    
    # Mean: E[S_T] = S_0 * exp(μ*T)
    mean_price = S0 * np.exp(mu_dt * T)
    
    # Median: exp(log_mean) = S_0 * exp((μ - 0.5*σ²)*T)
    median_price = np.exp(log_mean)
    
    # Variance: Var[S_T] = S_0² * exp(2*μ*T) * (exp(σ²*T) - 1)
    variance = S0**2 * np.exp(2 * mu_dt * T) * (np.exp(log_var) - 1)
    std_price = np.sqrt(variance)
    
    # Quantiles from log-normal distribution
    q_values = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    quantiles = {}
    for q_val in q_values:
        z_q = norm.ppf(q_val)
        quantile_price = np.exp(log_mean + z_q * log_std)
        quantiles[q_val] = quantile_price
    
    return {
        'mean': mean_price,
        'median': median_price,
        'std': std_price,
        'quantiles': quantiles
    }


def compute_merton_probability_analytical(S0, K, minutes_ahead, mu_dt, sigma_dt, lam, mu_J, delta, n_terms=None):
    """
    Compute probability P(S_T > K) analytically for Merton jump-diffusion.
    
    Uses series expansion: P(S_T > K) = Σ P(N jumps) * P(S_T > K | N jumps)
    where each conditional probability is computed using GBM formula with adjusted parameters.
    
    Args:
        S0: Initial price
        K: Strike/target price
        minutes_ahead: Number of minutes to forecast
        mu_dt: Per-minute drift
        sigma_dt: Per-minute volatility
        lam: Jump intensity (expected jumps per minute)
        mu_J: Mean of jump size distribution
        delta: Std dev of jump size distribution
        n_terms: Number of terms in series expansion (default: 20)
        
    Returns:
        Probability P(S_T > K)
    """
    from scipy.stats import poisson
    
    T = minutes_ahead
    lambda_T = lam * T  # Expected number of jumps
    
    # Auto-determine n_terms based on lambda_T (cover 99.9% of probability mass)
    if n_terms is None:
        # Use quantile of Poisson distribution: P(N <= n_terms) >= 0.999
        n_terms = max(int(lambda_T + 5 * np.sqrt(lambda_T)), 50)  # At least 50 terms
    
    prob_above = 0.0
    
    # Sum over possible number of jumps
    for n in range(n_terms):
        # Probability of n jumps
        p_n = poisson.pmf(n, lambda_T)
        
        if p_n < 1e-10:
            continue
        
        # Adjusted parameters given n jumps
        # In Merton model with compensator: log(S_T/S_0) ~ Normal((μ - 0.5*σ² - λ*κ)*T + n*μ_J, σ²*T + n*δ²)
        # where κ = exp(μ_J + 0.5*δ²) - 1 is the compensator
        kappa = np.exp(mu_J + 0.5 * delta**2) - 1.0
        drift_component = (mu_dt - 0.5 * sigma_dt**2 - lam * kappa) * T
        jump_mean = n * mu_J  # Total jump contribution to mean
        log_mean = np.log(S0) + drift_component + jump_mean
        
        # Variance: diffusion variance + jump variance
        log_var = sigma_dt**2 * T + n * delta**2
        log_std = np.sqrt(log_var)
        
        if log_std < 1e-10:
            prob_n = 1.0 if S0 > K else 0.0
        else:
            z_score = (np.log(K) - log_mean) / log_std
            prob_n = 1.0 - norm.cdf(z_score)
        
        prob_above += p_n * prob_n
    
    return prob_above


def compute_merton_statistics_analytical(S0, minutes_ahead, mu_dt, sigma_dt, lam, mu_J, delta, n_terms=None):
    """
    Compute terminal price statistics analytically for Merton jump-diffusion.
    
    Args:
        S0: Initial price
        minutes_ahead: Number of minutes to forecast
        mu_dt: Per-minute drift
        sigma_dt: Per-minute volatility
        lam: Jump intensity
        mu_J: Mean of jump size distribution
        delta: Std dev of jump size distribution
        n_terms: Number of terms in series expansion
        
    Returns:
        Dictionary with mean, median, std, and quantiles
    """
    from scipy.stats import poisson
    
    T = minutes_ahead
    lambda_T = lam * T
    
    # Auto-determine n_terms based on lambda_T
    if n_terms is None:
        n_terms = max(int(lambda_T + 5 * np.sqrt(lambda_T)), 50)
    
    # Mean calculation: The simulation uses drift = mu_dt - 0.5*sigma_dt^2 - lam*kappa
    # where kappa = exp(mu_J + 0.5*delta^2) - 1 is the compensator
    # The log-price evolution: d log(S) = (mu_dt - 0.5*sigma_dt^2 - lam*kappa) dt + sigma_dt dW + jumps
    # E[log(S_T)] = log(S_0) + (mu_dt - 0.5*sigma_dt^2 - lam*kappa)*T + lambda_T*mu_J
    # Var[log(S_T)] = sigma_dt^2*T + lambda_T*delta^2
    # E[S_T] = exp(E[log(S_T)] + 0.5*Var[log(S_T)])
    #        = S_0 * exp(mu_dt*T - lam*kappa*T + lambda_T*mu_J + 0.5*(sigma_dt^2*T + lambda_T*delta^2))
    #        = S_0 * exp(mu_dt*T + lambda_T*(mu_J + 0.5*delta^2 - kappa))
    #        = S_0 * exp(mu_dt*T + lambda_T*(mu_J + 0.5*delta^2 - (exp(mu_J + 0.5*delta^2) - 1)))
    #        = S_0 * exp(mu_dt*T + lambda_T*(1 + mu_J + 0.5*delta^2 - exp(mu_J + 0.5*delta^2)))
    kappa = np.exp(mu_J + 0.5 * delta**2) - 1.0
    mean_price = S0 * np.exp(mu_dt * T + lambda_T * (1 + mu_J + 0.5 * delta**2 - np.exp(mu_J + 0.5 * delta**2)))
    
    # Compute variance properly: Var[S_T] = E[S_T²] - E[S_T]²
    E_S2_total = 0.0
    for n in range(n_terms):
        p_n = poisson.pmf(n, lambda_T)
        if p_n < 1e-10:
            continue
        kappa = np.exp(mu_J + 0.5 * delta**2) - 1.0
        drift_comp = (mu_dt - 0.5 * sigma_dt**2 - lam * kappa) * T
        jump_mean = n * mu_J
        log_mean_n = np.log(S0) + drift_comp + jump_mean
        log_var_n = sigma_dt**2 * T + n * delta**2
        # E[S_T² | n jumps] = exp(2*log_mean_n + 2*log_var_n)
        E_S2_n = np.exp(2 * log_mean_n + 2 * log_var_n)
        E_S2_total += p_n * E_S2_n
    
    variance_total = E_S2_total - mean_price**2
    std_price = np.sqrt(max(variance_total, 0))
    
    # For quantiles, use log-normal approximation with computed mean and variance
    if variance_total > 0 and mean_price > 0:
        log_mean_approx = np.log(mean_price) - 0.5 * np.log(1 + variance_total / mean_price**2)
        log_var_approx = np.log(1 + variance_total / mean_price**2)
        log_std_approx = np.sqrt(log_var_approx)
    else:
        log_mean_approx = np.log(mean_price)
        log_std_approx = 0.0
    
    q_values = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    quantiles = {}
    for q_val in q_values:
        z_q = norm.ppf(q_val)
        quantile_price = np.exp(log_mean_approx + z_q * log_std_approx)
        quantiles[q_val] = quantile_price
    
    # Median
    median_price = np.exp(log_mean_approx)
    
    return {
        'mean': mean_price,
        'median': median_price,
        'std': std_price,
        'quantiles': quantiles
    }


# ---------- Main execution ----------

def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo simulation for Bitcoin price prediction'
    )
    parser.add_argument(
        '--csv', 
        type=str, 
        default='upload/btcusd_1-min_data.csv',
        help='Path to CSV file with BTC price data'
    )
    parser.add_argument(
        '--hours', 
        type=int, 
        default=6,
        help='Forecast horizon in hours (default: 6)'
    )
    parser.add_argument(
        '--paths', 
        type=int, 
        default=20000,
        help='Number of Monte Carlo paths (default: 20000)'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['gbm', 'merton', 'heston', 'regime', 'all'],
        default='merton',
        help='Model to use: gbm, merton, heston (stochastic vol), regime (regime-switching), or all (default: merton)'
    )
    parser.add_argument(
        '--strikes', 
        type=str, 
        nargs='+',
        default=['-5%', '-2%', '0%', '+2%', '+5%'],
        help='Strike levels to compute probabilities (default: -5%% -2%% 0%% +2%% +5%%)'
    )
    parser.add_argument(
        '--jump-threshold', 
        type=float, 
        default=3.5,
        help='Jump detection threshold multiplier (default: 3.5)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--risk-neutral',
        action='store_true',
        help='Use risk-neutral drift (0) instead of historical mean. Assumes BTC is fairly priced.'
    )
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.0,
        help='Risk-free rate per minute (default: 0.0). Only used with --risk-neutral.'
    )
    
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found at {args.csv}")
        print("Please run the data scraping script first or provide a valid CSV path.")
        return 1
    
    prices, last_ts, S0 = load_last_24h_prices(args.csv)
    
    if len(prices) < 60:
        print(f"Error: Insufficient data. Need at least 60 data points, got {len(prices)}")
        return 1
    
    # Estimate parameters
    print(f"\n{'='*60}")
    print("Parameter Estimation")
    if args.risk_neutral:
        print("Using RISK-NEUTRAL measure (drift = risk-free rate)")
    else:
        print("Using REAL-WORLD measure (drift = historical mean)")
    print(f"{'='*60}")
    
    mu_dt, sigma_dt = estimate_gbm_params(
        prices, 
        risk_neutral=args.risk_neutral, 
        risk_free_rate=args.risk_free_rate
    )
    print(f"\nGBM Parameters (per minute):")
    if args.risk_neutral:
        print(f"  Drift (μ):     {mu_dt:.8f} (risk-neutral)")
    else:
        historical_mu = log_returns(prices).mean()
        print(f"  Drift (μ):     {mu_dt:.8f} (historical: {historical_mu:.8f})")
    print(f"  Volatility (σ): {sigma_dt:.8f}")
    print(f"  Annualized σ:  {sigma_dt * np.sqrt(525600):.4f}")
    
    # Estimate parameters for all models that might be needed
    models_to_run = ['merton'] if args.model == 'merton' else \
                    ['gbm'] if args.model == 'gbm' else \
                    ['gbm', 'merton'] if args.model == 'both' else \
                    ['heston'] if args.model == 'heston' else \
                    ['regime'] if args.model == 'regime' else \
                    ['gbm', 'merton', 'heston', 'regime']  # all
    
    merton_params = None
    heston_params = None
    regime_params = None
    
    if 'merton' in models_to_run:
        merton_params = estimate_merton_params(
            prices, 
            c=args.jump_threshold,
            risk_neutral=args.risk_neutral,
            risk_free_rate=args.risk_free_rate
        )
        print(f"\nMerton Jump-Diffusion Parameters:")
        if args.risk_neutral:
            print(f"  Drift (μ):     {merton_params['mu_dt']:.8f} (risk-neutral)")
        else:
            historical_mu = log_returns(prices).mean()
            print(f"  Drift (μ):     {merton_params['mu_dt']:.8f} (historical: {historical_mu:.8f})")
        print(f"  Volatility (σ): {merton_params['sigma_dt']:.8f}")
        print(f"  Jump intensity (λ): {merton_params['lam']:.6f} per minute")
        print(f"  Jump mean (μ_J):    {merton_params['mu_J']:.8f}")
        print(f"  Jump std (δ):       {merton_params['delta']:.8f}")
    
    if 'heston' in models_to_run:
        heston_params = estimate_heston_params(
            prices,
            risk_neutral=args.risk_neutral,
            risk_free_rate=args.risk_free_rate
        )
        print(f"\nHeston Stochastic Volatility Parameters:")
        if args.risk_neutral:
            print(f"  Drift (μ):     {heston_params['mu_dt']:.8f} (risk-neutral)")
        else:
            historical_mu = log_returns(prices).mean()
            print(f"  Drift (μ):     {heston_params['mu_dt']:.8f} (historical: {historical_mu:.8f})")
        print(f"  Initial variance (v0): {heston_params['v0']:.8f}")
        print(f"  Mean reversion (κ):    {heston_params['kappa']:.6f}")
        print(f"  Long-term variance (θ): {heston_params['theta']:.8f}")
        print(f"  Vol of vol (σ_v):      {heston_params['sigma_v']:.6f}")
        print(f"  Correlation (ρ):       {heston_params['rho']:.4f}")
    
    if 'regime' in models_to_run:
        regime_params = estimate_regime_switching_params(
            prices,
            risk_neutral=args.risk_neutral,
            risk_free_rate=args.risk_free_rate
        )
        print(f"\nRegime-Switching Parameters:")
        if args.risk_neutral:
            print(f"  Drift (μ):     {regime_params['mu_dt']:.8f} (risk-neutral)")
        else:
            historical_mu = log_returns(prices).mean()
            print(f"  Drift (μ):     {regime_params['mu_dt']:.8f} (historical: {historical_mu:.8f})")
        print(f"  Low vol (σ_low):  {regime_params['sigma_low']:.8f}")
        print(f"  High vol (σ_high): {regime_params['sigma_high']:.8f}")
        print(f"  Transition probabilities:")
        print(f"    P(low→low):  {regime_params['p00']:.4f}")
        print(f"    P(low→high):  {regime_params['p01']:.4f}")
        print(f"    P(high→low):  {regime_params['p10']:.4f}")
        print(f"    P(high→high): {regime_params['p11']:.4f}")
        print(f"  Current regime: {'HIGH VOL' if regime_params['current_regime'] == 1 else 'LOW VOL'}")
    
    # Simulate paths
    minutes_ahead = args.hours * 60
    seed_base = args.seed if args.seed is not None else 42
    seeds = {
        'gbm': seed_base,
        'merton': seed_base + 1,
        'heston': seed_base + 2,
        'regime': seed_base + 3
    }
    
    results = {}
    
    if 'gbm' in models_to_run:
        print(f"\n{'='*60}")
        print(f"Simulating {args.paths:,} GBM paths for {args.hours}h ahead...")
        print(f"{'='*60}")
        S_gbm = simulate_gbm_paths(
            S0, minutes_ahead, mu_dt, sigma_dt, 
            n_paths=args.paths, seed=seeds['gbm']
        )
        results['gbm'] = analyze_paths(S_gbm, S0, args.hours, "GBM")
        results['gbm']['paths'] = S_gbm
    
    if 'merton' in models_to_run:
        print(f"\n{'='*60}")
        print(f"Simulating {args.paths:,} Merton paths for {args.hours}h ahead...")
        print(f"{'='*60}")
        S_merton = simulate_merton_paths(
            S0, minutes_ahead,
            mu_dt=merton_params['mu_dt'],
            sigma_dt=merton_params['sigma_dt'],
            lam=merton_params['lam'],
            mu_J=merton_params['mu_J'],
            delta=merton_params['delta'],
            n_paths=args.paths,
            seed=seeds['merton']
        )
        results['merton'] = analyze_paths(S_merton, S0, args.hours, "Merton Jump-Diffusion")
        results['merton']['paths'] = S_merton
    
    if 'heston' in models_to_run:
        print(f"\n{'='*60}")
        print(f"Simulating {args.paths:,} Heston paths for {args.hours}h ahead...")
        print(f"{'='*60}")
        S_heston = simulate_heston_paths(
            S0, minutes_ahead,
            mu_dt=heston_params['mu_dt'],
            v0=heston_params['v0'],
            kappa=heston_params['kappa'],
            theta=heston_params['theta'],
            sigma_v=heston_params['sigma_v'],
            rho=heston_params['rho'],
            n_paths=args.paths,
            seed=seeds['heston']
        )
        results['heston'] = analyze_paths(S_heston, S0, args.hours, "Heston Stochastic Volatility")
        results['heston']['paths'] = S_heston
    
    if 'regime' in models_to_run:
        print(f"\n{'='*60}")
        print(f"Simulating {args.paths:,} Regime-Switching paths for {args.hours}h ahead...")
        print(f"{'='*60}")
        S_regime = simulate_regime_switching_paths(
            S0, minutes_ahead,
            mu_dt=regime_params['mu_dt'],
            sigma_low=regime_params['sigma_low'],
            sigma_high=regime_params['sigma_high'],
            p00=regime_params['p00'],
            p01=regime_params['p01'],
            p10=regime_params['p10'],
            p11=regime_params['p11'],
            current_regime=regime_params['current_regime'],
            n_paths=args.paths,
            seed=seeds['regime']
        )
        results['regime'] = analyze_paths(S_regime, S0, args.hours, "Regime-Switching")
        results['regime']['paths'] = S_regime
    
    # Compute strike probabilities
    print(f"\n{'='*60}")
    print("Strike Probabilities")
    print(f"{'='*60}")
    
    for model_name in results.keys():
        if 'paths' not in results[model_name]:
            continue
        
        probs = compute_probabilities(results[model_name]['paths'], S0, args.strikes)
        print(f"\n{model_name.upper()} - P(S_T > K):")
        for strike, data in probs.items():
            print(f"  K = ${data['strike_price']:,.2f} ({data['strike_pct']:+.2f}%): "
                  f"{data['probability']*100:.2f}%")
        results[model_name]['probabilities'] = probs
    
    # Compare models if multiple were run
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("Model Comparison")
        print(f"{'='*60}")
        for model_name in sorted(results.keys()):
            print(f"{model_name.upper():20} Expected Return: {results[model_name]['expected_return_pct']:+.2f}%")
            print(f"{'':20} Terminal Std Dev:  ${results[model_name]['std']:,.2f}")
    
    print(f"\n{'='*60}")
    print("Simulation Complete")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())

