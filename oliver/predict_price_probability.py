"""
Predict probability of BTC price being above a target price at a specific time.

Uses risk-neutral Merton jump-diffusion model.
"""

import os
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from monte_carlo_btc import (
    load_last_24h_prices,
    estimate_gbm_params,
    estimate_merton_params,
    compute_gbm_probability_analytical,
    compute_gbm_statistics_analytical,
    compute_merton_probability_analytical,
    compute_merton_statistics_analytical
)

# Import update functions
from kaggle_update_bitcoin import (
    download_latest_dataset,
    download_latest_metadata,
    check_missing_data,
    fetch_and_append_missing_data
)

# Cache for loaded data and parameter estimates
_data_cache = {
    'csv_path': None,
    'csv_mtime': None,
    'prices': None,
    'last_ts': None,
    'S0': None,
    'params_cache': {}  # Cache parameter estimates per timeframe/model/weights
}


def update_dataset(csv_path, skip_update=False):
    """
    Update the BTC dataset from Kaggle and Bitstamp before making predictions.
    
    Args:
        csv_path: Path to CSV file
        skip_update: If True, skip the update step
        
    Returns:
        True if update was successful or skipped, False if failed
    """
    if skip_update:
        print("Skipping dataset update (--skip-update flag used)")
        return True
    
    print(f"{'='*80}")
    print("UPDATING DATASET")
    print(f"{'='*80}\n")
    print(f"Current time (UTC): {datetime.now(timezone.utc)}")
    
    dataset_slug = "mczielinski/bitcoin-historical-data"
    currency_pair = "btcusd"
    upload_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else "upload"
    
    # Ensure the upload directory exists
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    existing_data_filename = csv_path
    output_filename = csv_path
    
    try:
        # Step 1: Download the latest dataset and metadata from Kaggle
        print("\nDownloading dataset metadata from Kaggle...")
        download_latest_metadata(dataset_slug)
        
        print("Downloading dataset from Kaggle...")
        download_latest_dataset(dataset_slug)
        
        # Step 2: Check for missing data (only if file exists)
        if os.path.exists(existing_data_filename):
            print("Checking for missing data...")
            last_timestamp, current_timestamp = check_missing_data(existing_data_filename)
            
            # Step 3: Fetch and append missing data
            if last_timestamp is not None and current_timestamp is not None:
                print("Missing data detected. Fetching from Bitstamp API...")
                fetch_and_append_missing_data(
                    currency_pair, last_timestamp, current_timestamp, 
                    existing_data_filename, output_filename
                )
                print("Dataset updated successfully!")
            else:
                print("Dataset is already up to date.")
        else:
            print("Dataset downloaded from Kaggle. File created.")
        
        return True
        
    except Exception as e:
        print(f"\nWarning: Dataset update failed: {e}")
        print("Continuing with existing dataset (if available)...")
        return False


def extract_prices_for_window(prices_array, last_timestamp, minutes, data_step_seconds=60):
    """
    Extract prices from the last N minutes of a price array.
    
    Args:
        prices_array: Array of prices (most recent last)
        last_timestamp: Last timestamp in the array
        minutes: Number of minutes to extract
        data_step_seconds: Time interval between data points in seconds (default: 60 for 1-minute)
    
    Returns:
        Array of prices from the last N minutes
    """
    # Calculate how many data points represent N minutes
    # For per-second data: 60 points per minute
    # For per-5-second data: 12 points per minute
    # For per-minute data: 1 point per minute
    points_per_minute = 60 // data_step_seconds
    n_points = min(len(prices_array), minutes * points_per_minute)
    return prices_array[-n_points:]


def _load_and_cache_data(csv_path):
    """
    Load price data with caching based on file modification time.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Tuple of (prices, last_ts, S0)
    """
    global _data_cache
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Get file modification time
    csv_mtime = os.path.getmtime(csv_path)
    
    # Check if cache is valid
    if (_data_cache['csv_path'] == csv_path and 
        _data_cache['csv_mtime'] == csv_mtime and 
        _data_cache['prices'] is not None):
        # Cache hit - return cached data
        return _data_cache['prices'], _data_cache['last_ts'], _data_cache['S0']
    
    # Cache miss - load data
    prices, last_ts, S0 = load_last_24h_prices(csv_path)
    
    # Update cache
    _data_cache['csv_path'] = csv_path
    _data_cache['csv_mtime'] = csv_mtime
    _data_cache['prices'] = prices
    _data_cache['last_ts'] = last_ts
    _data_cache['S0'] = S0
    _data_cache['params_cache'] = {}  # Clear parameter cache when data changes
    
    return prices, last_ts, S0


def clear_parameter_cache():
    """
    Clear the parameter cache to force recalculation of volatility and other parameters.
    Should be called whenever price data is updated (e.g., when queue updates).
    """
    global _data_cache
    _data_cache['params_cache'] = {}


def predict_price_probability(csv_path=None, target_price=None, hours_ahead=None, 
                             lookback_hours=24, model='merton',
                             weight_5m=0.4, weight_15m=0.3, weight_1h=0.2, weight_6h=0.1,
                             verbose=False, prices_data=None, last_timestamp=None, current_price=None,
                             data_step_seconds=60, one_minute_prices_data=None, one_minute_last_timestamp=None):
    """
    Predict probability of BTC price being above target_price in hours_ahead.
    Uses analytical methods only (instant results, no Monte Carlo).
    
    Parameters are estimated using weighted averages from multiple timeframes:
    - Last 5 minutes (weight: configurable, default 40%)
    - Last 15 minutes (weight: configurable, default 30%)
    - Last 1 hour (weight: configurable, default 20%)
    - Last 6 hours (weight: configurable, default 10%)
    
    More recent data is weighted more heavily to capture current market conditions.
    
    Args:
        csv_path: Path to CSV file with BTC data (optional if prices_data provided)
        target_price: Target price to check (e.g., 102500)
        hours_ahead: Hours into the future
        lookback_hours: Hours of historical data to load (default: 24, must be >= 6)
        model: Model to use ('gbm' or 'merton')
        weight_5m: Weight for 5-minute timeframe (default: 0.4)
        weight_15m: Weight for 15-minute timeframe (default: 0.3)
        weight_1h: Weight for 1-hour timeframe (default: 0.2)
        weight_6h: Weight for 6-hour timeframe (default: 0.1)
        verbose: If True, print detailed output (default: False)
        prices_data: Optional array of prices (if provided, csv_path is ignored)
        last_timestamp: Optional last timestamp (required if prices_data provided)
        current_price: Optional current price (required if prices_data provided)
        data_step_seconds: Time interval between data points in seconds (default: 60 for 1-minute)
        one_minute_prices_data: Optional array of prices from 1-minute queue (per-second updates)
        one_minute_last_timestamp: Optional last timestamp for 1-minute queue
    
    Returns:
        Dictionary with probability and statistics
    """
    if verbose:
        print(f"{'='*80}")
        print("BTC PRICE PROBABILITY PREDICTION")
        print(f"{'='*80}\n")
    
    # Load current price and historical data
    if prices_data is not None and last_timestamp is not None and current_price is not None:
        # Use provided data (from queue) - already numpy array from optimized get_price_data_for_prediction
        if isinstance(prices_data, np.ndarray):
            prices = prices_data
        else:
            prices = np.array(prices_data, dtype=np.float64)
        last_ts = float(last_timestamp)
        S0 = float(current_price)
    elif csv_path is not None:
        # Load from CSV (cached)
        prices, last_ts, S0 = _load_and_cache_data(csv_path)
    else:
        raise ValueError("Must provide either csv_path or (prices_data, last_timestamp, current_price)")
    
    if verbose:
        print(f"\nCurrent BTC Price: ${S0:,.2f}")
        print(f"Target Price: ${target_price:,.2f}")
        print(f"Time Horizon: {hours_ahead} hours")
        print(f"Required Change: {((target_price / S0) - 1) * 100:+.2f}%")
    
    # Normalize weights to sum to 1.0
    total_weight = weight_5m + weight_15m + weight_1h + weight_6h
    if total_weight > 0:
        weight_5m_norm = weight_5m / total_weight
        weight_15m_norm = weight_15m / total_weight
        weight_1h_norm = weight_1h / total_weight
        weight_6h_norm = weight_6h / total_weight
    else:
        # Fallback to equal weights if all zero
        weight_5m_norm = weight_15m_norm = weight_1h_norm = weight_6h_norm = 1.0 / 4.0
    
    # Estimate parameters using weighted multi-timeframe approach
    if verbose:
        print(f"\n{'='*80}")
        print(f"ESTIMATING PARAMETERS (Risk-Neutral {model.upper()})")
        print(f"{'='*80}")
        print(f"Using weighted estimates from multiple timeframes:")
        if one_minute_prices_data is not None and len(one_minute_prices_data) >= 30:
            print(f"  - 1-minute queue (weight: 0.50) - highest priority")
            print(f"  - Last 5 minutes (weight: {weight_5m_norm * 0.5:.2f})")
            print(f"  - Last 15 minutes (weight: {weight_15m_norm * 0.5:.2f})")
            print(f"  - Last 1 hour (weight: {weight_1h_norm * 0.5:.2f})")
            print(f"  - Last 6 hours (weight: {weight_6h_norm * 0.5:.2f})")
        else:
            print(f"  - Last 5 minutes (weight: {weight_5m_norm:.2f})")
            print(f"  - Last 15 minutes (weight: {weight_15m_norm:.2f})")
            print(f"  - Last 1 hour (weight: {weight_1h_norm:.2f})")
            print(f"  - Last 6 hours (weight: {weight_6h_norm:.2f})")
    
    minutes_ahead = int(hours_ahead * 60)
    
    # Extract prices for each timeframe
    last_timestamp = last_ts
    prices_5m = extract_prices_for_window(prices, last_timestamp, minutes=5, data_step_seconds=data_step_seconds)
    prices_15m = extract_prices_for_window(prices, last_timestamp, minutes=15, data_step_seconds=data_step_seconds)
    prices_1h = extract_prices_for_window(prices, last_timestamp, minutes=60, data_step_seconds=data_step_seconds)
    prices_6h = extract_prices_for_window(prices, last_timestamp, minutes=360, data_step_seconds=data_step_seconds)
    
    if verbose:
        print(f"\nData points per timeframe:")
        if one_minute_prices_data is not None:
            print(f"  1-minute queue:  {len(one_minute_prices_data)} points (per-second)")
        print(f"  Last 5 minutes:  {len(prices_5m)} points")
        print(f"  Last 15 minutes: {len(prices_15m)} points")
        print(f"  Last 1 hour:     {len(prices_1h)} points")
        print(f"  Last 6 hours:    {len(prices_6h)} points")
        print(f"\n{'='*80}")
        print("USING ANALYTICAL METHOD (Instant Results)")
        print(f"{'='*80}")
    
    # Create cache key for parameter estimates
    # Use last timestamp as part of key - when queue updates, timestamp changes, so we recalculate
    # This ensures we cache parameters between queue updates (every 5 seconds) but recalculate
    # when new data arrives, which is exactly what we want
    # Include 1-minute queue timestamp if available
    one_min_ts_key = one_minute_last_timestamp if one_minute_prices_data is not None and len(one_minute_prices_data) >= 30 else None
    cache_key = (model, weight_5m_norm, weight_15m_norm, weight_1h_norm, weight_6h_norm, last_timestamp, one_min_ts_key)
    
    # Check parameter cache
    if cache_key in _data_cache['params_cache']:
        model_params = _data_cache['params_cache'][cache_key]
        if verbose:
            print(f"\nUsing cached parameter estimates")
    else:
        # Estimate parameters (not cached)
        if model == 'gbm':
            # Estimate GBM parameters from each timeframe
            params_5m = estimate_gbm_params(prices_5m, risk_neutral=True, drift_factor=0.0, time_period_seconds=data_step_seconds) if len(prices_5m) >= 5 else None
            params_15m = estimate_gbm_params(prices_15m, risk_neutral=True, drift_factor=0.0, time_period_seconds=data_step_seconds) if len(prices_15m) >= 10 else None
            params_1h = estimate_gbm_params(prices_1h, risk_neutral=True, drift_factor=0.0, time_period_seconds=data_step_seconds) if len(prices_1h) >= 30 else None
            params_6h = estimate_gbm_params(prices_6h, risk_neutral=True, drift_factor=0.0, time_period_seconds=data_step_seconds) if len(prices_6h) >= 100 else None
            
            # Weight and combine (more recent = higher weight)
            weights = [weight_5m_norm, weight_15m_norm, weight_1h_norm, weight_6h_norm]  # 5m, 15m, 1h, 6h
            params_list = [params_5m, params_15m, params_1h, params_6h]
            
            # Filter out None values and normalize weights
            valid_params = [(p, w) for p, w in zip(params_list, weights) if p is not None]
            if not valid_params:
                raise ValueError("Insufficient data for parameter estimation")
            
            # Normalize weights (in case some timeframes had insufficient data)
            total_weight = sum(w for _, w in valid_params)
            normalized_params = [(p, w / total_weight) for p, w in valid_params]
            
            # Weighted average
            mu_dt = sum(p[0] * w for p, w in normalized_params)
            sigma_dt = sum(p[1] * w for p, w in normalized_params)
            
            model_params = {'mu_dt': mu_dt, 'sigma_dt': sigma_dt}
            
            if verbose:
                print(f"\nWeighted Parameters (per minute):")
                print(f"  Drift (μ):     {mu_dt:.8f} (risk-neutral = 0)")
                print(f"  Volatility (σ): {sigma_dt:.8f}")
                print(f"\nContributions:")
                timeframe_names = ["5m", "15m", "1h", "6h"]
                for p, w in normalized_params:
                    idx = params_list.index(p)
                    timeframe = timeframe_names[idx]
                    print(f"  {timeframe}: σ={p[1]:.8f} (weight: {w:.2%})")
            
            # Cache parameters
            _data_cache['params_cache'][cache_key] = model_params
            
        elif model == 'merton':
            # Estimate Merton parameters from each timeframe
            params_5m = estimate_merton_params(prices_5m, risk_neutral=True, drift_factor=0.0, time_period_seconds=data_step_seconds) if len(prices_5m) >= 5 else None
            params_15m = estimate_merton_params(prices_15m, risk_neutral=True, drift_factor=0.0, time_period_seconds=data_step_seconds) if len(prices_15m) >= 10 else None
            params_1h = estimate_merton_params(prices_1h, risk_neutral=True, drift_factor=0.0, time_period_seconds=data_step_seconds) if len(prices_1h) >= 30 else None
            params_6h = estimate_merton_params(prices_6h, risk_neutral=True, drift_factor=0.0, time_period_seconds=data_step_seconds) if len(prices_6h) >= 100 else None
            
            # Estimate from 1-minute queue if available (per-second data)
            params_1min = None
            weight_1min_norm = 0.0
            if one_minute_prices_data is not None and len(one_minute_prices_data) >= 30:
                # 1-minute queue has per-second data, so time_period_seconds=1
                params_1min = estimate_merton_params(one_minute_prices_data, risk_neutral=True, drift_factor=0.0, time_period_seconds=1)
                # Give 1-minute queue highest weight (50% of total), redistribute others proportionally
                weight_1min_raw = 0.5
                remaining_weight = 1.0 - weight_1min_raw
                # Redistribute existing weights proportionally
                weight_5m_norm_adj = weight_5m_norm * remaining_weight
                weight_15m_norm_adj = weight_15m_norm * remaining_weight
                weight_1h_norm_adj = weight_1h_norm * remaining_weight
                weight_6h_norm_adj = weight_6h_norm * remaining_weight
                weight_1min_norm = weight_1min_raw
            else:
                # No 1-minute queue data, use original weights
                weight_5m_norm_adj = weight_5m_norm
                weight_15m_norm_adj = weight_15m_norm
                weight_1h_norm_adj = weight_1h_norm
                weight_6h_norm_adj = weight_6h_norm
            
            # Weight and combine (more recent = higher weight)
            if params_1min is not None:
                weights = [weight_1min_norm, weight_5m_norm_adj, weight_15m_norm_adj, weight_1h_norm_adj, weight_6h_norm_adj]  # 1min, 5m, 15m, 1h, 6h
                params_list = [params_1min, params_5m, params_15m, params_1h, params_6h]
            else:
                weights = [weight_5m_norm, weight_15m_norm, weight_1h_norm, weight_6h_norm]  # 5m, 15m, 1h, 6h
                params_list = [params_5m, params_15m, params_1h, params_6h]
            
            # Filter out None values and normalize weights
            valid_params = [(p, w) for p, w in zip(params_list, weights) if p is not None]
            if not valid_params:
                raise ValueError("Insufficient data for parameter estimation")
            
            # Normalize weights (in case some timeframes had insufficient data)
            total_weight = sum(w for _, w in valid_params)
            normalized_params = [(p, w / total_weight) for p, w in valid_params]
            
            # Weighted average for each parameter
            mu_dt = sum(p['mu_dt'] * w for p, w in normalized_params)
            sigma_dt = sum(p['sigma_dt'] * w for p, w in normalized_params)
            lam = sum(p['lam'] * w for p, w in normalized_params)
            mu_J = sum(p['mu_J'] * w for p, w in normalized_params)
            delta = sum(p['delta'] * w for p, w in normalized_params)
            
            model_params = {
                'mu_dt': mu_dt,
                'sigma_dt': sigma_dt,
                'lam': lam,
                'mu_J': mu_J,
                'delta': delta
            }
            
            if verbose:
                print(f"\nWeighted Parameters (per minute):")
                print(f"  Drift (μ):     {mu_dt:.8f} (risk-neutral = 0)")
                print(f"  Volatility (σ): {sigma_dt:.8f}")
                print(f"  Jump intensity (λ): {lam:.6f} per minute")
                print(f"  Jump mean (μ_J):    {mu_J:.8f}")
                print(f"  Jump std (δ):       {delta:.8f}")
                print(f"\nContributions:")
                if params_1min is not None:
                    timeframe_names = ["1min", "5m", "15m", "1h", "6h"]
                else:
                    timeframe_names = ["5m", "15m", "1h", "6h"]
                for p, w in normalized_params:
                    idx = params_list.index(p)
                    timeframe = timeframe_names[idx]
                    print(f"  {timeframe}: σ={p['sigma_dt']:.8f}, λ={p['lam']:.6f} (weight: {w:.2%})")
            
            # Cache parameters
            _data_cache['params_cache'][cache_key] = model_params
        else:
            raise ValueError(f"Unknown model: {model}. Use 'gbm' or 'merton'.")
    
    # Always compute new predictions (never cache these)
    if model == 'gbm':
        mu_dt = model_params['mu_dt']
        sigma_dt = model_params['sigma_dt']
        
        # Compute probability analytically
        prob_above = compute_gbm_probability_analytical(
            S0, target_price, minutes_ahead, mu_dt, sigma_dt
        )
        
        # Compute statistics analytically
        stats = compute_gbm_statistics_analytical(
            S0, minutes_ahead, mu_dt, sigma_dt
        )
        
    elif model == 'merton':
        mu_dt = model_params['mu_dt']
        sigma_dt = model_params['sigma_dt']
        lam = model_params['lam']
        mu_J = model_params['mu_J']
        delta = model_params['delta']
        
        # Compute probability analytically
        prob_above = compute_merton_probability_analytical(
            S0, target_price, minutes_ahead,
            mu_dt, sigma_dt, lam, mu_J, delta
        )
        
        # Compute statistics analytically
        stats = compute_merton_statistics_analytical(
            S0, minutes_ahead, mu_dt, sigma_dt, lam, mu_J, delta
        )
    
    prob_below = 1 - prob_above
    mean_price = stats['mean']
    median_price = stats['median']
    std_price = stats['std']
    quantiles = stats['quantiles']
    expected_return = (mean_price / S0 - 1) * 100
    
    # Results (only print if verbose)
    if verbose:
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"\nProbability of price > ${target_price:,.2f}: {prob_above*100:.2f}%")
        print(f"Probability of price ≤ ${target_price:,.2f}: {prob_below*100:.2f}%")
        
        print(f"\nTerminal Price Statistics:")
        print(f"  Mean:     ${mean_price:,.2f} ({expected_return:+.2f}%)")
        print(f"  Median:   ${median_price:,.2f}")
        print(f"  Std Dev:  ${std_price:,.2f}")
        
        print(f"\nPrice Quantiles:")
        for q_val, price in quantiles.items():
            pct_change = (price / S0 - 1) * 100
            print(f"  {q_val*100:4.1f}%: ${price:,.2f} ({pct_change:+.2f}%)")
    
    return {
        'current_price': S0,
        'target_price': target_price,
        'hours_ahead': hours_ahead,
        'probability_above': prob_above,
        'probability_below': prob_below,
        'mean_terminal': mean_price,
        'median_terminal': median_price,
        'std_terminal': std_price,
        'quantiles': quantiles,
        'parameters': model_params,
        'model': model,
        'method': 'analytical'
    }


def main():
    parser = argparse.ArgumentParser(
        description='Predict probability of BTC price being above target price. '
                    'Automatically updates dataset from Kaggle and Bitstamp before prediction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict probability using Merton model (default, analytical)
  python predict_price_probability.py --target 102500 --hours 26.5
  
  # Use GBM model (analytical)
  python predict_price_probability.py --target 102500 --hours 26.5 --model gbm
  
  # Skip dataset update (use existing CSV)
  python predict_price_probability.py --target 102500 --hours 26.5 --skip-update
        """
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='upload/btcusd_1-min_data.csv',
        help='Path to CSV file with BTC data (default: upload/btcusd_1-min_data.csv)'
    )
    parser.add_argument(
        '--target',
        type=float,
        required=True,
        help='Target price to check (e.g., 102500)'
    )
    parser.add_argument(
        '--hours',
        type=float,
        required=True,
        help='Hours into the future (e.g., 26.5)'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=24,
        help='Hours of historical data for parameter estimation (default: 24)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['gbm', 'merton'],
        default='merton',
        help='Model to use: gbm (geometric brownian motion) or merton (jump-diffusion) (default: merton)'
    )
    parser.add_argument(
        '--skip-update',
        action='store_true',
        help='Skip dataset update and use existing CSV file'
    )
    parser.add_argument(
        '--weight-5m',
        type=float,
        default=0.4,
        help='Weight for 5-minute timeframe (default: 0.4)'
    )
    parser.add_argument(
        '--weight-15m',
        type=float,
        default=0.3,
        help='Weight for 15-minute timeframe (default: 0.3)'
    )
    parser.add_argument(
        '--weight-1h',
        type=float,
        default=0.2,
        help='Weight for 1-hour timeframe (default: 0.2)'
    )
    parser.add_argument(
        '--weight-6h',
        type=float,
        default=0.1,
        help='Weight for 6-hour timeframe (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Update dataset first (unless skipped)
    if not args.skip_update:
        update_dataset(args.csv, skip_update=False)
    
    # Check if CSV exists
    if not os.path.exists(args.csv):
        print(f"\nError: CSV file not found at {args.csv}")
        print("Please run the data scraping script first or provide a valid CSV path.")
        return 1
    
    # Run prediction (verbose=True for CLI usage)
    results = predict_price_probability(
        args.csv,
        args.target,
        args.hours,
        lookback_hours=args.lookback,
        model=args.model,
        weight_5m=args.weight_5m,
        weight_15m=args.weight_15m,
        weight_1h=args.weight_1h,
        weight_6h=args.weight_6h,
        verbose=True
    )
    
    print(f"\n{'='*80}")
    print("PREDICTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nFinal Answer:")
    print(f"  Probability of BTC > ${args.target:,.2f} in {args.hours} hours: {results['probability_above']*100:.2f}%")
    
    return 0


if __name__ == "__main__":
    exit(main())

