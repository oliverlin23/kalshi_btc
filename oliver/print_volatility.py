#!/usr/bin/env python3
"""
Print current BTC volatility estimate.

This script initializes the price queue, fetches current data,
and calculates the volatility using the same method as the trading algorithm.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from oliver.config import DATA_STEP_SECONDS, PREDICTION_MODEL, PREDICTION_EXPONENTIAL_DECAY
from oliver.data import initialize_price_queue_from_bitstamp, get_price_data_for_prediction
from oliver.data.price_queue import jump_detection_price_queue, jump_detection_queue_lock
from oliver.trading.prediction import get_current_btc_price_estimate
import numpy as np

try:
    from predict_price_probability import predict_price_probability
except ImportError:
    from oliver.predict_price_probability import predict_price_probability


def main():
    """Print current BTC volatility."""
    print("=" * 60)
    print("BTC Volatility Calculator")
    print("=" * 60)
    
    # Initialize price queue
    print("\nInitializing price data queue from Bitstamp...")
    try:
        initialize_price_queue_from_bitstamp()
        print("✓ Price queue initialized successfully")
    except Exception as e:
        print(f"✗ ERROR: Failed to initialize price queue: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get current BTC price
    try:
        current_price = get_current_btc_price_estimate()
        print(f"\nCurrent BTC Price: ${current_price:,}")
    except Exception as e:
        print(f"\n⚠️  Could not fetch current BTC price: {e}")
        current_price = None
    
    # Get price data
    try:
        prices, last_ts, S0_queue = get_price_data_for_prediction()
        print(f"\nPrice data: {len(prices)} data points")
        print(f"  Last timestamp: {last_ts}")
        print(f"  Last price: ${S0_queue:,.2f}")
        
        if current_price is None:
            current_price = S0_queue
        
        # Get jump detection queue data if available
        jump_detection_prices = None
        jump_detection_last_ts = None
        with jump_detection_queue_lock:
            if len(jump_detection_price_queue) >= 30:
                jump_detection_queue_items = list(jump_detection_price_queue)
                jump_detection_prices = np.array([price for _, price in jump_detection_queue_items], dtype=np.float64)
                jump_detection_last_ts = jump_detection_queue_items[-1][0] if jump_detection_queue_items else None
                print(f"\nJump detection queue: {len(jump_detection_prices)} data points (per-second, max 300)")
            else:
                print(f"\nJump detection queue: {len(jump_detection_price_queue)} data points (need at least 30)")
        
        # Allow model override via environment variable
        model = os.environ.get('PREDICTION_MODEL', PREDICTION_MODEL)
        
        # Calculate volatility using the same method as trading algorithm
        print(f"\nCalculating volatility using {model.upper()} model...")
        print(f"  Using exponential decay: {PREDICTION_EXPONENTIAL_DECAY}")
        print(f"  Data step: {DATA_STEP_SECONDS} seconds")
        
        # Use a dummy target price and time horizon (we just want the volatility)
        target_price = current_price * 1.01  # 1% above current
        hours_ahead = 1.0  # 1 hour ahead
        
        result = predict_price_probability(
            csv_path=None,
            target_price=target_price,
            hours_ahead=hours_ahead,
            model=model,
            exponential_decay=PREDICTION_EXPONENTIAL_DECAY,
            prices_data=prices,
            last_timestamp=last_ts,
            current_price=current_price,
            data_step_seconds=DATA_STEP_SECONDS,
            one_minute_prices_data=jump_detection_prices,
            one_minute_last_timestamp=jump_detection_last_ts,
            verbose=True
        )
        
        # Extract volatility
        params = result.get('parameters', {})
        volatility_per_minute = params.get('sigma_dt', None)
        
        if volatility_per_minute is not None:
            # Convert to annualized volatility for display
            # Per-minute volatility * sqrt(minutes_per_year)
            minutes_per_year = 365.25 * 24 * 60
            annualized_vol = volatility_per_minute * np.sqrt(minutes_per_year)
            
            print("\n" + "=" * 60)
            print("VOLATILITY ESTIMATES")
            print("=" * 60)
            print(f"Per-minute volatility (σ_dt): {volatility_per_minute:.8f}")
            print(f"Annualized volatility:         {annualized_vol:.4f} ({annualized_vol*100:.2f}%)")
            
            # Also show other parameters if available
            if model == 'merton':
                lam = params.get('lam', None)
                mu_J = params.get('mu_J', None)
                delta = params.get('delta', None)
                if lam is not None:
                    print(f"\nJump Parameters:")
                    print(f"  Jump intensity (λ): {lam:.6f} per minute")
                    print(f"  Jump mean (μ_J):     {mu_J:.8f}")
                    print(f"  Jump std (δ):        {delta:.8f}")
            
            mu_dt = params.get('mu_dt', None)
            if mu_dt is not None:
                print(f"\nDrift (μ_dt): {mu_dt:.8f} per minute")
            
            print("=" * 60)
        else:
            print("\n✗ ERROR: Could not extract volatility from prediction result")
            print(f"Result keys: {list(result.keys())}")
            print(f"Parameters: {params}")
            
    except Exception as e:
        print(f"\n✗ ERROR: Failed to calculate volatility: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

