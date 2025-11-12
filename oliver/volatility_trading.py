"""
Volatility trading algorithm for Kalshi BTC markets.

Entry point for the trading system. Initializes components and runs trading cycles.
"""

import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from oliver.config import (
    BASE_SPREAD_CENTS,
    MAX_SPREAD_CENTS,
    VOLATILITY_MULTIPLIER,
    VOLUME,
    USE_KELLY_SIZING,
    KELLY_FRACTION,
    MAX_POSITION_PCT_OF_BALANCE,
    CYCLE_INTERVAL_SECONDS,
    PREDICTION_MODEL,
    DATA_STEP_SECONDS,
    DATA_HOURS_BACK,
    LOGS_DIR
)
from oliver.data import initialize_price_queue_from_bitstamp
from oliver.loggers.trading_logger import init_log_files
from oliver.trading.hourly_maintenance import handle_hour_change
from oliver.trading.prediction import get_current_btc_price_estimate
from oliver.trading.cycle import run_trading_cycle


def main():
    """Main trading algorithm - runs continuously."""
    init_log_files()
    
    print("=" * 60)
    print("Volatility Trading Algorithm - Continuous Mode")
    print("=" * 60)
    print(f"Dynamic Spread: {BASE_SPREAD_CENTS}-{MAX_SPREAD_CENTS} cents (based on volatility)")
    print(f"  Base: {BASE_SPREAD_CENTS}c, Multiplier: {VOLATILITY_MULTIPLIER}x volatility")
    if USE_KELLY_SIZING:
        print(f"Position Sizing: Kelly criterion ({KELLY_FRACTION*100:.0f}% Kelly, max {MAX_POSITION_PCT_OF_BALANCE*100:.0f}% of balance)")
    else:
        print(f"Position Sizing: Fixed volume ({VOLUME} contracts per side)")
    print(f"Interval: {CYCLE_INTERVAL_SECONDS}s")
    print(f"Prediction Model: {PREDICTION_MODEL}")
    print(f"Data Source: Bitstamp API (no CSV)")
    print(f"Data Frequency: {DATA_STEP_SECONDS}-second intervals")
    print(f"Historical Data: {DATA_HOURS_BACK} hours")
    print(f"Logs directory: {LOGS_DIR}")
    print(f"Log files rotate hourly by date (YYYY-MM-DD) and hour (HH)")
    print("=" * 60)
    
    print("\nInitializing price data queue from Bitstamp...")
    try:
        initialize_price_queue_from_bitstamp()
        print("✓ Price queue initialized successfully")
    except Exception as e:
        print(f"✗ ERROR: Failed to initialize price queue: {e}")
        import traceback
        traceback.print_exc()
        print("\nCannot start trading without price data. Exiting.")
        return
    
    btc_price = get_current_btc_price_estimate()
    if btc_price:
        print(f"Current BTC Price: ${btc_price:,}")
    else:
        print(f"⚠️  Could not fetch BTC price - will continue anyway")
    
    print("\nInitializing hour change logic...")
    handle_hour_change()
    
    print("\nStarting trading cycles...")
    print("Press Ctrl-C to stop\n")
    
    cycle_num = 1
    
    try:
        while True:
            run_trading_cycle(cycle_num)
            time.sleep(CYCLE_INTERVAL_SECONDS)
            cycle_num += 1
            
    except KeyboardInterrupt:
        print(f"\n\nStopped after {cycle_num - 1} cycle(s)")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
