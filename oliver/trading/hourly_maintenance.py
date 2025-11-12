"""
Hourly maintenance tasks for trading system.

Handles hour changes, balance floor updates, price queue reconciliation, and PnL calculations.
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple

from ..config import EST, BALANCE_FLOOR_BUFFER
from ..data.price_queue import price_data_queue, price_queue_lock
from ..trading.pnl_tracker import calculate_pnl_for_resolved_markets

# Import clear_parameter_cache from predict_price_probability
try:
    from predict_price_probability import clear_parameter_cache
except ImportError:
    from oliver.predict_price_probability import clear_parameter_cache
from ..loggers.trading_logger import update_log_files_for_current_hour

balance_floor: Optional[float] = None
current_hour_key: Optional[Tuple[int, int, int, int]] = None


def get_current_est_hour():
    """Get the NEXT top-of-hour in EST (ceil to next hour)."""
    est_now = datetime.now(EST)
    if est_now.minute == 0 and est_now.second == 0:
        target = est_now
    else:
        target = (est_now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    return target.year, target.month, target.day, target.hour


def get_previous_est_hour():
    """Get the PREVIOUS top-of-hour in EST (the hour that just resolved)."""
    est_now = datetime.now(EST)
    target = est_now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    return target.year, target.month, target.day, target.hour


def _update_balance_floor() -> Optional[float]:
    """Update the balance floor based on current available balance."""
    global balance_floor
    
    try:
        from app.utils import get_available_funds
        available_balance = get_available_funds()
        
        if available_balance is not None and available_balance > 0:
            balance_floor = available_balance - BALANCE_FLOOR_BUFFER
            print(f"  Balance floor updated: ${balance_floor:.2f} (available: ${available_balance:.2f}, buffer: ${BALANCE_FLOOR_BUFFER:.2f})")
            return balance_floor
        else:
            print(f"  Warning: Could not set balance floor - invalid balance: {available_balance}")
            return None
    except Exception as e:
        print(f"  Warning: Failed to update balance floor: {e}")
        return None


def _reconcile_price_queue_with_bitstamp():
    """Reconcile price queue with Bitstamp historical data."""
    global price_data_queue
    
    try:
        from bitstamp_data_fetcher import fetch_bitstamp_ohlc_historical
        
        print(f"  Reconciling price queue with Bitstamp (last 1 hour)...")
        ohlc_data = fetch_bitstamp_ohlc_historical(
            currency_pair="btcusd",
            hours_back=1,
            step=60
        )
        
        if not ohlc_data:
            print(f"  Warning: Could not fetch Bitstamp data for reconciliation")
            return
        
        with price_queue_lock:
            queue_dict = {ts: price for ts, price in price_data_queue}
            
            updates_count = 0
            new_count = 0
            for ts, price in ohlc_data:
                if ts in queue_dict:
                    if abs(queue_dict[ts] - price) > 0.01:
                        queue_dict[ts] = price
                        updates_count += 1
                else:
                    queue_dict[ts] = price
                    new_count += 1
            
            price_data_queue.clear()
            sorted_items = sorted(queue_dict.items())
            for ts, price in sorted_items:
                price_data_queue.append((ts, price))
            
            from ..data.price_queue import _price_data_cache
            _price_data_cache['queue_hash'] = None
            _price_data_cache['prices_array'] = None
            
            clear_parameter_cache()
            
            if updates_count > 0 or new_count > 0:
                print(f"  ✓ Reconciled: {updates_count} updates, {new_count} new entries, {len(price_data_queue)} total points")
            else:
                print(f"  ✓ Reconciliation complete: {len(price_data_queue)} points (no changes needed)")
                
    except Exception as e:
        print(f"  Warning: Price queue reconciliation failed: {e}")


def handle_hour_change() -> Tuple[int, int, int, int]:
    """Consolidated function to handle all logic that should occur at the top of each hour."""
    global current_hour_key
    
    year, month, day, hour = get_current_est_hour()
    current_hour = (year, month, day, hour)
    
    hour_changed = (current_hour_key is None or current_hour_key != current_hour)
    
    if hour_changed:
        print(f"\n{'='*60}")
        print(f"Hour change detected: {year}-{month:02d}-{day:02d} {hour:02d}:00")
        print(f"{'='*60}")
        
        _reconcile_price_queue_with_bitstamp()
        
        previous_hour_key = get_previous_est_hour()
        calculate_pnl_for_resolved_markets(previous_hour_key)
        
        _update_balance_floor()
        
        update_log_files_for_current_hour()
        
        current_hour_key = current_hour
        
        print(f"{'='*60}\n")
    
    return year, month, day, hour

