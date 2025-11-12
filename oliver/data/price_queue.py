"""
Price data queue management for volatility trading.

Manages rolling windows of price data from Bitstamp API, including:
- Main price queue (1-minute intervals) - used for volatility and Merton jump parameter estimation
- Jump detection queue (per-second updates, 300 seconds = 5 minutes) - used for real-time jump/spike detection
- Background thread for automatic updates
"""

import time
import threading
import numpy as np
from collections import deque
from typing import Tuple

import requests

from ..config import DATA_STEP_SECONDS, DATA_HOURS_BACK
try:
    from predict_price_probability import clear_parameter_cache
except ImportError:
    from oliver.predict_price_probability import clear_parameter_cache

_price_queue_maxlen = DATA_HOURS_BACK * (3600 // DATA_STEP_SECONDS)
price_data_queue = deque(maxlen=_price_queue_maxlen)
price_data_initialized = False
last_price_update_time = 0.0
price_queue_lock = threading.Lock()
price_queue_thread = None

jump_detection_price_queue = deque(maxlen=300)  # 300 seconds = 5 minutes of per-second data for jump detection
jump_detection_queue_lock = threading.Lock()

_price_data_cache = {
    'prices_array': None,
    'last_ts': None,
    'last_price': None,
    'queue_hash': None
}


def _fetch_bitstamp_price():
    """Fetch BTC price from Bitstamp API."""
    try:
        response = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/", timeout=2)
        response.raise_for_status()
        data = response.json()
        if "bid" in data and "ask" in data:
            return (float(data["bid"]) + float(data["ask"])) / 2
    except Exception:
        pass
    return None


def get_current_btc_price_estimate():
    """Get current BTC price from Bitstamp only."""
    bitstamp_price = _fetch_bitstamp_price()
    
    if bitstamp_price is not None:
        return int(bitstamp_price)
    else:
        print("  Warning: Could not fetch BTC price from Bitstamp")
        raise Exception("Could not fetch BTC price from Bitstamp")


def initialize_price_queue_from_bitstamp():
    """Initialize price data queue from Bitstamp API."""
    global price_data_queue, price_data_initialized
    
    if price_data_initialized:
        print("  Price queue already initialized, skipping...")
        return
    
    try:
        from bitstamp_data_fetcher import fetch_bitstamp_ohlc_historical, fetch_bitstamp_ticker
        
        print(f"  Fetching {DATA_HOURS_BACK} hours of 1-minute data from Bitstamp...")
        ohlc_data = fetch_bitstamp_ohlc_historical(
            currency_pair="btcusd",
            hours_back=DATA_HOURS_BACK,
            step=60
        )
        
        if not ohlc_data:
            raise ValueError("No data fetched from Bitstamp")
        
        price_data_queue.clear()
        
        global _price_data_cache
        _price_data_cache['queue_hash'] = None
        _price_data_cache['prices_array'] = None
        
        for ts, price in ohlc_data:
            price_data_queue.append((ts, price))
        
        print(f"  Loaded {len(price_data_queue)} data points (1-minute intervals)")
        
        current_ticker = fetch_bitstamp_ticker()
        if current_ticker:
            price_data_queue.append(current_ticker)
            with jump_detection_queue_lock:
                jump_detection_price_queue.append(current_ticker)
        
        price_data_initialized = True
        print(f"  ✓ Initialized cached price queue with {len(price_data_queue)} data points")
        
        start_price_queue_updater()
        print(f"  ✓ Started daemon thread to update queue every {DATA_STEP_SECONDS} seconds")
        
    except Exception as e:
        print(f"  ✗ Failed to initialize price queue from Bitstamp: {e}")
        raise


def update_price_queue():
    """Update price queue with new BTC price data."""
    global price_data_queue, last_price_update_time
    
    with price_queue_lock:
        now = time.time()
        current_interval = int(now // DATA_STEP_SECONDS)
        last_interval = int(last_price_update_time // DATA_STEP_SECONDS) if last_price_update_time > 0 else None
        
        if last_interval is None:
            needs_update = True
            intervals_to_catch_up = 0
        elif current_interval > last_interval:
            intervals_to_catch_up = current_interval - last_interval
            needs_update = True
        else:
            needs_update = False
            intervals_to_catch_up = 0
        
        if not needs_update:
            return False
        
        btc_price = None
        for attempt in range(3):
            try:
                btc_price = get_current_btc_price_estimate()
                if btc_price is not None:
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.5)
                else:
                    print(f"  Warning: Failed to fetch BTC price after 3 attempts: {e}")
        
        if btc_price is not None:
            if intervals_to_catch_up > 1:
                try:
                    from bitstamp_data_fetcher import fetch_bitstamp_ohlc_historical
                    hours_back = (intervals_to_catch_up * DATA_STEP_SECONDS) / 3600.0 + 0.02
                    ohlc_data = fetch_bitstamp_ohlc_historical(
                        currency_pair="btcusd",
                        hours_back=hours_back,
                        step=60
                    )
                    if ohlc_data:
                        last_ts_in_queue = price_data_queue[-1][0] if price_data_queue else 0
                        for ts, price in ohlc_data:
                            if ts > last_ts_in_queue and ts <= now:
                                price_data_queue.append((ts, price))
                        print(f"  Fetched {len([ts for ts, _ in ohlc_data if ts > last_ts_in_queue and ts <= now])} historical data points for catch-up")
                    else:
                        for i in range(intervals_to_catch_up):
                            timestamp = now - (intervals_to_catch_up - i - 1) * DATA_STEP_SECONDS
                            price_data_queue.append((timestamp, float(btc_price)))
                except Exception as e:
                    print(f"  Warning: Could not fetch historical data for catch-up: {e}")
                    for i in range(intervals_to_catch_up):
                        timestamp = now - (intervals_to_catch_up - i - 1) * DATA_STEP_SECONDS
                        price_data_queue.append((timestamp, float(btc_price)))
            elif intervals_to_catch_up == 1:
                timestamp = now
                price_data_queue.append((timestamp, float(btc_price)))
            else:
                timestamp = now
                price_data_queue.append((timestamp, float(btc_price)))
            
            last_price_update_time = now
            
            clear_parameter_cache()
            
            global _price_data_cache
            _price_data_cache['queue_hash'] = None
            _price_data_cache['prices_array'] = None
            
            if intervals_to_catch_up > 1:
                print(f"  Caught up {intervals_to_catch_up} missed interval(s) ({DATA_STEP_SECONDS}s each) in price queue")
            
            return True
    
    return False


def update_jump_detection_queue():
    """Update jump detection price queue with current BTC price (per-second updates for real-time jump/spike detection)."""
    global jump_detection_price_queue
    
    with jump_detection_queue_lock:
        btc_price = None
        for attempt in range(3):
            try:
                btc_price = get_current_btc_price_estimate()
                if btc_price is not None:
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.1)
                else:
                    return
        
        if btc_price is not None:
            current_time = time.time()
            jump_detection_price_queue.append((current_time, float(btc_price)))


def price_queue_updater_thread():
    """Background thread that updates price queue every DATA_STEP_SECONDS."""
    while True:
        try:
            update_price_queue()
            now = time.time()
            next_interval = int(now // DATA_STEP_SECONDS) + 1
            sleep_until = next_interval * DATA_STEP_SECONDS
            sleep_time = max(0.1, sleep_until - now)
            time.sleep(sleep_time)
        except Exception as e:
            print(f"  Warning: Error in price queue updater thread: {e}")
            time.sleep(DATA_STEP_SECONDS)


def start_price_queue_updater():
    """Start background thread for price queue updates."""
    global price_queue_thread
    
    if price_queue_thread is not None and price_queue_thread.is_alive():
        return
    
    price_queue_thread = threading.Thread(target=price_queue_updater_thread, daemon=True)
    price_queue_thread.start()
    print(f"  Started background price queue updater thread")


def get_price_data_for_prediction() -> Tuple[np.ndarray, float, float]:
    """Get price data from queue in format expected by predict_price_probability."""
    global price_data_queue, _price_data_cache
    
    with price_queue_lock:
        if len(price_data_queue) == 0:
            raise ValueError("Price queue is empty - cannot make predictions")
        
        queue_items = list(price_data_queue)
        if len(queue_items) >= 4:
            queue_hash = hash((
                queue_items[0][0], queue_items[0][1],
                queue_items[-1][0], queue_items[-1][1],
                len(queue_items)
            ))
        else:
            queue_hash = hash(tuple(queue_items))
        
        if (_price_data_cache['queue_hash'] == queue_hash and 
            _price_data_cache['prices_array'] is not None):
            return _price_data_cache['prices_array'], _price_data_cache['last_ts'], _price_data_cache['last_price']
        
        prices_list = [price for _, price in price_data_queue]
        prices_array = np.array(prices_list, dtype=np.float64)
        
        last_ts = queue_items[-1][0]
        last_price = queue_items[-1][1]
        
        _price_data_cache['prices_array'] = prices_array
        _price_data_cache['last_ts'] = last_ts
        _price_data_cache['last_price'] = last_price
        _price_data_cache['queue_hash'] = queue_hash
        
        return prices_array, last_ts, last_price

