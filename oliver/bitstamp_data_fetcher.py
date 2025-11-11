"""
Bitstamp API data fetcher for historical OHLC data.
Note: Bitstamp OHLC API only supports step=60 (per-minute).
For finer granularity, we fetch 1-minute data and can interpolate or poll ticker for per-second updates.
"""

import requests
import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional


def fetch_bitstamp_ohlc_historical(
    currency_pair: str = "btcusd",
    hours_back: int = 6,
    step: int = 60,
    limit: int = 1000
) -> List[Tuple[float, float]]:
    """
    Fetch historical OHLC data from Bitstamp API.
    
    Note: Bitstamp API only supports step=60 (per-minute) for OHLC data.
    For per-second data, use fetch_bitstamp_ticker_recent() or build up over time.
    
    Args:
        currency_pair: Trading pair (default: "btcusd")
        hours_back: Number of hours of historical data to fetch (default: 6)
        step: Time interval in seconds (must be 60, Bitstamp API limitation)
        limit: Maximum number of data points per request (default: 1000, Bitstamp API limit)
    
    Returns:
        List of (timestamp, price) tuples, where price is the closing price
        Timestamps are Unix timestamps (seconds since epoch)
        Data is sorted chronologically (oldest first)
    """
    if step != 60:
        raise ValueError(f"Bitstamp OHLC API only supports step=60 (per-minute). Got step={step}")
    
    end_timestamp = time.time()
    start_timestamp = end_timestamp - (hours_back * 3600)
    
    # Calculate number of API calls needed
    # Each call can fetch up to 1000 minutes of data
    minutes_per_call = min(limit, hours_back * 60)
    seconds_per_call = minutes_per_call * 60
    
    all_data = []
    current_start = start_timestamp
    
    url = f"https://www.bitstamp.net/api/v2/ohlc/{currency_pair}/"
    
    call_count = 0
    while current_start < end_timestamp:
        current_end = min(current_start + seconds_per_call, end_timestamp)
        
        params = {
            "step": 60,  # Must be 60
            "start": int(current_start),
            "end": int(current_end),
            "limit": limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            ohlc_data = data.get("data", {}).get("ohlc", [])
            
            if not ohlc_data:
                break
            
            # Extract timestamp and close price
            for point in ohlc_data:
                timestamp = float(point.get("timestamp", 0))
                close_price = float(point.get("close", 0))
                if timestamp > 0 and close_price > 0:
                    all_data.append((timestamp, close_price))
            
            call_count += 1
            
            # Be nice to the API - small delay between calls
            if current_end < end_timestamp:
                time.sleep(0.1)
            
            current_start = current_end
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Bitstamp data: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    
    # Sort by timestamp (should already be sorted, but ensure it)
    all_data.sort(key=lambda x: x[0])
    
    # Remove duplicates (keep first occurrence)
    seen = set()
    unique_data = []
    for ts, price in all_data:
        if ts not in seen:
            seen.add(ts)
            unique_data.append((ts, price))
    
    print(f"Fetched {len(unique_data)} data points from Bitstamp ({call_count} API calls, step=60s)")
    
    return unique_data


def fetch_bitstamp_ticker() -> Optional[Tuple[float, float]]:
    """
    Fetch current BTC price from Bitstamp ticker API.
    This is fast and can be called every second.
    
    Returns:
        Tuple of (timestamp, price) or None if fetch fails
    """
    try:
        response = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/", timeout=2)
        response.raise_for_status()
        data = response.json()
        if "last" in data:
            timestamp = time.time()
            price = float(data["last"])
            return (timestamp, price)
    except Exception:
        pass
    return None


def test_bitstamp_api_support() -> dict:
    """
    Test Bitstamp API to see what step values are supported.
    
    Returns:
        Dictionary with test results
    """
    results = {
        'step_1_supported': False,
        'step_5_supported': False,
        'step_60_supported': False,
        'rate_limit_info': None
    }
    
    # Test step=1 (per-second)
    try:
        end_ts = time.time()
        start_ts = end_ts - 100  # 100 seconds ago
        
        url = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
        params = {"step": 1, "start": int(start_ts), "end": int(end_ts), "limit": 100}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            ohlc = data.get("data", {}).get("ohlc", [])
            results['step_1_supported'] = len(ohlc) > 0
            if results['step_1_supported']:
                print(f"✓ Step=1 (per-second) supported: Got {len(ohlc)} points")
        else:
            print(f"✗ Step=1 returned status {response.status_code}")
    except Exception as e:
        print(f"✗ Step=1 test failed: {e}")
    
    # Test step=5 (per-5-second)
    try:
        end_ts = time.time()
        start_ts = end_ts - 500  # 500 seconds ago
        
        params = {"step": 5, "start": int(start_ts), "end": int(end_ts), "limit": 100}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            ohlc = data.get("data", {}).get("ohlc", [])
            results['step_5_supported'] = len(ohlc) > 0
            if results['step_5_supported']:
                print(f"✓ Step=5 (per-5-second) supported: Got {len(ohlc)} points")
        else:
            print(f"✗ Step=5 returned status {response.status_code}")
    except Exception as e:
        print(f"✗ Step=5 test failed: {e}")
    
    # Test step=60 (per-minute) - known to work
    try:
        end_ts = time.time()
        start_ts = end_ts - 3600  # 1 hour ago
        
        params = {"step": 60, "start": int(start_ts), "end": int(end_ts), "limit": 100}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            ohlc = data.get("data", {}).get("ohlc", [])
            results['step_60_supported'] = len(ohlc) > 0
            if results['step_60_supported']:
                print(f"✓ Step=60 (per-minute) supported: Got {len(ohlc)} points")
    except Exception as e:
        print(f"✗ Step=60 test failed: {e}")
    
    return results


if __name__ == "__main__":
    print("Testing Bitstamp API support...")
    results = test_bitstamp_api_support()
    print(f"\nResults: {results}")
    
    if results['step_1_supported'] or results['step_5_supported']:
        print("\nTesting data fetch...")
        test_data = fetch_bitstamp_ohlc_historical(hours_back=1, step=5)
        if test_data:
            print(f"Successfully fetched {len(test_data)} data points")
            print(f"First point: {datetime.fromtimestamp(test_data[0][0])} @ ${test_data[0][1]:,.2f}")
            print(f"Last point: {datetime.fromtimestamp(test_data[-1][0])} @ ${test_data[-1][1]:,.2f}")

