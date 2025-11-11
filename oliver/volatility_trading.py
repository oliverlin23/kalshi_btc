"""
Volatility trading algorithm for Kalshi BTC markets.

Monitors Bitcoin price, calculates time-averaged price over 5 seconds,
and places limit orders with configurable spread and volume.
"""

import os
import sys
import time
import csv
import requests
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.utils import (
    get_kalshi_price_by_ticker,
    execute_trade_by_ticker,
    get_kalshi_client,
    cancel_all_orders_for_ticker
)
from app.ticker_utils import (
    generate_range_ticker,
    round_to_range_midpoint,
    parse_range_ticker,
    parse_threshold_ticker
)
from app.market_search import get_available_btc_tickers_for_hour
import pytz

# Tunable global variables
SPREAD_CENTS = 20  # Total spread in cents (split: buy 10c below, sell 10c above)
VOLUME = 25  # Number of contracts to trade per side
AVERAGE_WINDOW_SIZE = 5  # Number of recent prices to average (rolling window)
CYCLE_INTERVAL_SECONDS = 5  # Time to wait between trading cycles (after placing order)

# Global rolling queue for price history
price_history = deque(maxlen=AVERAGE_WINDOW_SIZE)

# Track recent order IDs to monitor for fills
recent_order_ids = deque(maxlen=100)  # Keep last 100 order IDs

# BTC price cache (update every 5 minutes)
btc_price_cache = {
    'price': None,
    'timestamp': 0.0,
    'ttl_seconds': 300  # 5 minutes
}

# Cache the last ticker we traded (to cancel orders when switching tickers)
last_ticker = None


# Log file paths
CYCLES_LOG_FILE = os.path.join(os.path.dirname(__file__), "volatility_trading_cycles.log")
FILLS_LOG_FILE = os.path.join(os.path.dirname(__file__), "volatility_trading_fills.log")

EST = pytz.timezone('US/Eastern')


def init_log_files():
    """Initialize log files with headers."""
    # Cycles log: timestamp, cycle_num, ticker, btc_price, current_price, avg_price, limit_prices (buy/sell), order_ids (buy/sell), status, error
    if not os.path.exists(CYCLES_LOG_FILE):
        with open(CYCLES_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cycle_num', 'ticker', 'btc_price', 'current_price', 
                           'avg_price', 'limit_prices', 'order_ids', 'status', 'error'])
    
    # Fills log: timestamp, order_id, fill_id, ticker, action, side, count, price
    if not os.path.exists(FILLS_LOG_FILE):
        with open(FILLS_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'order_id', 'fill_id', 'ticker', 'action', 'side', 'count', 'price'])


def log_cycle(timestamp: str, cycle_num: int, ticker: str, btc_price: float, current_price: float,
              avg_price: float, buy_limit_price: Optional[float], sell_limit_price: Optional[float],
              buy_order_id: Optional[str], sell_order_id: Optional[str], 
              status: str, error: Optional[str] = None):
    """Log cycle action to cycles log file."""
    try:
        with open(CYCLES_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            # Include both order IDs and limit prices
            order_ids = f"{buy_order_id or ''}/{sell_order_id or ''}"
            limit_prices = f"{buy_limit_price or ''}/{sell_limit_price or ''}"
            writer.writerow([timestamp, cycle_num, ticker, btc_price, current_price, 
                           avg_price, limit_prices, order_ids, status, error or ''])
    except Exception as e:
        print(f"Warning: Could not write to cycles log: {e}")


def log_fill(timestamp: str, order_id: str, fill_id: str, ticker: str, action: str, 
             side: str, count: int, price: float):
    """Log fill execution to fills log file."""
    try:
        with open(FILLS_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, order_id, fill_id, ticker, action, side, count, price])
    except Exception as e:
        print(f"Warning: Could not write to fills log: {e}")


def get_btc_price_from_ticker(ticker: str) -> Optional[float]:
    """
    Derive BTC price from a Kalshi ticker.
    Range markets: use midpoint
    Threshold markets: use threshold
    
    Args:
        ticker: Kalshi ticker string
    
    Returns:
        BTC price estimate in USD, or None if ticker can't be parsed
    """
    range_info = parse_range_ticker(ticker)
    threshold_info = parse_threshold_ticker(ticker)
    
    if range_info:
        return float(range_info.midpoint)
    elif threshold_info:
        return threshold_info.threshold
    
    return None


def get_current_est_hour():
    """Get the NEXT top-of-hour in EST (ceil to next hour)."""
    est_now = datetime.now(EST)
    if est_now.minute == 0 and est_now.second == 0:
        target = est_now
    else:
        target = (est_now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    return target.year, target.month, target.day, target.hour


def get_current_btc_price_estimate() -> Optional[int]:
    """
    Get current BTC price from Bitstamp API, cached for 5 minutes.
    
    Returns:
        BTC price in USD (integer), or None if unavailable
    """
    global btc_price_cache
    
    # Check cache first
    now = time.time()
    if (btc_price_cache['price'] is not None and 
        (now - btc_price_cache['timestamp']) < btc_price_cache['ttl_seconds']):
        return btc_price_cache['price']
    
    # Fetch fresh price from Bitstamp (only if cache expired)
    try:
        response = requests.get(
            "https://www.bitstamp.net/api/v2/ticker/btcusd/",
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        if "last" in data:
            price = int(float(data["last"]))
            # Update cache
            btc_price_cache['price'] = price
            btc_price_cache['timestamp'] = now
            return price
    except requests.exceptions.HTTPError as e:
        # Rate limited or other HTTP error - use cached value if available
        if btc_price_cache['price'] is not None:
            return btc_price_cache['price']
    except Exception as e:
        # Other errors - use cached value if available
        if btc_price_cache['price'] is not None:
            return btc_price_cache['price']
    
    # Last resort: return cached value even if expired
    if btc_price_cache['price'] is not None:
        return btc_price_cache['price']
    
    return None


def find_available_ticker(year: int, month: int, day: int, hour: int, btc_price: Optional[int] = None) -> Optional[str]:
    """
    Find an available Kalshi BTC ticker for the given hour, close to current BTC price.
    If no market exists for the exact hour, falls back to finding the nearest available market.
    
    Args:
        year, month, day, hour: Target hour for the market
        btc_price: Optional BTC price in USD (integer). If None, fetches from CoinGecko.
    
    Returns:
        Ticker string or None if not found
    """
    # Get BTC price estimate if not provided
    if btc_price is None:
        btc_price = get_current_btc_price_estimate()
    
    try:
        # First, try to get tickers for the exact hour
        # Use higher limit to ensure we get markets near current BTC price
        tickers = get_available_btc_tickers_for_hour(
            year=year,
            month=month,
            day=day,
            hour=hour,
            btc_price=btc_price,
            limit=500  # Increased to get all markets for proper sorting
        )
        
        # If no tickers found for exact hour, search for any available markets
        if not tickers:
            from app.market_search import search_btc_markets
            from app.ticker_utils import parse_range_ticker, parse_threshold_ticker
            
            # Get all available BTC markets
            all_markets = search_btc_markets(limit=100)
            tickers = [m.get('ticker') for m in all_markets if m.get('ticker')]
            
            # Sort by proximity to BTC price if available
            if btc_price and tickers:
                tickers_with_distance = []
                for ticker in tickers:
                    range_info = parse_range_ticker(ticker)
                    threshold_info = parse_threshold_ticker(ticker)
                    
                    if range_info:
                        distance = abs(btc_price - range_info.midpoint)
                    elif threshold_info:
                        distance = abs(btc_price - threshold_info.threshold)
                    else:
                        distance = float('inf')
                    
                    tickers_with_distance.append((ticker, distance))
                
                tickers_with_distance.sort(key=lambda x: x[1])
                tickers = [t[0] for t in tickers_with_distance]
        
        # Try each ticker until we find one that exists (has prices)
        if tickers:
            for ticker in tickers:
                try:
                    # Quick check if market exists by trying to get price
                    test_price = get_kalshi_price_by_ticker(ticker, action="buy")
                    if test_price is not None:
                        return ticker
                except ValueError:
                    # Market doesn't exist, try next one
                    continue
                except Exception:
                    # Other error, try next one
                    continue
        
        return None
    except Exception as e:
        print(f"Error in find_available_ticker: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_current_mid_price(ticker: str) -> Optional[float]:
    """
    Get current mid price for a ticker.
    
    Args:
        ticker: Market ticker to get price for
    
    Returns:
        Mid price (average of buy and sell) or None if unavailable
    """
    try:
        buy_price = get_kalshi_price_by_ticker(ticker, action="buy")
        sell_price = get_kalshi_price_by_ticker(ticker, action="sell")
        mid_price = (buy_price + sell_price) / 2
        return mid_price
    except Exception as e:
        print(f"  Warning: Could not get price: {e}")
        return None


def add_price_to_history(price: float):
    """
    Add a price to the rolling history queue.
    
    Args:
        price: Price to add
    """
    global price_history
    price_history.append(price)


def check_recent_fills(ticker: str, since_seconds: int = 60) -> list:
    """
    Check for recent fills (executed trades) using Kalshi SDK get_fills API.
    
    Args:
        ticker: Market ticker to check fills for
        since_seconds: Only check fills from the last N seconds (default: 60)
    
    Returns:
        List of fill dictionaries with order_id, fill_id, price, count, timestamp, etc.
    """
    try:
        client = get_kalshi_client()
        fills_response = client.get_fills(limit=100)
        
        if not fills_response or not hasattr(fills_response, 'fills') or not fills_response.fills:
            return []
        
        # Filter fills for this ticker and recent timeframe
        cutoff_time = datetime.now() - timedelta(seconds=since_seconds)
        recent_fills = []
        
        for fill in fills_response.fills:
            # Check if fill matches our ticker
            if hasattr(fill, 'ticker') and fill.ticker == ticker:
                # Check if fill is recent (if timestamp available)
                fill_time = None
                if hasattr(fill, 'ts') and fill.ts:
                    # Timestamp might be in different formats
                    try:
                        if isinstance(fill.ts, (int, float)):
                            fill_time = datetime.fromtimestamp(fill.ts)
                        elif isinstance(fill.ts, str):
                            fill_time = datetime.fromisoformat(fill.ts.replace('Z', '+00:00'))
                    except Exception:
                        pass
                
                # If we can't parse timestamp, include it anyway (better safe than sorry)
                if fill_time is None or fill_time >= cutoff_time:
                    # Fill object has 'price' field (already in decimal format, not cents)
                    fill_price = getattr(fill, 'price', None)
                    if fill_price is None:
                        print(f"  Warning: Fill {getattr(fill, 'fill_id', 'unknown')} missing 'price' field")
                        fill_price = None  # Don't use misleading fallback
                    
                    fill_info = {
                        'order_id': getattr(fill, 'order_id', None),
                        'fill_id': getattr(fill, 'fill_id', None),
                        'ticker': fill.ticker,
                        'action': getattr(fill, 'action', None),
                        'side': getattr(fill, 'side', None),
                        'count': getattr(fill, 'count', None),  # Don't default to 0
                        'price': fill_price,
                        'price_decimal': fill_price,  # Keep None if missing, don't default to 0.0
                        'timestamp': fill_time.isoformat() if fill_time else None,
                    }
                    recent_fills.append(fill_info)
        
        return recent_fills
    except Exception as e:
        print(f"  Warning: Could not check fills: {e}")
        return []


def calculate_average_price() -> Optional[float]:
    """
    Calculate average price from the rolling history queue.
    
    Returns:
        Average price or None if queue is empty
    """
    global price_history
    
    if not price_history:
        return None
    
    return sum(price_history) / len(price_history)


def place_limit_order_with_spread(
    ticker: str,
    average_price: float,
    action: str = "buy"
) -> Tuple[Optional[dict], Optional[float], Optional[str]]:
    """
    Place a limit order with spread offset from average price.
    Buy orders: average_price - (SPREAD_CENTS/2)
    Sell orders: average_price + (SPREAD_CENTS/2)
    
    Args:
        ticker: Market ticker
        average_price: Average price to base limit on
        action: "buy" or "sell"
    
    Returns:
        Tuple of (order_result, limit_price, error_message)
    """
    # Split spread in half: buy 10c below, sell 10c above (for 20c total spread)
    half_spread_decimal = (SPREAD_CENTS / 2.0) / 100.0
    
    if action == "buy":
        limit_price = average_price - half_spread_decimal
    else:
        limit_price = average_price + half_spread_decimal
    
    # Skip order if limit price is invalid (don't clamp)
    if limit_price <= 0 or limit_price >= 1:
        return None, limit_price, f"Invalid limit price: {limit_price:.4f}"
    
    try:
        result = execute_trade_by_ticker(
            ticker=ticker,
            action=action,
            count=VOLUME,
            limit_price=limit_price
        )
        
        if result:
            order_id = result.get('order_id', None)
            if order_id:
                recent_order_ids.append(order_id)
            return result, limit_price, None
        else:
            return None, limit_price, "Order failed"
    except Exception as e:
        return None, limit_price, str(e)


def run_trading_cycle(cycle_num: int) -> bool:
    """
    Run a single trading cycle.
    
    Args:
        cycle_num: Cycle number for logging
    
    Returns:
        True if cycle completed successfully, False otherwise
    """
    timestamp = datetime.now().isoformat()
    
    # Find available ticker for current hour (no external BTC price needed)
    year, month, day, hour = get_current_est_hour()
    ticker = find_available_ticker(year, month, day, hour)
    
    if not ticker:
        print(f"[{cycle_num}] SKIP: No available markets for {year}-{month:02d}-{day:02d} {hour:02d}:00")
        return False
    
    # Derive BTC price from ticker (for logging purposes)
    btc_price = get_btc_price_from_ticker(ticker)
    if btc_price is None:
        btc_price = 0.0  # Fallback if can't parse
    
    # Get current price and add to rolling history
    current_price = get_current_mid_price(ticker)
    if current_price is None:
        # Market doesn't exist (404) - skip this cycle gracefully
        print(f"[{cycle_num}] SKIP: Market {ticker} not found on Kalshi")
        log_cycle(timestamp, cycle_num, ticker, btc_price, None, None, None, None, None, None, "market_not_found", f"Market {ticker} does not exist")
        return False
    
    add_price_to_history(current_price)
    
    # Calculate average price from rolling history
    average_price = calculate_average_price()
    if average_price is None:
        print(f"[{cycle_num}] ERROR: Could not calculate average")
        return False
    
    # Cancel all existing orders for the LAST ticker (if we switched tickers)
    global last_ticker
    if last_ticker is not None and last_ticker != ticker:
        cancelled_count, failed_count = cancel_all_orders_for_ticker(last_ticker)
        if cancelled_count > 0:
            print(f"[{cycle_num}] Cancelled {cancelled_count} order(s) for previous ticker {last_ticker}")
        if failed_count > 0:
            print(f"[{cycle_num}] Warning: Failed to cancel {failed_count} order(s) for {last_ticker}")
    
    # Cancel all existing orders for current ticker before placing new ones (market making)
    cancelled_count, failed_count = cancel_all_orders_for_ticker(ticker)
    if cancelled_count > 0:
        print(f"[{cycle_num}] Cancelled {cancelled_count} existing order(s) for {ticker}")
    if failed_count > 0:
        print(f"[{cycle_num}] Warning: Failed to cancel {failed_count} order(s) for {ticker}")
    
    # Update cached ticker
    last_ticker = ticker
    
    # Place both buy and sell limit orders
    buy_result, buy_limit_price, buy_error = place_limit_order_with_spread(ticker, average_price, action="buy")
    sell_result, sell_limit_price, sell_error = place_limit_order_with_spread(ticker, average_price, action="sell")
    
    buy_order_id = buy_result.get('order_id', None) if buy_result else None
    sell_order_id = sell_result.get('order_id', None) if sell_result else None
    
    # Determine overall status
    buy_status = "placed" if buy_result else ("skipped" if buy_error and "Invalid limit price" in buy_error else "failed")
    sell_status = "placed" if sell_result else ("skipped" if sell_error and "Invalid limit price" in sell_error else "failed")
    
    # Log both orders to file with both order IDs and limit prices
    status = "both_placed" if (buy_result and sell_result) else ("buy_only" if buy_result else ("sell_only" if sell_result else "failed"))
    error_msg = None
    if buy_error and sell_error:
        error_msg = f"Buy: {buy_error}; Sell: {sell_error}"
    elif buy_error:
        error_msg = f"Buy: {buy_error}"
    elif sell_error:
        error_msg = f"Sell: {sell_error}"
    
    log_cycle(timestamp, cycle_num, ticker, btc_price, current_price, 
              average_price, buy_limit_price, sell_limit_price, buy_order_id, sell_order_id, status, error_msg)
    
    # Console output (one line)
    if buy_result and sell_result:
        print(f"[{cycle_num}] {ticker} | BTC=${btc_price:,.0f} | Price={current_price:.4f} | Avg={average_price:.4f} | Buy@{buy_limit_price:.4f} Sell@{sell_limit_price:.4f} | Orders: {buy_order_id}/{sell_order_id}")
    elif buy_result:
        print(f"[{cycle_num}] {ticker} | BTC=${btc_price:,.0f} | Price={current_price:.4f} | Avg={average_price:.4f} | Buy@{buy_limit_price:.4f} ✓ | Sell SKIPPED")
    elif sell_result:
        print(f"[{cycle_num}] {ticker} | BTC=${btc_price:,.0f} | Price={current_price:.4f} | Avg={average_price:.4f} | Buy SKIPPED | Sell@{sell_limit_price:.4f} ✓")
    else:
        print(f"[{cycle_num}] {ticker} | BTC=${btc_price:,.0f} | Price={current_price:.4f} | Avg={average_price:.4f} | FAILED: {error_msg}")
    
    # Check for recent fills (executed trades) and log to CSV
    recent_fills = check_recent_fills(ticker, since_seconds=60)
    
    if recent_fills:
        for fill in recent_fills:
            fill_timestamp = fill.get('timestamp') or timestamp
            log_fill(
                fill_timestamp,
                fill.get('order_id', ''),
                fill.get('fill_id', ''),
                fill.get('ticker', ticker),
                fill.get('action', ''),
                fill.get('side', ''),
                fill.get('count', 0),
                fill.get('price_decimal', 0)
            )
    
    return status in ("both_placed", "buy_only", "sell_only")


def main():
    """Main trading algorithm - runs continuously."""
    # Initialize log files
    init_log_files()
    
    # Fetch initial BTC price
    btc_price = get_current_btc_price_estimate()
    
    print("=" * 60)
    print("Volatility Trading Algorithm - Continuous Mode")
    print("=" * 60)
    print(f"Spread: {SPREAD_CENTS} cents (buy {SPREAD_CENTS//2}c below, sell {SPREAD_CENTS//2}c above)")
    print(f"Volume: {VOLUME} contracts per side")
    print(f"Window: {AVERAGE_WINDOW_SIZE} prices | Interval: {CYCLE_INTERVAL_SECONDS}s")
    print(f"BTC Price: Updated every 5 minutes from Bitstamp")
    if btc_price:
        print(f"Current BTC Price: ${btc_price:,} (will trade contracts close to this price)")
    else:
        print(f"⚠️  Could not fetch BTC price - will trade any available contract")
    print(f"Cycles log: {CYCLES_LOG_FILE}")
    print(f"Fills log: {FILLS_LOG_FILE}")
    print("Press Ctrl-C to stop")
    print("=" * 60)
    
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

