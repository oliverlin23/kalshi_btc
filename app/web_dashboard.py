"""
Web-based dashboard for Kalshi BTC binary contracts.
Run with: python3 app/web_dashboard.py
"""

import os
import sys
from flask import Flask, render_template, jsonify
from datetime import datetime, timedelta
import time
import pytz

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.utils import get_kalshi_price_by_ticker, get_kalshi_spread_by_ticker
from app.ticker_utils import (
    generate_range_ticker,
    generate_threshold_ticker,
    parse_range_ticker,
    parse_threshold_ticker
)
from app.market_search import get_available_btc_tickers_for_hour
from app.utils import get_kalshi_orderbook_by_ticker

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

EST = pytz.timezone('US/Eastern')

MARKET_CACHE = {
    'hour_key': None,      # (year, month, day, hour)
    'tickers': [],         # cached tickers for the hour
    'last_fetch_ts': 0.0,  # epoch seconds
}


def get_current_est_hour():
    """Get the NEXT top-of-hour in EST (ceil to next hour).

    Example: at 11:11 AM EST -> returns 12:00 hour.
    At exactly 11:00:00 -> returns 11:00 hour.
    """
    est_now = datetime.now(EST)
    if est_now.minute == 0 and est_now.second == 0:
        target = est_now
    else:
        target = (est_now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    return target.year, target.month, target.day, target.hour


def get_or_refresh_tickers(btc_price: int):
    """Return cached tickers, refreshing only when the hour changes or once per minute.

    - Uses USE_KALSHI_MARKET_API=true to fetch from Kalshi; otherwise generates locally.
    - Caches result keyed by the next top-of-hour.
    - TTL: 60 seconds to avoid excessive API calls.
    """
    year, month, day, hour = get_current_est_hour()
    hour_key = (year, month, day, hour)
    now_ts = time.time()

    # Decide if we need to refresh: hour changed or TTL expired
    needs_refresh = (
        MARKET_CACHE['hour_key'] != hour_key or
        (now_ts - MARKET_CACHE['last_fetch_ts']) >= 60.0 or
        not MARKET_CACHE['tickers']
    )

    if not needs_refresh:
        return MARKET_CACHE['tickers'], year, month, day, hour

    # Refresh cache
    use_api_markets = os.getenv("USE_KALSHI_MARKET_API", "false").lower() == "true"
    try:
        if use_api_markets:
            try:
                available_tickers = get_available_btc_tickers_for_hour(
                    year, month, day, hour, btc_price=btc_price, limit=100
                )
                if available_tickers and len(available_tickers) > 0:
                    tickers = available_tickers[:6]
                else:
                    # No markets found, fall back to generated
                    tickers = generate_closest_tickers(btc_price, year, month, day, hour)
            except Exception as e:
                # Fallback to local generation if API fails
                print(f"Warning: Could not fetch markets from API: {e}. Using generated tickers.")
                tickers = generate_closest_tickers(btc_price, year, month, day, hour)
        else:
            tickers = generate_closest_tickers(btc_price, year, month, day, hour)
    except Exception as e:
        # Final fallback - this should never fail but just in case
        print(f"Error generating tickers: {e}")
        tickers = []

    MARKET_CACHE['hour_key'] = hour_key
    MARKET_CACHE['tickers'] = tickers
    MARKET_CACHE['last_fetch_ts'] = now_ts

    return tickers, year, month, day, hour


def round_to_range_midpoint(price: int) -> int:
    """
    Round price to nearest valid range market midpoint.
    Range market midpoints must be 125 mod 250 (e.g., 101875, 102125, 102375).
    """
    # Round to nearest 250
    base = round(price / 250) * 250
    
    # Find the two closest values that are 125 mod 250
    candidate1 = base - 125  # If base was 250 mod 250, this gives 125 mod 250
    candidate2 = base + 125  # If base was 0 mod 250, this gives 125 mod 250
    
    # Ensure both are 125 mod 250
    if candidate1 % 250 != 125:
        candidate1 = candidate1 - 250
    if candidate2 % 250 != 125:
        candidate2 = candidate2 + 250
    
    # Return the one closest to original price
    if abs(price - candidate1) <= abs(price - candidate2):
        return candidate1
    else:
        return candidate2


def round_to_threshold(price: int) -> float:
    """
    Round price to nearest valid threshold market value.
    Threshold markets must be one cent less than a multiple of 250
    (e.g., 101999.99, 102249.99, 102499.99).
    """
    # Round to nearest 250, then subtract 0.01
    base = round(price / 250) * 250
    return base - 0.01


def generate_closest_tickers(btc_price: int, year: int, month: int, day: int, hour: int):
    """
    Generate 6 tickers (3 range, 3 threshold) closest to BTC price.
    
    Rules:
    - Range markets (B): midpoints must be 125 mod 250 (e.g., 101875, 102125)
    - Threshold markets (T): must be one cent less than multiple of 250 (e.g., 101999.99)
    
    Returns list of tickers ordered by proximity to BTC price.
    """
    tickers_with_distance = []
    
    # Generate 3 range markets closest to BTC price
    # Range market midpoints must be 125 mod 250
    base_midpoint = round_to_range_midpoint(btc_price)
    midpoint1 = base_midpoint - 250
    midpoint2 = base_midpoint
    midpoint3 = base_midpoint + 250
    
    # Ensure all are 125 mod 250
    midpoint1 = ((midpoint1 // 250) * 250) + 125
    midpoint2 = ((midpoint2 // 250) * 250) + 125
    midpoint3 = ((midpoint3 // 250) * 250) + 125
    
    # Generate range tickers
    for midpoint in [midpoint1, midpoint2, midpoint3]:
        ticker = generate_range_ticker(year, month, day, hour, midpoint)
        # Distance is absolute difference from BTC price to midpoint
        distance = abs(btc_price - midpoint)
        tickers_with_distance.append((ticker, distance, 'range', midpoint))
    
    # Generate 3 threshold markets closest to BTC price
    # Threshold markets must be one cent less than a multiple of 250
    base_threshold = round_to_threshold(btc_price)
    threshold1 = base_threshold - 250
    threshold2 = base_threshold
    threshold3 = base_threshold + 250
    
    # Generate threshold tickers
    for threshold in [threshold1, threshold2, threshold3]:
        ticker = generate_threshold_ticker(year, month, day, hour, threshold)
        # Distance is absolute difference from BTC price to threshold
        distance = abs(btc_price - threshold)
        tickers_with_distance.append((ticker, distance, 'threshold', threshold))
    
    # Sort by distance and return just tickers
    tickers_with_distance.sort(key=lambda x: x[1])
    return [t[0] for t in tickers_with_distance]


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/tickers')
def get_tickers():
    """API endpoint to get current tickers based on EST hour and BTC price."""
    try:
        btc_price_str = os.getenv("BTC_CURRENT_PRICE", "")
        if not btc_price_str:
            return jsonify({"error": "BTC_CURRENT_PRICE environment variable not set"}), 400
        
        btc_price = int(btc_price_str)
        tickers, year, month, day, hour = get_or_refresh_tickers(btc_price)
        
        return jsonify({
            "tickers": tickers,
            "btc_price": btc_price,
            "est_time": datetime.now(EST).isoformat(),
            "year": year,
            "month": month,
            "day": day,
            "hour": hour
        })
    except ValueError as e:
        return jsonify({"error": f"Invalid BTC_CURRENT_PRICE: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/prices')
def get_prices():
    """API endpoint to get current prices for all tickers."""
    try:
        # Get tickers (cached; refreshes only on hour change or every 60s)
        btc_price_str = os.getenv("BTC_CURRENT_PRICE", "")
        if not btc_price_str:
            return jsonify({
                "error": "BTC_CURRENT_PRICE environment variable not set. Set it with: export BTC_CURRENT_PRICE=101875",
                "prices": [],
                "btc_price": None,
                "timestamp": datetime.now().isoformat(),
                "est_time": datetime.now(EST).isoformat()
            }), 200  # Return 200 so frontend can display the error
        
        btc_price = int(btc_price_str)
        tickers, year, month, day, hour = get_or_refresh_tickers(btc_price)
        
        if not tickers or len(tickers) == 0:
            return jsonify({
                "error": "No tickers available",
                "prices": [],
                "btc_price": btc_price,
                "timestamp": datetime.now().isoformat(),
                "est_time": datetime.now(EST).isoformat()
            }), 200  # Return 200 with empty list instead of error
        
        prices = []
        for ticker in tickers:
            try:
                buy_price = get_kalshi_price_by_ticker(ticker, action="buy")
                sell_price = get_kalshi_price_by_ticker(ticker, action="sell")
                spread = get_kalshi_spread_by_ticker(ticker)
                # Fetch orderbook to extract top-of-book volumes
                ob = get_kalshi_orderbook_by_ticker(ticker)
                
                buy_volume = None
                sell_volume = None
                
                # Orderbook structure: each entry is [price_cents, volume]
                # For buying YES: use NO side (buying NO = buying YES), prices are inverted
                # For selling YES: use YES side directly
                
                # Buying YES: use NO side, convert prices
                no_orders = ob.get('no', []) or []
                if no_orders:
                    # Convert NO prices to YES prices: YES_price = 100 - NO_price
                    # Best ask for buying YES is the lowest YES price (highest NO price)
                    yes_asks = [(100 - price_cents, volume) for price_cents, volume in no_orders]
                    if yes_asks:
                        best_ask = min(yes_asks, key=lambda x: x[0])  # Lowest YES price = best ask
                        buy_volume = int(best_ask[1]) if isinstance(best_ask[1], (int, float)) else None
                
                # Selling YES: use YES side directly
                yes_orders = ob.get('yes', []) or []
                if yes_orders:
                    # Best bid for selling YES: maximum price (highest bid)
                    best_bid = max(yes_orders, key=lambda x: x[0])
                    sell_volume = int(best_bid[1]) if isinstance(best_bid[1], (int, float)) else None
                
                # Parse ticker to get market info
                range_info = parse_range_ticker(ticker)
                threshold_info = parse_threshold_ticker(ticker)
                
                market_type = None
                market_value = None
                if range_info:
                    market_type = "range"
                    market_value = range_info.midpoint
                elif threshold_info:
                    market_type = "threshold"
                    market_value = threshold_info.threshold
                
                prices.append({
                    "ticker": ticker,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "spread": spread,
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "mid_price": (buy_price + sell_price) / 2 if buy_price and sell_price else None,
                    "status": "ok",
                    "market_type": market_type,
                    "market_value": market_value,
                    "distance_from_btc": abs(btc_price - market_value) if market_value else None
                })
            except Exception as e:
                error_str = str(e)
                # Parse ticker to get market info even if market doesn't exist
                range_info = parse_range_ticker(ticker)
                threshold_info = parse_threshold_ticker(ticker)
                
                market_type = None
                market_value = None
                if range_info:
                    market_type = "range"
                    market_value = range_info.midpoint
                elif threshold_info:
                    market_type = "threshold"
                    market_value = threshold_info.threshold
                
                # Check if it's a 404 (market not found)
                is_not_found = "404" in error_str or "not found" in error_str.lower() or "not_found" in error_str.lower()
                
                prices.append({
                    "ticker": ticker,
                    "buy_price": None,
                    "sell_price": None,
                    "spread": None,
                    "buy_volume": None,
                    "sell_volume": None,
                    "mid_price": None,
                    "status": "not_found" if is_not_found else "error",
                    "error": "Market not found on Kalshi" if is_not_found else error_str,
                    "market_type": market_type,
                    "market_value": market_value,
                    "distance_from_btc": abs(btc_price - market_value) if market_value else None
                })
        
        # Sort by distance from BTC price
        prices.sort(key=lambda x: x.get("distance_from_btc") or float('inf'))
        
        return jsonify({
            "prices": prices,
            "btc_price": btc_price,
            "timestamp": datetime.now().isoformat(),
            "est_time": datetime.now(EST).isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv("DASHBOARD_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    btc_price = os.getenv("BTC_CURRENT_PRICE", "")
    if not btc_price:
        print("WARNING: BTC_CURRENT_PRICE not set. Set it with: export BTC_CURRENT_PRICE=101875")
    else:
        print(f"BTC Current Price: ${int(btc_price):,}")
    
    year, month, day, hour = get_current_est_hour()
    print(f"Current EST: {year}-{month:02d}-{day:02d} {hour:02d}:00")
    
    print(f"Starting dashboard on http://localhost:{port}")
    print(f"Dashboard will show 6 markets (3 range, 3 threshold) closest to BTC price")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

