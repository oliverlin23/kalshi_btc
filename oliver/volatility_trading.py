"""
Volatility trading algorithm for Kalshi BTC markets.

Uses Monte Carlo price prediction to determine fair value for markets,
then places limit orders with configurable spread around the prediction.
"""

import os
import sys
import time
import csv
import requests
import threading
import random
import math
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Optional, Tuple, List

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
    parse_threshold_ticker
)
from app.market_search import get_available_btc_tickers_for_hour
import pytz

# Import prediction functions - required for trading
try:
    from predict_price_probability import predict_price_probability, clear_parameter_cache
except ImportError:
    print("ERROR: predict_price_probability module not available. Cannot trade without predictions.")
    sys.exit(1)

# Tunable global variables
BASE_SPREAD_CENTS = 8  # Base spread in cents (minimum spread)
VOLATILITY_MULTIPLIER = 2500  # Multiplier to convert volatility to spread (tune this)
MAX_SPREAD_CENTS = 50  # Maximum spread in cents (safety limit)
VOLUME = 25  # Base number of contracts to trade per side (used as fallback)
MAX_POSITION_PCT_OF_BALANCE = 0.20  # Maximum position size as % of available balance (20%)
KELLY_FRACTION = 0.25  # Fraction of Kelly criterion to use (25% = quarter Kelly, more conservative)
MIN_POSITION_SIZE = 5  # Minimum contracts to trade (for small balances)
CYCLE_INTERVAL_SECONDS = 1  # Time to wait between trading cycles
MARKET_MAKING_BUFFER_CENTS = 1  # Minimum distance from market price to ensure market making (in cents)
MARKET_TAKING_FEE_RATE = 0.07  # Fee rate for market taking: 0.07 * p * (1-p)

# Data frequency settings
# Maintains a rolling window of price data to get model parameters from
# like volatility, drift, etc.
DATA_STEP_SECONDS = 5  # Size of the rolling window intervals in seconds
DATA_HOURS_BACK = 6    # Initializes the cache

# Prediction settings
PREDICTION_MODEL = 'gbm'  # 'merton' or 'gbm' (analytical only)
# Weighted averages for parameter estimation (must sum to 1.0, will be normalized if not)
PREDICTION_WEIGHT_15M = 0.5  # Weight for 15-minute timeframe
PREDICTION_WEIGHT_1H = 0.3   # Weight for 1-hour timeframe
PREDICTION_WEIGHT_6H = 0.2   # Weight for 6-hour timeframe

# Price data queue - rolling window of (timestamp, price) tuples
# Size depends on DATA_STEP_SECONDS and DATA_HOURS_BACK
# For per-second: maxlen = DATA_HOURS_BACK * 3600
# For per-5-second: maxlen = DATA_HOURS_BACK * 720
# For per-minute: maxlen = DATA_HOURS_BACK * 60
_price_queue_maxlen = DATA_HOURS_BACK * (3600 // DATA_STEP_SECONDS)
price_data_queue = deque(maxlen=_price_queue_maxlen)
price_data_initialized = False
last_price_update_time = 0.0  # Track when we last updated the price queue
price_queue_lock = threading.Lock()  # Lock for thread-safe queue access
price_queue_thread = None  # Background thread for queue updates

# Cache for price data arrays (numpy arrays for efficiency)
_price_data_cache = {
    'prices_array': None,
    'last_ts': None,
    'last_price': None,
    'queue_hash': None  # Hash of queue contents to detect changes
}

# Track recent order IDs to monitor for fills
recent_order_ids = deque(maxlen=100)  # Keep last 100 order IDs

# Track logged fill IDs to avoid duplicate logging
logged_fill_ids = set()  # Set of fill_ids that have already been logged

# Cache the last ticker we traded (to cancel orders when switching tickers)
last_ticker = None

# Track last order details to avoid unnecessary cancellations
last_order_details = {
    'ticker': None,
    'predicted_prob': None,
    'volatility': None,
    'buy_limit_price': None,
    'sell_limit_price': None
}


# Log file paths - all logs go to logs/ directory
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
CYCLES_LOG_FILE = os.path.join(LOGS_DIR, "volatility_trading_cycles.csv")
FILLS_LOG_FILE = os.path.join(LOGS_DIR, "volatility_trading_fills.csv")

# Track positions per ticker: {ticker: {'position': int, 'total_buy_cost': float, 'total_sell_revenue': float, 'buy_count': int, 'sell_count': int}}
positions = {}
total_realized_pnl = 0.0
last_resolved_hour = None  # Track last hour we calculated PnL for

EST = pytz.timezone('US/Eastern')


def init_log_files():
    """Initialize log files with headers."""
    # Ensure logs directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Cycles log: timestamp, cycle_num, ticker, btc_price_from_ticker, btc_price_current, threshold_price, 
    # predicted_price, volatility, spread_cents, hours_until_resolution, resolution_datetime, time_to_resolution,
    # limit_prices (buy/sell), order_ids (buy/sell), buy_position_size, sell_position_size, available_balance, market_price, status, error
    if not os.path.exists(CYCLES_LOG_FILE):
        with open(CYCLES_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cycle_num', 'ticker', 'btc_price_from_ticker', 'btc_price_current', 
                           'threshold_price', 'predicted_price', 'volatility', 'spread_cents', 'hours_until_resolution',
                           'resolution_datetime', 'time_to_resolution', 'limit_prices', 'order_ids', 
                           'buy_position_size', 'sell_position_size', 'available_balance', 'market_price', 'status', 'error'])
    
    # Fills log: timestamp, order_id, fill_id, ticker, action, side, count, price
    if not os.path.exists(FILLS_LOG_FILE):
        with open(FILLS_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'order_id', 'fill_id', 'ticker', 'action', 'side', 'count', 'price'])


def format_time_to_resolution(hours: float) -> str:
    """
    Format hours until resolution as a human-readable string.
    
    Args:
        hours: Hours until resolution (can be fractional)
    
    Returns:
        Formatted string like "2h 30m" or "45m" or "1d 2h"
    """
    if hours < 0:
        return "RESOLVED"
    
    total_minutes = int(hours * 60)
    days = total_minutes // (24 * 60)
    hours_remaining = (total_minutes % (24 * 60)) // 60
    minutes_remaining = total_minutes % 60
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours_remaining > 0:
        parts.append(f"{hours_remaining}h")
    if minutes_remaining > 0 or len(parts) == 0:
        parts.append(f"{minutes_remaining}m")
    
    return " ".join(parts)


def log_cycle(timestamp: str, cycle_num: int, ticker: str, btc_price_from_ticker: float,
              btc_price_current: float, threshold_price: float, predicted_price: float,
              volatility: float, spread_cents: float, hours_until_resolution: float,
              resolution_datetime: str, time_to_resolution: str,
              buy_limit_price: Optional[float], sell_limit_price: Optional[float],
              buy_order_id: Optional[str], sell_order_id: Optional[str],
              buy_position_size: Optional[int] = None, sell_position_size: Optional[int] = None,
              available_balance: Optional[float] = None, market_price: Optional[float] = None,
              status: str = '', error: Optional[str] = None):
    """Log cycle action to cycles log file."""
    try:
        with open(CYCLES_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            # Include both order IDs and limit prices
            order_ids = f"{buy_order_id or ''}/{sell_order_id or ''}"
            limit_prices = f"{buy_limit_price or ''}/{sell_limit_price or ''}"
            writer.writerow([timestamp, cycle_num, ticker, btc_price_from_ticker, btc_price_current,
                           threshold_price, predicted_price, volatility, spread_cents, hours_until_resolution,
                           resolution_datetime, time_to_resolution, limit_prices, order_ids,
                           buy_position_size or '', sell_position_size or '', available_balance or '', market_price or '',
                           status, error or ''])
    except Exception as e:
        print(f"Warning: Could not write to cycles log: {e}")


def log_fill(timestamp: str, order_id: str, fill_id: str, ticker: str, action: str, 
             side: str, count: int, price: float):
    """Log fill execution to fills log file and update position tracking."""
    global positions
    
    try:
        with open(FILLS_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, order_id, fill_id, ticker, action, side, count, price])
        
        # Update position tracking
        if ticker not in positions:
            positions[ticker] = {
                'position': 0,
                'total_buy_cost': 0.0,
                'total_sell_revenue': 0.0,
                'buy_count': 0,
                'sell_count': 0
            }
        
        # Track positions: buying YES increases position, selling YES decreases position
        # side='yes' means trading YES side, action='buy' means buying YES
        if side == 'yes':
            if action == 'buy':
                positions[ticker]['position'] += count
                positions[ticker]['total_buy_cost'] += price * count
                positions[ticker]['buy_count'] += count
            elif action == 'sell':
                positions[ticker]['position'] -= count
                positions[ticker]['total_sell_revenue'] += price * count
                positions[ticker]['sell_count'] += count
        # side='no' means trading NO side - for binary options, this is inverse
        elif side == 'no':
            if action == 'buy':  # Buying NO = selling YES
                positions[ticker]['position'] -= count
                positions[ticker]['total_sell_revenue'] += (1.0 - price) * count
                positions[ticker]['sell_count'] += count
            elif action == 'sell':  # Selling NO = buying YES
                positions[ticker]['position'] += count
                positions[ticker]['total_buy_cost'] += (1.0 - price) * count
                positions[ticker]['buy_count'] += count
    except Exception as e:
        print(f"Warning: Could not write to fills log: {e}")


def get_btc_price_from_ticker(ticker: str) -> Optional[float]:
    """
    Derive BTC price from a Kalshi threshold ticker.
    
    Args:
        ticker: Kalshi threshold ticker string
    
    Returns:
        BTC price estimate in USD (threshold value), or None if ticker can't be parsed
    """
    threshold_info = parse_threshold_ticker(ticker)
    
    if threshold_info:
        return threshold_info.threshold
    
    return None


def predict_market_resolution_probability(ticker: str, current_price_override: Optional[float] = None) -> Optional[Tuple[float, float]]:
    """
    Predict the probability that a Kalshi threshold market will resolve YES and get volatility.
    
    Predicts P(BTC > threshold)
    
    Args:
        ticker: Kalshi threshold ticker string
        current_price_override: Optional current price to use instead of queue price (for fresh data)
    
    Returns:
        Tuple of (predicted_probability, volatility) or None if prediction fails
        - predicted_probability: (0-1) that market resolves YES
        - volatility: per-minute volatility (sigma_dt) for spread calculation
    """
    # Parse ticker to get market details
    threshold_info = parse_threshold_ticker(ticker)
    
    if not threshold_info:
        print(f"  Warning: Could not parse ticker {ticker}")
        return None
    
    # Calculate hours until resolution using the ticker's expiry time
    # Note: expiry_datetime from ticker is naive, but represents EST time
    # Use localize() instead of replace() to properly handle EST/EDT transitions
    if threshold_info.expiry_datetime.tzinfo is None:
        resolution_time = EST.localize(threshold_info.expiry_datetime)
    else:
        resolution_time = threshold_info.expiry_datetime.astimezone(EST)
    
    current_time = datetime.now(EST)
    hours_until_resolution = (resolution_time - current_time).total_seconds() / 3600.0
    
    # If market already resolved or very close (less than 1 minute), skip prediction
    # Markets resolve at the top of the hour, so we need a small buffer
    if hours_until_resolution < (1/60):  # Less than 1 minute
        print(f"  Warning: Market {ticker} already resolved or too close (hours: {hours_until_resolution:.2f})")
        return None
    
    try:
        # Get price data from queue (for historical data)
        prices, last_ts, S0_queue = get_price_data_for_prediction()
        
        if len(prices) < 10:
            print(f"  Warning: Insufficient price data in queue ({len(prices)} points, need at least 10)")
            return None
        
        # Use override price if provided (fresh current price), otherwise use queue price
        S0 = current_price_override if current_price_override is not None else S0_queue
        
        if S0 is None or S0 <= 0:
            print(f"  Warning: Invalid current price: {S0}")
            return None
        
        # Threshold market: Predict P(BTC > threshold)
        target_price = threshold_info.threshold
        prob_above = predict_price_probability(
            csv_path=None,  # Use queue data instead
            target_price=target_price,
            hours_ahead=hours_until_resolution,
            model=PREDICTION_MODEL,
            weight_15m=PREDICTION_WEIGHT_15M,
            weight_1h=PREDICTION_WEIGHT_1H,
            weight_6h=PREDICTION_WEIGHT_6H,
            prices_data=prices,
            last_timestamp=last_ts,
            current_price=S0,  # Use fresh current price if provided
            data_step_seconds=DATA_STEP_SECONDS
        )
        predicted_prob = prob_above['probability_above']
        volatility = prob_above['parameters']['sigma_dt']  # per-minute volatility
        
        return predicted_prob, volatility
        
    except Exception as e:
        # Prediction failed, return None (skip trading this cycle)
        import traceback
        print(f"  Warning: Prediction failed for {ticker}: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return None


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


def get_market_resolution_price(ticker: str) -> Optional[float]:
    """
    Get the resolution price for a market (1.0 for YES, 0.0 for NO, or None if not resolved).
    
    Args:
        ticker: Market ticker
    
    Returns:
        Resolution price (1.0 for YES, 0.0 for NO) or None if market hasn't resolved
    """
    try:
        client = get_kalshi_client()
        market = client.get_market(ticker=ticker)
        
        # Check if market is closed/resolved
        if hasattr(market.market, 'status'):
            status = market.market.status
            if status == 'closed':
                # Market resolved - check settlement price
                if hasattr(market.market, 'settlement_price'):
                    settlement = market.market.settlement_price
                    if settlement is not None:
                        return settlement / 100.0 if settlement > 1 else settlement
                # Fallback: check if there's a last_price that indicates resolution
                if hasattr(market.market, 'last_price') and market.market.last_price is not None:
                    last_price = market.market.last_price / 100.0 if market.market.last_price > 1 else market.market.last_price
                    # If last price is very close to 0 or 1, market likely resolved
                    if last_price < 0.01:
                        return 0.0
                    elif last_price > 0.99:
                        return 1.0
        
        return None  # Market hasn't resolved yet
    except Exception as e:
        return None


def calculate_pnl_for_resolved_markets():
    """
    Calculate PnL for markets that resolved in the previous hour.
    Called at the start of each new hour.
    """
    global positions, total_realized_pnl, last_resolved_hour
    
    year, month, day, hour = get_previous_est_hour()
    current_hour_key = (year, month, day, hour)
    
    # Skip if we already calculated PnL for this hour
    if last_resolved_hour == current_hour_key:
        return
    
    timestamp = datetime.now().isoformat()
    
    # Find all tickers that resolved in the previous hour
    resolved_tickers = []
    if not positions:
        # No positions to check
        return
    
    for ticker in list(positions.keys()):
        # Parse ticker to check if it matches the previous hour
        threshold_info = parse_threshold_ticker(ticker)
        
        if not threshold_info:
            # Can't parse ticker, skip
            continue
        
        ticker_hour = (threshold_info.year, threshold_info.month, threshold_info.day, threshold_info.hour)
        
        if ticker_hour == current_hour_key:
            resolved_tickers.append(ticker)
    
    # Calculate PnL for each resolved market
    for ticker in resolved_tickers:
        pos_data = positions[ticker]
        position = pos_data['position']
        
        if position == 0:
            # No position, skip
            continue
        
        # Get resolution price
        resolution_price = get_market_resolution_price(ticker)
        if resolution_price is None:
            # Market hasn't resolved yet, skip
            continue
        
        # Calculate average prices
        avg_buy_price = (pos_data['total_buy_cost'] / pos_data['buy_count']) if pos_data['buy_count'] > 0 else 0.0
        avg_sell_price = (pos_data['total_sell_revenue'] / pos_data['sell_count']) if pos_data['sell_count'] > 0 else 0.0
        
        # Calculate realized PnL
        # For long positions: PnL = (resolution_price - avg_buy_price) * position
        # For short positions: PnL = (avg_sell_price - resolution_price) * abs(position)
        if position > 0:
            # Long position
            realized_pnl = (resolution_price - avg_buy_price) * position
        else:
            # Short position
            realized_pnl = (avg_sell_price - resolution_price) * abs(position)
        
        total_realized_pnl += realized_pnl
        
        # Print PnL to console (no file logging)
        print(f"  PnL: {ticker} | Position: {position} | Resolution: {resolution_price:.4f} | PnL: ${realized_pnl:.2f} | Total: ${total_realized_pnl:.2f}")
        
        # Remove resolved position from tracking
        del positions[ticker]
    
    last_resolved_hour = current_hour_key


def _fetch_bitstamp_price() -> Optional[float]:
    """Fetch BTC price from Bitstamp API."""
    try:
        response = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/", timeout=2)
        response.raise_for_status()
        data = response.json()
        if "last" in data:
            return float(data["last"])
    except Exception:
        pass
    return None


def _fetch_kraken_price() -> Optional[float]:
    """Fetch BTC price from Kraken API."""
    try:
        response = requests.get("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", timeout=2)
        response.raise_for_status()
        data = response.json()
        if "result" in data and "XXBTZUSD" in data["result"]:
            return float(data["result"]["XXBTZUSD"]["c"][0])
    except Exception:
        pass
    return None


def get_current_btc_price_estimate() -> Optional[int]:
    """
    Get current BTC price from Bitstamp only.
    Always fetches fresh data (no caching).
    
    Returns:
        BTC price in USD (integer), or None if Bitstamp unavailable
    """
    bitstamp_price = _fetch_bitstamp_price()
    
    if bitstamp_price is not None:
        return int(bitstamp_price)
    
    return None


def find_available_ticker_threshold_only(year: int, month: int, day: int, hour: int, btc_price: Optional[int] = None) -> Optional[str]:
    """
    Find an available Kalshi BTC threshold ticker for the given hour, close to current BTC price.
    Only searches threshold markets (not interval/range markets).
    
    Args:
        year, month, day, hour: Target hour for the market
        btc_price: Optional BTC price in USD. If None, fetches current price.
    
    Returns:
        Ticker string or None if not found
    """
    if btc_price is None:
        btc_price = get_current_btc_price_estimate()
    
    try:
        tickers = get_available_btc_tickers_for_hour(
            year=year,
            month=month,
            day=day,
            hour=hour,
            btc_price=btc_price,
            limit=500,
            market_type='threshold'  # Only threshold markets
        )
        
        # Find first ticker that has prices
        for ticker in tickers:
            try:
                test_price = get_kalshi_price_by_ticker(ticker, action="buy")
                if test_price is not None:
                    return ticker
            except (ValueError, Exception):
                continue
        
        return None
    except Exception as e:
        print(f"Error finding ticker: {e}")
        return None


def check_recent_fills(ticker: str, since_seconds: int = 60) -> list:
    """
    Check for recent fills (executed trades) for a ticker.
    
    Args:
        ticker: Market ticker to check fills for
        since_seconds: Only check fills from the last N seconds
    
    Returns:
        List of fill dictionaries
    """
    try:
        client = get_kalshi_client()
        fills_response = client.get_fills(limit=100)
        
        if not fills_response or not hasattr(fills_response, 'fills') or not fills_response.fills:
            return []
        
        cutoff_time = datetime.now() - timedelta(seconds=since_seconds)
        recent_fills = []
        
        for fill in fills_response.fills:
            if not (hasattr(fill, 'ticker') and fill.ticker == ticker):
                continue
            
            # Parse timestamp
            fill_time = None
            if hasattr(fill, 'ts') and fill.ts:
                try:
                    if isinstance(fill.ts, (int, float)):
                        fill_time = datetime.fromtimestamp(fill.ts)
                    elif isinstance(fill.ts, str):
                        fill_time = datetime.fromisoformat(fill.ts.replace('Z', '+00:00'))
                except Exception:
                    pass
            
            if fill_time is None or fill_time >= cutoff_time:
                fill_price = getattr(fill, 'price', None)
                recent_fills.append({
                    'order_id': getattr(fill, 'order_id', None),
                    'fill_id': getattr(fill, 'fill_id', None),
                    'ticker': fill.ticker,
                    'action': getattr(fill, 'action', None),
                    'side': getattr(fill, 'side', None),
                    'count': getattr(fill, 'count', None),
                    'price': fill_price,
                    'price_decimal': fill_price,
                    'timestamp': fill_time.isoformat() if fill_time else None,
                })
        
        return recent_fills
    except Exception as e:
        print(f"  Warning: Could not check fills: {e}")
        return []


def calculate_dynamic_spread(volatility: float) -> float:
    """
    Calculate dynamic spread in cents based on volatility.
    
    Args:
        volatility: Per-minute volatility (sigma_dt)
    
    Returns:
        Spread in cents
    """
    # Convert per-minute volatility to spread
    # Higher volatility = wider spread
    # Formula: base_spread + volatility_multiplier * volatility
    spread_cents = BASE_SPREAD_CENTS + (VOLATILITY_MULTIPLIER * volatility)
    
    # Clamp between min and max
    spread_cents = max(BASE_SPREAD_CENTS, min(spread_cents, MAX_SPREAD_CENTS))
    
    return spread_cents


def calculate_market_taking_fee(price: float) -> float:
    """
    Calculate market taking fee for a given price.
    Fee formula: 0.07 * p * (1-p)
    
    Args:
        price: Contract price (0-1)
    
    Returns:
        Fee as a decimal (e.g., 0.0175 for 1.75%)
    """
    return MARKET_TAKING_FEE_RATE * price * (1.0 - price)


def calculate_optimal_position_size(
    predicted_prob: float,
    market_price: Optional[float],
    available_balance: Optional[float] = None,
    volatility: Optional[float] = None
) -> int:
    """
    Calculate optimal position size using Kelly criterion with risk limits.
    
    Kelly formula: f* = (p * b - q) / b
    where:
    - p = probability of winning (predicted_prob)
    - q = probability of losing (1 - p)
    - b = net odds received on the wager
    
    For binary options:
    - If we buy at price P, we win (1-P) if correct, lose P if wrong
    - Net odds b = (1-P) / P = (1/P) - 1
    
    Args:
        predicted_prob: Our predicted probability (0-1)
        market_price: Current market price (0-1), None if unavailable
        available_balance: Available balance in dollars, None to skip balance check
        volatility: Volatility estimate (for confidence adjustment)
    
    Returns:
        Optimal number of contracts to trade
    """
    # Fallback to base volume if we can't calculate optimal size
    if market_price is None:
        return VOLUME
    
    # Calculate edge: difference between our prediction and market price
    edge = abs(predicted_prob - market_price)
    
    # If edge is too small (< 1%), use base volume
    if edge < 0.01:
        return VOLUME
    
    # Determine which side has edge
    if predicted_prob > market_price:
        # We think it's more likely than market - buy YES
        # Win probability = predicted_prob
        # If we buy at market_price, we win (1 - market_price) if correct, lose market_price if wrong
        p = predicted_prob
        win_amount = 1.0 - market_price  # What we get if we win
        lose_amount = market_price  # What we lose if we're wrong
    else:
        # We think it's less likely than market - sell YES (or buy NO)
        # Win probability = 1 - predicted_prob
        p = 1.0 - predicted_prob
        win_amount = market_price  # What we get if we win (selling YES at market_price)
        lose_amount = 1.0 - market_price  # What we lose if we're wrong
    
    # Net odds: b = win_amount / lose_amount
    if lose_amount == 0:
        return VOLUME  # Avoid division by zero
    
    b = win_amount / lose_amount
    
    # Kelly criterion: f* = (p * b - q) / b
    # where q = 1 - p
    q = 1.0 - p
    kelly_fraction = (p * b - q) / b
    
    # Apply Kelly fraction multiplier (use quarter Kelly for safety)
    kelly_fraction *= KELLY_FRACTION
    
    # Ensure non-negative
    kelly_fraction = max(0.0, kelly_fraction)
    
    # If Kelly suggests very small position, use base volume
    if kelly_fraction < 0.01:
        return VOLUME
    
    # Calculate position size based on balance
    if available_balance is not None and available_balance > 0:
        # Maximum we're willing to risk per contract
        # For buying: risk = market_price per contract
        # For selling: risk = (1 - market_price) per contract
        if predicted_prob > market_price:
            risk_per_contract = market_price
        else:
            risk_per_contract = 1.0 - market_price
        
        # Calculate position size from Kelly fraction
        # Position value = kelly_fraction * available_balance
        position_value = kelly_fraction * available_balance
        
        # Number of contracts = position_value / risk_per_contract
        contracts_from_kelly = int(position_value / risk_per_contract)
        
        # Apply maximum position size limit
        max_contracts_from_balance = int((MAX_POSITION_PCT_OF_BALANCE * available_balance) / risk_per_contract)
        
        # Take minimum of Kelly suggestion and max limit
        optimal_contracts = min(contracts_from_kelly, max_contracts_from_balance)
        
        # Ensure minimum position size
        optimal_contracts = max(optimal_contracts, MIN_POSITION_SIZE)
        
        return optimal_contracts
    else:
        # No balance info - scale base volume by edge and Kelly fraction
        # Scale by edge: larger edge = larger position
        edge_multiplier = min(edge / 0.05, 2.0)  # Cap at 2x for edges > 5%
        scaled_volume = int(VOLUME * edge_multiplier * kelly_fraction * 4)  # *4 to scale from fraction to reasonable size
        return max(scaled_volume, MIN_POSITION_SIZE)


def place_limit_order_with_spread(
    ticker: str,
    predicted_price: float,
    volatility: float,
    action: str = "buy",
    position_size: Optional[int] = None
) -> Tuple[Optional[dict], Optional[float], Optional[str]]:
    """
    Place a limit order with dynamic spread offset from predicted price.
    Ensures orders are placed away from market price to avoid market taking fees.
    Buy orders: predicted_price - (spread/2), but at least MARKET_MAKING_BUFFER_CENTS below market
    Sell orders: predicted_price + (spread/2), but at least MARKET_MAKING_BUFFER_CENTS above market
    
    Args:
        ticker: Market ticker
        predicted_price: Predicted probability (0-1) to base limit on
        volatility: Per-minute volatility for spread calculation
        action: "buy" or "sell"
    
    Returns:
        Tuple of (order_result, limit_price, error_message)
    """
    # Calculate dynamic spread based on volatility
    spread_cents = calculate_dynamic_spread(volatility)
    
    # Split spread in half: buy below, sell above
    half_spread_decimal = (spread_cents / 2.0) / 100.0
    
    if action == "buy":
        limit_price_raw = predicted_price - half_spread_decimal
    else:
        limit_price_raw = predicted_price + half_spread_decimal
    
    # Get current market price to ensure we're market making (not market taking)
    try:
        current_market_price = get_kalshi_price_by_ticker(ticker, action=action)
        if current_market_price is not None:
            buffer_decimal = MARKET_MAKING_BUFFER_CENTS / 100.0
            
            if action == "buy":
                # For buy orders, ensure we're at least buffer_cents below market (market making)
                max_buy_price = current_market_price - buffer_decimal
                if limit_price_raw > max_buy_price:
                    limit_price_raw = max_buy_price
            else:  # sell
                # For sell orders, ensure we're at least buffer_cents above market (market making)
                min_sell_price = current_market_price + buffer_decimal
                if limit_price_raw < min_sell_price:
                    limit_price_raw = min_sell_price
    except Exception:
        # If we can't get market price, proceed with original limit price
        pass
    
    # Round to nearest cent (0.01) - same as Kalshi API does
    def round_to_cent(price: float) -> Optional[float]:
        """Round price to nearest cent (0.01)."""
        if price <= 0 or price >= 1:
            return None
        return round(price * 100) / 100.0
    
    limit_price = round_to_cent(limit_price_raw)
    
    # Skip order if limit price is invalid (don't clamp)
    if limit_price is None:
        return None, limit_price_raw, f"Invalid limit price: {limit_price_raw:.4f}"
    
    # Use provided position size or fallback to base volume
    count = position_size if position_size is not None else VOLUME
    
    try:
        result = execute_trade_by_ticker(
            ticker=ticker,
            action=action,
            count=count,
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


def initialize_price_queue_from_bitstamp():
    """
    Initialize the cached price data queue from Bitstamp API.
    Called BEFORE the first trading cycle to collect all historical data.
    Loads the last DATA_HOURS_BACK hours of data into the cached queue.
    
    For per-second or per-5-second data:
    - Fetches 1-minute OHLC data from Bitstamp (only supported granularity)
    - Interpolates to finer granularity if needed
    - Starts daemon thread to update queue every DATA_STEP_SECONDS
    """
    global price_data_queue, price_data_initialized
    
    if price_data_initialized:
        print("  Price queue already initialized, skipping...")
        return  # Already initialized
    
    try:
        from bitstamp_data_fetcher import fetch_bitstamp_ohlc_historical, fetch_bitstamp_ticker
        
        # Fetch 1-minute OHLC data from Bitstamp (this is what's available)
        print(f"  Fetching {DATA_HOURS_BACK} hours of 1-minute data from Bitstamp...")
        ohlc_data = fetch_bitstamp_ohlc_historical(
            currency_pair="btcusd",
            hours_back=DATA_HOURS_BACK,
            step=60  # Bitstamp only supports 60-second intervals
        )
        
        if not ohlc_data:
            raise ValueError("No data fetched from Bitstamp")
        
        price_data_queue.clear()
        
        # Clear price data cache since we're reinitializing
        global _price_data_cache
        _price_data_cache['queue_hash'] = None
        _price_data_cache['prices_array'] = None
        
        # If we want per-second or per-5-second data, we need to interpolate from 1-minute data
        # Use volatility-preserving interpolation to maintain reasonable parameters on cold start
        if DATA_STEP_SECONDS < 60:
            # First, estimate volatility from 1-minute data to guide interpolation
            if len(ohlc_data) >= 2:
                # Extract prices as numpy array for efficient calculation
                prices_array = np.array([price for _, price in ohlc_data[:100]], dtype=np.float64)  # Use up to 100 points
                
                if len(prices_array) >= 2:
                    # Calculate log returns using numpy (more efficient)
                    log_returns = np.diff(np.log(prices_array))
                    
                    # Estimate per-minute volatility using numpy std (sample std, ddof=1)
                    if len(log_returns) >= 2:
                        minute_volatility = log_returns.std(ddof=1)
                        # Scale to per-interval volatility (for DATA_STEP_SECONDS)
                        interval_volatility = minute_volatility * np.sqrt(DATA_STEP_SECONDS / 60.0)
                    else:
                        # Fallback: use a reasonable default volatility
                        interval_volatility = 0.001 * np.sqrt(DATA_STEP_SECONDS / 60.0)  # ~0.1% per minute
                else:
                    interval_volatility = 0.001 * np.sqrt(DATA_STEP_SECONDS / 60.0)
            else:
                interval_volatility = 0.001 * np.sqrt(DATA_STEP_SECONDS / 60.0)
            
            # For finer granularity, interpolate from 1-minute data using Brownian bridge
            # Each minute has 60 seconds, so we'll create DATA_STEP_SECONDS intervals per minute
            points_per_minute = 60 // DATA_STEP_SECONDS
            
            for i, (ts, price) in enumerate(ohlc_data):
                # Add the minute data point
                price_data_queue.append((ts, price))
                
                # If not the last point, interpolate intermediate points using Brownian bridge
                if i < len(ohlc_data) - 1:
                    next_ts, next_price = ohlc_data[i + 1]
                    time_diff = next_ts - ts
                    price_diff = next_price - price
                    
                    # Use Brownian bridge interpolation to preserve volatility
                    # This creates a random walk between endpoints that preserves variance
                    # Brownian bridge variance at time t: sigma^2 * t * (1 - t)
                    num_intervals = points_per_minute - 1
                    if num_intervals > 0:
                        # Generate deviations from linear interpolation with proper variance
                        # The variance of a Brownian bridge at time t is sigma^2 * t * (1 - t)
                        bridge_deviations = []
                        for j in range(1, num_intervals + 1):
                            t = j / points_per_minute
                            
                            # Brownian bridge variance: var(t) = sigma^2 * t * (1 - t)
                            # This ensures variance is 0 at endpoints and maximum in the middle
                            variance_bridge = interval_volatility**2 * t * (1 - t)
                            std_bridge = math.sqrt(max(0, variance_bridge))
                            
                            if j < num_intervals:
                                # Generate deviation from linear interpolation
                                deviation = random.gauss(0, std_bridge)
                                bridge_deviations.append(deviation)
                            else:
                                # Last point: ensure bridge ends at next_price (sum of deviations = 0)
                                bridge_deviations.append(-sum(bridge_deviations))
                        
                        # Build cumulative bridge path (deviations from linear interpolation)
                        bridge_path = [0.0]  # Start at 0 (no deviation at start)
                        for dev in bridge_deviations:
                            bridge_path.append(bridge_path[-1] + dev)
                        
                        # Add interpolated points
                        for j in range(1, points_per_minute):
                            interp_ts = ts + (j * DATA_STEP_SECONDS)
                            if interp_ts < next_ts:
                                # Linear interpolation component
                                t = j / points_per_minute
                                linear_component = price + t * price_diff
                                
                                # Bridge deviation component (preserves volatility)
                                bridge_component = bridge_path[j] if j < len(bridge_path) else 0.0
                                
                                interp_price = linear_component + bridge_component
                                
                                # Ensure price stays reasonable (within 1% bounds)
                                min_price = min(price, next_price) * 0.99
                                max_price = max(price, next_price) * 1.01
                                interp_price = max(min_price, min(max_price, interp_price))
                                
                                price_data_queue.append((interp_ts, interp_price))
            
            print(f"  Interpolated to {DATA_STEP_SECONDS}-second intervals using volatility-preserving method: {len(price_data_queue)} points")
            
            # Fetch current price to add most recent point
            current_ticker = fetch_bitstamp_ticker()
            if current_ticker:
                price_data_queue.append(current_ticker)
        else:
            # For per-minute data, use OHLC data directly
            for ts, price in ohlc_data:
                price_data_queue.append((ts, price))
            
            print(f"  Loaded {len(price_data_queue)} data points (1-minute intervals)")
        
        price_data_initialized = True
        print(f"  ✓ Initialized cached price queue with {len(price_data_queue)} data points")
        
        # Start daemon thread for queue updates (runs every DATA_STEP_SECONDS)
        start_price_queue_updater()
        print(f"  ✓ Started daemon thread to update queue every {DATA_STEP_SECONDS} seconds")
        
    except Exception as e:
        print(f"  ✗ Failed to initialize price queue from Bitstamp: {e}")
        raise


def update_price_queue():
    """
    Update the price queue with new BTC price data.
    Adds new data point if DATA_STEP_SECONDS have passed since last update.
    Thread-safe and handles missed intervals gracefully.
    """
    global price_data_queue, last_price_update_time
    
    with price_queue_lock:
        now = time.time()
        current_interval = int(now // DATA_STEP_SECONDS)  # Current interval since epoch
        last_interval = int(last_price_update_time // DATA_STEP_SECONDS) if last_price_update_time > 0 else None
        
        # Check if we need to update
        if last_interval is None:
            # First update - always do it
            needs_update = True
            intervals_to_catch_up = 0
        elif current_interval > last_interval:
            # New interval(s) have passed
            intervals_to_catch_up = current_interval - last_interval
            needs_update = True
        else:
            # Same interval - no update needed
            needs_update = False
            intervals_to_catch_up = 0
        
        if not needs_update:
            return False
        
        # Fetch current BTC price (with retry on failure)
        btc_price = None
        for attempt in range(3):  # Try up to 3 times
            try:
                btc_price = get_current_btc_price_estimate()
                if btc_price is not None:
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.5)  # Brief delay before retry
                else:
                    print(f"  Warning: Failed to fetch BTC price after 3 attempts: {e}")
        
        if btc_price is not None:
            # If we missed intervals, add the current price for each missed interval
            # (using same price for missed intervals is better than having gaps)
            for i in range(intervals_to_catch_up):
                # Use current time minus the number of intervals we're catching up
                timestamp = now - (intervals_to_catch_up - i - 1) * DATA_STEP_SECONDS
                price_data_queue.append((timestamp, float(btc_price)))
            
            # Update last update time to current interval
            last_price_update_time = now
            
            # Clear parameter cache to force volatility recalculation with fresh data
            clear_parameter_cache()
            
            # Clear price data cache since queue has changed
            global _price_data_cache
            _price_data_cache['queue_hash'] = None
            _price_data_cache['prices_array'] = None
            
            if intervals_to_catch_up > 1:
                print(f"  Caught up {intervals_to_catch_up} missed interval(s) ({DATA_STEP_SECONDS}s each) in price queue")
            
            return True
    
    return False


def price_queue_updater_thread():
    """
    Background thread that updates the price queue every DATA_STEP_SECONDS.
    Runs independently of trading cycles to ensure timely updates.
    """
    while True:
        try:
            update_price_queue()
            # Sleep until the start of the next interval
            now = time.time()
            next_interval = int(now // DATA_STEP_SECONDS) + 1
            sleep_until = next_interval * DATA_STEP_SECONDS
            sleep_time = max(0.1, sleep_until - now)  # At least 100ms sleep
            time.sleep(sleep_time)
        except Exception as e:
            print(f"  Warning: Error in price queue updater thread: {e}")
            time.sleep(DATA_STEP_SECONDS)  # Sleep one interval before retrying on error


def start_price_queue_updater():
    """
    Start the background thread for price queue updates.
    Should be called once after initializing the queue.
    """
    global price_queue_thread
    
    if price_queue_thread is not None and price_queue_thread.is_alive():
        return  # Already running
    
    price_queue_thread = threading.Thread(target=price_queue_updater_thread, daemon=True)
    price_queue_thread.start()
    print(f"  Started background price queue updater thread")


def get_price_data_for_prediction() -> Tuple[np.ndarray, float, float]:
    """
    Get price data from queue in format expected by predict_price_probability.
    Thread-safe access to the queue with caching for efficiency.
    
    Returns:
        Tuple of (prices_array as numpy array, last_timestamp, last_price)
    """
    global price_data_queue, _price_data_cache
    
    with price_queue_lock:
        if len(price_data_queue) == 0:
            raise ValueError("Price queue is empty - cannot make predictions")
        
        # Calculate hash of queue contents to detect changes
        # Use first and last few timestamps/prices as a simple hash
        queue_items = list(price_data_queue)
        if len(queue_items) >= 4:
            queue_hash = hash((
                queue_items[0][0], queue_items[0][1],  # First timestamp, price
                queue_items[-1][0], queue_items[-1][1],  # Last timestamp, price
                len(queue_items)  # Length
            ))
        else:
            queue_hash = hash(tuple(queue_items))
        
        # Check cache
        if (_price_data_cache['queue_hash'] == queue_hash and 
            _price_data_cache['prices_array'] is not None):
            # Cache hit - return cached numpy array
            return _price_data_cache['prices_array'], _price_data_cache['last_ts'], _price_data_cache['last_price']
        
        # Cache miss - convert queue to numpy array
        prices_list = [price for _, price in price_data_queue]
        prices_array = np.array(prices_list, dtype=np.float64)
        
        last_ts = queue_items[-1][0]
        last_price = queue_items[-1][1]
        
        # Update cache
        _price_data_cache['prices_array'] = prices_array
        _price_data_cache['last_ts'] = last_ts
        _price_data_cache['last_price'] = last_price
        _price_data_cache['queue_hash'] = queue_hash
        
        return prices_array, last_ts, last_price


def run_trading_cycle(cycle_num: int) -> bool:
    """
    Run a single trading cycle.
    
    Args:
        cycle_num: Cycle number for logging
    
    Returns:
        True if cycle completed successfully, False otherwise
    """
    global price_data_initialized
    
    timestamp = datetime.now().isoformat()
    
    # Price queue should already be initialized before first cycle
    # This check is just a safety fallback
    if not price_data_initialized:
        print(f"[{cycle_num}] ERROR: Price queue not initialized! This should not happen.")
        return False
    
    # Note: Price queue updates are now handled by background thread
    # No need to call update_price_queue() here anymore
    
    # Calculate PnL for markets that resolved in the previous hour (at start of new hour)
    calculate_pnl_for_resolved_markets()
    
    # Find available ticker for current hour (no external BTC price needed)
    year, month, day, hour = get_current_est_hour()
    ticker = find_available_ticker_threshold_only(year, month, day, hour)
    
    if not ticker:
        print(f"[{cycle_num}] SKIP: No available markets for {year}-{month:02d}-{day:02d} {hour:02d}:00")
        return False
    
    # Derive BTC price from ticker (for logging)
    btc_price_from_ticker = get_btc_price_from_ticker(ticker) or 0.0
    
    # Parse ticker to get threshold price and calculate hours until resolution
    threshold_info = parse_threshold_ticker(ticker)
    if not threshold_info:
        print(f"[{cycle_num}] SKIP: Could not parse ticker {ticker}")
        return False
    
    threshold_price = threshold_info.threshold
    # Convert naive datetime to EST-aware datetime (Bitcoin markets use EST)
    # Use localize() instead of replace() to properly handle EST/EDT transitions
    if threshold_info.expiry_datetime.tzinfo is None:
        resolution_time = EST.localize(threshold_info.expiry_datetime)
    else:
        resolution_time = threshold_info.expiry_datetime.astimezone(EST)
    current_time = datetime.now(EST)
    hours_until_resolution = (resolution_time - current_time).total_seconds() / 3600.0
    resolution_datetime_str = resolution_time.isoformat()
    time_to_resolution_str = format_time_to_resolution(hours_until_resolution)
    
    # Get current BTC price fresh (for accurate logging and predictions)
    # Note: Queue is updated every minute, but we want current price for each cycle
    btc_price_current = get_current_btc_price_estimate()
    if btc_price_current is None:
        # Fallback to queue price if fetch fails
        try:
            _, _, btc_price_current = get_price_data_for_prediction()
        except Exception as e:
            print(f"[{cycle_num}] WARNING: Could not get current BTC price: {e}")
            btc_price_current = btc_price_from_ticker  # Final fallback to ticker price
    
    # Get predicted probability and volatility (required for trading)
    # Pass fresh current price to ensure predictions use up-to-date price
    prediction_result = predict_market_resolution_probability(ticker, current_price_override=btc_price_current)
    if prediction_result is None:
        # Error details are already printed by predict_market_resolution_probability
        print(f"[{cycle_num}] SKIP: Could not get prediction for {ticker}")
        return False
    
    predicted_prob, volatility = prediction_result
    
    # Validate prediction is reasonable (between 0 and 1, not too extreme)
    if not (0.01 <= predicted_prob <= 0.99):
        print(f"[{cycle_num}] SKIP: Prediction {predicted_prob:.3f} too extreme for {ticker}")
        return False
    
    # Calculate dynamic spread based on volatility
    spread_cents = calculate_dynamic_spread(volatility)
    
    # Calculate what the new limit prices would be (before placing orders)
    half_spread_decimal = (spread_cents / 2.0) / 100.0
    new_buy_limit_price_raw = predicted_prob - half_spread_decimal
    new_sell_limit_price_raw = predicted_prob + half_spread_decimal
    
    # Round to nearest cent (0.01) - same as Kalshi API does
    def round_to_cent(price: float) -> Optional[float]:
        """Round price to nearest cent (0.01)."""
        if price <= 0 or price >= 1:
            return None
        return round(price * 100) / 100.0
    
    new_buy_limit_price = round_to_cent(new_buy_limit_price_raw)
    new_sell_limit_price = round_to_cent(new_sell_limit_price_raw)
    
    # Check if we need to update orders
    global last_ticker, last_order_details
    need_to_update = False
    
    # Always update if ticker changed
    if last_ticker is not None and last_ticker != ticker:
        cancel_all_orders_for_ticker(last_ticker)
        need_to_update = True
    elif last_ticker != ticker:
        # First time for this ticker
        need_to_update = True
    else:
        # Same ticker - check if rounded limit prices changed
        # Only update if the actual rounded prices sent to Kalshi would be different
        old_buy_limit_price = round_to_cent(last_order_details['buy_limit_price']) if last_order_details['buy_limit_price'] is not None else None
        old_sell_limit_price = round_to_cent(last_order_details['sell_limit_price']) if last_order_details['sell_limit_price'] is not None else None
        
        # Check if rounded limit prices changed (this is what actually matters for Kalshi)
        buy_price_changed = (old_buy_limit_price != new_buy_limit_price)
        sell_price_changed = (old_sell_limit_price != new_sell_limit_price)
        
        # Only update if rounded prices changed (don't care about raw prediction/spread if rounded prices are same)
        if buy_price_changed or sell_price_changed:
            need_to_update = True
    
    # Initialize position size and balance variables (will be set if orders are updated)
    buy_position_size = None
    sell_position_size = None
    available_balance = None
    market_price = None
    
    # Only cancel and replace if needed
    if need_to_update:
        # Cancel existing orders for current ticker
        cancel_all_orders_for_ticker(ticker)
        last_ticker = ticker
        
        # Calculate optimal position sizes based on edge and balance
        try:
            from app.utils import get_available_funds, get_kalshi_price_by_ticker
            available_balance = get_available_funds()
            market_price = get_kalshi_price_by_ticker(ticker, action="buy")  # Use ask price as reference
        except Exception:
            available_balance = None
            market_price = None
        
        buy_position_size = calculate_optimal_position_size(
            predicted_prob=predicted_prob,
            market_price=market_price,
            available_balance=available_balance,
            volatility=volatility
        )
        
        # For sell side, use inverse probability
        sell_position_size = calculate_optimal_position_size(
            predicted_prob=1.0 - predicted_prob,  # Inverse for sell side
            market_price=1.0 - market_price if market_price else None,
            available_balance=available_balance,
            volatility=volatility
        )
        
        # Log position sizing decision
        if market_price:
            edge = abs(predicted_prob - market_price)
            edge_pct = edge * 100
            print(f"  Position sizing: edge={edge_pct:.2f}%, buy={buy_position_size}, sell={sell_position_size}, balance=${available_balance:.2f}" if available_balance else f"  Position sizing: edge={edge_pct:.2f}%, buy={buy_position_size}, sell={sell_position_size}")
        
        # Place both buy and sell limit orders using predicted probability and volatility
        buy_result, buy_limit_price, buy_error = place_limit_order_with_spread(
            ticker, predicted_prob, volatility, action="buy", position_size=buy_position_size
        )
        sell_result, sell_limit_price, sell_error = place_limit_order_with_spread(
            ticker, predicted_prob, volatility, action="sell", position_size=sell_position_size
        )
        
        # Update tracking
        last_order_details = {
            'ticker': ticker,
            'predicted_prob': predicted_prob,
            'volatility': volatility,
            'buy_limit_price': buy_limit_price,
            'sell_limit_price': sell_limit_price
        }
    else:
        # Orders are still valid - skip placement
        buy_result = None
        sell_result = None
        buy_limit_price = last_order_details['buy_limit_price']
        sell_limit_price = last_order_details['sell_limit_price']
        buy_error = None
        sell_error = None
        status = "orders_unchanged"
    
    buy_order_id = buy_result.get('order_id', None) if buy_result else None
    sell_order_id = sell_result.get('order_id', None) if sell_result else None
    
    # Determine status
    if need_to_update:
        # We placed new orders
        if buy_result and sell_result:
            status = "both_placed"
        elif buy_result:
            status = "buy_only"
        elif sell_result:
            status = "sell_only"
        else:
            status = "failed"
    else:
        # Orders unchanged - status already set above
        pass  # status = "orders_unchanged" already set
    
    error_msg = None
    if buy_error and sell_error:
        error_msg = f"Buy: {buy_error}; Sell: {sell_error}"
    elif buy_error:
        error_msg = f"Buy: {buy_error}"
    elif sell_error:
        error_msg = f"Sell: {sell_error}"
    
    log_cycle(timestamp, cycle_num, ticker, btc_price_from_ticker, btc_price_current,
              threshold_price, predicted_prob, volatility, spread_cents, hours_until_resolution,
              resolution_datetime_str, time_to_resolution_str,
              buy_limit_price, sell_limit_price, buy_order_id, sell_order_id,
              buy_position_size, sell_position_size, available_balance, market_price,
              status, error_msg)
    
    # Console output (one line)
    pred_str = f"Pred={predicted_prob:.3f}"
    vol_str = f"σ={volatility:.6f}"
    spread_str = f"spread={spread_cents:.1f}c"
    
    if buy_result and sell_result:
        print(f"[{cycle_num}] {ticker} | BTC=${btc_price_current:,.0f} | {pred_str} | {vol_str} | {spread_str} | Buy@{buy_limit_price:.4f} Sell@{sell_limit_price:.4f} | Orders: {buy_order_id}/{sell_order_id}")
    elif buy_result:
        print(f"[{cycle_num}] {ticker} | BTC=${btc_price_current:,.0f} | {pred_str} | {vol_str} | {spread_str} | Buy@{buy_limit_price:.4f} ✓ | Sell SKIPPED")
    elif sell_result:
        print(f"[{cycle_num}] {ticker} | BTC=${btc_price_current:,.0f} | {pred_str} | {vol_str} | {spread_str} | Buy SKIPPED | Sell@{sell_limit_price:.4f} ✓")
    else:
        print(f"[{cycle_num}] {ticker} | BTC=${btc_price_current:,.0f} | {pred_str} | {vol_str} | {spread_str} | FAILED: {error_msg}")
    
    # Check for recent fills and log new ones
    global logged_fill_ids
    recent_fills = check_recent_fills(ticker, since_seconds=60)
    
    for fill in recent_fills:
        fill_id = fill.get('fill_id')
        if fill_id and fill_id not in logged_fill_ids:
            log_fill(
                fill.get('timestamp') or timestamp,
                fill.get('order_id', ''),
                fill_id,
                fill.get('ticker', ticker),
                fill.get('action', ''),
                fill.get('side', ''),
                fill.get('count', 0),
                fill.get('price_decimal', 0)
            )
            logged_fill_ids.add(fill_id)
            if len(logged_fill_ids) > 1000:
                logged_fill_ids = set(list(logged_fill_ids)[500:])
    
    return status != "failed"


def main():
    """Main trading algorithm - runs continuously."""
    # Initialize log files
    init_log_files()
    
    print("=" * 60)
    print("Volatility Trading Algorithm - Continuous Mode")
    print("=" * 60)
    print(f"Dynamic Spread: {BASE_SPREAD_CENTS}-{MAX_SPREAD_CENTS} cents (based on volatility)")
    print(f"  Base: {BASE_SPREAD_CENTS}c, Multiplier: {VOLATILITY_MULTIPLIER}x volatility")
    print(f"Volume: {VOLUME} contracts per side")
    print(f"Interval: {CYCLE_INTERVAL_SECONDS}s")
    print(f"Prediction Model: {PREDICTION_MODEL}")
    print(f"Data Source: Bitstamp API (no CSV)")
    print(f"Data Frequency: {DATA_STEP_SECONDS}-second intervals")
    print(f"Historical Data: {DATA_HOURS_BACK} hours")
    print(f"Cycles log: {CYCLES_LOG_FILE}")
    print(f"Fills log: {FILLS_LOG_FILE}")
    print("=" * 60)
    
    # Initialize price queue from Bitstamp BEFORE first cycle
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
    
    # Fetch initial BTC price for display
    btc_price = get_current_btc_price_estimate()
    if btc_price:
        print(f"Current BTC Price: ${btc_price:,}")
    else:
        print(f"⚠️  Could not fetch BTC price - will continue anyway")
    
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

