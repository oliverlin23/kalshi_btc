"""
Functions to search and list available Kalshi markets.
"""

import os
import sys
import requests
from datetime import datetime
from typing import List, Optional, Dict
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import base64

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.utils import get_kalshi_client
from app.ticker_utils import parse_range_ticker, parse_threshold_ticker


def get_kalshi_markets(
    event_ticker: Optional[str] = None,
    series_ticker: Optional[str] = None,
    limit: int = 100,
    cursor: Optional[str] = None,
    status: Optional[str] = None
) -> Dict:
    """
    Get list of markets from Kalshi API.
    
    Args:
        event_ticker: Optional event ticker to filter by
        series_ticker: Optional series ticker to filter by (e.g., 'KXBTC', 'KXBTCD')
        limit: Maximum number of markets to return (default: 100, max: 1000)
        cursor: Pagination cursor
        status: Filter by status (e.g., 'open', 'closed')
    
    Returns:
        Dictionary with markets data (JSON format)
    """
    # Always use direct API call to get JSON dicts instead of Market objects
    params = {}
    if event_ticker:
        params['event_ticker'] = event_ticker
    if series_ticker:
        params['series_ticker'] = series_ticker
    if limit:
        params['limit'] = min(limit, 1000)  # API max is 1000
    if cursor:
        params['cursor'] = cursor
    if status:
        params['status'] = status
    
    return _get_markets_direct(params)


def _get_markets_direct(params: Dict) -> Dict:
    """Make direct API call to get markets."""
    api_key_id = os.getenv("KALSHI_API_KEY_ID")
    default_key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "private_key.txt")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", default_key_path)
    
    if not api_key_id:
        raise ValueError("Error: Set KALSHI_API_KEY_ID environment variable")
    
    with open(private_key_path, "r") as f:
        private_key_pem = f.read()
    
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(),
        password=None
    )
    
    # Build request
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    timestamp = str(int(datetime.now().timestamp()))
    
    # Build query string
    query_parts = []
    for key, value in params.items():
        if value is not None:
            query_parts.append(f"{key}={value}")
    query_string = "&".join(query_parts)
    path = f"/trade-api/v2/markets" + (f"?{query_string}" if query_string else "")
    
    message = f"{timestamp}GET{path}"
    
    # Sign the message
    signature = private_key.sign(
        message.encode(),
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    
    signature_b64 = base64.b64encode(signature).decode()
    
    headers = {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-SIGNATURE": signature_b64,
        "KALSHI-ACCESS-TIMESTAMP": timestamp
    }
    
    full_url = url + (f"?{query_string}" if query_string else "")
    response = requests.get(full_url, headers=headers)
    response.raise_for_status()
    
    return response.json()


def search_btc_markets(
    year: Optional[int] = None,
    month: Optional[int] = None,
    day: Optional[int] = None,
    hour: Optional[int] = None,
    limit: int = 100,
    market_type: Optional[str] = None  # 'range' or 'threshold' or None for both
) -> List[Dict]:
    """
    Search for BTC markets matching specific criteria.
    
    Args:
        year: Filter by year
        month: Filter by month (1-12)
        day: Filter by day
        hour: Filter by hour (0-23)
        limit: Maximum number of results
        market_type: 'range' for KXBTC, 'threshold' for KXBTCD, None for both
    
    Returns:
        List of market dictionaries with ticker and other info
    """
    try:
        all_markets = []
        
        # Fetch range markets (KXBTC)
        if market_type is None or market_type == 'range':
            try:
                range_response = get_kalshi_markets(limit=limit, series_ticker='KXBTC', status='open')
                # Handle different response structures
                markets_list = None
                if isinstance(range_response, dict):
                    markets_list = range_response.get('markets', [])
                elif hasattr(range_response, 'markets'):
                    markets_list = range_response.markets
                
                if markets_list:
                    all_markets.extend(markets_list)
            except Exception as e:
                print(f"Warning: Could not fetch range markets: {e}")
        
        # Fetch threshold markets (KXBTCD)
        if market_type is None or market_type == 'threshold':
            try:
                threshold_response = get_kalshi_markets(limit=limit, series_ticker='KXBTCD', status='open')
                # Handle different response structures
                markets_list = None
                if isinstance(threshold_response, dict):
                    markets_list = threshold_response.get('markets', [])
                elif hasattr(threshold_response, 'markets'):
                    markets_list = threshold_response.markets
                
                if markets_list:
                    all_markets.extend(markets_list)
            except Exception as e:
                print(f"Warning: Could not fetch threshold markets: {e}")
        
        # Filter by date/time if specified
        btc_markets = []
        for market in all_markets:
            # Convert Market objects to dicts if needed
            if not isinstance(market, dict):
                # Try to convert Market object to dict
                try:
                    # Check if it's a Market object with attributes
                    if hasattr(market, 'ticker'):
                        # Convert to dict manually
                        market = {
                            'ticker': getattr(market, 'ticker', None),
                            'ticker_symbol': getattr(market, 'ticker_symbol', None),
                        }
                    elif hasattr(market, '__dict__'):
                        market = market.__dict__
                    else:
                        # Skip if we can't convert
                        continue
                except Exception as e:
                    print(f"Warning: Could not convert market object: {e}")
                    continue
            
            # Now market should be a dict
            ticker = market.get('ticker') or market.get('ticker_symbol', '')
            
            if not ticker:
                continue
            
            # Parse ticker to check date/time
            range_info = parse_range_ticker(ticker)
            threshold_info = parse_threshold_ticker(ticker)
            
            if range_info:
                info = range_info
            elif threshold_info:
                info = threshold_info
            else:
                continue
            
            # Filter by date/time if specified
            if year and info.year != year:
                continue
            if month and info.month != month:
                continue
            if day and info.day != day:
                continue
            if hour is not None and info.hour != hour:
                continue
            
            btc_markets.append({
                'ticker': ticker,
                'type': 'range' if range_info else 'threshold',
                'expiry': info.expiry_datetime.isoformat(),
                'year': info.year,
                'month': info.month,
                'day': info.day,
                'hour': info.hour,
                'market_data': market
            })
        
        return btc_markets
        
    except Exception as e:
        raise ValueError(f"Error searching BTC markets: {e}")


def get_available_btc_tickers_for_hour(
    year: int,
    month: int,
    day: int,
    hour: int,
    btc_price: Optional[int] = None,
    limit: int = 100,
    market_type: Optional[str] = None  # 'range' or 'threshold' or None for both
) -> List[str]:
    """
    Get available BTC tickers for a specific hour, optionally filtered by proximity to BTC price.
    
    Args:
        year: Year
        month: Month (1-12)
        day: Day
        hour: Hour (0-23)
        btc_price: Optional BTC price to filter by proximity
        limit: Maximum number of markets to fetch
        market_type: 'range' for interval markets, 'threshold' for threshold markets, None for both
    
    Returns:
        List of ticker strings, optionally sorted by proximity to btc_price
    """
    markets = search_btc_markets(year=year, month=month, day=day, hour=hour, limit=limit, market_type=market_type)
    
    tickers = [m['ticker'] for m in markets]
    
    # If btc_price provided, sort by proximity
    if btc_price:
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
    
    return tickers

