"""
Market discovery and utility functions.
"""

from datetime import datetime, timedelta
from typing import Optional

from app.market_search import get_available_btc_tickers_for_hour
from app.utils import get_kalshi_price_by_ticker, get_kalshi_client
from app.ticker_utils import parse_threshold_ticker
from .prediction import get_current_btc_price_estimate


def format_time_to_resolution(hours: float) -> str:
    """Format hours until resolution as a human-readable string."""
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


def find_available_ticker_threshold_only(year: int, month: int, day: int, hour: int, btc_price: Optional[int] = None) -> Optional[str]:
    """Find an available Kalshi BTC threshold ticker for the given hour."""
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
            market_type='threshold'
        )
        
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
    """Check for recent fills (executed trades) for a ticker."""
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



