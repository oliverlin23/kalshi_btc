"""
Order placement logic with dynamic spread and market making protection.
"""

from collections import deque
from typing import Optional, Tuple

from ..config import MARKET_MAKING_BUFFER_CENTS, VOLUME
from .position_sizing import calculate_dynamic_spread
from app.utils import get_kalshi_price_by_ticker, execute_trade_by_ticker

recent_order_ids = deque(maxlen=100)
last_order_details = {
    'ticker': None,
    'predicted_prob': None,
    'volatility': None,
    'buy_limit_price': None,
    'sell_limit_price': None
}


def place_limit_order_with_spread(
    ticker: str,
    predicted_price: float,
    volatility: float,
    action: str = "buy",
    position_size: Optional[int] = None
) -> Tuple[Optional[dict], Optional[float], Optional[str]]:
    """Place a limit order with dynamic spread offset from predicted price."""
    spread_cents = calculate_dynamic_spread(volatility)
    
    half_spread_decimal = (spread_cents / 2.0) / 100.0
    
    if action == "buy":
        limit_price_raw = predicted_price - half_spread_decimal
    else:
        limit_price_raw = predicted_price + half_spread_decimal
    
    try:
        current_market_price = get_kalshi_price_by_ticker(ticker, action=action)
        if current_market_price is not None:
            buffer_decimal = MARKET_MAKING_BUFFER_CENTS / 100.0
            
            if action == "buy":
                max_buy_price = current_market_price - buffer_decimal
                if limit_price_raw > max_buy_price:
                    limit_price_raw = max_buy_price
            else:
                min_sell_price = current_market_price + buffer_decimal
                if limit_price_raw < min_sell_price:
                    limit_price_raw = min_sell_price
    except Exception:
        pass
    
    def round_to_cent(price: float) -> Optional[float]:
        """Round price to nearest cent (0.01)."""
        if price <= 0 or price >= 1:
            return None
        return round(price * 100) / 100.0
    
    limit_price = round_to_cent(limit_price_raw)
    
    if limit_price is None:
        return None, limit_price_raw, f"Invalid limit price: {limit_price_raw:.4f}"
    
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

