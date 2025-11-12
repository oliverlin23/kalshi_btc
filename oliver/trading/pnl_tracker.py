"""
Position and PnL tracking for resolved markets.
"""

from datetime import datetime
from typing import Dict, Optional, Tuple

from app.utils import get_kalshi_client
from app.ticker_utils import parse_threshold_ticker

positions: Dict[str, Dict] = {}
total_realized_pnl = 0.0
last_resolved_hour: Optional[Tuple[int, int, int, int]] = None


def update_position_from_fill(ticker: str, action: str, side: str, count: int, price: float):
    """Update position tracking from a fill."""
    if ticker not in positions:
        positions[ticker] = {
            'position': 0,
            'total_buy_cost': 0.0,
            'total_sell_revenue': 0.0,
            'buy_count': 0,
            'sell_count': 0
        }
    
    if side == 'yes':
        if action == 'buy':
            positions[ticker]['position'] += count
            positions[ticker]['total_buy_cost'] += price * count
            positions[ticker]['buy_count'] += count
        elif action == 'sell':
            positions[ticker]['position'] -= count
            positions[ticker]['total_sell_revenue'] += price * count
            positions[ticker]['sell_count'] += count
    elif side == 'no':
        if action == 'buy':
            positions[ticker]['position'] -= count
            positions[ticker]['total_sell_revenue'] += (1.0 - price) * count
            positions[ticker]['sell_count'] += count
        elif action == 'sell':
            positions[ticker]['position'] += count
            positions[ticker]['total_buy_cost'] += (1.0 - price) * count
            positions[ticker]['buy_count'] += count


def get_market_resolution_price(ticker: str) -> Optional[float]:
    """Get resolution price for a market (1.0 for YES, 0.0 for NO, or None if not resolved)."""
    try:
        client = get_kalshi_client()
        market = client.get_market(ticker=ticker)
        
        if hasattr(market.market, 'status'):
            status = market.market.status
            if status == 'closed':
                if hasattr(market.market, 'settlement_price'):
                    settlement = market.market.settlement_price
                    if settlement is not None:
                        return settlement / 100.0 if settlement > 1 else settlement
                if hasattr(market.market, 'last_price') and market.market.last_price is not None:
                    last_price = market.market.last_price / 100.0 if market.market.last_price > 1 else market.market.last_price
                    if last_price < 0.01:
                        return 0.0
                    elif last_price > 0.99:
                        return 1.0
        
        return None
    except Exception:
        return None


def calculate_pnl_for_resolved_markets(previous_hour_key: Tuple[int, int, int, int]):
    """Calculate PnL for markets that resolved in the previous hour."""
    global positions, total_realized_pnl, last_resolved_hour
    
    if last_resolved_hour == previous_hour_key:
        return
    
    timestamp = datetime.now().isoformat()
    
    resolved_tickers = []
    if not positions:
        last_resolved_hour = previous_hour_key
        return
    
    for ticker in list(positions.keys()):
        threshold_info = parse_threshold_ticker(ticker)
        
        if not threshold_info:
            continue
        
        ticker_hour = (threshold_info.year, threshold_info.month, threshold_info.day, threshold_info.hour)
        
        if ticker_hour == previous_hour_key:
            resolved_tickers.append(ticker)
    
    for ticker in resolved_tickers:
        pos_data = positions[ticker]
        position = pos_data['position']
        
        if position == 0:
            continue
        
        resolution_price = get_market_resolution_price(ticker)
        if resolution_price is None:
            continue
        
        avg_buy_price = (pos_data['total_buy_cost'] / pos_data['buy_count']) if pos_data['buy_count'] > 0 else 0.0
        avg_sell_price = (pos_data['total_sell_revenue'] / pos_data['sell_count']) if pos_data['sell_count'] > 0 else 0.0
        
        if position > 0:
            realized_pnl = (resolution_price - avg_buy_price) * position
        else:
            realized_pnl = (avg_sell_price - resolution_price) * abs(position)
        
        total_realized_pnl += realized_pnl
        
        print(f"  PnL: {ticker} | Position: {position} | Resolution: {resolution_price:.4f} | PnL: ${realized_pnl:.2f} | Total: ${total_realized_pnl:.2f}")
        
        del positions[ticker]
    
    last_resolved_hour = previous_hour_key

