"""
Trading-related logging functions.

Handles all CSV logging for trading cycles, predictions, fills, and spike events.
"""

import os
import csv
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

from ..config import LOGS_DIR, EST

_current_log_date_hour: Optional[Tuple[str, int]] = None
_current_cycles_log_file: Optional[str] = None
_current_fills_log_file: Optional[str] = None
_current_predictions_log_file: Optional[str] = None
_current_events_log_file: Optional[str] = None


def get_log_file_paths(date_str: str, hour: int) -> Tuple[str, str, str, str]:
    """Get log file paths for a specific date and hour.
    
    Structure: LOGS_DIR/YYYY-MM-DD/HH/volatility_trading_*.csv
    """
    date_dir = os.path.join(LOGS_DIR, date_str)
    hour_dir = os.path.join(date_dir, f"{hour:02d}")
    os.makedirs(hour_dir, exist_ok=True)
    
    cycles_file = os.path.join(hour_dir, f"volatility_trading_cycles_{date_str}_{hour:02d}.csv")
    fills_file = os.path.join(hour_dir, f"volatility_trading_fills_{date_str}_{hour:02d}.csv")
    predictions_file = os.path.join(hour_dir, f"volatility_trading_predictions_{date_str}_{hour:02d}.csv")
    events_file = os.path.join(hour_dir, f"volatility_trading_events_{date_str}_{hour:02d}.csv")
    
    return cycles_file, fills_file, predictions_file, events_file


def ensure_log_file_headers(log_file: str, headers: list):
    """Ensure log file exists with headers if it's a new file."""
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def update_log_files_for_current_hour():
    """Update log file paths for the current hour, creating files if needed."""
    global _current_log_date_hour, _current_cycles_log_file, _current_fills_log_file, _current_predictions_log_file, _current_events_log_file
    
    est_now = datetime.now(EST)
    date_str = est_now.strftime('%Y-%m-%d')
    hour = est_now.hour
    
    if _current_log_date_hour != (date_str, hour):
        _current_log_date_hour = (date_str, hour)
        _current_cycles_log_file, _current_fills_log_file, _current_predictions_log_file, _current_events_log_file = get_log_file_paths(date_str, hour)
        
        ensure_log_file_headers(_current_cycles_log_file, [
            'timestamp', 'cycle_num', 'ticker', 'btc_price_from_ticker', 'btc_price_current', 
            'threshold_price', 'predicted_price', 'volatility', 'spread_cents', 'hours_until_resolution',
            'resolution_datetime', 'time_to_resolution', 'limit_prices', 'order_ids', 
            'buy_position_size', 'sell_position_size', 'available_balance', 'market_price', 'status', 'error'
        ])
        
        ensure_log_file_headers(_current_fills_log_file, [
            'timestamp', 'order_id', 'fill_id', 'ticker', 'action', 'side', 'count', 'price'
        ])
        
        ensure_log_file_headers(_current_predictions_log_file, [
            'timestamp', 'ticker', 'current_btc_price', 'target_price', 'predicted_probability', 
            'market_price', 'edge', 'spread_cents', 'volatility', 'hours_until_resolution',
            'resolution_datetime', 'model', 'mu_dt', 'sigma_dt', 'lam', 'mu_J', 'delta',
            'mean_terminal', 'median_terminal', 'std_terminal', 'quantile_5', 'quantile_95',
            'actual_volatility_6h', 'volatility_ratio'
        ])
        
        ensure_log_file_headers(_current_events_log_file, [
            'timestamp', 'event_type', 'zscore', 'current_price', 'mean_price', 'std_dev', 
            'price_change_pct', 'action_taken', 'resume_time'
        ])


def init_log_files():
    """Initialize log files with headers for current hour."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    update_log_files_for_current_hour()


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
    update_log_files_for_current_hour()
    
    try:
        if _current_cycles_log_file:
            with open(_current_cycles_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                order_ids = f"{buy_order_id or ''}/{sell_order_id or ''}"
                limit_prices = f"{buy_limit_price or ''}/{sell_limit_price or ''}"
                writer.writerow([timestamp, cycle_num, ticker, btc_price_from_ticker, btc_price_current,
                               threshold_price, predicted_price, volatility, spread_cents, hours_until_resolution,
                               resolution_datetime, time_to_resolution, limit_prices, order_ids,
                               buy_position_size or '', sell_position_size or '', available_balance or '', market_price or '',
                               status, error or ''])
    except Exception as e:
        print(f"Warning: Could not write to cycles log: {e}")


def calculate_actual_volatility_from_bitstamp(hours_back: int = 6) -> Optional[float]:
    """Calculate actual per-minute volatility from Bitstamp data over the last N hours."""
    try:
        from bitstamp_data_fetcher import fetch_bitstamp_ohlc_historical
        
        ohlc_data = fetch_bitstamp_ohlc_historical(
            currency_pair="btcusd",
            hours_back=hours_back,
            step=60
        )
        
        if not ohlc_data or len(ohlc_data) < 2:
            return None
        
        prices = np.array([price for _, price in ohlc_data], dtype=np.float64)
        
        if len(prices) < 2:
            return None
        
        log_returns = np.diff(np.log(prices))
        
        if len(log_returns) < 1:
            return None
        
        sigma_dt = log_returns.std(ddof=1)
        
        return float(sigma_dt)
        
    except Exception as e:
        return None


def log_prediction(timestamp: str, ticker: str, current_btc_price: float, target_price: float,
                  predicted_probability: float, market_price: Optional[float], edge: Optional[float],
                  spread_cents: float, volatility: float, hours_until_resolution: float,
                  resolution_datetime: str, model: str, model_params: dict,
                  mean_terminal: Optional[float] = None, median_terminal: Optional[float] = None,
                  std_terminal: Optional[float] = None, quantiles: Optional[dict] = None):
    """Log prediction data for performance analysis."""
    update_log_files_for_current_hour()
    
    actual_volatility_6h = calculate_actual_volatility_from_bitstamp(hours_back=6)
    volatility_ratio = (volatility / actual_volatility_6h) if actual_volatility_6h and actual_volatility_6h > 0 else None
    
    try:
        if _current_predictions_log_file:
            with open(_current_predictions_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                mu_dt = model_params.get('mu_dt', '')
                sigma_dt = model_params.get('sigma_dt', '')
                lam = model_params.get('lam', '') if model == 'merton' else ''
                mu_J = model_params.get('mu_J', '') if model == 'merton' else ''
                delta = model_params.get('delta', '') if model == 'merton' else ''
                
                quantile_5 = quantiles.get(0.05, '') if quantiles else ''
                quantile_95 = quantiles.get(0.95, '') if quantiles else ''
                
                writer.writerow([
                    timestamp, ticker, current_btc_price, target_price, predicted_probability,
                    market_price or '', edge or '', spread_cents, volatility, hours_until_resolution,
                    resolution_datetime, model, mu_dt, sigma_dt, lam, mu_J, delta,
                    mean_terminal or '', median_terminal or '', std_terminal or '', quantile_5, quantile_95,
                    actual_volatility_6h or '', volatility_ratio or ''
                ])
    except Exception as e:
        print(f"Warning: Could not write to predictions log: {e}")


def log_fill(timestamp: str, order_id: str, fill_id: str, ticker: str, action: str, 
             side: str, count: int, price: float):
    """Log fill execution to fills log file."""
    update_log_files_for_current_hour()
    
    try:
        if _current_fills_log_file:
            with open(_current_fills_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, order_id, fill_id, ticker, action, side, count, price])
    except Exception as e:
        print(f"Warning: Could not write to fills log: {e}")


def log_spike_event(event_type: str, zscore: float, current_price: float, mean_price: float, 
                    std_dev: float, action_taken: str, resume_time: Optional[str] = None):
    """Log spike detection and resume events to events log file."""
    update_log_files_for_current_hour()
    
    try:
        if _current_events_log_file:
            timestamp = datetime.now().isoformat()
            price_change_pct = ((current_price - mean_price) / mean_price * 100) if mean_price > 0 else 0.0
            
            with open(_current_events_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, event_type, zscore, current_price, mean_price, std_dev,
                    price_change_pct, action_taken, resume_time or ''
                ])
    except Exception as e:
        print(f"Warning: Could not write to events log: {e}")

