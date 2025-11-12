"""
Prediction wrapper and price fetching utilities.
"""

from typing import Optional, Tuple

from ..config import EST, PREDICTION_MODEL, PREDICTION_EXPONENTIAL_DECAY, DATA_STEP_SECONDS
from ..data.price_queue import get_price_data_for_prediction, jump_detection_price_queue, jump_detection_queue_lock
from app.ticker_utils import parse_threshold_ticker
from datetime import datetime
import numpy as np

try:
    from predict_price_probability import predict_price_probability
except ImportError:
    try:
        from oliver.predict_price_probability import predict_price_probability
    except ImportError:
        print("ERROR: predict_price_probability module not available. Cannot trade without predictions.")
        predict_price_probability = None


# Use get_current_btc_price_estimate from price_queue to avoid duplication
from ..data.price_queue import get_current_btc_price_estimate


def get_btc_price_from_ticker(ticker: str) -> Optional[float]:
    """Derive BTC price from a Kalshi threshold ticker."""
    threshold_info = parse_threshold_ticker(ticker)
    
    if threshold_info:
        return threshold_info.threshold
    
    return None


def predict_market_resolution_probability(ticker: str, current_price_override: Optional[float] = None) -> Optional[Tuple[float, float, dict]]:
    """Predict the probability that a Kalshi threshold market will resolve YES and get volatility."""
    threshold_info = parse_threshold_ticker(ticker)
    
    if not threshold_info:
        print(f"  Warning: Could not parse ticker {ticker}")
        return None
    
    if threshold_info.expiry_datetime.tzinfo is None:
        resolution_time = EST.localize(threshold_info.expiry_datetime)
    else:
        resolution_time = threshold_info.expiry_datetime.astimezone(EST)
    
    current_time = datetime.now(EST)
    hours_until_resolution = (resolution_time - current_time).total_seconds() / 3600.0
    
    if hours_until_resolution < (1/60):
        print(f"  Warning: Market {ticker} already resolved or too close (hours: {hours_until_resolution:.2f})")
        return None
    
    try:
        prices, last_ts, S0_queue = get_price_data_for_prediction()
        
        if len(prices) < 10:
            print(f"  Warning: Insufficient price data in queue ({len(prices)} points, need at least 10)")
            return None
        
        S0 = current_price_override if current_price_override is not None else S0_queue
        
        if S0 is None or S0 <= 0:
            print(f"  Warning: Invalid current price: {S0}")
            return None
        
        jump_detection_prices = None
        jump_detection_last_ts = None
        with jump_detection_queue_lock:
            if len(jump_detection_price_queue) >= 30:
                jump_detection_queue_items = list(jump_detection_price_queue)
                jump_detection_prices = np.array([price for _, price in jump_detection_queue_items], dtype=np.float64)
                jump_detection_last_ts = jump_detection_queue_items[-1][0] if jump_detection_queue_items else None
        
        target_price = threshold_info.threshold
        prob_above = predict_price_probability(
            csv_path=None,
            target_price=target_price,
            hours_ahead=hours_until_resolution,
            model=PREDICTION_MODEL,
            exponential_decay=PREDICTION_EXPONENTIAL_DECAY,
            prices_data=prices,
            last_timestamp=last_ts,
            current_price=S0,
            data_step_seconds=DATA_STEP_SECONDS,
            one_minute_prices_data=jump_detection_prices,
            one_minute_last_timestamp=jump_detection_last_ts
        )
        predicted_prob = prob_above['probability_above']
        volatility = prob_above['parameters']['sigma_dt']
        
        return predicted_prob, volatility, prob_above
        
    except Exception as e:
        import traceback
        print(f"  Warning: Prediction failed for {ticker}: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return None

