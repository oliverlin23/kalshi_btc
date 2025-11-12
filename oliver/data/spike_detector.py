"""
Price spike detection and trading pause logic.

Detects price spikes using z-score analysis and cumulative deviation,
then pauses trading until price settles.
"""

import time
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

from ..config import (
    SPIKE_DETECTION_ZSCORE_THRESHOLD,
    SPIKE_RESUME_ZSCORE_THRESHOLD,
    SPIKE_MIN_DATA_POINTS,
    SPIKE_PAUSE_DURATION_SECONDS,
    SPIKE_CUMULATIVE_DEVIATION_THRESHOLD,
    SPIKE_CUMULATIVE_RESUME_THRESHOLD
)
from ..data.price_queue import jump_detection_price_queue, jump_detection_queue_lock, get_price_data_for_prediction

trading_paused: bool = False
spike_detected_time: Optional[float] = None
last_spike_zscore: Optional[float] = None


def detect_price_spike(prices: np.ndarray, window_seconds: int = 60) -> Tuple[bool, Optional[float], Optional[float], Optional[float]]:
    """Detect price spike using z-score analysis on price data."""
    if len(prices) < SPIKE_MIN_DATA_POINTS:
        return False, None, None, None
    
    window_size = min(window_seconds, len(prices))
    window_prices = prices[-window_size:]
    
    if len(window_prices) < SPIKE_MIN_DATA_POINTS:
        return False, None, None, None
    
    try:
        mean_price = float(np.mean(window_prices))
        std_dev = float(np.std(window_prices, ddof=1))
        
        if std_dev < 1e-6:
            return False, None, mean_price, std_dev
        
        current_price = float(prices[-1])
        zscore = (current_price - mean_price) / std_dev
        
        is_spike = abs(zscore) > SPIKE_DETECTION_ZSCORE_THRESHOLD
        
        return is_spike, zscore, mean_price, std_dev
        
    except Exception as e:
        print(f"  Warning: Spike detection calculation failed: {e}")
        return False, None, None, None


def detect_cumulative_deviation(current_price: float) -> Tuple[bool, Optional[float], Optional[float]]:
    """Detect cumulative price deviation from baseline."""
    try:
        prices, _, _ = get_price_data_for_prediction()
        
        if len(prices) < 5:
            return False, None, None
        
        baseline_mean = float(np.mean(prices))
        deviation_amount = current_price - baseline_mean
        
        is_deviation = abs(deviation_amount) > SPIKE_CUMULATIVE_DEVIATION_THRESHOLD
        
        return is_deviation, deviation_amount, baseline_mean
        
    except Exception as e:
        print(f"  Warning: Cumulative deviation detection failed: {e}")
        return False, None, None


def check_and_handle_price_spike(last_ticker=None, log_spike_event_fn=None) -> bool:
    """Check for price spikes and manage trading pause/resume logic."""
    global trading_paused, spike_detected_time, last_spike_zscore
    
    with jump_detection_queue_lock:
        if len(jump_detection_price_queue) < SPIKE_MIN_DATA_POINTS:
            return False
        
        queue_items = list(jump_detection_price_queue)
        prices = np.array([price for _, price in queue_items], dtype=np.float64)
    
    is_spike, zscore, mean_price, std_dev = detect_price_spike(prices, window_seconds=60)
    
    current_price = float(prices[-1]) if len(prices) > 0 else 0.0
    is_cumulative_deviation, deviation_amount, baseline_mean = detect_cumulative_deviation(current_price)
    
    is_any_spike = is_spike or (is_cumulative_deviation and deviation_amount is not None)
    
    if zscore is None and deviation_amount is None:
        if trading_paused and spike_detected_time is not None:
            elapsed = time.time() - spike_detected_time
            if elapsed >= SPIKE_PAUSE_DURATION_SECONDS:
                trading_paused = False
                resume_time = datetime.now().isoformat()
                if log_spike_event_fn:
                    log_spike_event_fn(
                        event_type='trading_resumed',
                        zscore=last_spike_zscore or 0.0,
                        current_price=current_price,
                        mean_price=mean_price or 0.0,
                        std_dev=std_dev or 0.0,
                        action_taken='max_pause_duration_exceeded',
                        resume_time=resume_time
                    )
                print(f"  ✓ Trading resumed: Max pause duration ({SPIKE_PAUSE_DURATION_SECONDS}s) exceeded")
                spike_detected_time = None
                last_spike_zscore = None
                return False
        return trading_paused
    
    if zscore is not None:
        last_spike_zscore = zscore
    elif deviation_amount is not None:
        last_spike_zscore = deviation_amount / baseline_mean if baseline_mean > 0 else deviation_amount / 100.0
    
    if is_any_spike and not trading_paused:
        trading_paused = True
        spike_detected_time = time.time()
        
        if last_ticker is not None:
            from app.utils import cancel_all_orders_for_ticker
            try:
                cancelled_count, failed_count = cancel_all_orders_for_ticker(last_ticker)
                action_msg = f"cancelled_{cancelled_count}_orders_paused_trading"
                if failed_count > 0:
                    action_msg += f"_{failed_count}_failed"
            except Exception as e:
                action_msg = f"cancelled_orders_paused_trading_error_{str(e)}"
        else:
            action_msg = "paused_trading_no_active_ticker"
        
        if log_spike_event_fn:
            log_spike_event_fn(
                event_type='spike_detected',
                zscore=zscore or last_spike_zscore or 0.0,
                current_price=current_price,
                mean_price=mean_price or baseline_mean or 0.0,
                std_dev=std_dev or 0.0,
                action_taken=action_msg
            )
        
        print(f"  ⚠️  PRICE SPIKE DETECTED: z-score={zscore or last_spike_zscore:.2f}, price=${current_price:,.0f}, mean=${mean_price or baseline_mean:,.0f}, std=${std_dev:.2f}")
        print(f"  ⚠️  Trading PAUSED - orders cancelled, waiting for price to settle...")
        
        return True
    
    elif trading_paused:
        elapsed = time.time() - spike_detected_time if spike_detected_time else 0
        current_zscore = zscore if zscore is not None else (abs(deviation_amount) / baseline_mean if deviation_amount is not None and baseline_mean else 0.0)
        
        if current_zscore < SPIKE_RESUME_ZSCORE_THRESHOLD or elapsed >= SPIKE_PAUSE_DURATION_SECONDS:
            trading_paused = False
            resume_time = datetime.now().isoformat()
            resume_reason = 'price_settled' if current_zscore < SPIKE_RESUME_ZSCORE_THRESHOLD else 'max_pause_duration_exceeded'
            
            if log_spike_event_fn:
                log_spike_event_fn(
                    event_type='trading_resumed',
                    zscore=current_zscore,
                    current_price=current_price,
                    mean_price=mean_price or baseline_mean or 0.0,
                    std_dev=std_dev or 0.0,
                    action_taken=resume_reason,
                    resume_time=resume_time
                )
            
            print(f"  ✓ Trading RESUMED: z-score={current_zscore:.2f} (threshold: {SPIKE_RESUME_ZSCORE_THRESHOLD}), reason: {resume_reason}")
            spike_detected_time = None
            last_spike_zscore = None
            return False
        
        return True
    
    return False

