"""
Main trading cycle orchestration.

Coordinates all trading components: prediction, position sizing, order placement,
logging, and fill tracking.
"""

from datetime import datetime
from typing import Optional

from ..config import EST, PREDICTION_MODEL
from ..data import price_queue as data_price_queue
from ..data.price_queue import get_price_data_for_prediction, update_jump_detection_queue
from ..data.spike_detector import check_and_handle_price_spike
from ..loggers.trading_logger import log_spike_event
from ..loggers.trading_logger import log_cycle, log_prediction, log_fill
from ..trading.hourly_maintenance import handle_hour_change
from ..trading.market_discovery import (
    find_available_ticker_threshold_only,
    check_recent_fills,
    format_time_to_resolution
)
from ..trading.prediction import predict_market_resolution_probability, get_current_btc_price_estimate, get_btc_price_from_ticker
from ..trading.position_sizing import calculate_optimal_position_size, calculate_dynamic_spread
from ..trading.order_placement import place_limit_order_with_spread, last_order_details
from ..trading.hourly_maintenance import balance_floor
from app.ticker_utils import parse_threshold_ticker
from app.utils import cancel_all_orders_for_ticker, get_position_by_ticker, get_available_funds, get_kalshi_mid_price_by_ticker

last_ticker: Optional[str] = None
logged_fill_ids = set()


def run_trading_cycle(cycle_num: int) -> bool:
    """Run a single trading cycle."""
    global last_ticker, logged_fill_ids
    
    timestamp = datetime.now().isoformat()
    
    if not data_price_queue.price_data_initialized:
        print(f"[{cycle_num}] ERROR: Price queue not initialized! This should not happen.")
        return False
    
    update_jump_detection_queue()
    
    is_paused = check_and_handle_price_spike(
        last_ticker=last_ticker,
        log_spike_event_fn=log_spike_event
    )
    
    year, month, day, hour = handle_hour_change()
    
    ticker = find_available_ticker_threshold_only(year, month, day, hour)
    
    if not ticker:
        print(f"[{cycle_num}] SKIP: No available markets for {year}-{month:02d}-{day:02d} {hour:02d}:00")
        return False
    
    if is_paused:
        btc_price_current = get_current_btc_price_estimate()
        if btc_price_current is None:
            try:
                _, _, btc_price_current = get_price_data_for_prediction()
            except Exception:
                btc_price_current = 0.0
        
        threshold_info = parse_threshold_ticker(ticker)
        threshold_price = threshold_info.threshold if threshold_info else 0.0
        
        log_cycle(
            timestamp=timestamp,
            cycle_num=cycle_num,
            ticker=ticker,
            btc_price_from_ticker=get_btc_price_from_ticker(ticker) or 0.0,
            btc_price_current=btc_price_current,
            threshold_price=threshold_price,
            predicted_price=0.0,
            volatility=0.0,
            spread_cents=0.0,
            hours_until_resolution=0.0,
            resolution_datetime='',
            time_to_resolution='',
            buy_limit_price=None,
            sell_limit_price=None,
            buy_order_id=None,
            sell_order_id=None,
            buy_position_size=None,
            sell_position_size=None,
            available_balance=None,
            market_price=None,
            status='trading_paused_spike',
            error='Price spike detected - trading paused'
        )
        
        print(f"[{cycle_num}] {ticker} | BTC=${btc_price_current:,.0f} | ⚠️  TRADING PAUSED (spike detected)")
        return True
    
    btc_price_from_ticker = get_btc_price_from_ticker(ticker) or 0.0
    
    threshold_info = parse_threshold_ticker(ticker)
    if not threshold_info:
        print(f"[{cycle_num}] SKIP: Could not parse ticker {ticker}")
        return False
    
    threshold_price = threshold_info.threshold
    if threshold_info.expiry_datetime.tzinfo is None:
        resolution_time = EST.localize(threshold_info.expiry_datetime)
    else:
        resolution_time = threshold_info.expiry_datetime.astimezone(EST)
    current_time = datetime.now(EST)
    hours_until_resolution = (resolution_time - current_time).total_seconds() / 3600.0
    resolution_datetime_str = resolution_time.isoformat()
    time_to_resolution_str = format_time_to_resolution(hours_until_resolution)
    
    btc_price_current = get_current_btc_price_estimate()
    if btc_price_current is None:
        try:
            _, _, btc_price_current = get_price_data_for_prediction()
        except Exception as e:
            print(f"[{cycle_num}] WARNING: Could not get current BTC price: {e}")
            btc_price_current = btc_price_from_ticker
    
    prediction_result = predict_market_resolution_probability(ticker, current_price_override=btc_price_current)
    if prediction_result is None:
        print(f"[{cycle_num}] SKIP: Could not get prediction for {ticker}")
        return False
    
    predicted_prob, volatility, prediction_result_dict = prediction_result
    
    if not (0.01 <= predicted_prob <= 0.99):
        print(f"[{cycle_num}] SKIP: Prediction {predicted_prob:.3f} too extreme for {ticker}")
        return False
    
    spread_cents = calculate_dynamic_spread(volatility)
    
    def round_to_cent(price: float) -> Optional[float]:
        """Round price to nearest cent (0.01)."""
        if price <= 0 or price >= 1:
            return None
        return round(price * 100) / 100.0
    
    half_spread_decimal = (spread_cents / 2.0) / 100.0
    new_buy_limit_price_raw = predicted_prob - half_spread_decimal
    new_sell_limit_price_raw = predicted_prob + half_spread_decimal
    
    new_buy_limit_price = round_to_cent(new_buy_limit_price_raw)
    new_sell_limit_price = round_to_cent(new_sell_limit_price_raw)
    
    need_to_update = False
    
    if last_ticker is not None and last_ticker != ticker:
        cancel_all_orders_for_ticker(last_ticker)
        need_to_update = True
    elif last_ticker != ticker:
        need_to_update = True
    else:
        old_buy_limit_price = round_to_cent(last_order_details['buy_limit_price']) if last_order_details['buy_limit_price'] is not None else None
        old_sell_limit_price = round_to_cent(last_order_details['sell_limit_price']) if last_order_details['sell_limit_price'] is not None else None
        
        buy_price_changed = (old_buy_limit_price != new_buy_limit_price)
        sell_price_changed = (old_sell_limit_price != new_sell_limit_price)
        
        if buy_price_changed or sell_price_changed:
            need_to_update = True
    
    buy_position_size = None
    sell_position_size = None
    available_balance = None
    market_price = None
    
    try:
        market_price_for_log = get_kalshi_mid_price_by_ticker(ticker)
        edge_for_log = (predicted_prob - market_price_for_log) if market_price_for_log is not None else None
    except Exception:
        market_price_for_log = None
        edge_for_log = None
    
    log_prediction(
        timestamp=timestamp,
        ticker=ticker,
        current_btc_price=btc_price_current,
        target_price=threshold_price,
        predicted_probability=predicted_prob,
        market_price=market_price_for_log,
        edge=edge_for_log,
        spread_cents=spread_cents,
        volatility=volatility,
        hours_until_resolution=hours_until_resolution,
        resolution_datetime=resolution_datetime_str,
        model=PREDICTION_MODEL,
        model_params=prediction_result_dict.get('parameters', {}),
        mean_terminal=prediction_result_dict.get('mean_terminal'),
        median_terminal=prediction_result_dict.get('median_terminal'),
        std_terminal=prediction_result_dict.get('std_terminal'),
        quantiles=prediction_result_dict.get('quantiles')
    )
    
    if need_to_update:
        cancel_all_orders_for_ticker(ticker)
        last_ticker = ticker
        
        try:
            available_balance = get_available_funds()
            market_price = get_kalshi_mid_price_by_ticker(ticker)
        except Exception:
            available_balance = None
            market_price = None
        
        if balance_floor is not None and available_balance is not None and available_balance < balance_floor:
            try:
                current_position = get_position_by_ticker(ticker)
            except Exception as e:
                print(f"  Warning: Could not get position for balance floor check: {e}")
                current_position = 0
            
            print(f"  Balance floor protection: balance=${available_balance:.2f} < floor=${balance_floor:.2f}, position={current_position}")
            
            buy_position_size_normal = calculate_optimal_position_size(
                predicted_prob=predicted_prob,
                market_price=market_price,
                available_balance=available_balance,
                volatility=volatility
            )
            sell_position_size_normal = calculate_optimal_position_size(
                predicted_prob=1.0 - predicted_prob,
                market_price=market_price,
                available_balance=available_balance,
                volatility=volatility
            )
            
            if current_position > 0:
                buy_position_size = 0
                sell_position_size = sell_position_size_normal
                print(f"  Balance floor: Blocking buy orders (would increase exposure), allowing sell orders (reducing exposure)")
            elif current_position < 0:
                buy_position_size = buy_position_size_normal
                sell_position_size = 0
                print(f"  Balance floor: Allowing buy orders (reducing exposure), blocking sell orders (would increase exposure)")
            else:
                buy_position_size = 0
                sell_position_size = 0
                print(f"  Balance floor: Blocking both orders (flat position, any order would increase exposure)")
        else:
            buy_position_size = calculate_optimal_position_size(
                predicted_prob=predicted_prob,
                market_price=market_price,
                available_balance=available_balance,
                volatility=volatility
            )
            
            sell_position_size = calculate_optimal_position_size(
                predicted_prob=1.0 - predicted_prob,
                market_price=market_price,
                available_balance=available_balance,
                volatility=volatility
            )
        
        if market_price:
            edge = abs(predicted_prob - market_price)
            edge_pct = edge * 100
            print(f"  Position sizing: edge={edge_pct:.2f}%, buy={buy_position_size}, sell={sell_position_size}, balance=${available_balance:.2f}" if available_balance else f"  Position sizing: edge={edge_pct:.2f}%, buy={buy_position_size}, sell={sell_position_size}")
        
        buy_result, buy_limit_price, buy_error = place_limit_order_with_spread(
            ticker, predicted_prob, volatility, action="buy", position_size=buy_position_size
        )
        sell_result, sell_limit_price, sell_error = place_limit_order_with_spread(
            ticker, predicted_prob, volatility, action="sell", position_size=sell_position_size
        )
        
        last_order_details.update({
            'ticker': ticker,
            'predicted_prob': predicted_prob,
            'volatility': volatility,
            'buy_limit_price': buy_limit_price,
            'sell_limit_price': sell_limit_price
        })
    else:
        buy_result = None
        sell_result = None
        buy_limit_price = last_order_details['buy_limit_price']
        sell_limit_price = last_order_details['sell_limit_price']
        buy_error = None
        sell_error = None
        status = "orders_unchanged"
    
    buy_order_id = buy_result.get('order_id', None) if buy_result else None
    sell_order_id = sell_result.get('order_id', None) if sell_result else None
    
    if need_to_update:
        if buy_result and sell_result:
            status = "both_placed"
        elif buy_result:
            status = "buy_only"
        elif sell_result:
            status = "sell_only"
        else:
            status = "failed"
    
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
    
    recent_fills = check_recent_fills(ticker, since_seconds=60)
    
    for fill in recent_fills:
        fill_id = fill.get('fill_id')
        if fill_id and fill_id not in logged_fill_ids:
            from ..trading.pnl_tracker import update_position_from_fill
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
            update_position_from_fill(
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

