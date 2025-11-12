"""
Position sizing calculations using Kelly criterion and risk limits.
"""

from typing import Optional

from ..config import (
    BASE_SPREAD_CENTS,
    VOLATILITY_MULTIPLIER,
    MAX_SPREAD_CENTS,
    VOLUME,
    USE_KELLY_SIZING,
    MAX_POSITION_PCT_OF_BALANCE,
    KELLY_FRACTION,
    MIN_POSITION_SIZE,
    MARKET_TAKING_FEE_RATE
)


def calculate_dynamic_spread(volatility: float) -> float:
    """Calculate dynamic spread in cents based on volatility."""
    spread_cents = BASE_SPREAD_CENTS + (VOLATILITY_MULTIPLIER * volatility)
    spread_cents = max(BASE_SPREAD_CENTS, min(spread_cents, MAX_SPREAD_CENTS))
    return spread_cents


def calculate_market_taking_fee(price: float) -> float:
    """Calculate market taking fee for a given price."""
    return MARKET_TAKING_FEE_RATE * price * (1.0 - price)


def calculate_optimal_position_size(
    predicted_prob: float,
    market_price: Optional[float],
    available_balance: Optional[float] = None,
    volatility: Optional[float] = None
) -> int:
    """Calculate optimal position size using Kelly criterion with risk limits, or fixed volume."""
    if not USE_KELLY_SIZING:
        return VOLUME
    
    if market_price is None:
        return VOLUME
    
    edge = abs(predicted_prob - market_price)
    
    if edge < 0.01:
        return VOLUME
    
    if predicted_prob > market_price:
        p = predicted_prob
        win_amount = 1.0 - market_price
        lose_amount = market_price
    else:
        p = 1.0 - predicted_prob
        win_amount = market_price
        lose_amount = 1.0 - market_price
    
    if lose_amount == 0:
        return VOLUME
    
    b = win_amount / lose_amount
    
    q = 1.0 - p
    kelly_fraction = (p * b - q) / b
    
    kelly_fraction *= KELLY_FRACTION
    
    kelly_fraction = max(0.0, kelly_fraction)
    
    if kelly_fraction < 0.01:
        return VOLUME
    
    if available_balance is not None and available_balance > 0:
        if predicted_prob > market_price:
            risk_per_contract = market_price
        else:
            risk_per_contract = 1.0 - market_price
        
        position_value = kelly_fraction * available_balance
        contracts_from_kelly = int(position_value / risk_per_contract)
        
        max_contracts_from_balance = int((MAX_POSITION_PCT_OF_BALANCE * available_balance) / risk_per_contract)
        
        optimal_contracts = min(contracts_from_kelly, max_contracts_from_balance)
        optimal_contracts = max(optimal_contracts, MIN_POSITION_SIZE)
        
        return optimal_contracts
    else:
        edge_multiplier = min(edge / 0.05, 2.0)
        scaled_volume = int(VOLUME * edge_multiplier * kelly_fraction * 4)
        return max(scaled_volume, MIN_POSITION_SIZE)

