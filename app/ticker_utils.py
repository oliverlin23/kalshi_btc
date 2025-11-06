"""
Utilities for parsing and generating Kalshi BTC ticker codes.

Supports two market types:
1. Range markets: KXBTC-[YY][MON][DD][HH]-B[midpoint]
2. Threshold markets: KXBTCD-[YY][MON][DD][HH]-T[threshold]
"""

import re
from datetime import datetime
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


MONTH_CODES = {
    1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
    7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'
}

MONTH_NAMES = {v: k for k, v in MONTH_CODES.items()}


@dataclass
class RangeMarketInfo:
    """Information parsed from a range market ticker."""
    ticker: str
    year: int
    month: int
    day: int
    hour: int
    midpoint: int
    lower_bound: int
    upper_bound: int
    expiry_datetime: datetime

    def __str__(self):
        return f"Range: ${self.lower_bound:,} - ${self.upper_bound:,} (mid: ${self.midpoint:,})"


@dataclass
class ThresholdMarketInfo:
    """Information parsed from a threshold market ticker."""
    ticker: str
    year: int
    month: int
    day: int
    hour: int
    threshold: float
    expiry_datetime: datetime

    def __str__(self):
        return f"Threshold: > ${self.threshold:,.2f}"


def parse_range_ticker(ticker: str) -> Optional[RangeMarketInfo]:
    """
    Parse a range market ticker (KXBTC-*).
    
    Args:
        ticker: Ticker string (e.g., "KXBTC-25NOV0612-B101875")
    
    Returns:
        RangeMarketInfo object or None if parsing fails
    """
    pattern = r'KXBTC-(\d{2})([A-Z]{3})(\d{2})(\d{2})-B(\d+)'
    match = re.match(pattern, ticker)
    
    if not match:
        return None
    
    yy, mon, dd, hh, midpoint_str = match.groups()
    
    try:
        year = 2000 + int(yy)
        month = MONTH_NAMES[mon]
        day = int(dd)
        hour = int(hh)
        midpoint = int(midpoint_str)
        
        expiry_datetime = datetime(year, month, day, hour)
        
        return RangeMarketInfo(
            ticker=ticker,
            year=year,
            month=month,
            day=day,
            hour=hour,
            midpoint=midpoint,
            lower_bound=midpoint - 125,
            upper_bound=midpoint + 125,
            expiry_datetime=expiry_datetime
        )
    except (ValueError, KeyError):
        return None


def parse_threshold_ticker(ticker: str) -> Optional[ThresholdMarketInfo]:
    """
    Parse a threshold market ticker (KXBTCD-*).
    
    Args:
        ticker: Ticker string (e.g., "KXBTCD-25NOV0612-T101999.99")
    
    Returns:
        ThresholdMarketInfo object or None if parsing fails
    """
    pattern = r'KXBTCD-(\d{2})([A-Z]{3})(\d{2})(\d{2})-T([\d.]+)'
    match = re.match(pattern, ticker)
    
    if not match:
        return None
    
    yy, mon, dd, hh, threshold_str = match.groups()
    
    try:
        year = 2000 + int(yy)
        month = MONTH_NAMES[mon]
        day = int(dd)
        hour = int(hh)
        threshold = float(threshold_str)
        
        expiry_datetime = datetime(year, month, day, hour)
        
        return ThresholdMarketInfo(
            ticker=ticker,
            year=year,
            month=month,
            day=day,
            hour=hour,
            threshold=threshold,
            expiry_datetime=expiry_datetime
        )
    except (ValueError, KeyError):
        return None


def parse_ticker(ticker: str) -> Optional[Tuple[RangeMarketInfo, None] | Tuple[None, ThresholdMarketInfo]]:
    """
    Parse any BTC ticker (range or threshold).
    
    Args:
        ticker: Ticker string
    
    Returns:
        Tuple of (RangeMarketInfo, ThresholdMarketInfo) with one None,
        or (None, None) if parsing fails
    """
    if ticker.startswith('KXBTC-'):
        info = parse_range_ticker(ticker)
        return (info, None) if info else (None, None)
    elif ticker.startswith('KXBTCD-'):
        info = parse_threshold_ticker(ticker)
        return (None, info) if info else (None, None)
    else:
        return (None, None)


def generate_range_ticker(
    year: int,
    month: int,
    day: int,
    hour: int,
    midpoint: int
) -> str:
    """
    Generate a range market ticker.
    
    Args:
        year: Full year (e.g., 2025)
        month: Month (1-12)
        day: Day (1-31)
        hour: Hour in 24-hour format (0-23)
        midpoint: Midpoint price (integer, no decimals)
    
    Returns:
        Ticker string (e.g., "KXBTC-25NOV0612-B101875")
    
    Example:
        >>> generate_range_ticker(2025, 11, 6, 12, 101875)
        'KXBTC-25NOV0612-B101875'
    """
    yy = str(year)[-2:]
    mon = MONTH_CODES[month]
    dd = f"{day:02d}"
    hh = f"{hour:02d}"
    
    return f"KXBTC-{yy}{mon}{dd}{hh}-B{midpoint}"


def generate_threshold_ticker(
    year: int,
    month: int,
    day: int,
    hour: int,
    threshold: float
) -> str:
    """
    Generate a threshold market ticker.
    
    Args:
        year: Full year (e.g., 2025)
        month: Month (1-12)
        day: Day (1-31)
        hour: Hour in 24-hour format (0-23)
        threshold: Threshold price (can be float)
    
    Returns:
        Ticker string (e.g., "KXBTCD-25NOV0612-T101999.99")
    
    Example:
        >>> generate_threshold_ticker(2025, 11, 6, 12, 101999.99)
        'KXBTCD-25NOV0612-T101999.99'
    """
    yy = str(year)[-2:]
    mon = MONTH_CODES[month]
    dd = f"{day:02d}"
    hh = f"{hour:02d}"
    
    # Format threshold - remove trailing zeros if it's a whole number
    if threshold == int(threshold):
        threshold_str = str(int(threshold))
    else:
        threshold_str = str(threshold)
    
    return f"KXBTCD-{yy}{mon}{dd}{hh}-T{threshold_str}"


def generate_range_ticker_from_datetime(dt: datetime, midpoint: int) -> str:
    """
    Generate a range market ticker from a datetime object.
    
    Args:
        dt: Datetime object (hour will be used, minutes/seconds ignored)
        midpoint: Midpoint price (integer)
    
    Returns:
        Ticker string
    """
    return generate_range_ticker(dt.year, dt.month, dt.day, dt.hour, midpoint)


def generate_threshold_ticker_from_datetime(dt: datetime, threshold: float) -> str:
    """
    Generate a threshold market ticker from a datetime object.
    
    Args:
        dt: Datetime object (hour will be used, minutes/seconds ignored)
        threshold: Threshold price (float)
    
    Returns:
        Ticker string
    """
    return generate_threshold_ticker(dt.year, dt.month, dt.day, dt.hour, threshold)


def get_ticker_info(ticker: str) -> Optional[str]:
    """
    Get a human-readable description of a ticker.
    
    Args:
        ticker: Ticker string
    
    Returns:
        Description string or None if parsing fails
    """
    range_info, threshold_info = parse_ticker(ticker)
    
    if range_info:
        return f"{ticker}: {range_info} (expires {range_info.expiry_datetime.strftime('%Y-%m-%d %H:00 UTC')})"
    elif threshold_info:
        return f"{ticker}: {threshold_info} (expires {threshold_info.expiry_datetime.strftime('%Y-%m-%d %H:00 UTC')})"
    else:
        return None

