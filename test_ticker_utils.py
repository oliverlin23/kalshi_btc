"""
Test script for ticker utilities.
"""

from app.ticker_utils import (
    parse_range_ticker,
    parse_threshold_ticker,
    parse_ticker,
    generate_range_ticker,
    generate_threshold_ticker,
    generate_range_ticker_from_datetime,
    generate_threshold_ticker_from_datetime,
    get_ticker_info
)
from datetime import datetime


def test_parsing():
    """Test parsing existing tickers."""
    print("=" * 60)
    print("TEST: Parsing Tickers")
    print("=" * 60)
    
    # Test range market
    ticker1 = "KXBTC-25NOV0612-B101875"
    print(f"\nParsing: {ticker1}")
    range_info = parse_range_ticker(ticker1)
    if range_info:
        print(f"  Year: {range_info.year}")
        print(f"  Date: {range_info.month}/{range_info.day}/{range_info.year}")
        print(f"  Hour: {range_info.hour}:00")
        print(f"  Midpoint: ${range_info.midpoint:,}")
        print(f"  Range: ${range_info.lower_bound:,} - ${range_info.upper_bound:,}")
        print(f"  Expiry: {range_info.expiry_datetime}")
    else:
        print("  Failed to parse")
    
    # Test threshold market
    ticker2 = "KXBTCD-25NOV0612-T101999.99"
    print(f"\nParsing: {ticker2}")
    threshold_info = parse_threshold_ticker(ticker2)
    if threshold_info:
        print(f"  Year: {threshold_info.year}")
        print(f"  Date: {threshold_info.month}/{threshold_info.day}/{threshold_info.year}")
        print(f"  Hour: {threshold_info.hour}:00")
        print(f"  Threshold: ${threshold_info.threshold:,.2f}")
        print(f"  Expiry: {threshold_info.expiry_datetime}")
    else:
        print("  Failed to parse")
    
    # Test generic parser
    print(f"\nUsing generic parser:")
    range_info, threshold_info = parse_ticker(ticker1)
    if range_info:
        print(f"  {ticker1} -> Range market")
    range_info, threshold_info = parse_ticker(ticker2)
    if threshold_info:
        print(f"  {ticker2} -> Threshold market")


def test_generation():
    """Test generating tickers."""
    print("\n" + "=" * 60)
    print("TEST: Generating Tickers")
    print("=" * 60)
    
    # Generate range ticker
    ticker1 = generate_range_ticker(2025, 11, 6, 12, 101875)
    print(f"\nGenerated range ticker: {ticker1}")
    print(f"  Expected: KXBTC-25NOV0612-B101875")
    print(f"  Match: {ticker1 == 'KXBTC-25NOV0612-B101875'}")
    
    # Generate threshold ticker
    ticker2 = generate_threshold_ticker(2025, 11, 6, 12, 101999.99)
    print(f"\nGenerated threshold ticker: {ticker2}")
    print(f"  Expected: KXBTCD-25NOV0612-T101999.99")
    print(f"  Match: {ticker2 == 'KXBTCD-25NOV0612-T101999.99'}")
    
    # Generate from datetime
    dt = datetime(2025, 11, 6, 12, 0, 0)
    ticker3 = generate_range_ticker_from_datetime(dt, 101875)
    print(f"\nGenerated from datetime: {ticker3}")
    
    ticker4 = generate_threshold_ticker_from_datetime(dt, 101999.99)
    print(f"Generated from datetime: {ticker4}")


def test_info():
    """Test getting ticker info."""
    print("\n" + "=" * 60)
    print("TEST: Getting Ticker Info")
    print("=" * 60)
    
    ticker1 = "KXBTC-25NOV0612-B101875"
    info1 = get_ticker_info(ticker1)
    print(f"\n{ticker1}")
    print(f"  {info1}")
    
    ticker2 = "KXBTCD-25NOV0612-T101999.99"
    info2 = get_ticker_info(ticker2)
    print(f"\n{ticker2}")
    print(f"  {info2}")


def test_round_trip():
    """Test that parsing and generating are consistent."""
    print("\n" + "=" * 60)
    print("TEST: Round-trip (Generate -> Parse)")
    print("=" * 60)
    
    # Test range market
    original_range = generate_range_ticker(2025, 11, 6, 12, 101875)
    parsed_range = parse_range_ticker(original_range)
    if parsed_range:
        regenerated = generate_range_ticker(
            parsed_range.year, parsed_range.month, parsed_range.day,
            parsed_range.hour, parsed_range.midpoint
        )
        print(f"\nRange market:")
        print(f"  Original: {original_range}")
        print(f"  Regenerated: {regenerated}")
        print(f"  Match: {original_range == regenerated}")
    
    # Test threshold market
    original_threshold = generate_threshold_ticker(2025, 11, 6, 12, 101999.99)
    parsed_threshold = parse_threshold_ticker(original_threshold)
    if parsed_threshold:
        regenerated = generate_threshold_ticker(
            parsed_threshold.year, parsed_threshold.month, parsed_threshold.day,
            parsed_threshold.hour, parsed_threshold.threshold
        )
        print(f"\nThreshold market:")
        print(f"  Original: {original_threshold}")
        print(f"  Regenerated: {regenerated}")
        print(f"  Match: {original_threshold == regenerated}")


if __name__ == "__main__":
    test_parsing()
    test_generation()
    test_info()
    test_round_trip()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

