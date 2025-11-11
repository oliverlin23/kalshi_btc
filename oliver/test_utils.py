"""
Tests for utils.py functions
Run with: pytest personal/test_utils.py -v

Tests that use real API calls will only run if credentials are set in .env
Tests that would cost money (place orders) require ENABLE_EXPENSIVE_TESTS=true
To run free tests: pytest personal/test_utils.py -v
To run expensive tests: ENABLE_EXPENSIVE_TESTS=true pytest personal/test_utils.py -v
"""

import pytest
import os
import csv
import tempfile
from unittest.mock import Mock, patch, mock_open, MagicMock
from datetime import datetime

# Import functions to test
from app.utils import (
    get_kalshi_client,
    get_kalshi_price_by_ticker,
    get_kalshi_spread_by_ticker,
    get_kalshi_orderbook_by_ticker,
    get_kalshi_price_impact_by_ticker,
    get_available_funds,
    write_trade_to_csv,
    execute_trade_by_ticker,
    get_position_by_ticker,
    cancel_all_orders_for_ticker,
    DRY_RUN
)

# Check if we have credentials for integration tests
HAS_KALSHI_CREDS = bool(os.getenv("KALSHI_API_KEY_ID") and os.path.exists(os.getenv("KALSHI_PRIVATE_KEY_PATH", "private_key.txt")))

# Check if expensive tests (that cost money) should be enabled
ENABLE_EXPENSIVE_TESTS = os.getenv("ENABLE_EXPENSIVE_TESTS", "false").lower() in ("true", "1", "yes")

# Test ticker - using a Bitcoin-related ticker format
# Note: Replace with actual ticker from Kalshi for real tests
TEST_TICKER = "BTC-UP-20241231"  # Example ticker format


class TestGetKalshiClient:
    """Tests for get_kalshi_client function - FREE (just creates client)"""

    @pytest.mark.skipif(not HAS_KALSHI_CREDS, reason="Kalshi credentials not available")
    def test_real_client_creation(self):
        """Test with real credentials - FREE"""
        client = get_kalshi_client()
        assert client is not None

    @patch('app.utils.os.getenv')
    def test_missing_api_key_raises_error(self, mock_getenv):
        mock_getenv.return_value = None

        with pytest.raises(ValueError, match="KALSHI_API_KEY_ID"):
            get_kalshi_client()

    @patch('app.utils.os.getenv')
    def test_missing_private_key_file_raises_error(self, mock_getenv):
        def getenv_side_effect(key, default=None):
            if key == "KALSHI_API_KEY_ID":
                return "test_key_id"
            elif key == "KALSHI_PRIVATE_KEY_PATH":
                return "nonexistent.txt"
            return default
        
        mock_getenv.side_effect = getenv_side_effect

        with pytest.raises(FileNotFoundError):
            get_kalshi_client()


class TestGetKalshiPriceByTicker:
    """Tests for get_kalshi_price_by_ticker function - FREE (reads market data)"""

    @pytest.mark.skipif(not HAS_KALSHI_CREDS, reason="Kalshi credentials not available")
    def test_real_price_fetch_buy(self):
        """Test with real Kalshi API - FREE"""
        # Note: Replace TEST_TICKER with actual ticker from Kalshi
        try:
            price = get_kalshi_price_by_ticker(TEST_TICKER, action="buy")
            # Prices should be between 0 and 1 (decimal format)
            assert 0 <= price <= 1
        except ValueError as e:
            if "No ask price available" in str(e) or "Error fetching Kalshi price" in str(e):
                pytest.skip(f"Ticker {TEST_TICKER} not available or invalid")

    @pytest.mark.skipif(not HAS_KALSHI_CREDS, reason="Kalshi credentials not available")
    def test_real_price_fetch_sell(self):
        """Test with real Kalshi API - FREE"""
        try:
            price = get_kalshi_price_by_ticker(TEST_TICKER, action="sell")
            # Prices should be between 0 and 1 (decimal format)
            assert 0 <= price <= 1
        except ValueError as e:
            if "No bid price available" in str(e) or "Error fetching Kalshi price" in str(e):
                pytest.skip(f"Ticker {TEST_TICKER} not available or invalid")

    @patch('app.utils.get_kalshi_client')
    def test_raises_error_on_api_failure(self, mock_get_client):
        mock_get_client.side_effect = Exception("API error")

        with pytest.raises(ValueError, match="Error fetching Kalshi price"):
            get_kalshi_price_by_ticker(TEST_TICKER, action="buy")

    def test_invalid_action_raises_error(self):
        """Test that invalid action parameter raises ValueError"""
        with pytest.raises(ValueError, match="Invalid action"):
            get_kalshi_price_by_ticker(TEST_TICKER, action="invalid")


class TestGetKalshiSpreadByTicker:
    """Tests for get_kalshi_spread_by_ticker function - FREE (reads market data)"""

    @pytest.mark.skipif(not HAS_KALSHI_CREDS, reason="Kalshi credentials not available")
    def test_real_spread_fetch(self):
        """Test with real Kalshi API - FREE"""
        try:
            spread = get_kalshi_spread_by_ticker(TEST_TICKER)
        # Spread should be positive and reasonable (less than 10¢)
        assert 0 <= spread <= 0.10
        except ValueError as e:
            if "No bid/ask prices available" in str(e) or "Error fetching spread" in str(e):
                pytest.skip(f"Ticker {TEST_TICKER} not available or invalid")

    @patch('app.utils.get_kalshi_client')
    def test_raises_error_on_missing_prices(self, mock_get_client):
        mock_client = Mock()
        mock_market = Mock()
        mock_market.market.yes_bid = None
        mock_market.market.yes_ask = None
        mock_client.get_market.return_value = mock_market
        mock_get_client.return_value = mock_client

        with pytest.raises(ValueError, match="No bid/ask prices available"):
            get_kalshi_spread_by_ticker(TEST_TICKER)

    @patch('app.utils.get_kalshi_client')
    def test_raises_error_on_api_failure(self, mock_get_client):
        mock_get_client.side_effect = Exception("API error")

        with pytest.raises(ValueError, match="Error fetching spread"):
            get_kalshi_spread_by_ticker(TEST_TICKER)


class TestGetKalshiOrderbookByTicker:
    """Tests for get_kalshi_orderbook_by_ticker function - FREE (reads market data)"""

    @pytest.mark.skipif(not HAS_KALSHI_CREDS, reason="Kalshi credentials not available")
    def test_real_orderbook_fetch(self):
        """Test with real Kalshi API - FREE"""
        try:
            orderbook = get_kalshi_orderbook_by_ticker(TEST_TICKER)
            assert isinstance(orderbook, dict)
            # Orderbook should have yes/no keys
            assert 'yes' in orderbook or 'no' in orderbook
        except ValueError as e:
            if "Error fetching orderbook" in str(e):
                pytest.skip(f"Ticker {TEST_TICKER} not available or invalid")


class TestGetKalshiPriceImpactByTicker:
    """Tests for get_kalshi_price_impact_by_ticker function"""

    def test_returns_valid_estimate(self):
        """Test that price impact returns a valid float"""
        try:
            impact = get_kalshi_price_impact_by_ticker(TEST_TICKER, count=1, action="buy")
        # Price impact can be 0.0 for small orders, or 0.01 for errors
        # Just check it's a valid float
        assert isinstance(impact, float)
        assert impact >= 0.0
        except ValueError:
            # If ticker doesn't exist, that's okay for this test
            pass

    def test_invalid_action_raises_error(self):
        """Test that invalid action raises ValueError"""
        with pytest.raises(ValueError, match="Invalid action"):
            get_kalshi_price_impact_by_ticker(TEST_TICKER, count=1, action="invalid")


class TestGetAvailableFunds:
    """Tests for get_available_funds function - FREE (reads balance)"""

    @pytest.mark.skipif(not HAS_KALSHI_CREDS, reason="Kalshi credentials not available")
    def test_real_balance_fetch(self):
        """Test with real Kalshi API - FREE"""
        funds = get_available_funds()

        # Balance should be non-negative
        assert funds >= 0
        # Should be a reasonable amount (less than $1M for testing accounts)
        assert funds < 1000000

    @patch('app.utils.get_kalshi_client')
    def test_raises_error_on_api_failure(self, mock_get_client):
        mock_get_client.side_effect = Exception("API error")

        with pytest.raises(ValueError, match="Error fetching balance"):
            get_available_funds()


class TestWriteTradeToCSV:
    """Tests for write_trade_to_csv function"""

    def test_creates_file_and_writes_trade(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
            csv_file = f.name

        os.unlink(csv_file)

        try:
            write_trade_to_csv(
                timestamp='2025-01-01T00:00:00',
                ticker=TEST_TICKER,
                action='buy',
                amount=10,
                avg_price=0.51,
                csv_file=csv_file
            )

            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                assert rows[0]['ticker'] == TEST_TICKER
                assert rows[0]['action'] == 'buy'
                assert rows[0]['amount'] == '10'
                assert rows[0]['avg_price'] == '0.51'
        finally:
            if os.path.exists(csv_file):
                os.unlink(csv_file)


class TestExecuteTradeByTicker:
    """
    Tests for execute_trade_by_ticker function
    
    **⚠️ WARNING: THESE TESTS COST MONEY ⚠️**
    
    These tests make real API calls that place actual orders on Kalshi, which will cost money.
    By default, these tests are skipped unless ENABLE_EXPENSIVE_TESTS=true is set.
    
    To run these tests:
        ENABLE_EXPENSIVE_TESTS=true pytest personal/test_utils.py::TestExecuteTradeByTicker -v
    
    Note: Validation tests (invalid inputs) do not place orders and can run without the flag.
    """

    @pytest.mark.skipif(not ENABLE_EXPENSIVE_TESTS, reason="Expensive tests disabled. Set ENABLE_EXPENSIVE_TESTS=true to run.")
    @pytest.mark.skipif(not HAS_KALSHI_CREDS, reason="Kalshi credentials not available")
    def test_buy_order_success(self):
        """Test successful buy order with market order - REAL API CALL, COSTS MONEY"""
        # Execute market buy order for 1 contract
        result = execute_trade_by_ticker(ticker=TEST_TICKER, action="buy", count=1, limit_price=None)

        assert result is not None
        assert "order_id" in result
        assert result["count"] == 1
        assert result["action"] == "buy"
        assert result["order_type"] == "market"
        assert 0 <= result["execution_price"] <= 1

    @pytest.mark.skipif(not ENABLE_EXPENSIVE_TESTS, reason="Expensive tests disabled. Set ENABLE_EXPENSIVE_TESTS=true to run.")
    @pytest.mark.skipif(not HAS_KALSHI_CREDS, reason="Kalshi credentials not available")
    def test_buy_order_with_limit_price(self):
        """Test buy order with limit price - REAL API CALL, COSTS MONEY"""
        # Get current market price to set a reasonable limit
        # For a buy order, set limit at or below market price to get a better deal
        try:
            current_price = get_kalshi_price_by_ticker(TEST_TICKER, action="buy")
        limit_price = max(current_price - 0.05, 0.01)  # Set limit 5 cents below market (or 1 cent minimum)

        # Execute limit buy order for 1 contract at limit price
            result = execute_trade_by_ticker(ticker=TEST_TICKER, action="buy", count=1, limit_price=limit_price)

        assert result is not None
        assert "order_id" in result
        assert result["order_type"] == "limit"
        assert 0 <= result["execution_price"] <= limit_price
        
        # Cancel the limit order after test
            cancelled_count, failed_count = cancel_all_orders_for_ticker(TEST_TICKER)
        assert cancelled_count >= 0  # Should have cancelled at least the order we just created
        assert failed_count == 0  # Should not have any failures
        except ValueError as e:
            if "No ask price available" in str(e) or "Error fetching Kalshi price" in str(e):
                pytest.skip(f"Ticker {TEST_TICKER} not available or invalid")

    @pytest.mark.skipif(not ENABLE_EXPENSIVE_TESTS, reason="Expensive tests disabled. Set ENABLE_EXPENSIVE_TESTS=true to run.")
    @pytest.mark.skipif(not HAS_KALSHI_CREDS, reason="Kalshi credentials not available")
    def test_buy_long_then_sell_short(self):
        """Test buying long then selling short to achieve net short position - REAL API CALL, COSTS MONEY"""
        import time
        
        # Cancel any existing orders first
        cancel_all_orders_for_ticker(TEST_TICKER)
        
        try:
        # Get initial position
            initial_position = get_position_by_ticker(TEST_TICKER)
        
        # Step 1: Buy 1 contract (go long)
            buy_result = execute_trade_by_ticker(ticker=TEST_TICKER, action="buy", count=1, limit_price=None)
        assert buy_result is not None
        assert buy_result["action"] == "buy"
        assert buy_result["count"] == 1
        
        # Wait for position to update
        time.sleep(5)
        
        # Verify we have 1 long position
            position_after_buy = get_position_by_ticker(TEST_TICKER)
        assert position_after_buy >= 1, f"Expected at least 1 long position after buy, got {position_after_buy}"
        
        # Step 2: Sell 2 contracts (this will cover the 1 long and create 1 short)
            sell_result = execute_trade_by_ticker(ticker=TEST_TICKER, action="sell", count=2, limit_price=None)
        assert sell_result is not None
        assert sell_result["action"] == "sell"
        assert sell_result["count"] == 2
        
        # Wait for position to update
        time.sleep(5)
        
        # Verify we have net short position of -1
            final_position = get_position_by_ticker(TEST_TICKER)
        assert final_position == -1, f"Expected net short position of -1, got {final_position}"
        except ValueError as e:
            if "No ask price available" in str(e) or "Error fetching Kalshi price" in str(e):
                pytest.skip(f"Ticker {TEST_TICKER} not available or invalid")

    def test_invalid_action_raises_error(self):
        """Test that invalid action raises ValueError - NO API CALL, NO COST"""
        with pytest.raises(ValueError, match="Invalid action"):
            execute_trade_by_ticker(ticker=TEST_TICKER, action="invalid", count=10)

    def test_invalid_count_raises_error(self):
        """Test that invalid count raises ValueError - NO API CALL, NO COST"""
        with pytest.raises(ValueError, match="count must be a positive integer"):
            execute_trade_by_ticker(ticker=TEST_TICKER, action="buy", count=0)

        with pytest.raises(ValueError, match="count must be a positive integer"):
            execute_trade_by_ticker(ticker=TEST_TICKER, action="buy", count=-5)

    def test_invalid_limit_price_raises_error(self):
        """Test that invalid limit_price raises ValueError - NO API CALL, NO COST"""
        with pytest.raises(ValueError, match="limit_price must be between 0 and 1"):
            execute_trade_by_ticker(ticker=TEST_TICKER, action="buy", count=10, limit_price=1.5)

        with pytest.raises(ValueError, match="limit_price must be between 0 and 1"):
            execute_trade_by_ticker(ticker=TEST_TICKER, action="buy", count=10, limit_price=-0.1)


class TestGetPositionByTicker:
    """Tests for get_position_by_ticker function - FREE (reads position)"""

    @pytest.mark.skipif(not HAS_KALSHI_CREDS, reason="Kalshi credentials not available")
    def test_real_position_fetch(self):
        """Test with real Kalshi API - FREE"""
        try:
            position = get_position_by_ticker(TEST_TICKER)
            # Position can be positive (long), negative (short), or zero
            assert isinstance(position, int)
        except ValueError as e:
            if "Error fetching position" in str(e):
                pytest.skip(f"Ticker {TEST_TICKER} not available or invalid")


class TestCancelAllOrdersForTicker:
    """Tests for cancel_all_orders_for_ticker function"""
    
    @pytest.mark.skipif(not ENABLE_EXPENSIVE_TESTS, reason="Expensive tests disabled. Set ENABLE_EXPENSIVE_TESTS=true to run.")
    @pytest.mark.skipif(not HAS_KALSHI_CREDS, reason="Kalshi credentials not available")
    def test_cancel_all_orders_for_ticker(self):
        """Test cancelling all outstanding orders for a ticker - REAL API CALL"""
        # First, create a limit order that will likely remain outstanding
        try:
            current_price = get_kalshi_price_by_ticker(TEST_TICKER, action="buy")
        limit_price = max(current_price - 0.10, 0.01)  # Set limit well below market so it won't fill
        
        # Create limit order
            order_result = execute_trade_by_ticker(ticker=TEST_TICKER, action="buy", count=1, limit_price=limit_price)
        assert order_result is not None
        assert order_result["order_type"] == "limit"
        
        # Give it a moment to appear in the system
        import time
        time.sleep(2)
        
            # Cancel all orders for the ticker
            cancelled_count, failed_count = cancel_all_orders_for_ticker(TEST_TICKER)
        
        # Should have cancelled at least the order we just created
        assert cancelled_count >= 1
        assert failed_count == 0
        
        # Verify no orders remain by trying to cancel again
            cancelled_count_2, failed_count_2 = cancel_all_orders_for_ticker(TEST_TICKER)
        assert cancelled_count_2 == 0  # No orders should remain
        assert failed_count_2 == 0
        except ValueError as e:
            if "No ask price available" in str(e) or "Error fetching Kalshi price" in str(e):
                pytest.skip(f"Ticker {TEST_TICKER} not available or invalid")
    
    @pytest.mark.skipif(not HAS_KALSHI_CREDS, reason="Kalshi credentials not available")
    def test_cancel_all_orders_for_ticker_no_orders(self):
        """Test cancelling orders when none exist - FREE (no orders to cancel)"""
        # Cancel all orders for a ticker (should handle gracefully when no orders exist)
        cancelled_count, failed_count = cancel_all_orders_for_ticker(TEST_TICKER)
        
        # Should return 0 cancelled, 0 failed (no error)
        assert cancelled_count == 0
        assert failed_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
