from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

import requests
import os
import csv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from datetime import datetime
import base64

load_dotenv()

# Dry run configuration
DRY_RUN = False  # Set to False to execute real trades


def get_kalshi_client():
    """
    Initialize and return a Kalshi client with API credentials.
    Uses production API by default.
    """
    from kalshi_python import Configuration, KalshiClient
    
    host = "https://api.elections.kalshi.com/trade-api/v2"
    api_key_id = os.getenv("KALSHI_API_KEY_ID")
    # Default to private_key.txt in project root (parent of app/)
    default_key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "private_key.txt")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", default_key_path)
    
    config = Configuration(host=host)
    
    if not api_key_id:
        raise ValueError("Error: Set KALSHI_API_KEY_ID environment variable")
    
    try:
        with open(private_key_path, "r") as f:
            private_key = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Private key file not found at {private_key_path}")
    
    config.api_key_id = api_key_id
    config.private_key_pem = private_key
    
    return KalshiClient(config)


def get_kalshi_price_by_ticker(ticker, action="buy"):
    """
    Fetch the latest Kalshi price for a given ticker.

    Args:
        ticker: Market ticker (e.g., "BTC-UP-20241231")
        action: "buy" or "sell" - determines which price to return (default: "buy")
                - "buy": returns ask price (what you'd pay)
                - "sell": returns bid price (what you'd receive)

    Returns:
        The yes price in decimal format (0-1).

    Raises:
        ValueError: If action is invalid or price data is unavailable
    """
    action_lower = action.lower()
    if action_lower not in ("buy", "sell"):
        raise ValueError(f"Error: Invalid action {action}. Must be 'buy' or 'sell'")

    try:
        client = get_kalshi_client()
        market = client.get_market(ticker=ticker)

        if action_lower == "buy":
            # For buying, return ask price (what we'd pay)
            if market.market.yes_ask:
                return market.market.yes_ask / 100  # Convert cents to decimal
            elif market.market.last_price:
                return market.market.last_price / 100
            else:
                raise ValueError(f"Error: No ask price available for market {ticker}")
        else:
            if market.market.yes_bid:
                return market.market.yes_bid / 100  # Convert cents to decimal
            elif market.market.last_price:
                return market.market.last_price / 100
            else:
                raise ValueError(f"Error: No bid price available for market {ticker}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error fetching Kalshi price: {e}")


def get_kalshi_spread_by_ticker(ticker):
    """
    Get the bid-ask spread for a market.

    Args:
        ticker: Market ticker (e.g., "BTC-UP-20241231")

    Returns:
        The bid-ask spread as a decimal (0-1).

    Raises:
        ValueError: If bid/ask prices are unavailable
    """
    try:
        client = get_kalshi_client()
        market = client.get_market(ticker=ticker)

        if market.market.yes_bid and market.market.yes_ask:
            spread = (market.market.yes_ask - market.market.yes_bid) / 100
            return spread
        else:
            raise ValueError(f"Error: No bid/ask prices available for market {ticker}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error fetching spread: {e}")


def get_kalshi_orderbook_by_ticker(ticker):
    """
    Get the orderbook for a specific market.
    
    Args:
        ticker: Market ticker (e.g., "BTC-UP-20241231")
    
    Returns:
        Dict with orderbook data containing 'yes', 'yes_dollars', 'no', 'no_dollars'
    
    Raises:
        ValueError: If API call fails
    """
    try:
        # Get credentials
        api_key_id = os.getenv("KALSHI_API_KEY_ID")
        default_key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "private_key.txt")
        private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", default_key_path)
        
        if not api_key_id:
            raise ValueError("Error: Set KALSHI_API_KEY_ID environment variable")
        
        with open(private_key_path, "r") as f:
            private_key_pem = f.read()
        
        # Parse private key
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=None
        )
        
        # Build request
        url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}/orderbook"
        timestamp = str(int(datetime.now().timestamp()))
        path = f"/trade-api/v2/markets/{ticker}/orderbook"
        message = f"{timestamp}GET{path}"
        
        # Sign the message
        signature = private_key.sign(
            message.encode(),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        
        # Encode signature
        signature_b64 = base64.b64encode(signature).decode()
        
        # Set headers
        headers = {
            "KALSHI-ACCESS-KEY": api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp
        }
        
        # Make request
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        return data.get('orderbook', {})
        
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error fetching orderbook: {e}")


def get_kalshi_price_impact_by_ticker(ticker, count=1, action="buy"):
    """
    Estimate price impact based on orderbook depth.
    
    Args:
        ticker: Market ticker (e.g., "BTC-UP-20241231")
        count: Number of contracts to estimate impact for (default: 1)
        action: "buy" or "sell" - determines which side of orderbook to check for price impact
    
    Returns:
        Estimated price impact as a decimal representing the difference
        between average execution price and current market price. 
        Positive when buying, negative when selling.
    """
    try:
        if action not in ("buy", "sell"):
            raise ValueError(f"Invalid action '{action}'. Must be 'buy' or 'sell'")
        
        orderbook = get_kalshi_orderbook_by_ticker(ticker)
        
        # Get current market price
        current_price = get_kalshi_price_by_ticker(ticker, action=action)
        current_price_cents = round(current_price * 100)
        
        # Select which side of orderbook to use:
        # - Buying YES: use NO side (buying NO = buying YES), but prices are inverted
        # - Selling YES: use YES side (selling YES)
        if action == "buy":
            # Use NO side for buying YES
            # NO prices are inverted: NO price = 100 - YES price
            orders = orderbook.get('no', [])
            # Convert NO prices to YES prices: YES_price = 100 - NO_price
            orders_converted = [(100 - price_cents, volume) for price_cents, volume in orders]
            orders_sorted = sorted(orders_converted, key=lambda x: x[0])
        else:  # sell
            # Use YES side for selling YES
            orders = orderbook.get('yes', [])
            # For selling, sort by price descending (highest bids first)
            orders_sorted = sorted(orders, key=lambda x: x[0], reverse=True)
        
        if not orders_sorted:
            return 0.01  # Fallback to conservative estimate
        
        # Filter to asks (prices at or above current market price for buying)
        # or bids (prices at or below current market price for selling)
        remaining = count
        total_cost = 0
        
        for price_cents, volume in orders_sorted:
            if action == "buy":
                # For buying: only consider prices >= current (asks)
                if price_cents < current_price_cents:
                    continue
            else:  # sell
                # For selling: only consider prices <= current (bids)
                if price_cents > current_price_cents:
                    continue
            
            filled = min(remaining, volume)
            total_cost += filled * price_cents
            remaining -= filled
            
            if remaining <= 0:
                break
        
        if remaining > 0:
            # Not enough liquidity at or near current price
            if action == "buy":
                # For buying: use current price + conservative estimate
                avg_price_cents = (total_cost + remaining * (current_price_cents + 2)) / count
            else:  # sell
                # For selling: use current price - conservative estimate (lower price)
                avg_price_cents = (total_cost + remaining * (current_price_cents - 2)) / count
        else:
            avg_price_cents = total_cost / count if count > 0 else current_price_cents
        
        # Calculate impact as difference from current market price
        impact = (avg_price_cents - current_price_cents) / 100
        if action == "buy":
            return max(impact, 0.0)  # Return at least 0 (no negative impact for buying)
        else:  # sell
            return min(impact, 0.0)  # Can be negative for selling (getting less per contract)
        
    except ValueError:
        raise
    except Exception as e:
        # Fallback to conservative estimate on error
        return 0.01


def get_available_funds():
    """
    Get available balance from Kalshi account.
    
    Returns:
        Balance in dollars.
    
    Raises:
        ValueError: If balance cannot be fetched from the API
    """
    try:
        client = get_kalshi_client()
        balance = client.get_balance()
        return balance.balance / 100  # Convert cents to dollars
    except Exception as e:
        raise ValueError(f"Error fetching balance: {e}")


def write_trade_to_csv(timestamp, ticker, action, amount, avg_price, csv_file="trades.csv"):
    """
    Write trade details to CSV file.

    Args:
        timestamp: Timestamp string
        ticker: Market ticker (e.g., "BTC-UP-20241231")
        action: "buy" or "sell"
        amount: Number of contracts
        avg_price: Average execution price in decimal format (0-1)
        csv_file: Path to CSV file (default: "trades.csv")
    """
    file_exists = os.path.exists(csv_file)

    try:
        with open(csv_file, 'a', newline='') as f:
            fieldnames = ['timestamp', 'ticker', 'action', 'amount', 'avg_price']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'timestamp': timestamp,
                'ticker': ticker,
                'action': action,
                'amount': amount,
                'avg_price': avg_price
            })
    except Exception as e:
        print(f"Error writing trade to CSV: {e}")


def execute_trade_by_ticker(ticker, action, count, limit_price=None):
    """
    Execute a trade on Kalshi for the specified ticker.

    Args:
        ticker: Market ticker (e.g., "BTC-UP-20241231")
        action: "buy" or "sell"
        count: Number of contracts to trade
        limit_price: Optional limit price in decimal format (0-1). If None, uses market order.
                    For buy: will not pay more than this price
                    For sell: will not accept less than this price

    Returns:
        Dict with order details if successful, None otherwise

    Raises:
        ValueError: For invalid inputs
    """
    import logging
    logger = logging.getLogger(__name__)

    # Validate inputs
    if action not in ("buy", "sell"):
        raise ValueError(f"Invalid action '{action}'. Must be 'buy' or 'sell'")

    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"count must be a positive integer, got {count}")

    if limit_price is not None and (limit_price <= 0 or limit_price >= 1):
        raise ValueError(f"limit_price must be between 0 and 1, got {limit_price}")

    # DRY RUN MODE: Return simulated successful execution
    if DRY_RUN:
        # Get actual current price for more realistic simulation
        try:
            current_price = get_kalshi_price_by_ticker(ticker, action=action) if limit_price is None else limit_price
        except Exception:
            # Fallback to placeholder if price fetch fails
            current_price = limit_price or 0.50
        
        logger.info(f"[DRY RUN] Would execute: {action.upper()} {count} contracts of {ticker} @ ${current_price:.4f} ({'MARKET' if limit_price is None else 'LIMIT'})")
        return {
            "order_id": "DRY_RUN",
            "ticker": ticker,
            "action": action,
            "count": count,
            "execution_price": current_price,
            "order_type": "limit" if limit_price else "market"
        }

    try:
        client = get_kalshi_client()

        # Determine order type and price
        if limit_price is None:
            # Market order - use current market price (ask for buy, bid for sell)
            order_type = "market"
            # Get current market price for the action
            current_price = get_kalshi_price_by_ticker(ticker, action=action)
            yes_price = round(current_price * 100)  # Convert to cents (round to avoid truncation)
        else:
            # Limit order - only execute at limit_price or better
            order_type = "limit"
            yes_price = round(limit_price * 100)

        # Note: We don't check balance here - let Kalshi handle margin/balance checks
        # Kalshi's system accounts for existing positions, so selling when long or buying when short
        # (reducing exposure) should be allowed even with low balance.
        # If there are truly insufficient funds, Kalshi will reject the order with an appropriate error.

        # Create and execute order
        order_kwargs = {
            "ticker": ticker,
            "action": action,
            "side": "yes",  # We always trade the yes side
            "count": count,
            "type": order_type,
            "yes_price": yes_price  # Need to provide price (required by API)
        }

        order_response = client.create_order(**order_kwargs)

        # Get actual execution price from response or market
        if hasattr(order_response.order, 'yes_price') and order_response.order.yes_price:
            execution_price = order_response.order.yes_price / 100
        else:
            # Fallback to current market price
            execution_price = get_kalshi_price_by_ticker(ticker, action=action)

        print(f"{action.capitalize()} order executed: {count} contracts at ~${execution_price:.3f}")

        # Log trade to CSV
        write_trade_to_csv(
            timestamp=datetime.now().isoformat(),
            ticker=ticker,
            action=action,
            amount=count,
            avg_price=execution_price
        )

        return {
            "order_id": order_response.order.order_id,
            "ticker": ticker,
            "action": action,
            "count": count,
            "execution_price": execution_price,
            "order_type": order_type
        }

    except ValueError:
        raise  # Re-raise ValueError with original message
    except Exception as e:
        print(f"Error executing trade: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_position_by_ticker(ticker):
    """
    Get current position for a specific ticker.
    
    Args:
        ticker: Market ticker (e.g., "BTC-UP-20241231")
    
    Returns:
        Integer representing the current position (number of contracts).
        Positive numbers mean long position, negative means short.
    
    Raises:
        ValueError: If position cannot be fetched
    """
    try:
        client = get_kalshi_client()
        
        # Try to get position from positions API
        # Note: positions API may require event_ticker, so we'll try fills as primary
        current_position = 0
        
        try:
            fills = client.get_fills(limit=1000)
            if fills.fills:
                relevant_fills = [f for f in fills.fills if f.ticker == ticker]
                # Calculate net position for YES side:
                # - side='yes': Add to position (regardless of buy/sell action)
                # - side='no': Subtract from position (regardless of buy/sell action)
                for f in relevant_fills:
                    side = getattr(f, 'side', None)
                    if side is None:
                        print(f"Warning: Fill missing 'side' attribute - fill_id: {getattr(f, 'fill_id', 'unknown')}, action: {f.action}, ticker: {f.ticker}")
                        continue
                    
                    if side == 'yes':
                        current_position += f.count
                    elif side == 'no':
                        current_position -= f.count
        except Exception:
            # If fills API fails, try positions API
            try:
                # Try to get from positions (may need event_ticker)
                positions = client.get_positions()
                position_list = None
                if hasattr(positions, 'market_positions') and positions.market_positions:
                    position_list = positions.market_positions
                elif positions.positions:
                    position_list = positions.positions
                
                if position_list:
                    for pos in position_list:
                        if pos.ticker == ticker:
                            current_position = pos.position
                            break
            except Exception:
                pass
        
        return current_position
    
    except Exception as e:
        raise ValueError(f"Error fetching position: {e}")


def cancel_all_orders_for_ticker(ticker):
    """
    Cancel all outstanding (resting) orders for a specific ticker.
    
    Args:
        ticker: Market ticker (e.g., "BTC-UP-20241231")
    
    Returns:
        Tuple (cancelled_count, failed_count)
    """
    try:
        client = get_kalshi_client()
        
        # Get all resting orders
        orders_response = client.get_orders(status="resting")
        
        if not hasattr(orders_response, 'orders') or not orders_response.orders:
            return (0, 0)
        
        # Filter orders for this ticker
        ticker_orders = [
            o for o in orders_response.orders 
            if o.ticker == ticker
        ]
        
        if not ticker_orders:
            return (0, 0)
        
        print(f"Found {len(ticker_orders)} outstanding order(s) for ticker {ticker}")
        
        # Cancel each order individually
        cancelled_count = 0
        failed_count = 0
        
        for order in ticker_orders:
            try:
                client.cancel_order(order_id=order.order_id)
                cancelled_count += 1
                # Order objects use 'remaining_count', not 'count'
                count = getattr(order, 'remaining_count', None)
                if count is None:
                    count = getattr(order, 'count', None)
                if count is None:
                    print(f"  Warning: Order {order.order_id} missing 'remaining_count' and 'count' fields")
                    count = 'MISSING'
                
                action = getattr(order, 'action', None)
                if action is None:
                    print(f"  Warning: Order {order.order_id} missing 'action' field")
                    action = 'MISSING'
                
                print(f"✓ Cancelled order {order.order_id}: {action} {count} contracts of {order.ticker}")
            except Exception as e:
                failed_count += 1
                print(f"✗ Failed to cancel order {order.order_id}: {e}")
        
        return (cancelled_count, failed_count)
    
    except Exception as e:
        print(f"Error cancelling orders for ticker {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return (0, 0)
