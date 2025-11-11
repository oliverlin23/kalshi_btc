# kalshi_btc

Utilities for trading Bitcoin markets on Kalshi.

## Overview

This module provides general-purpose functions for interacting with the Kalshi exchange API. All functions work directly with market tickers (e.g., `"BTC-UP-20241231"`), making them suitable for trading any market, including Bitcoin price movements.

## Functions

### Client Management

#### `get_kalshi_client()`

Initialize and return a Kalshi client with API credentials.

**Why useful for Bitcoin trading:**

- Essential for all API interactions
- Handles authentication using your API key and private key
- Required by all other functions in this module

---

### Price Discovery

#### `get_kalshi_price_by_ticker(ticker, action="buy")`

Fetch the latest Kalshi price for a given ticker.

**Parameters:**

- `ticker`: Market ticker (e.g., `"BTC-UP-20241231"`)
- `action`: `"buy"` returns ask price (what you'd pay), `"sell"` returns bid price (what you'd receive)

**Returns:** Price in decimal format (0-1)

**Why useful for Bitcoin trading:**

- Get real-time Bitcoin market prices before placing orders
- Understand current market valuation
- Compare bid/ask spreads to identify opportunities
- Essential for limit order price selection

#### `get_kalshi_spread_by_ticker(ticker)`

Get the bid-ask spread for a market.

**Parameters:**

- `ticker`: Market ticker

**Returns:** Spread as a decimal (0-1)

**Why useful for Bitcoin trading:**

- Measure market liquidity (tight spreads = better liquidity)
- Estimate transaction costs before trading
- Identify arbitrage opportunities when spreads are unusually wide
- Monitor market conditions in real-time

---

### Orderbook Analysis

#### `get_kalshi_orderbook_by_ticker(ticker)`

Get the full orderbook for a specific market.

**Returns:** Dict with orderbook data containing `'yes'`, `'yes_dollars'`, `'no'`, `'no_dollars'`

**Why useful for Bitcoin trading:**

- Analyze market depth and liquidity
- Understand where large orders are sitting
- Identify support/resistance levels from orderbook walls
- Essential for sophisticated trading strategies

#### `get_kalshi_price_impact_by_ticker(ticker, count=1, action="buy")`

Estimate price impact based on orderbook depth.

**Parameters:**

- `ticker`: Market ticker
- `count`: Number of contracts to estimate impact for
- `action`: `"buy"` or `"sell"`

**Returns:** Estimated price impact as a decimal (positive when buying, negative when selling)

**Why useful for Bitcoin trading:**

- Calculate expected execution price before placing large orders
- Avoid slippage by sizing orders appropriately
- Estimate transaction costs including market impact
- Critical for risk management when trading larger positions

---

### Account Management

#### `get_available_funds()`

Get available balance from Kalshi account.

**Returns:** Balance in dollars

**Why useful for Bitcoin trading:**

- Check account balance before placing trades
- Ensure sufficient funds for planned positions
- Monitor account health and margin requirements
- Essential for position sizing and risk management

---

### Trade Execution

#### `execute_trade_by_ticker(ticker, action, count, limit_price=None)`

Execute a trade on Kalshi for the specified ticker.

**Parameters:**

- `ticker`: Market ticker
- `action`: `"buy"` or `"sell"`
- `count`: Number of contracts to trade
- `limit_price`: Optional limit price (0-1). If None, uses market order.

**Returns:** Dict with order details if successful

**Why useful for Bitcoin trading:**

- Execute market and limit orders for Bitcoin positions
- Automate trade execution based on your trading strategy
- Supports both market orders (instant execution) and limit orders (price control)
- Automatically checks balance before executing
- Logs trades to CSV for record keeping

#### `cancel_all_orders_for_ticker(ticker)`

Cancel all outstanding (resting) orders for a specific ticker.

**Returns:** Tuple `(cancelled_count, failed_count)`

**Why useful for Bitcoin trading:**

- Quickly exit all pending orders when market conditions change
- Manage risk by canceling orders before market events
- Clean up stale limit orders
- Essential for dynamic trading strategies

---

### Position Management

#### `get_position_by_ticker(ticker)`

Get current position for a specific ticker.

**Returns:** Integer (positive = long, negative = short)

**Why useful for Bitcoin trading:**

- Monitor current exposure to Bitcoin markets
- Track position sizes across different contracts
- Calculate net exposure when trading multiple Bitcoin-related markets
- Essential for risk management and portfolio monitoring

---

### Trade Logging

#### `write_trade_to_csv(timestamp, ticker, action, amount, avg_price, csv_file="trades.csv")`

Write trade details to CSV file for record keeping.

**Why useful for Bitcoin trading:**

- Maintain a complete trading history
- Analyze performance over time
- Calculate P&L and track profitability
- Required for tax reporting and compliance
- Essential for backtesting and strategy refinement

---

## Configuration

### Environment Variables

Set these in your `.env` file:

- `KALSHI_API_KEY_ID`: Your Kalshi API key ID
- `KALSHI_PRIVATE_KEY_PATH`: Path to your private key file (defaults to `private_key.txt` in project root)

### Dry Run Mode

Set `DRY_RUN = True` in `utils.py` to simulate trades without executing them. Useful for testing strategies.

---

## Usage Example

```python
from app.utils import (
    get_kalshi_client,
    get_kalshi_price_by_ticker,
    get_kalshi_price_impact_by_ticker,
    execute_trade_by_ticker,
    get_position_by_ticker
)

# Get current Bitcoin market price
ticker = "BTC-UP-20241231"
current_price = get_kalshi_price_by_ticker(ticker, action="buy")
print(f"Current ask price: ${current_price:.4f}")

# Check price impact before placing large order
impact = get_kalshi_price_impact_by_ticker(ticker, count=100, action="buy")
print(f"Expected price impact: {impact:.4f}")

# Execute a trade
result = execute_trade_by_ticker(
    ticker=ticker,
    action="buy",
    count=10,
    limit_price=0.55  # Only buy if price is 0.55 or better
)

# Check position
position = get_position_by_ticker(ticker)
print(f"Current position: {position} contracts")
```

---

## Key Design Decisions

1. **Ticker-based API**: All functions work with tickers directly, not event configs. This makes the code flexible for any market type, including Bitcoin.

2. **No event configs**: Removed all election-specific event configuration logic. You specify tickers directly when calling functions.

3. **General-purpose utilities**: Functions are designed to be reusable across different trading strategies and market types.

4. **Price impact estimation**: Includes sophisticated orderbook analysis to help with large order execution.

5. **Error handling**: Functions raise clear `ValueError` exceptions with descriptive messages for debugging.

---

## Dashboards

The project includes two dashboard interfaces for monitoring Bitcoin markets:

### CLI Dashboard (`app/cli_dashboard.py`)

A terminal-based live dashboard for monitoring specific tickers.

**Usage:**
```bash
# Run directly
python -m app.cli_dashboard --tickers "BTC-UP-20241231,BTC-DOWN-20241231"

# Or use the runner script
python run_dashboard.py --tickers "BTC-UP-20241231,BTC-DOWN-20241231"

# With environment variables
export BTC_DASHBOARD_TICKERS="BTC-UP-20241231,BTC-DOWN-20241231"
export BTC_DASHBOARD_INTERVAL=0.5
python run_dashboard.py
```

**Features:**
- Real-time price updates (configurable refresh interval, default 0.5s)
- Shows buy (ask), sell (bid), and spread for each ticker
- Simple text table output
- Press Ctrl-C to exit

**Options:**
- `--tickers`: Comma-separated list of tickers to monitor
- `--interval`: Refresh interval in seconds (default: 0.5)
- `--action`: Show "buy", "sell", or "both" prices (default: both)

### Web Dashboard (`app/web_dashboard.py`)

A Flask-based web dashboard with auto-discovery of markets.

**Usage:**
```bash
# Set BTC current price (required)
export BTC_CURRENT_PRICE=101875

# Run the web server
python -m app.web_dashboard

# Or with custom port
export DASHBOARD_PORT=8080
python -m app.web_dashboard
```

Then open `http://localhost:5000` (or your custom port) in your browser.

**Features:**
- Auto-generates 6 markets (3 range, 3 threshold) closest to current BTC price
- Real-time price updates via JSON API
- Shows orderbook volumes, spreads, and market types
- Caching (refreshes every 60 seconds or on hour change)
- Can fetch from Kalshi API or generate tickers locally

**API Endpoints:**
- `GET /` - Main dashboard HTML page
- `GET /api/tickers` - Get available tickers for current EST hour
- `GET /api/prices` - Get real-time prices for all tickers

**Environment Variables:**
- `BTC_CURRENT_PRICE` (required): Current BTC price in dollars (e.g., 101875)
- `DASHBOARD_PORT`: Port to run server on (default: 5000)
- `FLASK_DEBUG`: Set to "true" for debug mode
- `USE_KALSHI_MARKET_API`: Set to "true" to fetch markets from Kalshi API instead of generating locally


