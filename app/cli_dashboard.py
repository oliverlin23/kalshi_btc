import os
import sys
import time
import argparse
from datetime import datetime

# Add project root to path so we can import app modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.utils import get_kalshi_price_by_ticker, get_kalshi_spread_by_ticker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live dashboard for Kalshi BTC binary contracts"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=os.getenv("BTC_DASHBOARD_TICKERS", ""),
        help="Comma-separated list of Kalshi tickers to display (e.g., BTC-UP-20251231,BTC-DOWN-20251231)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.getenv("BTC_DASHBOARD_INTERVAL", "0.5")),
        help="Refresh interval in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--action",
        type=str,
        default="both",
        choices=["buy", "sell", "both"],
        help="Which side price to show (default: both)",
    )
    return parser.parse_args()


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def format_price(p):
    return f"${p:.4f}" if p is not None else "-"


def render_table(rows):
    headers = [
        "Ticker",
        "Buy (Ask)",
        "Sell (Bid)",
        "Spread",
        "Timestamp",
    ]
    col_widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)
    ]
    col_widths = [
        max(col_widths[i], len(headers[i])) for i in range(len(headers))
    ]

    # Header
    header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "  ".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for r in rows:
        print("  ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))))


def main():
    args = parse_args()

    # Resolve tickers list
    if args.tickers.strip():
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        # Default: put your BTC tickers here if not supplied via --tickers or env
        tickers = [
            # Example placeholders – replace with your actual Kalshi BTC market tickers
            # "BTC-UP-20251231",
            # "BTC-DOWN-20251231",
        ]
        if not tickers:
            print("No tickers provided. Use --tickers or set BTC_DASHBOARD_TICKERS env var.")
            sys.exit(1)

    interval = max(0.1, float(args.interval))

    try:
        while True:
            clear_screen()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Kalshi BTC Dashboard   |   {now}   |   interval={interval:.2f}s")
            print()

            rows = []
            for ticker in tickers:
                try:
                    buy_price = None
                    sell_price = None

                    if args.action in ("buy", "both"):
                        buy_price = get_kalshi_price_by_ticker(ticker, action="buy")
                    if args.action in ("sell", "both"):
                        sell_price = get_kalshi_price_by_ticker(ticker, action="sell")

                    try:
                        spread = get_kalshi_spread_by_ticker(ticker)
                    except Exception:
                        spread = None

                    rows.append([
                        ticker,
                        format_price(buy_price),
                        format_price(sell_price),
                        f"{spread:.4f}" if spread is not None else "-",
                        now,
                    ])
                except Exception as e:
                    rows.append([ticker, "-", "-", "-", f"ERR: {e}"])

            render_table(rows)
            print()
            print("Press Ctrl-C to exit. Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH in .env.")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nExiting dashboard.")


if __name__ == "__main__":
    main()
