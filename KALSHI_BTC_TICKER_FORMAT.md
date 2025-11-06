# Kalshi BTC Ticker Format Documentation

Kalshi has two types of Bitcoin binary markets that trade on the hour.

## Format 1: Range Markets (`KXBTC-*`)

**Format**: `KXBTC-[YY][MON][DD][HH]-B[midpoint]`

**Example**: `KXBTC-25NOV0612-B101875`

### Components:
- `KXBTC` - Prefix for range markets
- `[YY]` - Last two digits of year (e.g., `25` for 2025)
- `[MON]` - Three-letter month abbreviation (e.g., `NOV` for November)
- `[DD]` - Two-digit day (e.g., `06` for 6th)
- `[HH]` - Two-digit hour in 24-hour format (e.g., `12` for noon)
- `B[midpoint]` - Midpoint price (no decimal point, integer)

### Resolution:
- Resolves to **YES** if Bitcoin price is between `midpoint - 125` and `midpoint + 125`
- Resolves to **NO** otherwise

### Example Breakdown:
`KXBTC-25NOV0612-B101875`
- Date: November 6, 2025, 12:00 (noon)
- Midpoint: 101,875
- Range: 101,750 to 102,000 (101,875 ± 125)
- Resolves YES if BTC price is between $101,750 and $102,000

---

## Format 2: Threshold Markets (`KXBTCD-*`)

**Format**: `KXBTCD-[YY][MON][DD][HH]-T[threshold]`

**Example**: `KXBTCD-25NOV0612-T101999.99`

### Components:
- `KXBTCD` - Prefix for threshold markets
- `[YY]` - Last two digits of year (e.g., `25` for 2025)
- `[MON]` - Three-letter month abbreviation (e.g., `NOV` for November)
- `[DD]` - Two-digit day (e.g., `06` for 6th)
- `[HH]` - Two-digit hour in 24-hour format (e.g., `12` for noon)
- `T[threshold]` - Threshold price (can include decimal point)

### Resolution:
- Resolves to **YES** if Bitcoin price is **above** the threshold
- Resolves to **NO** if Bitcoin price is at or below the threshold

### Example Breakdown:
`KXBTCD-25NOV0612-T101999.99`
- Date: November 6, 2025, 12:00 (noon)
- Threshold: 101,999.99
- Resolves YES if BTC price > $101,999.99

---

## Month Abbreviations

| Month | Code |
|-------|------|
| January | JAN |
| February | FEB |
| March | MAR |
| April | APR |
| May | MAY |
| June | JUN |
| July | JUL |
| August | AUG |
| September | SEP |
| October | OCT |
| November | NOV |
| December | DEC |

---

## Notes

- All times are in UTC
- Range markets have a fixed ±125 range around the midpoint
- Threshold markets can have decimal prices
- Markets typically expire on the hour specified in the ticker

