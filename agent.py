"""
Advanced Swing Trading Agent v4
Improved Version of Your Existing Strategy

MAJOR IMPROVEMENTS:
✔ BUY-only mode
✔ Stronger ADX filtering
✔ No after-hours trading
✔ Reduced HOLD spam
✔ Breakout confirmation
✔ Volume confirmation
✔ ATR trailing stop framework
✔ Better trend validation
✔ Cleaner signal generation
✔ Safer risk structure
✔ Reduced lagging indicators

Designed for:
- Indian Equity Swing Trading
- 15m timeframe
- 1–3 day holding period
"""

import os
import json
import time
import datetime
import pytz
import pandas as pd
import numpy as np
import gspread

from google.oauth2.service_account import Credentials

IST = pytz.timezone("Asia/Kolkata")

# =========================================================
# CONFIG
# =========================================================

STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BAJFINANCE.NS",
    "KOTAKBANK.NS",
    "WIPRO.NS",
    "AXISBANK.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "ADANIENT.NS",
    "BHARTIARTL.NS",
    "LTIM.NS",
]

GOOGLE_SHEET_NAME = "Trading data"

# =========================================================
# STRATEGY SETTINGS
# =========================================================

ADX_MIN = 25
BREAKOUT_LOOKBACK = 20
VOLUME_SPIKE = 1.5

ATR_SL_MULT = 1.8
ATR_TRAIL_MULT = 2.2

EMA_FAST = 9
EMA_SLOW = 21

COOLDOWN_HOURS = 6

# =========================================================
# MARKET STATUS
# =========================================================

def market_open():
    now = datetime.datetime.now(IST)

    if now.weekday() >= 5:
        return False

    market_open = now.replace(hour=9, minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)

    return market_open <= now <= market_close


# =========================================================
# INDICATORS
# =========================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)

    return tr.ewm(span=period, adjust=False).mean()


def adx(df, period=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff() * -1

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)

    atr_val = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_val)

    dx = (
        abs(plus_di - minus_di) /
        (plus_di + minus_di)
    ) * 100

    return dx.rolling(period).mean().iloc[-1]


# =========================================================
# BREAKOUT LOGIC
# =========================================================

def breakout_signal(df):

    recent_high = df["High"].rolling(BREAKOUT_LOOKBACK).max().iloc[-2]

    current_close = df["Close"].iloc[-1]

    return current_close > recent_high


# =========================================================
# VOLUME CONFIRMATION
# =========================================================

def volume_spike(df):

    avg_volume = df["Volume"].rolling(20).mean().iloc[-1]

    current_volume = df["Volume"].iloc[-1]

    ratio = current_volume / avg_volume

    return ratio >= VOLUME_SPIKE, round(ratio, 2)


# =========================================================
# TREND FILTER
# =========================================================

def trend_strength(df):

    close = df["Close"]

    ema_fast = ema(close, EMA_FAST).iloc[-1]
    ema_slow = ema(close, EMA_SLOW).iloc[-1]

    bullish = ema_fast > ema_slow

    return bullish, ema_fast, ema_slow


# =========================================================
# SIGNAL ENGINE
# =========================================================

def generate_signal(df):

    if len(df) < 50:
        return None

    current_price = float(df["Close"].iloc[-1])

    bullish_trend, ema9, ema21 = trend_strength(df)

    adx_value = adx(df)

    breakout = breakout_signal(df)

    volume_ok, vol_ratio = volume_spike(df)

    atr_value = atr(df).iloc[-1]

    score = 0
    reasons = []

    # Trend
    if bullish_trend:
        score += 1
        reasons.append("EMA bullish")

    # ADX
    if adx_value >= ADX_MIN:
        score += 1
        reasons.append(f"ADX strong {round(adx_value,1)}")

    # Breakout
    if breakout:
        score += 1.5
        reasons.append("20-candle breakout")

    # Volume
    if volume_ok:
        score += 1
        reasons.append(f"Volume spike x{vol_ratio}")

    # Final Decision
    if score >= 3:
        signal = "BUY"

        confidence = (
            "HIGH"
            if score >= 4
            else "MEDIUM"
        )

    else:
        signal = "HOLD"
        confidence = "LOW"

    stop_loss = round(
        current_price - (atr_value * ATR_SL_MULT),
        2
    )

    trailing_stop = round(
        current_price - (atr_value * ATR_TRAIL_MULT),
        2
    )

    return {
        "signal": signal,
        "confidence": confidence,
        "price": round(current_price, 2),
        "ema9": round(ema9, 2),
        "ema21": round(ema21, 2),
        "adx": round(adx_value, 2),
        "volume_ratio": vol_ratio,
        "atr": round(atr_value, 2),
        "stop_loss": stop_loss,
        "trailing_stop": trailing_stop,
        "notes": " | ".join(reasons)
    }


# =========================================================
# DATA FETCH
# =========================================================

def fetch_data(symbol):

    import requests

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=15m&range=30d"
    )

    try:

        r = requests.get(url, headers=headers, timeout=15)

        data = r.json()

        result = data["chart"]["result"][0]

        q = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "Open": q["open"],
            "High": q["high"],
            "Low": q["low"],
            "Close": q["close"],
            "Volume": q["volume"]
        })

        df.dropna(inplace=True)

        return df

    except Exception as e:

        print(f"ERROR {symbol}: {e}")

        return pd.DataFrame()


# =========================================================
# GOOGLE SHEETS
# =========================================================

def get_sheet():

    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    creds_json = os.environ["GOOGLE_CREDENTIALS_JSON"]

    creds = Credentials.from_service_account_info(
        json.loads(creds_json),
        scopes=scopes
    )

    client = gspread.authorize(creds)

    spreadsheet = client.open(GOOGLE_SHEET_NAME)

    try:

        sheet = spreadsheet.worksheet("Signals")

    except:

        sheet = spreadsheet.add_worksheet(
            title="Signals",
            rows=5000,
            cols=20
        )

    return sheet


# =========================================================
# MAIN AGENT
# =========================================================

def run():

    print("\n==============================")
    print("Advanced Swing Agent v4")
    print("==============================\n")

    if not market_open():

        print("Market closed. No signals generated.")

        return

    sheet = get_sheet()

    now = datetime.datetime.now(IST)

    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    rows = []

    for symbol in STOCKS:

        print(f"\nProcessing {symbol}")

        df = fetch_data(symbol)

        if df.empty:
            continue

        result = generate_signal(df)

        if not result:
            continue

        # Skip HOLD logging
        if result["signal"] == "HOLD":
            print("Skipping HOLD")
            continue

        row = [
            date_str,
            time_str,
            symbol.replace(".NS", ""),
            result["price"],
            result["signal"],
            result["confidence"],
            result["ema9"],
            result["ema21"],
            result["adx"],
            result["volume_ratio"],
            result["atr"],
            result["stop_loss"],
            result["trailing_stop"],
            result["notes"]
        ]

        rows.append(row)

        print(
            f"BUY [{result['confidence']}] "
            f"Price={result['price']} "
            f"SL={result['stop_loss']} "
            f"Trail={result['trailing_stop']} "
            f"{result['notes']}"
        )

        time.sleep(1)

    if rows:

        sheet.append_rows(
            rows,
            value_input_option="USER_ENTERED"
        )

        print(f"\nSaved {len(rows)} signals")

    else:

        print("\nNo valid signals")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    run()
