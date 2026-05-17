"""
Advanced Swing Trading Agent v5
Fully Optimized Google Sheets Compatible Version

Features:
✔ BUY-only swing trading system
✔ Strong ADX trend filter
✔ Breakout confirmation
✔ Volume confirmation
✔ ATR stop-loss
✔ Risk reward calculation
✔ Google Sheet full-column compatibility
✔ Clean HOLD filtering
✔ Future analytics ready
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
# SETTINGS
# =========================================================

ADX_MIN = 25
BREAKOUT_LOOKBACK = 20
VOLUME_SPIKE = 1.5

ATR_SL_MULT = 1.8
TARGET_MULT = 3.0

EMA_FAST = 9
EMA_SLOW = 21

# =========================================================
# SECTOR MAP
# =========================================================

SECTOR_MAP = {
    "RELIANCE": "OIL_GAS",
    "TCS": "IT",
    "HDFCBANK": "BANK",
    "INFY": "IT",
    "ICICIBANK": "BANK",
    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "SBIN": "BANK",
    "BAJFINANCE": "FINANCE",
    "KOTAKBANK": "BANK",
    "WIPRO": "IT",
    "AXISBANK": "BANK",
    "MARUTI": "AUTO",
    "SUNPHARMA": "PHARMA",
    "ADANIENT": "CONGLOMERATE",
    "BHARTIARTL": "TELECOM",
    "LTIM": "IT",
}

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
        abs(plus_di - minus_di)
        / (plus_di + minus_di)
    ) * 100

    return dx.rolling(period).mean().iloc[-1]


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def macd(series):

    ema12 = ema(series, 12)
    ema26 = ema(series, 26)

    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)

    return macd_line, signal


def bollinger(series, period=20):

    mid = series.rolling(period).mean()

    std = series.rolling(period).std()

    upper = mid + (std * 2)
    lower = mid - (std * 2)

    return upper.iloc[-1], mid.iloc[-1], lower.iloc[-1]

# =========================================================
# SIGNAL LOGIC
# =========================================================

def breakout_signal(df):

    recent_high = (
        df["High"]
        .rolling(BREAKOUT_LOOKBACK)
        .max()
        .iloc[-2]
    )

    current_close = df["Close"].iloc[-1]

    return current_close > recent_high


def volume_spike(df):

    avg_volume = (
        df["Volume"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    current_volume = df["Volume"].iloc[-1]

    ratio = current_volume / avg_volume

    return ratio >= VOLUME_SPIKE, round(ratio, 2)


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

    close = df["Close"]

    current_price = round(float(close.iloc[-1]), 2)

    bullish_trend, ema9, ema21 = trend_strength(df)

    adx_value = adx(df)

    breakout = breakout_signal(df)

    volume_ok, vol_ratio = volume_spike(df)

    atr_value = round(float(atr(df).iloc[-1]), 2)

    rsi_value = round(float(rsi(close).iloc[-1]), 2)

    macd_line, macd_signal = macd(close)

    macd_value = round(float(macd_line.iloc[-1]), 4)
    macd_signal_value = round(float(macd_signal.iloc[-1]), 4)

    bb_upper, bb_mid, bb_lower = bollinger(close)

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

    # Final Signal
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

    target = round(
        current_price + (atr_value * TARGET_MULT),
        2
    )

    risk = current_price - stop_loss
    reward = target - current_price

    rr = (
        f"1:{round(reward / risk, 2)}"
        if risk > 0
        else ""
    )

    return {
        "signal": signal,
        "confidence": confidence,

        "price": current_price,

        "rsi": rsi_value,

        "macd": macd_value,
        "macd_signal": macd_signal_value,

        "ema9": round(ema9, 2),
        "ema21": round(ema21, 2),

        "volume": int(df["Volume"].iloc[-1]),

        "avg_volume": int(
            df["Volume"]
            .rolling(20)
            .mean()
            .iloc[-1]
        ),

        "volume_ratio": vol_ratio,

        "adx": round(adx_value, 2),

        "bb_upper": round(bb_upper, 2),
        "bb_mid": round(bb_mid, 2),
        "bb_lower": round(bb_lower, 2),

        "atr": atr_value,

        "stop_loss": stop_loss,

        "target": target,

        "risk_reward": rr,

        "trend": (
            "UP"
            if ema9 > ema21
            else "DOWN"
        ),

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

        r = requests.get(
            url,
            headers=headers,
            timeout=15
        )

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
            rows=10000,
            cols=30
        )

    return sheet

# =========================================================
# MAIN
# =========================================================

def run():

    print("\n==============================")
    print("Advanced Swing Agent v5")
    print("==============================\n")

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

        # Skip HOLD rows
        if result["signal"] == "HOLD":

            print("Skipping HOLD")

            continue

        clean_symbol = symbol.replace(".NS", "")

        sector = SECTOR_MAP.get(
            clean_symbol,
            ""
        )

        row = [
            date_str,
            time_str,
            clean_symbol,
            sector,

            result["price"],
            result["signal"],

            result["rsi"],

            result["macd"],
            result["macd_signal"],

            result["ema9"],
            result["ema21"],

            result["volume"],
            result["avg_volume"],

            result["volume_ratio"],

            result["confidence"],

            result["notes"],

            result["adx"],

            result["bb_upper"],
            result["bb_mid"],
            result["bb_lower"],

            result["atr"],

            result["stop_loss"],

            result["target"],

            result["risk_reward"],

            result["trend"],

            "",

            "MARKET_OPEN"
        ]

        rows.append(row)

        print(
            f"{result['signal']} "
            f"[{result['confidence']}] "
            f"Price={result['price']} "
            f"Target={result['target']} "
            f"SL={result['stop_loss']}"
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
