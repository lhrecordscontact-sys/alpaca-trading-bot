import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# =========================================================
# ALPACA SETTINGS
# =========================================================

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

DATA_URL = "https://data.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

# =========================================================
# SCANNER SETTINGS
# =========================================================

TIMEFRAME = "4Min"

MIN_PRICE = 5.00
MAX_PRICE = 1000.00

MIN_AVG_VOLUME = 500_000
MIN_RELATIVE_VOLUME = 1.20

# Number of candidates to scan from Alpaca's active-stock list
TOP_ACTIVE_STOCKS = 50

# Require this many candles before calculations
MIN_BARS = 40

# Minimum score required to return a stock
MIN_SCORE = 70


# =========================================================
# INDICATORS
# =========================================================

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def calculate_vwap(df):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3

    cumulative_price_volume = (
        typical_price * df["volume"]
    ).cumsum()

    cumulative_volume = df["volume"].cumsum()

    return cumulative_price_volume / cumulative_volume


# =========================================================
# GET MOST ACTIVE STOCKS
# =========================================================

def get_most_active_stocks():
    """
    Gets highly active US stocks from Alpaca.
    """

    url = (
        f"{DATA_URL}/v1beta1/screener/stocks/most-actives"
        f"?top={TOP_ACTIVE_STOCKS}&by=volume"
    )

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        stocks = data.get("most_actives", [])

        symbols = []

        for stock in stocks:
            symbol = stock.get("symbol")

            if symbol:
                symbols.append(symbol)

        return symbols

    except Exception as error:
        print(f"Error getting active stocks: {error}")

        # Backup list if screener endpoint is unavailable
        return [
            "SPY",
            "IWM",
            "QQQ",
            "NVDA",
            "AMD",
            "META",
            "TSLA",
            "AMZN",
            "AAPL",
            "MSFT",
            "GOOGL",
            "NFLX",
            "AVGO",
            "ARM",
            "PLTR",
        ]


# =========================================================
# GET HISTORICAL BARS
# =========================================================

def get_stock_bars(symbol):
    """
    Downloads recent 4-minute candles.
    """

    end_time = datetime.now(timezone.utc)

    start_time = end_time - timedelta(days=10)

    url = f"{DATA_URL}/v2/stocks/{symbol}/bars"

    params = {
        "timeframe": TIMEFRAME,
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "limit": 1000,
        "adjustment": "raw",
        "feed": "iex",
    }

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=15,
        )

        response.raise_for_status()

        data = response.json()

        bars = data.get("bars", [])

        if not bars:
            return None

        df = pd.DataFrame(bars)

        df.rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            },
            inplace=True,
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True
        )

        df = df.sort_values("timestamp")

        return df

    except Exception as error:
        print(f"{symbol}: bar error: {error}")

        return None


# =========================================================
# PREPARE INDICATORS
# =========================================================

def prepare_indicators(df):

    df = df.copy()

    df["ema5"] = calculate_ema(df["close"], 5)
    df["ema9"] = calculate_ema(df["close"], 9)
    df["ema30"] = calculate_ema(df["close"], 30)

    df["vwap"] = calculate_vwap(df)

    df["avg_volume_20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["relative_volume"] = (
        df["volume"] /
        df["avg_volume_20"]
    )

    return df


# =========================================================
# CALL SETUP
# =========================================================

def score_call_setup(df):

    latest = df.iloc[-1]
    previous = df.iloc[-2]
    candle3 = df.iloc[-3]

    score = 0
    reasons = []

    # ---------------------------------------------
    # EMA alignment
    # ---------------------------------------------

    if latest["ema5"] > latest["ema9"]:
        score += 15
        reasons.append("5 EMA above 9 EMA")

    if latest["ema9"] > latest["ema30"]:
        score += 15
        reasons.append("9 EMA above 30 EMA")

    # ---------------------------------------------
    # Price above VWAP
    # ---------------------------------------------

    if latest["close"] > latest["vwap"]:
        score += 15
        reasons.append("Price above VWAP")

    # ---------------------------------------------
    # EMAs above VWAP
    # ---------------------------------------------

    if (
        latest["ema5"] > latest["vwap"]
        and latest["ema9"] > latest["vwap"]
    ):
        score += 10
        reasons.append("5/9 EMA above VWAP")

    # ---------------------------------------------
    # Bullish candle
    # ---------------------------------------------

    if latest["close"] > latest["open"]:
        score += 5
        reasons.append("Bullish confirmation candle")

    # ---------------------------------------------
    # Pullback toward 5/9 EMA
    # ---------------------------------------------

    pullback = (
        previous["low"] <= previous["ema9"] * 1.002
        or previous["low"] <= previous["ema5"] * 1.002
    )

    if pullback:
        score += 10
        reasons.append("Pullback into EMA area")

    # ---------------------------------------------
    # Pullback held instead of breaking down
    # ---------------------------------------------

    if (
        previous["close"] > previous["ema9"]
        and latest["close"] > previous["high"]
    ):
        score += 10
        reasons.append("Bullish pullback confirmation")

    # ---------------------------------------------
    # Momentum
    # ---------------------------------------------

    if latest["close"] > previous["close"] > candle3["close"]:
        score += 5
        reasons.append("Increasing price momentum")

    # ---------------------------------------------
    # Relative volume
    # ---------------------------------------------

    if latest["relative_volume"] >= MIN_RELATIVE_VOLUME:
        score += 10
        reasons.append(
            f"Relative volume {latest['relative_volume']:.2f}x"
        )

    # ---------------------------------------------
    # EMA slopes
    # ---------------------------------------------

    if (
        latest["ema5"] > previous["ema5"]
        and latest["ema9"] > previous["ema9"]
    ):
        score += 5
        reasons.append("EMAs rising")

    return score, reasons


# =========================================================
# PUT SETUP
# =========================================================

def score_put_setup(df):

    latest = df.iloc[-1]
    previous = df.iloc[-2]
    candle3 = df.iloc[-3]

    score = 0
    reasons = []

    # ---------------------------------------------
    # EMA alignment
    # ---------------------------------------------

    if latest["ema5"] < latest["ema9"]:
        score += 15
        reasons.append("5 EMA below 9 EMA")

    if latest["ema9"] < latest["ema30"]:
        score += 15
        reasons.append("9 EMA below 30 EMA")

    # ---------------------------------------------
    # Price below VWAP
    # ---------------------------------------------

    if latest["close"] < latest["vwap"]:
        score += 15
        reasons.append("Price below VWAP")

    # ---------------------------------------------
    # EMAs below VWAP
    # ---------------------------------------------

    if (
        latest["ema5"] < latest["vwap"]
        and latest["ema9"] < latest["vwap"]
    ):
        score += 10
        reasons.append("5/9 EMA below VWAP")

    # ---------------------------------------------
    # Bearish candle
    # ---------------------------------------------

    if latest["close"] < latest["open"]:
        score += 5
        reasons.append("Bearish confirmation candle")

    # ---------------------------------------------
    # Pullback upward toward EMAs
    # ---------------------------------------------

    pullback = (
        previous["high"] >= previous["ema9"] * 0.998
        or previous["high"] >= previous["ema5"] * 0.998
    )

    if pullback:
        score += 10
        reasons.append("Pullback into EMA area")

    # ---------------------------------------------
    # Rejection + confirmation
    # ---------------------------------------------

    if (
        previous["close"] < previous["ema9"]
        and latest["close"] < previous["low"]
    ):
        score += 10
        reasons.append("Bearish pullback confirmation")

    # ---------------------------------------------
    # Momentum
    # ---------------------------------------------

    if latest["close"] < previous["close"] < candle3["close"]:
        score += 5
        reasons.append("Increasing downside momentum")

    # ---------------------------------------------
    # Relative volume
    # ---------------------------------------------

    if latest["relative_volume"] >= MIN_RELATIVE_VOLUME:
        score += 10
        reasons.append(
            f"Relative volume {latest['relative_volume']:.2f}x"
        )

    # ---------------------------------------------
    # EMA slopes
    # ---------------------------------------------

    if (
        latest["ema5"] < previous["ema5"]
        and latest["ema9"] < previous["ema9"]
    ):
        score += 5
        reasons.append("EMAs falling")

    return score, reasons


# =========================================================
# ANALYZE ONE STOCK
# =========================================================

def analyze_stock(symbol):

    df = get_stock_bars(symbol)

    if df is None:
        return None

    if len(df) < MIN_BARS:
        return None

    df = prepare_indicators(df)

    latest = df.iloc[-1]

    price = latest["close"]

    # Price filter
    if price < MIN_PRICE or price > MAX_PRICE:
        return None

    avg_volume = latest["avg_volume_20"]

    if pd.isna(avg_volume):
        return None

    # Volume filter
    if avg_volume < MIN_AVG_VOLUME:
        return None

    call_score, call_reasons = score_call_setup(df)

    put_score, put_reasons = score_put_setup(df)

    if call_score >= put_score:
        direction = "CALL"
        score = call_score
        reasons = call_reasons

    else:
        direction = "PUT"
        score = put_score
        reasons = put_reasons

    if score < MIN_SCORE:
        return None

    result = {
        "symbol": symbol,
        "direction": direction,
        "score": score,
        "price": round(float(price), 2),
        "relative_volume": round(
            float(latest["relative_volume"]),
            2
        ),
        "ema5": round(float(latest["ema5"]), 2),
        "ema9": round(float(latest["ema9"]), 2),
        "ema30": round(float(latest["ema30"]), 2),
        "vwap": round(float(latest["vwap"]), 2),
        "reasons": reasons,
    }

    return result


# =========================================================
# MAIN SCANNER
# =========================================================

def scan_market():

    print("\n========================================")
    print("PURGATORY AI STOCK SCANNER")
    print("========================================\n")

    symbols = get_most_active_stocks()

    print(f"Scanning {len(symbols)} active stocks...\n")

    results = []

    for symbol in symbols:

        print(f"Scanning {symbol}...")

        result = analyze_stock(symbol)

        if result:
            results.append(result)

    # Highest score first
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    print("\n========================================")
    print("TOP SETUPS")
    print("========================================\n")

    if not results:
        print("No high-quality setups found.")

        return []

    for result in results:

        print(
            f"{result['symbol']} | "
            f"{result['direction']} | "
            f"SCORE: {result['score']}/100"
        )

        print(
            f"Price: ${result['price']} | "
            f"RVOL: {result['relative_volume']}x"
        )

        print(
            f"EMA5: {result['ema5']} | "
            f"EMA9: {result['ema9']} | "
            f"EMA30: {result['ema30']} | "
            f"VWAP: {result['vwap']}"
        )

        print("Reasons:")

        for reason in result["reasons"]:
            print(f"  - {reason}")

        print("----------------------------------------")

    return results


# =========================================================
# ALLOW MAIN.PY TO IMPORT SCANNER
# =========================================================

def get_best_setups(limit=10):

    results = scan_market()

    return results[:limit]


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    scan_market()
