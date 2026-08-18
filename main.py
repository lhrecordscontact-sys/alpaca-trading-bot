import os
import time
import math
import threading
import traceback
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np
from flask import Flask, jsonify


# ============================================================
# APP / TIME
# ============================================================

app = Flask(__name__)

NY = ZoneInfo("America/New_York")

TRADING_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

SERVICE_NAME = "alpaca-0dte-paper-bot"


# ============================================================
# ENVIRONMENT
# ============================================================

def clean_env(value):
    if value is None:
        return ""

    value = str(value).strip()

    value = (
        value.replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
        .replace("\xa0", "")
        .replace('"', "")
        .replace("'", "")
        .strip()
    )

    return value


ALPACA_API_KEY = clean_env(os.getenv("ALPACA_API_KEY", ""))
ALPACA_SECRET_KEY = clean_env(os.getenv("ALPACA_SECRET_KEY", ""))

AUTO_TRADE = clean_env(
    os.getenv("AUTO_TRADE", "false")
).lower() == "true"

RUN_BOT_LOOP = clean_env(
    os.getenv("RUN_BOT_LOOP", "true")
).lower() == "true"


# ============================================================
# STRATEGY SETTINGS
# ============================================================

TIMEFRAME = "4Min"

# Number of completed historical setups used to score each stock
BACKTEST_TRADES = 64

# Pull enough historical data to FIND 64 setups.
# Increase if a symbol doesn't generate enough setups.
HISTORY_DAYS = 90

# Scanner ranking
TOP_STOCKS = 15

# To keep Render / Alpaca requests manageable we first scan a
# liquid stock universe, then fully backtest the candidates.
MAX_UNIVERSE = 250

MIN_PRICE = 5.00
MAX_PRICE = 1000.00

MIN_AVG_DAILY_VOLUME = 1_000_000

# Require this many historical setups before accepting a stock
MIN_BACKTEST_TRADES = 30

# Minimum historical win rate to qualify
MIN_WIN_RATE = 0.55

# Minimum expectancy per historical trade
MIN_EXPECTANCY = 0.00

# Risk / option position
POSITION_DOLLARS = 500.00
MAX_OPEN_POSITIONS = 3
MAX_NEW_TRADES_PER_CYCLE = 1

STOP_LOSS = 0.20

# Sell 50% at +30%
TAKE_PROFIT = 0.30
TAKE_PROFIT_FRACTION = 0.50

# Runner follows 9 EMA after first TP
RUNNER_TRAIL = 0.15


# ============================================================
# SCANNER TIMING
# ============================================================

PREMARKET_SCAN_START = dt_time(4, 0)
MARKET_OPEN_TIME = dt_time(9, 30)
MARKET_CLOSE_TIME = dt_time(16, 0)

# Repeat probability scan during premarket
PREMARKET_RESCAN_MINUTES = 20

# Live scanner interval after open
LIVE_SCAN_SECONDS = 30


# ============================================================
# GLOBAL STATUS
# ============================================================

status_lock = threading.Lock()

BOT_STATUS = {
    "service": SERVICE_NAME,
    "running": True,
    "paper": True,
    "credentials_ok": False,

    "stock_feed": "iex",
    "option_feed": "inactive",

    "auto_trade": AUTO_TRADE,

    "market_open": False,
    "premarket": False,

    "timeframe": TIMEFRAME,
    "backtest_trades_target": BACKTEST_TRADES,

    "stocks_scanned": 0,
    "stocks_tested": 0,

    "last_scan": None,
    "last_cycle": None,

    "top_probability_stocks": [],
    "signals": [],

    "managed_positions": {},

    "errors": [],
}


# ============================================================
# SESSION
# ============================================================

session = requests.Session()


def headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Accept": "application/json",
    }


# ============================================================
# STATUS HELPERS
# ============================================================

def set_status(key, value):
    with status_lock:
        BOT_STATUS[key] = value


def add_error(message):
    text = str(message)

    with status_lock:
        BOT_STATUS["errors"].append(text)
        BOT_STATUS["errors"] = BOT_STATUS["errors"][-20:]

    print("ERROR:", text, flush=True)


def clear_errors():
    with status_lock:
        BOT_STATUS["errors"] = []


# ============================================================
# HTTP
# ============================================================

def api_get(url, params=None, timeout=30):
    try:
        response = session.get(
            url,
            headers=headers(),
            params=params,
            timeout=timeout,
        )

        if response.status_code == 429:
            time.sleep(3)

            response = session.get(
                url,
                headers=headers(),
                params=params,
                timeout=timeout,
            )

        if not response.ok:
            add_error(
                f"GET {response.status_code} "
                f"{url} | {response.text[:300]}"
            )
            return None

        return response.json()

    except Exception as exc:
        add_error(f"GET FAILED {url} | {exc}")
        return None


def api_post(url, payload=None, timeout=30):
    try:
        response = session.post(
            url,
            headers={
                **headers(),
                "Content-Type": "application/json",
            },
            json=payload or {},
            timeout=timeout,
        )

        if not response.ok:
            add_error(
                f"POST {response.status_code} "
                f"{url} | {response.text[:300]}"
            )
            return None

        return response.json()

    except Exception as exc:
        add_error(f"POST FAILED {url} | {exc}")
        return None


def api_delete(url, timeout=30):
    try:
        response = session.delete(
            url,
            headers=headers(),
            timeout=timeout,
        )

        return response.ok

    except Exception as exc:
        add_error(f"DELETE FAILED {url} | {exc}")
        return False


# ============================================================
# CREDENTIAL CHECK
# ============================================================

def verify_credentials():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        set_status("credentials_ok", False)
        add_error("ALPACA API credentials missing")
        return False

    data = api_get(f"{TRADING_BASE_URL}/v2/account")

    ok = bool(data and data.get("id"))

    set_status("credentials_ok", ok)

    if ok:
        print("ALPACA CREDENTIALS OK", flush=True)

    return ok


# ============================================================
# CLOCK
# ============================================================

def get_clock():
    data = api_get(f"{TRADING_BASE_URL}/v2/clock")

    if not data:
        return None

    set_status(
        "market_open",
        bool(data.get("is_open", False)),
    )

    return data


def time_state():
    now = datetime.now(NY)
    t = now.time()

    weekday = now.weekday() < 5

    premarket = (
        weekday
        and PREMARKET_SCAN_START <= t < MARKET_OPEN_TIME
    )

    regular = (
        weekday
        and MARKET_OPEN_TIME <= t < MARKET_CLOSE_TIME
    )

    set_status("premarket", premarket)

    return premarket, regular


# ============================================================
# ASSET UNIVERSE
# ============================================================

def get_stock_universe():
    """
    Pull active tradable US equities from Alpaca.

    The code then ranks them using daily activity so we don't try
    to download months of 4-minute bars for thousands of symbols
    simultaneously.
    """

    data = api_get(
        f"{TRADING_BASE_URL}/v2/assets",
        params={
            "status": "active",
            "asset_class": "us_equity",
        },
        timeout=60,
    )

    if not isinstance(data, list):
        add_error("ASSET UNIVERSE EMPTY")
        return []

    symbols = []

    for asset in data:
        symbol = asset.get("symbol")

        if not symbol:
            continue

        if not asset.get("tradable", False):
            continue

        # Avoid strange OTC / warrant / preferred symbols where possible
        if len(symbol) > 6:
            continue

        if "." in symbol or "/" in symbol:
            continue

        symbols.append(symbol)

    # Put our most important index ETFs first
    priority = [
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "AAPL",
        "NVDA",
        "TSLA",
        "AMD",
        "META",
        "AMZN",
        "MSFT",
        "GOOGL",
        "AVGO",
        "NFLX",
        "PLTR",
    ]

    ordered = []

    for symbol in priority:
        if symbol in symbols and symbol not in ordered:
            ordered.append(symbol)

    for symbol in symbols:
        if symbol not in ordered:
            ordered.append(symbol)

    return ordered


# ============================================================
# DAILY BARS FOR LIQUIDITY FILTER
# ============================================================

def get_daily_bars(symbol, days=15):
    end = datetime.now(NY)
    start = end - timedelta(days=days * 2)

    data = api_get(
        f"{DATA_BASE_URL}/v2/stocks/{symbol}/bars",
        params={
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 100,
            "adjustment": "raw",
            "feed": "iex",
            "sort": "asc",
        },
    )

    if not data:
        return None

    bars = data.get("bars", [])

    if not bars:
        return None

    return bars


def score_liquidity(symbol):
    bars = get_daily_bars(symbol)

    if not bars:
        return None

    recent = bars[-10:]

    closes = [
        float(x["c"])
        for x in recent
        if x.get("c") is not None
    ]

    volumes = [
        float(x["v"])
        for x in recent
        if x.get("v") is not None
    ]

    if not closes or not volumes:
        return None

    price = closes[-1]
    avg_volume = sum(volumes) / len(volumes)

    if not MIN_PRICE <= price <= MAX_PRICE:
        return None

    if avg_volume < MIN_AVG_DAILY_VOLUME:
        return None

    dollar_volume = avg_volume * price

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "avg_volume": int(avg_volume),
        "dollar_volume": dollar_volume,
    }


# ============================================================
# HISTORICAL 4-MINUTE BARS
# ============================================================

def get_historical_bars(symbol, days=HISTORY_DAYS):
    """
    Historical 4-minute bars used for the 64-trade backtest.
    Handles Alpaca pagination.
    """

    now = datetime.now(NY)

    start = now - timedelta(days=days)
    end = now

    all_bars = []
    page_token = None

    while True:
        params = {
            "timeframe": TIMEFRAME,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
            "adjustment": "raw",
            "feed": "iex",
            "sort": "asc",
        }

        if page_token:
            params["page_token"] = page_token

        data = api_get(
            f"{DATA_BASE_URL}/v2/stocks/{symbol}/bars",
            params=params,
            timeout=45,
        )

        if not data:
            break

        bars = data.get("bars", [])

        if bars:
            all_bars.extend(bars)

        page_token = data.get("next_page_token")

        if not page_token:
            break

        # Keep request rate gentle
        time.sleep(0.10)

    if not all_bars:
        return None

    df = pd.DataFrame(all_bars)

    required = {"t", "o", "h", "l", "c", "v"}

    if not required.issubset(df.columns):
        return None

    df["timestamp"] = pd.to_datetime(
        df["t"],
        utc=True,
    ).dt.tz_convert(NY)

    df = df.set_index("timestamp")

    df = df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    # Strategy is based on regular-session candles
    df = df.between_time(
        "09:30",
        "15:59",
        inclusive="both",
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):
    if df is None or len(df) < 50:
        return None

    df = df.copy()

    df["ema5"] = df["close"].ewm(
        span=5,
        adjust=False,
    ).mean()

    df["ema9"] = df["close"].ewm(
        span=9,
        adjust=False,
    ).mean()

    df["ema30"] = df["close"].ewm(
        span=30,
        adjust=False,
    ).mean()

    # Session VWAP resets every trading day
    session_date = df.index.date

    typical = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3.0

    pv = typical * df["volume"]

    cumulative_pv = pv.groupby(
        session_date
    ).cumsum()

    cumulative_volume = df["volume"].groupby(
        session_date
    ).cumsum()

    df["vwap"] = (
        cumulative_pv
        / cumulative_volume.replace(0, np.nan)
    )

    # ATR
    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] - previous_close
    ).abs()

    tr3 = (
        df["low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    return df


# ============================================================
# PURGATORY SETUP
# ============================================================

def bullish_setup(df, i):
    """
    CALL setup:

    5 EMA > 9 EMA
    5 + 9 above VWAP
    5 + 9 above 30 EMA

    Requires bullish confirmation instead of entering
    merely because the averages crossed.
    """

    if i < 2:
        return False

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    trend = (
        row["ema5"] > row["ema9"]
        and row["ema5"] > row["vwap"]
        and row["ema9"] > row["vwap"]
        and row["ema5"] > row["ema30"]
        and row["ema9"] > row["ema30"]
    )

    momentum = (
        row["close"] > row["ema5"]
        and row["close"] > row["open"]
    )

    confirmation = (
        row["close"] > prev["high"]
        or (
            prev["low"] <= prev["ema9"]
            and row["close"] > prev["close"]
        )
    )

    return bool(
        trend
        and momentum
        and confirmation
    )


def bearish_setup(df, i):
    """
    PUT setup.
    """

    if i < 2:
        return False

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    trend = (
        row["ema5"] < row["ema9"]
        and row["ema5"] < row["vwap"]
        and row["ema9"] < row["vwap"]
        and row["ema5"] < row["ema30"]
        and row["ema9"] < row["ema30"]
    )

    momentum = (
        row["close"] < row["ema5"]
        and row["close"] < row["open"]
    )

    confirmation = (
        row["close"] < prev["low"]
        or (
            prev["high"] >= prev["ema9"]
            and row["close"] < prev["close"]
        )
    )

    return bool(
        trend
        and momentum
        and confirmation
    )


# ============================================================
# HISTORICAL TRADE SIMULATION
# ============================================================

def simulate_trade(df, entry_index, direction):
    """
    Simulates the UNDERLYING stock movement.

    This determines whether the chart setup historically worked.
    It does NOT pretend we know historical 0DTE option premium.

    Winner:
        underlying moves +0.30 ATR in trade direction

    Loser:
        candle closes through 9 EMA

    Also exits at session end.
    """

    entry = df.iloc[entry_index]

    entry_price = float(entry["close"])

    atr = entry["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    target_distance = atr * 0.30

    entry_day = df.index[entry_index].date()

    max_bars = 30

    last_index = min(
        len(df),
        entry_index + max_bars + 1,
    )

    best_move = 0.0

    for j in range(
        entry_index + 1,
        last_index,
    ):
        row = df.iloc[j]

        if df.index[j].date() != entry_day:
            break

        if direction == "CALL":

            favorable = (
                float(row["high"])
                - entry_price
            )

            best_move = max(
                best_move,
                favorable,
            )

            # Target hit
            if favorable >= target_distance:
                return {
                    "win": True,
                    "return_r": favorable / atr,
                    "bars": j - entry_index,
                }

            # 9 EMA close exit
            if row["close"] < row["ema9"]:
                move = (
                    float(row["close"])
                    - entry_price
                )

                return {
                    "win": move > 0,
                    "return_r": move / atr,
                    "bars": j - entry_index,
                }

        else:

            favorable = (
                entry_price
                - float(row["low"])
            )

            best_move = max(
                best_move,
                favorable,
            )

            if favorable >= target_distance:
                return {
                    "win": True,
                    "return_r": favorable / atr,
                    "bars": j - entry_index,
                }

            if row["close"] > row["ema9"]:
                move = (
                    entry_price
                    - float(row["close"])
                )

                return {
                    "win": move > 0,
                    "return_r": move / atr,
                    "bars": j - entry_index,
                }

    # Time exit
    final_idx = min(
        last_index - 1,
        len(df) - 1,
    )

    final_price = float(
        df.iloc[final_idx]["close"]
    )

    if direction == "CALL":
        move = final_price - entry_price
    else:
        move = entry_price - final_price

    return {
        "win": move > 0,
        "return_r": move / atr,
        "bars": final_idx - entry_index,
    }


# ============================================================
# 64 TRADE BACKTEST
# ============================================================

def backtest_symbol(symbol):
    df = get_historical_bars(symbol)

    if df is None or len(df) < 100:
        return None

    df = calculate_indicators(df)

    if df is None:
        return None

    trades = []

    # Walk backward because we want the MOST RECENT
    # 64 completed qualifying setups.
    for i in range(
        len(df) - 2,
        35,
        -1,
    ):

        direction = None

        if bullish_setup(df, i):
            direction = "CALL"

        elif bearish_setup(df, i):
            direction = "PUT"

        if direction is None:
            continue

        result = simulate_trade(
            df,
            i,
            direction,
        )

        if result is None:
            continue

        result["direction"] = direction
        result["timestamp"] = (
            df.index[i].isoformat()
        )

        result["entry"] = round(
            float(df.iloc[i]["close"]),
            4,
        )

        trades.append(result)

        if len(trades) >= BACKTEST_TRADES:
            break

    if len(trades) < MIN_BACKTEST_TRADES:
        return None

    wins = sum(
        1
        for trade in trades
        if trade["win"]
    )

    losses = len(trades) - wins

    win_rate = wins / len(trades)

    expectancy = float(
        np.mean(
            [
                trade["return_r"]
                for trade in trades
            ]
        )
    )

    avg_hold = float(
        np.mean(
            [
                trade["bars"]
                for trade in trades
            ]
        )
    )

    calls = [
        t
        for t in trades
        if t["direction"] == "CALL"
    ]

    puts = [
        t
        for t in trades
        if t["direction"] == "PUT"
    ]

    call_wins = sum(
        1
        for t in calls
        if t["win"]
    )

    put_wins = sum(
        1
        for t in puts
        if t["win"]
    )

    call_win_rate = (
        call_wins / len(calls)
        if calls
        else 0
    )

    put_win_rate = (
        put_wins / len(puts)
        if puts
        else 0
    )

    # Probability ranking score:
    # win rate is most important,
    # expectancy breaks ties.
    probability_score = (
        win_rate * 100
        + max(-10, min(10, expectancy * 10))
    )

    return {
        "symbol": symbol,

        "trades": len(trades),

        "wins": wins,
        "losses": losses,

        "win_rate": round(
            win_rate * 100,
            2,
        ),

        "call_win_rate": round(
            call_win_rate * 100,
            2,
        ),

        "put_win_rate": round(
            put_win_rate * 100,
            2,
        ),

        "expectancy_r": round(
            expectancy,
            3,
        ),

        "avg_hold_bars": round(
            avg_hold,
            1,
        ),

        "score": round(
            probability_score,
            2,
        ),

        "recent_trades": trades[:10],
    }


# ============================================================
# PRE-FILTER STOCKS
# ============================================================

def build_liquid_candidates():
    universe = get_stock_universe()

    if not universe:
        return []

    print(
        f"FOUND {len(universe)} ACTIVE STOCKS",
        flush=True,
    )

    candidates = []

    # We don't need to backtest thousands of illiquid stocks.
    # Check liquidity until we have enough candidates.
    for index, symbol in enumerate(universe):

        if len(candidates) >= MAX_UNIVERSE:
            break

        try:
            result = score_liquidity(symbol)

            if result:
                candidates.append(result)

        except Exception as exc:
            print(
                f"LIQUIDITY FILTER ERROR {symbol}: {exc}",
                flush=True,
            )

        # Slow requests slightly to avoid hammering API
        time.sleep(0.05)

    candidates.sort(
        key=lambda x: x["dollar_volume"],
        reverse=True,
    )

    print(
        f"{len(candidates)} LIQUID STOCKS QUALIFIED",
        flush=True,
    )

    return candidates


# ============================================================
# FULL PROBABILITY SCAN
# ============================================================

def probability_scan():
    scan_started = datetime.now(NY)

    print(
        "\n"
        "============================================\n"
        "STARTING PREMARKET 64-TRADE PROBABILITY SCAN\n"
        "============================================",
        flush=True,
    )

    set_status(
        "last_scan",
        scan_started.isoformat(),
    )

    set_status(
        "stocks_scanned",
        0,
    )

    set_status(
        "stocks_tested",
        0,
    )

    clear_errors()

    candidates = build_liquid_candidates()

    if not candidates:
        add_error("NO LIQUID CANDIDATES FOUND")
        return []

    results = []

    for number, candidate in enumerate(
        candidates,
        start=1,
    ):
        symbol = candidate["symbol"]

        print(
            f"[{number}/{len(candidates)}] "
            f"BACKTESTING {symbol}...",
            flush=True,
        )

        try:
            result = backtest_symbol(symbol)

            set_status(
                "stocks_scanned",
                number,
            )

            if result:
                set_status(
                    "stocks_tested",
                    BOT_STATUS["stocks_tested"] + 1,
                )

                result["price"] = candidate["price"]

                result["avg_volume"] = (
                    candidate["avg_volume"]
                )

                results.append(result)

                print(
                    f"{symbol} | "
                    f"{result['trades']} trades | "
                    f"{result['win_rate']}% wins | "
                    f"score {result['score']}",
                    flush=True,
                )

            else:
                print(
                    f"{symbol} | NOT ENOUGH VALID TRADES",
                    flush=True,
                )

        except Exception:
            add_error(
                f"BACKTEST ERROR {symbol}: "
                f"{traceback.format_exc()[-500:]}"
            )

        time.sleep(0.10)

    # Highest probability first
    results.sort(
        key=lambda x: (
            x["score"],
            x["win_rate"],
            x["expectancy_r"],
        ),
        reverse=True,
    )

    qualified = [
        result
        for result in results
        if (
            result["win_rate"]
            >= MIN_WIN_RATE * 100
            and result["expectancy_r"]
            >= MIN_EXPECTANCY
        )
    ]

    top = qualified[:TOP_STOCKS]

    set_status(
        "top_probability_stocks",
        top,
    )

    finished = datetime.now(NY)

    set_status(
        "last_scan",
        finished.isoformat(),
    )

    print(
        "\n"
        "============================================",
        flush=True,
    )

    print(
        "TOP PROBABILITY STOCKS",
        flush=True,
    )

    for rank, stock in enumerate(
        top,
        start=1,
    ):
        print