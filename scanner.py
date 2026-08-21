import os
import time
import threading
import logging
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from flask import Flask, jsonify, render_template_string


# ============================================================
# APP / LOGGING
# ============================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


# ============================================================
# ALPACA SETTINGS
# ============================================================

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "").strip()

DATA_URL = "https://data.alpaca.markets"

DATA_FEED = os.getenv(
    "DATA_FEED",
    "iex"
).strip().lower()

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}


# ============================================================
# YOUR WATCHLIST
#
# You can change this later from Render with:
#
# WATCHLIST=TSLA,NVDA,AAPL,MU,META,...
# ============================================================

DEFAULT_WATCHLIST = [
    "TSLA",
    "NVDA",
    "AAPL",
    "SPCX",
    "MU",
    "META",
    "AMZN",
    "MSFT",
    "AMD",
    "PLTR",
    "SNDK",
    "MRNA",
    "GOOGL",
    "MSTR",
    "SPY",
    "IWM",
    "QQQ",
]

WATCHLIST = [
    symbol.strip().upper()
    for symbol in os.getenv(
        "WATCHLIST",
        ",".join(DEFAULT_WATCHLIST)
    ).split(",")
    if symbol.strip()
]


# ============================================================
# STRATEGY SETTINGS
# ============================================================

TIMEFRAME = "4Min"

EMA_FAST = 5
EMA_SLOW = 9
EMA_TREND = 30

PREMARKET_START = dt_time(4, 0)
PREMARKET_END = dt_time(9, 30)

RTH_START = dt_time(9, 30)
RTH_END = dt_time(16, 0)

SPY_TARGET = 1.00
IWM_TARGET = 0.50
DEFAULT_TARGET = 1.00

MAX_TRADES_PER_DAY = 2

MIN_WIN_RATE = float(
    os.getenv(
        "MIN_WIN_RATE",
        "90"
    )
)

# Keep this at 1 to behave like your Pine stats.
# Later you can raise it to 10, 20, etc. if wanted.
MIN_SIDE_TRADES = int(
    os.getenv(
        "MIN_SIDE_TRADES",
        "1"
    )
)

# Number of calendar days downloaded from Alpaca.
HISTORY_DAYS = int(
    os.getenv(
        "HISTORY_DAYS",
        "120"
    )
)

# Rescan frequency.
SCAN_EVERY_MINUTES = int(
    os.getenv(
        "SCAN_EVERY_MINUTES",
        "15"
    )
)


# ============================================================
# SCANNER STATE
# ============================================================

lock = threading.Lock()

STATE = {
    "status": "STARTING",
    "last_scan": None,
    "results": [],
    "qualified": [],
    "error": None,
}


# ============================================================
# TARGET MOVE
# ============================================================

def target_move(symbol):

    if symbol == "IWM":
        return IWM_TARGET

    if symbol == "SPY":
        return SPY_TARGET

    return DEFAULT_TARGET


# ============================================================
# ALPACA REQUEST
# ============================================================

def alpaca_request(path, params=None):

    response = requests.get(
        f"{DATA_URL}{path}",
        headers=HEADERS,
        params=params,
        timeout=45,
    )

    if not response.ok:
        raise RuntimeError(
            f"Alpaca error "
            f"{response.status_code}: "
            f"{response.text[:500]}"
        )

    return response.json()


# ============================================================
# DOWNLOAD 4-MINUTE BARS
# ============================================================

def get_bars(symbol):

    end = datetime.now(UTC)

    start = (
        end -
        timedelta(days=HISTORY_DAYS)
    )

    params = {
        "symbols": symbol,
        "timeframe": TIMEFRAME,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "adjustment": "raw",
        "feed": DATA_FEED,
        "sort": "asc",
        "limit": 10000,
    }

    all_bars = []

    page_token = None

    while True:

        if page_token:
            params["page_token"] = page_token

        result = alpaca_request(
            "/v2/stocks/bars",
            params
        )

        bars_by_symbol = result.get(
            "bars",
            {}
        )

        bars = bars_by_symbol.get(
            symbol,
            []
        )

        all_bars.extend(bars)

        page_token = result.get(
            "next_page_token"
        )

        if not page_token:
            break

    return all_bars


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

def bars_to_dataframe(bars):

    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)

    needed = {
        "t",
        "o",
        "h",
        "l",
        "c",
        "v",
    }

    if not needed.issubset(df.columns):
        return pd.DataFrame()

    df = df.rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.set_index(
        "timestamp"
    )

    df = df.tz_convert(NY)

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    return df.sort_index()


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    if df.empty:
        return df

    df = df.copy()

    df["ema5"] = (
        df["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False
        )
        .mean()
    )

    df["ema9"] = (
        df["close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False
        )
        .mean()
    )

    df["ema30"] = (
        df["close"]
        .ewm(
            span=EMA_TREND,
            adjust=False
        )
        .mean()
    )

    typical_price = (
        df["high"] +
        df["low"] +
        df["close"]
    ) / 3.0

    trading_date = pd.Series(
        df.index.date,
        index=df.index
    )

    pv = (
        typical_price *
        df["volume"]
    )

    cumulative_pv = (
        pv.groupby(
            trading_date
        ).cumsum()
    )

    cumulative_volume = (
        df["volume"]
        .groupby(
            trading_date
        )
        .cumsum()
        .replace(
            0,
            float("nan")
        )
    )

    df["vwap"] = (
        cumulative_pv /
        cumulative_volume
    )

    return df


# ============================================================
# EXACT PINE-STYLE BACKTEST
# ============================================================

def calculate_stats(symbol, raw_df):

    if raw_df.empty:
        return None

    df = add_indicators(
        raw_df
    )

    target = target_move(
        symbol
    )

    # ----------------------------------------
    # Pine-style persistent variables
    # ----------------------------------------

    in_trade = False
    long_trade = False

    entry_price = None
    target_price = None

    trades_today = 0

    wins = 0
    losses = 0
    total_trades = 0

    call_wins = 0
    call_losses = 0
    call_trades = 0

    put_wins = 0
    put_losses = 0
    put_trades = 0

    total_move = 0.0

    pm_high = None
    pm_low = None

    previous_close = None
    previous_date = None

    last_signal = "WAITING"

    latest_timestamp = None

    # ----------------------------------------
    # Process every 4-minute candle
    # ----------------------------------------

    for timestamp, row in df.iterrows():

        latest_timestamp = timestamp

        current_date = timestamp.date()
        current_time = timestamp.time()

        close = float(
            row["close"]
        )

        high = float(
            row["high"]
        )

        low = float(
            row["low"]
        )

        ema5 = float(
            row["ema5"]
        )

        ema9 = float(
            row["ema9"]
        )

        ema30 = float(
            row["ema30"]
        )

        vwap = float(
            row["vwap"]
        )

        # ====================================
        # NEW DAY
        # ====================================

        if (
            previous_date is None
            or current_date != previous_date
        ):

            trades_today = 0

            pm_high = None
            pm_low = None

        # ====================================
        # PREMARKET HIGH / LOW
        # ====================================

        in_premarket = (
            current_time >= PREMARKET_START
            and current_time < PREMARKET_END
        )

        if in_premarket:

            if pm_high is None:
                pm_high = high
            else:
                pm_high = max(
                    pm_high,
                    high
                )

            if pm_low is None:
                pm_low = low
            else:
                pm_low = min(
                    pm_low,
                    low
                )

        # ====================================
        # REGULAR SESSION
        # ====================================

        in_rth = (
            current_time >= RTH_START
            and current_time < RTH_END
        )

        # ====================================
        # TREND FILTERS
        # ====================================

        bull_trend = (
            ema5 > ema9
            and ema9 > ema30
        )

        bear_trend = (
            ema5 < ema9
            and ema9 < ema30
        )

        bull_vwap = (
            close > vwap
        )

        bear_vwap = (
            close < vwap
        )

        # ====================================
        # BREAK CONDITIONS
        # ====================================

        long_break = False
        short_break = False

        if (
            in_rth
            and pm_high is not None
            and previous_close is not None
        ):

            long_break = (
                close > pm_high
                and previous_close <= pm_high
                and bull_trend
                and bull_vwap
            )

        if (
            in_rth
            and pm_low is not None
            and previous_close is not None
        ):

            short_break = (
                close < pm_low
                and previous_close >= pm_low
                and bear_trend
                and bear_vwap
            )

        # ====================================
        # ENTRY
        #
        # Pine enters first, then checks TP /
        # EMA9 exit on the same candle.
        # ====================================

        can_trade = (
            not in_trade
            and trades_today < MAX_TRADES_PER_DAY
        )

        long_entry = (
            long_break
            and can_trade
        )

        short_entry = (
            short_break
            and can_trade
        )

        if long_entry:

            in_trade = True
            long_trade = True

            entry_price = close

            target_price = (
                close +
                target
            )

            trades_today += 1

            last_signal = "CALL SIGNAL"

        elif short_entry:

            in_trade = True
            long_trade = False

            entry_price = close

            target_price = (
                close -
                target
            )

            trades_today += 1

            last_signal = "PUT SIGNAL"

        # ====================================
        # EXIT CONDITIONS
        # ====================================

        if in_trade:

            # --------------------------------
            # CALL
            # --------------------------------

            if long_trade:

                long_tp = (
                    high >= target_price
                )

                long_stop = (
                    close <= ema9
                )

                if long_tp:

                    profit_move = (
                        target_price -
                        entry_price
                    )

                    wins += 1
                    total_trades += 1

                    call_wins += 1
                    call_trades += 1

                    total_move += (
                        profit_move
                    )

                    in_trade = False

                elif long_stop:

                    exit_move = (
                        close -
                        entry_price
                    )

                    if exit_move > 0:

                        wins += 1
                        call_wins += 1

                    else:

                        losses += 1
                        call_losses += 1

                    total_trades += 1
                    call_trades += 1

                    total_move += (
                        exit_move
                    )

                    in_trade = False

            # --------------------------------
            # PUT
            # --------------------------------

            else:

                short_tp = (
                    low <= target_price
                )

                short_stop = (
                    close >= ema9
                )

                if short_tp:

                    profit_move = (
                        entry_price -
                        target_price
                    )

                    wins += 1
                    total_trades += 1

                    put_wins += 1
                    put_trades += 1

                    total_move += (
                        profit_move
                    )

                    in_trade = False

                elif short_stop:

                    exit_move = (
                        entry_price -
                        close
                    )

                    if exit_move > 0:

                        wins += 1
                        put_wins += 1

                    else:

                        losses += 1
                        put_losses += 1

                    total_trades += 1
                    put_trades += 1

                    total_move += (
                        exit_move
                    )

                    in_trade = False

        previous_close = close
        previous_date = current_date

    # ========================================
    # WIN RATES
    # ========================================

    overall_rate = (
        wins * 100.0 /
        total_trades
        if total_trades > 0
        else 0.0
    )

    call_rate = (
        call_wins * 100.0 /
        call_trades
        if call_trades > 0
        else 0.0
    )

    put_rate = (
        put_wins * 100.0 /
        put_trades
        if put_trades > 0
        else 0.0
    )

    # ========================================
    # 90% SIDE QUALIFICATION
    #
    # THIS IS THE IMPORTANT CHANGE:
    #
    # CALL must itself be >=90%
    # PUT must itself be >=90%
    # ========================================

    call_qualified = (
        call_trades >= MIN_SIDE_TRADES
        and call_rate >= MIN_WIN_RATE
    )

    put_qualified = (
        put_trades >= MIN_SIDE_TRADES
        and put_rate >= MIN_WIN_RATE
    )

    if (
        call_qualified
        and put_qualified
    ):

        qualification = "CALL + PUT"

    elif call_qualified:

        qualification = "CALL"

    elif put_qualified:

        qualification = "PUT"

    else:

        qualification = "SKIP"

    if in_trade:

        last_signal = (
            "CALL ACTIVE"
            if long_trade
            else "PUT ACTIVE"
        )

    return {
        "symbol": symbol,

        "overall": round(
            overall_rate,
            1
        ),

        "call_rate": round(
            call_rate,
            1
        ),

        "put_rate": round(
            put_rate,
            1
        ),

        "call_wins": call_wins,
        "call_losses": call_losses,
        "call_trades": call_trades,

        "put_wins": put_wins,
        "put_losses": put_losses,
        "put_trades": put_trades,

        "wins": wins,
        "losses": losses,
        "total_trades": total_trades,

        "net_move": round(
            total_move,
            2
        ),

        "call_qualified": call_qualified,
        "put_qualified": put_qualified,

        "qualification": qualification,

        "signal": last_signal,

        "latest_bar": (
            latest_timestamp.isoformat()
            if latest_timestamp
            else None
        ),
    }


# ============================================================
# SCAN ONE SYMBOL
# ============================================================

def scan_symbol(symbol):

    logging.info(
        "Scanning %s",
        symbol
    )

    bars = get_bars(
        symbol
    )

    df = bars_to_dataframe(
        bars
    )

    if df.empty:

        logging.warning(
            "%s has no usable bars",
            symbol
        )

        return None

    result = calculate_stats(
        symbol,
        df
    )

    if result:

        logging.info(
            "%s | CALL %.1f%% | PUT %.1f%% | %s",
            symbol,
            result["call_rate"],
            result["put_rate"],
            result["qualification"],
        )

    return result


# ============================================================
# RUN FULL WATCHLIST SCAN
# ============================================================

def run_scan():

    with lock:

        STATE["status"] = "SCANNING"
        STATE["error"] = None

    logging.info(
        "===================================="
    )

    logging.info(
        "Starting 90%% CALL/PUT scanner"
    )

    logging.info(
        "Watchlist: %s",
        WATCH