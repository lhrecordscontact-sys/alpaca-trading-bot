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
# ALPACA
# ============================================================

ALPACA_API_KEY = os.getenv(
    "ALPACA_API_KEY",
    ""
).strip()

ALPACA_SECRET_KEY = os.getenv(
    "ALPACA_SECRET_KEY",
    ""
).strip()

TRADING_URL = "https://paper-api.alpaca.markets"
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


# ============================================================
# SEPARATE CALL / PUT ROLLING SAMPLE
# ============================================================

ROLLING_TRADES = int(
    os.getenv(
        "ROLLING_TRADES",
        "64"
    )
)

REQUIRE_FULL_SAMPLE = True

# 58 / 64 = 90.625%
REQUIRED_WINS = 58


# ============================================================
# HISTORY / BATCH SETTINGS
# ============================================================

HISTORY_DAYS = int(
    os.getenv(
        "HISTORY_DAYS",
        "365"
    )
)

BATCH_SIZE = int(
    os.getenv(
        "BATCH_SIZE",
        "40"
    )
)

REQUEST_PAUSE = float(
    os.getenv(
        "REQUEST_PAUSE",
        "0.30"
    )
)


# ============================================================
# DAILY SCAN TIME
#
# Runs ONCE per weekday after market close.
# Default = 4:15 PM Eastern.
# ============================================================

SCAN_HOUR_ET = int(
    os.getenv(
        "SCAN_HOUR_ET",
        "16"
    )
)

SCAN_MINUTE_ET = int(
    os.getenv(
        "SCAN_MINUTE_ET",
        "15"
    )
)


# ============================================================
# STATE
# ============================================================

lock = threading.Lock()

STATE = {
    "status": "WAITING FOR DAILY SCAN",
    "universe_count": 0,
    "scanned_count": 0,
    "qualified_count": 0,
    "call_qualified_count": 0,
    "put_qualified_count": 0,
    "last_scan": None,
    "scan_started": None,
    "results": [],
    "qualified": [],
    "error": None,
}


# ============================================================
# API REQUEST
# ============================================================

def api_request(url, params=None, timeout=60):

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=timeout,
    )

    if not response.ok:
        raise RuntimeError(
            f"{response.status_code}: "
            f"{response.text[:500]}"
        )

    return response.json()


# ============================================================
# ENTIRE ACTIVE U.S. STOCK UNIVERSE
# ============================================================

def get_entire_market():

    logging.info(
        "Downloading full Alpaca stock universe..."
    )

    result = api_request(
        f"{TRADING_URL}/v2/assets",
        params={
            "status": "active",
            "asset_class": "us_equity",
        }
    )

    symbols = []

    for asset in result:

        symbol = str(
            asset.get(
                "symbol",
                ""
            )
        ).upper().strip()

        if not symbol:
            continue

        if not asset.get(
            "tradable",
            False
        ):
            continue

        if "/" in symbol:
            continue

        symbols.append(symbol)

    symbols = sorted(
        set(symbols)
    )

    logging.info(
        "Full market universe: %s symbols",
        len(symbols)
    )

    return symbols


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
# DOWNLOAD MULTIPLE STOCKS AT ONCE
# ============================================================

def get_batch_bars(symbols):

    if not symbols:
        return {}

    end = datetime.now(UTC)

    start = (
        end -
        timedelta(
            days=HISTORY_DAYS
        )
    )

    params = {
        "symbols": ",".join(symbols),
        "timeframe": TIMEFRAME,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "adjustment": "raw",
        "feed": DATA_FEED,
        "sort": "asc",
        "limit": 10000,
    }

    collected = {
        symbol: []
        for symbol in symbols
    }

    page_token = None

    while True:

        if page_token:
            params["page_token"] = page_token

        elif "page_token" in params:
            del params["page_token"]

        result = api_request(
            f"{DATA_URL}/v2/stocks/bars",
            params=params
        )

        bars_by_symbol = result.get(
            "bars",
            {}
        )

        for symbol, bars in bars_by_symbol.items():

            if symbol not in collected:
                collected[symbol] = []

            collected[symbol].extend(
                bars
            )

        page_token = result.get(
            "next_page_token"
        )

        if not page_token:
            break

        time.sleep(
            REQUEST_PAUSE
        )

    return collected


# ============================================================
# DATAFRAME
# ============================================================

def bars_to_dataframe(bars):

    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(
        bars
    )

    required = {
        "t",
        "o",
        "h",
        "l",
        "c",
        "v",
    }

    if not required.issubset(
        set(df.columns)
    ):
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

    df = df.tz_convert(
        NY
    )

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

    typical = (
        df["high"] +
        df["low"] +
        df["close"]
    ) / 3.0

    dates = pd.Series(
        df.index.date,
        index=df.index
    )

    pv = (
        typical *
        df["volume"]
    )

    cumulative_pv = (
        pv.groupby(
            dates
        ).cumsum()
    )

    cumulative_volume = (
        df["volume"]
        .groupby(
            dates
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
# TRADE RECORDER
# ============================================================

def make_trade_record(
    side,
    entry_price,
    exit_price,
    entry_time,
    exit_time,
    reason
):

    if side == "CALL":

        move = (
            exit_price -
            entry_price
        )

    else:

        move = (
            entry_price -
            exit_price
        )

    return {
        "side": side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_time": entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "move": move,
        "win": move > 0,
        "reason": reason,
    }


# ============================================================
# STRATEGY CALCULATION
#
# IMPORTANT:
#
# CALLS AND PUTS ARE NOW MEASURED SEPARATELY.
#
# CALL qualification:
# last 64 CALL trades only
#
# PUT qualification:
# last 64 PUT trades only
#
# Combined win rate DOES NOT control qualification.
# ============================================================

def calculate_stats(
    symbol,
    raw_df
):

    if raw_df.empty:
        return None

    df = add_indicators(
        raw_df
    )

    target = target_move(
        symbol
    )

    in_trade = False
    long_trade = False

    entry_price = None
    entry_timestamp = None
    target_price = None

    trades_today = 0

    pm_high = None
    pm_low = None

    previous_close = None
    previous_date = None

    current_signal = "WAITING"

    completed_trades = []

    for timestamp, row in df.iterrows():

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

        # ====================================================
        # NEW DAY
        # ====================================================

        if (
            previous_date is None
            or
            current_date != previous_date
        ):

            trades_today = 0

            pm_high = None
            pm_low = None

        # ====================================================
        # PREMARKET
        # ====================================================

        in_pm = (
            current_time >= PREMARKET_START
            and
            current_time < PREMARKET_END
        )

        if in_pm:

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

        # ====================================================
        # REGULAR SESSION
        # ====================================================

        in_rth = (
            current_time >= RTH_START
            and
            current_time < RTH_END
        )

        bull_trend = (
            ema5 > ema9
            and
            ema9 > ema30
        )

        bear_trend = (
            ema5 < ema9
            and
            ema9 < ema30
        )

        bull_vwap = (
            close > vwap
        )

        bear_vwap = (
            close < vwap
        )

        # ====================================================
        # MANAGE EXISTING TRADE FIRST
        # ====================================================

        if in_trade:

            if long_trade:

                if high >= target_price:

                    completed_trades.append(
                        make_trade_record(
                            side="CALL",
                            entry_price=entry_price,
                            exit_price=target_price,
                            entry_time=entry_timestamp,
                            exit_time=timestamp,
                            reason="TARGET"
                        )
                    )

                    in_trade = False
                    current_signal = "WAITING"

                elif close <= ema9:

                    completed_trades.append(
                        make_trade_record(
                            side="CALL",
                            entry_price=entry_price,
                            exit_price=close,
                            entry_time=entry_timestamp,
                            exit_time=timestamp,
                            reason="EMA9"
                        )
                    )

                    in_trade = False
                    current_signal = "WAITING"

            else:

                if low <= target_price:

                    completed_trades.append(
                        make_trade_record(
                            side="PUT",
                            entry_price=entry_price,
                            exit_price=target_price,
                            entry_time=entry_timestamp,
                            exit_time=timestamp,
                            reason="TARGET"
                        )
                    )

                    in_trade = False
                    current_signal = "WAITING"

                elif close >= ema9:

                    completed_trades.append(
                        make_trade_record(
                            side="PUT",
                            entry_price=entry_price,
                            exit_price=close,
                            entry_time=entry_timestamp,
                            exit_time=timestamp,
                            reason="EMA9"
                        )
                    )

                    in_trade = False
                    current_signal = "WAITING"

        # ====================================================
        # BREAKOUT CONDITIONS
        # ====================================================

        long_break = False
        short_break = False

        if (
            in_rth
            and
            pm_high is not None
            and
            previous_close is not None
        ):

            long_break = (
                close > pm_high
                and
                previous_close <= pm_high
                and
                bull_trend
                and
                bull_vwap
            )

        if (
            in_rth
            and
            pm_low is not None
            and
            previous_close is not None
        ):

            short_break = (
                close < pm_low
                and
                previous_close >= pm_low
                and
                bear_trend
                and
                bear_vwap
            )

        # ====================================================
        # NEW ENTRY
        # ====================================================

        can_trade = (
            not in_trade
            and
            trades_today < MAX_TRADES_PER_DAY
        )

        long_entry = (
            long_break
            and
            can_trade
        )

        short_entry = (
            short_break
            and
            can_trade
        )

        if long_entry:

            in_trade = True
            long_trade = True

            entry_price = close
            entry_timestamp = timestamp

            target_price = (
                close +
                target
            )

            trades_today += 1

            current_signal = (
                "CALL SIGNAL"
            )

        elif short_entry:

            in_trade = True
            long_trade = False

            entry_price = close
            entry_timestamp = timestamp

            target_price = (
                close -
                target
            )

            trades_today += 1

            current_signal = (
                "PUT SIGNAL"
            )

        previous_close = close
        previous_date = current_date

    # ========================================================
    # ALL HISTORICAL CALL / PUT TRADES
    # ========================================================

    historical_calls = [
        trade
        for trade in completed_trades
        if trade["side"] == "CALL"
    ]

    historical_puts = [
        trade
        for trade in completed_trades
        if trade["side"] == "PUT"
    ]

    historical_trade_count = len(
        completed_trades
    )

    historical_call_count = len(
        historical_calls
    )

    historical_put_count = len(
        historical_puts
    )

    # ========================================================
    # EXACT LAST 64 CALL TRADES
    # ========================================================

    rolling_calls = historical_calls[
        -ROLLING_TRADES:
    ]

    call_trades = len(
        rolling_calls
    )

    call_full_sample = (
        call_trades ==
        ROLLING_TRADES
    )

    call_wins = sum(
        1
        for trade in rolling_calls
        if trade["win"]
    )

    call_losses = (
        call_trades -
        call_wins
    )

    call_rate = (
        call_wins *
        100.0 /
        call_trades
        if call_trades
        else 0.0
    )

    # ========================================================
    # EXACT LAST 64 PUT TRADES
    # ========================================================

    rolling_puts = historical_puts[
        -ROLLING_TRADES:
    ]

    put_trades = len(
        rolling_puts
    )

    put_full_sample = (
        put_trades ==
        ROLLING_TRADES
    )

    put_wins = sum(
        1
        for trade in rolling_puts
        if trade["win"]
    )

    put_losses = (
        put_trades -
        put_wins
    )

    put_rate = (
        put_wins *
        100.0 /
        put_trades
        if put_trades
        else 0.0
    )

    # ========================================================
    # SEPARATE 90% QUALIFICATION
    # ========================================================

    call_qualified = (
        call_full_sample
        and
        call_wins >= REQUIRED_WINS
        and
        call_rate >= MIN_WIN_RATE
    )

    put_qualified = (
        put_full_sample
        and
        put_wins >= REQUIRED_WINS
        and
        put_rate >= MIN_WIN_RATE
    )

    qualified = (
        call_qualified
        or
        put_qualified
    )

    if (
        call_qualified
        and
        put_qualified
    ):

        qualification = (
            "CALL + PUT"
        )

    elif call_qualified:

        qualification = (
            "CALL"
        )

    elif put_qualified:

        qualification = (
            "PUT"
        )

    else:

        qualification = (
            "SKIP"
        )

    # ========================================================
    # DISPLAY / SORT RATE
    # ========================================================

    best_rate = max(
        call_rate
        if call_full_sample
        else 0.0,
        put_rate
        if put_full_sample
        else 0.0
    )

    # ========================================================
    # NET MOVE
    # ========================================================

    call_net_move = sum(
        trade["move"]
        for trade in rolling_calls
    )

    put_net_move = sum(
        trade["move"]
        for trade in rolling_puts
    )

    # ========================================================
    # CURRENT SIGNAL
    # ========================================================

    if in_trade:

        current_signal = (
            "CALL ACTIVE"
            if long_trade
            else "PUT ACTIVE"
        )

    return {

        "symbol": symbol,

        # ------------------------------------
        # BEST QUALIFYING RATE
        # ------------------------------------

        "overall": round(
            best_rate,
            1
        ),

        "rolling_win_rate": round(
            best_rate,
            1
        ),

        # ------------------------------------
        # CALL
        # ------------------------------------

        "call_rate": round(
            call_rate,
            1
        ),

        "call_wins":
            call_wins,

        "call_losses":
            call_losses,

        "call_trades":
            call_trades,

        "call_full_sample":
            call_full_sample,

        "call_qualified":
            call_qualified,

        "historical_call_count":
            historical_call_count,

        "call_net_move": round(
            call_net_move,
            2
        ),

        # ------------------------------------
        # PUT
        # ------------------------------------

        "put_rate": round(
            put_rate,
            1
        ),

        "put_wins":
            put_wins,

        "put_losses":
            put_losses,

        "put_trades":
            put_trades,

        "put_full_sample":
            put_full_sample,

        "put_qualified":
            put_qualified,

        "historical_put_count":
            historical_put_count,

        "put_net_move": round(
            put_net_move,
            2
        ),

        # ------------------------------------
        # TOTAL HISTORY
        # ------------------------------------

        "historical_trade_count":
            historical_trade_count,

        "sample_required":
            ROLLING_TRADES,

        "required_wins":
            REQUIRED_WINS,

        # ------------------------------------
        # RESULT
        # ------------------------------------

        "qualified":
            qualified,

        "qualification":
            qualification,

        "signal":
            current_signal,
    }


# ============================================================
# PROCESS ONE MARKET BATCH
# ============================================================

def process_batch(
    symbols
):

    bars_by_symbol = (
        get_batch_bars(
            symbols
        )
    )

    results = []

    for symbol in symbols:

        try:

            bars = (
                bars_by_symbol.get(
                    symbol,
                    []
                )
            )

            df = (
                bars_to_dataframe(
                    bars
                )
            )

            stats = (
                calculate_stats(
                    symbol,
                    df
                )
            )

            with lock:

                STATE[
                    "scanned_count"
                ] += 1

            if not stats:
                continue

            results.append(
                stats
            )

            logging.info(
                "%s | CALL %s/64 %.1f%% %s | "
                "PUT %s/64 %.1f%% %s | %s",
                symbol,
                stats["call_trades"],
                stats["call_rate"],
                "QUALIFIED"
                if stats["call_qualified"]
                else "SKIP",
                stats["put_trades"],
                stats["put_rate"],
                "QUALIFIED"
                if stats["put_qualified"]
                else "SKIP",
                stats["qualification"],
            )

        except Exception as exc:

            logging.warning(
                "%s failed: %s",
                symbol,
                exc
            )

    return results


# ============================================================
# FULL MARKET SCAN
# ============================================================

def run_full_market_scan():

    with lock:

        if STATE[
            "status"
        ] == "SCANNING":

            return

        STATE[
            "status"
        ] = "SCANNING"

        STATE[
            "scanned_count"
        ] = 0

        STATE[
            "qualified_count"
        ] = 0

        STATE[
            "call_qualified_count"
        ] = 0

        STATE[
            "put_qualified_count"
        ] = 0

        STATE[
            "error"
        ] = None

        STATE[
            "scan_started"
        ] = datetime.now(
            NY
        ).isoformat()

    try:

        universe = (
            get_entire_market()
        )

        with lock:

            STATE[
                "universe_count"
            ] = len(
                universe
            )

        all_results = []

        batches = [

            universe[
                i:
                i + BATCH_SIZE
            ]

            for i in range(
                0,
                len(universe),
                BATCH_SIZE
            )
        ]

        logging.info(
            "Scanning %s stocks in %s batches",
            len(universe),
            len(batches)
        )

        for number, batch in enumerate(
            batches,
            start=1
        ):

            logging.info(
                "Batch %s/%s",
                number,
                len(batches)
            )

            batch_results = (
                process_batch(
                    batch
                )
            )

            all_results.extend(
                batch_results
            )

            all_results.sort(
                key=lambda stock: (
                    stock["qualified"],
                    stock["rolling_win_rate"],
                    max(
                        stock["call_wins"],
                        stock["put_wins"]
                    )
                ),
                reverse=True
            )

            qualified = [
                stock
                for stock in all_results
                if stock["qualified"]
            ]

            call_qualified = [
                stock
                for stock in all_results
                if stock["call_qualified"]
            ]

            put_qualified = [
                stock
                for stock in all_results
                if stock["put_qualified"]
            ]

            with lock:

                STATE[
                    "results"
                ] = all_results.copy()

                STATE[
                    "qualified"
                ] = qualified.copy()

                STATE[
                    "qualified_count"
                ] = len(
                    qualified
                )

                STATE[
                    "call_qualified_count"
                ] = len(
                    call_qualified
                )

                STATE[
                    "put_qualified_count"
                ] = len(
                    put_qualified
                )

            time.sleep(
                REQUEST_PAUSE
            )

        now = datetime.now(
            NY
        )

        with lock:

            STATE[
                "status"
            ] = "READY"

            STATE[
                "last_scan"
            ] = now.isoformat()

        logging.info(
            "FULL MARKET DAILY SCAN COMPLETE"
        )

        logging.info(
            "Scanned: %s",
            STATE["scanned_count"]
        )

        logging.info(
            "90%%+ CALL qualified: %s",
            STATE["call_qualified_count"]
        )

        logging.info(
            "90%%+ PUT qualified: %s",
            STATE["put_qualified_count"]
        )

        logging.info(
            "Total qualifying symbols: %s",
            STATE["qualified_count"]
        )

    except Exception as exc:

        logging.exception(
            "Full-market scan failed"
        )

        with lock:

            STATE[
                "status"
            ] = "ERROR"

            STATE[
                "error"
            ] = str(
                exc
            )


# ============================================================
# NEXT WEEKDAY SCAN TIME
# ============================================================

def get_next_scan_time():

    now = datetime.now(
        NY
    )

    target = now.replace(
        hour=SCAN_HOUR_ET,
        minute=SCAN_MINUTE_ET,
        second=0,
        microsecond=0
    )

    # If today's time has already passed,
    # move to tomorrow.
    if target <= now:

        target += timedelta(
            days=1
        )

    # Skip Saturday / Sunday.
    while target.weekday() >= 5:

        target += timedelta(
            days=1
        )

    return target


# ============================================================
# AUTOMATIC DAILY SCANNER LOOP
#
# ONE scan each weekday at 4:15 PM ET.
#
# It does NOT immediately scan every time Render restarts.
# It does NOT scan every hour.
# ============================================================

def scanner_loop():

    while True:

        try:

            next_scan = (
                get_next_scan_time()
            )

            with lock:

                if STATE[
                    "status"
                ] != "SCANNING":

                    STATE[
                        "status"
                    ] = "WAITING FOR DAILY SCAN"

            logging.info(
                "Next automatic scan: %s",
                next_scan.isoformat()
            )

            while True:

                now = datetime.now(
                    NY
                )

                seconds_remaining = (
                    next_scan -
                    now
                ).total_seconds()

                if seconds_remaining <= 0:
                    break

                time.sleep(
                    min(
                        60,
                        max(
                            1,
                            seconds_remaining
                        )
                    )
                )

            logging.info(
                "Starting scheduled after-close scan..."
            )

            run_full_market_scan()

            # Prevent duplicate run on same minute.
            time.sleep(
                90
            )

        except Exception:

            logging.exception(
                "Scanner loop error"
            )

            time.sleep(
                60
            )


# ============================================================
# IPHONE WEBPAGE
# ============================================================

HTML = """
<!doctype html>

<html>

<head>

<meta
name="viewport"
content="width=device-width, initial-scale=1">

<meta
http-equiv="refresh"
content="60">

<title>
90% CALL / PUT Scanner
</title>

<style>

body {
    margin: 0;
    padding: 14px;
    background: #050505;
    color: white;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        Arial,
        sans-serif;
}

h1 {
    font-size: 23px;
    margin-bottom: 4px;
}

.subtitle {
    color: #999;
    font-size: 13px;
    margin-bottom: 15px;
}

.summary {
    background: #171717;
    padding: 13px;
    border-radius: 13px;
    margin-bottom: 16px;
    line-height: 1.7;
}

.stock {
    background: #171717;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 10px;
}

.top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.symbol {
    font-size: 25px;
    font-weight: 900;
}

.direction {
    font-size: 17px;
    font-weight: 900;
}

.call {
    color: #39d353;
}

.put {
    color: #ff4d4f;
}

.both {
    color: #ffd43b;
}

.rates {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 12px;
}

.rate {
    background: #0d0d0d;
    padding: 12px;
    border-radius: 10px;
    text-align: center;
}

.label {
    color: #999;
    font-size: 10px;
}

.number {
    font-size: 27px;
    font-weight: 900;
    margin-top: 3px;
}

.small {
    color: #999;
    font-size: 11px;
}

.good {
    color: #39d353;
}

.bad {
    color: #777;
}

.record {
    margin-top: 10px;
    font-size: 12px;
    color: #aaa;
    line-height: 1.6;
}

.status {
    margin-top: 9px;
    font-size: 12px;
    font-weight: 700;
}

.empty {
    padding: 25px;
    border-radius: 14px;
    background: #171717;
    color: #999;
    text-align: center;
}

</style>

</head>

<body>

<h1>
90% CALL / PUT MARKET SCANNER
</h1>

<div class="subtitle">
Separate last 64 CALL trades and last 64 PUT trades
</div>

<div class="summary">

<strong>Status:</strong>
{{ status }}

<br>

<strong>Market symbols:</strong>
{{ universe_count }}

<br>

<strong>Scanned:</strong>
{{ scanned_count }}

<br>

<strong>Total qualifying stocks:</strong>
{{ qualified_count }}

<br>

<span class="call">
<strong>90%+ CALL stocks:</strong>
{{ call_qualified_count }}
</span>

<br>

<span class="put">
<strong>90%+ PUT stocks:</strong>
{{ put_qualified_count }}
</span>

<br>

<strong>Required CALL sample:</strong>
64 CALL trades

<br>

<strong>Required PUT sample:</strong>
64 PUT trades

<br>

<strong>Required wins:</strong>
58 / 64

<br>

<strong>Automatic scan:</strong>
4:15 PM ET weekdays

<br>

<strong>Last completed scan:</strong>
{{ last_scan or "Not completed yet" }}

{% if error %}

<br><br>

<span class="put">
{{ error }}
</span>

{% endif %}

</div>


{% if qualified %}

{% for stock in qualified %}

<div class="stock">

<div class="top">

<div class="symbol">
{{ stock.symbol }}
</div>

{% if stock.qualification == "CALL" %}

<div class="direction call">
90% CALL
</div>

{% elif stock.qualification == "PUT" %}

<div class="direction put">
90% PUT
</div>

{% else %}

<div class="direction both">
90% CALL + PUT
</div>

{% endif %}

</div>


<div class="rates">

<div class="rate">

<div class="label">
LAST 64 CALL TRADES
</div>

<div
class="number
{% if stock.call_qualified %}
good
{% else %}
bad
{% endif %}"
>

{{ stock.call_rate }}%

</div>

<div class="small">
{{ stock.call_wins }}W /
{{ stock.call_losses }}L /
{{ stock.call_trades }}/64
</div>

</div>


<div class="rate">

<div class="label">
LAST 64 PUT TRADES
</div>

<div
class="number
{% if stock.put_qualified %}
good
{% else %}
bad
{% endif %}"
>

{{ stock.put_rate }}%

</div>

<div class="small">
{{ stock.put_wins }}W /
{{ stock.put_losses }}L /
{{ stock.put_trades }}/64
</div>

</div>

</div>


<div class="record">

Historical CALL trades:
{{ stock.historical_call_count }}

<br>

Historical PUT trades:
{{ stock.historical_put_count }}

<br>

CALL net move:
{{ stock.call_net_move }}

<br>

PUT net move:
{{ stock.put_net_move }}

</div>


<div class="status">

CURRENT STATUS:
{{ stock.signal }}

</div>

</div>

{% endfor %}

{% else %}

<div class="empty">

No CALL or PUT stocks currently meet
the separate 90% / 64-trade requirement.

<br><br>

A CALL must have 64 CALL trades.

<br>

A PUT must have 64 PUT trades.

</div>

{% endif %}


</body>

</html>
"""


# ============================================================
# WEBSITE ROUTES
# ============================================================

@app.route("/")
def home():

    return jsonify({

        "scanner":
            "Separate 90% CALL / PUT Full Market Scanner",

        "timeframe":
            TIMEFRAME,

        "minimum_win_rate":
            MIN_WIN_RATE,

        "required_completed_trades_per_side":
            ROLLING_TRADES,

        "required_wins":
            REQUIRED_WINS,

        "automatic_scan_time":
            f"{SCAN_HOUR_ET:02d}:{SCAN_MINUTE_ET:02d} ET",

        "watchlist":
            "/watchlist",

        "api":
            "/api/watchlist",

        "all_results":
            "/api/all",

        "manual_rescan":
            "/rescan",
    })


@app.route("/watchlist")
def watchlist():

    with lock:

        snapshot = {

            "status":
                STATE["status"],

            "universe_count":
                STATE["universe_count"],

            "scanned_count":
                STATE["scanned_count"],

            "qualified_count":
                STATE["qualified_count"],

            "call_qualified_count":
                STATE["call_qualified_count"],

            "put_qualified_count":
                STATE["put_qualified_count"],

            "qualified":
                STATE["qualified"].copy(),

            "last_scan":
                STATE["last_scan"],

            "error":
                STATE["error"],
        }

    return render_template_string(
        HTML,
        **snapshot
    )


@app.route("/api/watchlist")
def api_watchlist():

    with lock:

        return jsonify({

            "status":
                STATE["status"],

            "universe_count":
                STATE["universe_count"],

            "scanned_count":
                STATE["scanned_count"],

            "qualified_count":
                STATE["qualified_count"],

            "call_qualified_count":
                STATE["call_qualified_count"],

            "put_qualified_count":
                STATE["put_qualified_count"],

            "minimum_win_rate":
                MIN_WIN_RATE,

            "required_sample_per_side":
                ROLLING_TRADES,

            "required_wins":
                REQUIRED_WINS,

            "timeframe":
                TIMEFRAME,

            "automatic_scan_time_et":
                f"{SCAN_HOUR_ET:02d}:{SCAN_MINUTE_ET:02d}",

            "last_scan":
                STATE["last_scan"],

            "qualified":
                STATE["qualified"],

            "error":
                STATE["error"],
        })


@app.route("/api/all")
def api_all():

    with lock:

        return jsonify({

            "status":
                STATE["status"],

            "results":
                STATE["results"],

            "count":
                len(
                    STATE["results"]
                ),
        })


# ============================================================
# OPTIONAL MANUAL RESCAN
#
# Automatic scanning remains once daily.
# This route lets YOU manually force a scan if needed.
# ============================================================

@app.route("/rescan")
def rescan():

    with lock:

        already_scanning = (
            STATE[
                "status"
            ] == "SCANNING"
        )

    if already_scanning:

        return jsonify({
            "status":
                "already scanning"
        })

    thread = threading.Thread(
        target=run_full_market_scan,
        daemon=True
    )

    thread.start()

    return jsonify({
        "status":
            "manual full-market scan started"
    })


@app.route("/health")
def health():

    return jsonify({

        "status":
            "healthy",

        "alpaca_key_loaded":
            bool(
                ALPACA_API_KEY
            ),

        "alpaca_secret_loaded":
            bool(
                ALPACA_SECRET_KEY
            ),

        "scanner_status":
            STATE["status"],

        "rolling_sample_per_side":
            ROLLING_TRADES,

        "required_wins":
            REQUIRED_WINS,

        "minimum_win_rate":
            MIN_WIN_RATE,

        "daily_scan_time_et":
            f"{SCAN_HOUR_ET:02d}:{SCAN_MINUTE_ET:02d}",
    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    scanner_thread = (
        threading.Thread(
            target=scanner_loop,
            daemon=True
        )
    )

    scanner_thread.start()

    port = int(
        os.getenv(
            "PORT",
            "10000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )