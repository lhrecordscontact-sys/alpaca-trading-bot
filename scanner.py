import os
import time
import threading
import logging
from collections import deque
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

# Same target used for all symbols except IWM.
DEFAULT_TARGET = 1.00

MAX_TRADES_PER_DAY = 2

MIN_WIN_RATE = float(
    os.getenv(
        "MIN_WIN_RATE",
        "90"
    )
)

# ============================================================
# EXACT ROLLING SAMPLE
# ============================================================

ROLLING_TRADES = int(
    os.getenv(
        "ROLLING_TRADES",
        "64"
    )
)

# A stock CANNOT qualify unless all 64 completed trades exist.
REQUIRE_FULL_SAMPLE = True


# ============================================================
# HISTORY
# ============================================================

HISTORY_DAYS = int(
    os.getenv(
        "HISTORY_DAYS",
        "180"
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

SCAN_EVERY_MINUTES = int(
    os.getenv(
        "SCAN_EVERY_MINUTES",
        "60"
    )
)


# ============================================================
# STATE
# ============================================================

lock = threading.Lock()

STATE = {
    "status": "STARTING",
    "universe_count": 0,
    "scanned_count": 0,
    "qualified_count": 0,
    "last_scan": None,
    "scan_started": None,
    "results": [],
    "qualified": [],
    "error": None,
}


# ============================================================
# API REQUEST
# ============================================================

def api_request(
    url,
    params=None,
    timeout=60
):

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
# GET ENTIRE ACTIVE U.S. STOCK UNIVERSE
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

        symbols.append(
            symbol
        )

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

            params[
                "page_token"
            ] = page_token

        elif "page_token" in params:

            del params[
                "page_token"
            ]

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

                collected[
                    symbol
                ] = []

            collected[
                symbol
            ].extend(
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

    df[
        "timestamp"
    ] = pd.to_datetime(
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

        df[
            column
        ] = pd.to_numeric(
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
# EXACT ROLLING-64 STRATEGY CALCULATION
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

    # --------------------------------------------------------
    # IMPORTANT:
    # Stores completed trades chronologically.
    # Qualification will ONLY use the final 64.
    # --------------------------------------------------------

    completed_trades = []

    for timestamp, row in df.iterrows():

        current_date = (
            timestamp.date()
        )

        current_time = (
            timestamp.time()
        )

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
        # FIRST HANDLE EXISTING TRADE
        #
        # This is intentionally BEFORE new entries.
        # It prevents a new entry from "winning" from a high/low
        # that already occurred earlier inside the entry candle.
        # ====================================================

        if in_trade:

            if long_trade:

                # CALL TARGET
                if high >= target_price:

                    trade = make_trade_record(
                        side="CALL",
                        entry_price=entry_price,
                        exit_price=target_price,
                        entry_time=entry_timestamp,
                        exit_time=timestamp,
                        reason="TARGET"
                    )

                    completed_trades.append(
                        trade
                    )

                    in_trade = False
                    current_signal = "WAITING"

                # CALL EMA9 EXIT
                elif close <= ema9:

                    trade = make_trade_record(
                        side="CALL",
                        entry_price=entry_price,
                        exit_price=close,
                        entry_time=entry_timestamp,
                        exit_time=timestamp,
                        reason="EMA9"
                    )

                    completed_trades.append(
                        trade
                    )

                    in_trade = False
                    current_signal = "WAITING"

            else:

                # PUT TARGET
                if low <= target_price:

                    trade = make_trade_record(
                        side="PUT",
                        entry_price=entry_price,
                        exit_price=target_price,
                        entry_time=entry_timestamp,
                        exit_time=timestamp,
                        reason="TARGET"
                    )

                    completed_trades.append(
                        trade
                    )

                    in_trade = False
                    current_signal = "WAITING"

                # PUT EMA9 EXIT
                elif close >= ema9:

                    trade = make_trade_record(
                        side="PUT",
                        entry_price=entry_price,
                        exit_price=close,
                        entry_time=entry_timestamp,
                        exit_time=timestamp,
                        reason="EMA9"
                    )

                    completed_trades.append(
                        trade
                    )

                    in_trade = False
                    current_signal = "WAITING"

        # ====================================================
        # BREAKOUTS
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
    # TOTAL HISTORICAL COMPLETED TRADES
    # ========================================================

    historical_trade_count = len(
        completed_trades
    )

    # ========================================================
    # EXACT LAST 64 COMPLETED TRADES
    # ========================================================

    rolling = completed_trades[
        -ROLLING_TRADES:
    ]

    rolling_count = len(
        rolling
    )

    full_sample = (
        rolling_count ==
        ROLLING_TRADES
    )

    rolling_wins = sum(
        1
        for trade in rolling
        if trade["win"]
    )

    rolling_losses = (
        rolling_count -
        rolling_wins
    )

    if rolling_count:

        rolling_win_rate = (
            rolling_wins *
            100.0 /
            rolling_count
        )

    else:

        rolling_win_rate = 0.0

    # ========================================================
    # ROLLING CALL / PUT BREAKDOWN
    #
    # These are informational only.
    # Qualification is based on the SAME combined rolling 64.
    # ========================================================

    rolling_calls = [
        trade
        for trade in rolling
        if trade["side"] == "CALL"
    ]

    rolling_puts = [
        trade
        for trade in rolling
        if trade["side"] == "PUT"
    ]

    call_trades = len(
        rolling_calls
    )

    put_trades = len(
        rolling_puts
    )

    call_wins = sum(
        1
        for trade in rolling_calls
        if trade["win"]
    )

    put_wins = sum(
        1
        for trade in rolling_puts
        if trade["win"]
    )

    call_losses = (
        call_trades -
        call_wins
    )

    put_losses = (
        put_trades -
        put_wins
    )

    call_rate = (
        call_wins *
        100.0 /
        call_trades
        if call_trades
        else 0.0
    )

    put_rate = (
        put_wins *
        100.0 /
        put_trades
        if put_trades
        else 0.0
    )

    # ========================================================
    # EXACT QUALIFICATION RULE
    #
    # 1. Must have FULL 64 completed trades
    # 2. Only LAST 64 are measured
    # 3. Rolling-64 win rate must be >= 90%
    # ========================================================

    qualified = (
        full_sample
        and
        rolling_win_rate >= MIN_WIN_RATE
    )

    # ========================================================
    # DIRECTION
    #
    # The 90% qualification comes from the combined rolling 64.
    # Direction tells the bot which side has performed better.
    # ========================================================

    if qualified:

        if (
            call_trades > 0
            and
            put_trades > 0
        ):

            if call_rate > put_rate:

                qualification = "CALL"

            elif put_rate > call_rate:

                qualification = "PUT"

            else:

                qualification = "CALL + PUT"

        elif call_trades > 0:

            qualification = "CALL"

        elif put_trades > 0:

            qualification = "PUT"

        else:

            qualification = "SKIP"

    else:

        qualification = "SKIP"

    call_qualified = (
        qualified
        and
        qualification in (
            "CALL",
            "CALL + PUT"
        )
    )

    put_qualified = (
        qualified
        and
        qualification in (
            "PUT",
            "CALL + PUT"
        )
    )

    # ========================================================
    # NET MOVE FROM ROLLING 64
    # ========================================================

    total_move = sum(
        trade["move"]
        for trade in rolling
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

    # ========================================================
    # REQUIRED WINS FOR 90% OVER 64
    #
    # 58 / 64 = 90.625%
    # ========================================================

    required_wins = 58

    return {

        "symbol": symbol,

        # ------------------------------------
        # EXACT ROLLING 64
        # ------------------------------------

        "overall": round(
            rolling_win_rate,
            1
        ),

        "rolling_win_rate": round(
            rolling_win_rate,
            1
        ),

        "rolling_trades": rolling_count,

        "sample_required": ROLLING_TRADES,

        "full_sample": full_sample,

        "wins": rolling_wins,

        "losses": rolling_losses,

        "total_trades": rolling_count,

        "historical_trade_count":
            historical_trade_count,

        "required_wins":
            required_wins,

        # ------------------------------------
        # CALL / PUT BREAKDOWN INSIDE
        # THE SAME LAST 64
        # ------------------------------------

        "call_rate": round(
            call_rate,
            1
        ),

        "put_rate": round(
            put_rate,
            1
        ),

        "call_wins":
            call_wins,

        "call_losses":
            call_losses,

        "call_trades":
            call_trades,

        "put_wins":
            put_wins,

        "put_losses":
            put_losses,

        "put_trades":
            put_trades,

        # ------------------------------------
        # RESULT
        # ------------------------------------

        "qualified":
            qualified,

        "call_qualified":
            call_qualified,

        "put_qualified":
            put_qualified,

        "qualification":
            qualification,

        "net_move": round(
            total_move,
            2
        ),

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
                "%s | ROLLING %s/%s | "
                "%.1f%% | %s",
                symbol,
                stats[
                    "rolling_trades"
                ],
                ROLLING_TRADES,
                stats[
                    "rolling_win_rate"
                ],
                stats[
                    "qualification"
                ],
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

            # Highest true rolling-64 rate first.
            all_results.sort(
                key=lambda stock: (
                    stock[
                        "full_sample"
                    ],
                    stock[
                        "rolling_win_rate"
                    ],
                    stock[
                        "wins"
                    ]
                ),
                reverse=True
            )

            qualified = [

                stock

                for stock
                in all_results

                if stock[
                    "qualified"
                ]
            ]

            with lock:

                STATE[
                    "results"
                ] = (
                    all_results.copy()
                )

                STATE[
                    "qualified"
                ] = (
                    qualified.copy()
                )

                STATE[
                    "qualified_count"
                ] = len(
                    qualified
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
            "FULL MARKET SCAN COMPLETE"
        )

        logging.info(
            "Scanned: %s",
            STATE[
                "scanned_count"
            ]
        )

        logging.info(
            "90%%+ ROLLING-64 qualifying: %s",
            STATE[
                "qualified_count"
            ]
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
# AUTOMATIC SCANNER LOOP
# ============================================================

def scanner_loop():

    time.sleep(
        3
    )

    while True:

        try:

            run_full_market_scan()

        except Exception:

            logging.exception(
                "Scanner loop error"
            )

        time.sleep(
            SCAN_EVERY_MINUTES *
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
90% Rolling 64 Market Scanner
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
    font-size: 24px;
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
    line-height: 1.6;
}

.stock {
    background: #171717;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 10px;
}

.top {
    display: flex;
    justify-content:
        space-between;
    align-items: center;
}

.symbol {
    font-size: 24px;
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

.winrate {
    margin-top: 12px;
    background: #0d0d0d;
    padding: 13px;
    border-radius: 12px;
    text-align: center;
}

.big {
    font-size: 31px;
    font-weight: 900;
    color: #39d353;
}

.small {
    color: #999;
    font-size: 11px;
}

.rates {
    display: grid;
    grid-template-columns:
        1fr 1fr;
    gap: 8px;
    margin-top: 10px;
}

.rate {
    background: #0d0d0d;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}

.label {
    color: #999;
    font-size: 10px;
}

.number {
    font-size: 20px;
    font-weight: 900;
    margin-top: 3px;
}

.record {
    margin-top: 10px;
    font-size: 12px;
    color: #aaa;
    line-height: 1.6;
}

.status {
    margin-top: 8px;
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
90% ROLLING-64 MARKET SCANNER
</h1>

<div class="subtitle">
4-minute strategy • EXACT last 64 completed trades
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

<strong>90%+ qualified:</strong>
{{ qualified_count }}

<br>

<strong>Required sample:</strong>
{{ sample_size }} completed trades

<br>

<strong>Minimum rate:</strong>
{{ minimum }}%

<br>

<strong>Minimum wins:</strong>
58 / 64

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
CALL
</div>

{% elif stock.qualification == "PUT" %}

<div class="direction put">
PUT
</div>

{% else %}

<div class="direction both">
CALL + PUT
</div>

{% endif %}

</div>


<div class="winrate">

<div class="small">
EXACT LAST 64 WIN RATE
</div>

<div class="big">
{{ stock.rolling_win_rate }}%
</div>

<div class="small">
{{ stock.wins }} wins /
{{ stock.losses }} losses /
{{ stock.rolling_trades }} trades
</div>

</div>


<div class="rates">

<div class="rate">

<div class="label">
CALLS INSIDE LAST 64
</div>

<div class="number">
{{ stock.call_rate }}%
</div>

<div class="small">
{{ stock.call_wins }}W /
{{ stock.call_losses }}L
</div>

</div>


<div class="rate">

<div class="label">
PUTS INSIDE LAST 64
</div>

<div class="number">
{{ stock.put_rate }}%
</div>

<div class="small">
{{ stock.put_wins }}W /
{{ stock.put_losses }}L
</div>

</div>

</div>


<div class="record">

ROLLING SAMPLE:
{{ stock.rolling_trades }} / 64

<br>

HISTORICAL COMPLETED TRADES FOUND:
{{ stock.historical_trade_count }}

<br>

NET UNDERLYING MOVE:
{{ stock.net_move }}

</div>


<div class="status">

STATUS:
{{ stock.signal }}

</div>

</div>

{% endfor %}

{% else %}

<div class="empty">

No stocks currently meet the
90% rolling-64 requirement.

<br><br>

A stock must have a FULL
64 completed trades before it can appear here.

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
            "Exact Rolling 64 Full Market Scanner",

        "timeframe":
            TIMEFRAME,

        "minimum_win_rate":
            MIN_WIN_RATE,

        "required_completed_trades":
            ROLLING_TRADES,

        "required_wins_for_90_percent":
            58,

        "watchlist":
            "/watchlist",

        "api":
            "/api/watchlist",

        "rescan":
            "/rescan",
    })


@app.route("/watchlist")
def watchlist():

    with lock:

        snapshot = {

            "status":
                STATE[
                    "status"
                ],

            "universe_count":
                STATE[
                    "universe_count"
                ],

            "scanned_count":
                STATE[
                    "scanned_count"
                ],

            "qualified_count":
                STATE[
                    "qualified_count"
                ],

            "qualified":
                STATE[
                    "qualified"
                ].copy(),

            "last_scan":
                STATE[
                    "last_scan"
                ],

            "error":
                STATE[
                    "error"
                ],
        }

    return render_template_string(
        HTML,
        minimum=MIN_WIN_RATE,
        sample_size=ROLLING_TRADES,
        **snapshot
    )


@app.route("/api/watchlist")
def api_watchlist():

    with lock:

        return jsonify({

            "status":
                STATE[
                    "status"
                ],

            "universe_count":
                STATE[
                    "universe_count"
                ],

            "scanned_count":
                STATE[
                    "scanned_count"
                ],

            "qualified_count":
                STATE[
                    "qualified_count"
                ],

            "minimum_win_rate":
                MIN_WIN_RATE,

            "required_sample":
                ROLLING_TRADES,

            "required_wins":
                58,

            "timeframe":
                TIMEFRAME,

            "last_scan":
                STATE[
                    "last_scan"
                ],

            "qualified":
                STATE[
                    "qualified"
                ],

            "error":
                STATE[
                    "error"
                ],
        })


@app.route("/api/all")
def api_all():

    with lock:

        return jsonify({

            "status":
                STATE[
                    "status"
                ],

            "results":
                STATE[
                    "results"
                ],

            "count":
                len(
                    STATE[
                        "results"
                    ]
                ),
        })


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
        target=
            run_full_market_scan,
        daemon=True
    )

    thread.start()

    return jsonify({
        "status":
            "full market rolling-64 scan started"
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
            STATE[
                "status"
            ],

        "rolling_sample":
            ROLLING_TRADES,

        "minimum_win_rate":
            MIN_WIN_RATE,
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