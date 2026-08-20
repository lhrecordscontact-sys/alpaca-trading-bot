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
# ALPACA CONFIG
# ============================================================

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "").strip()

TRADING_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

DATA_FEED = os.getenv("DATA_FEED", "iex").strip().lower()

RUN_SCANNER = (
    os.getenv("RUN_SCANNER", "true")
    .strip()
    .lower()
    == "true"
)


# ============================================================
# YOUR STRATEGY SETTINGS
# ============================================================

TIMEFRAME = "4Min"

EMA_FAST = 5
EMA_SLOW = 9
EMA_TREND = 30

PM_START = dt_time(4, 0)
PM_END = dt_time(9, 30)

RTH_START = dt_time(9, 30)
RTH_END = dt_time(16, 0)

SPY_TARGET = 1.00
IWM_TARGET = 0.50

# Same behavior as your current Pine:
# non-IWM symbols use the $1 target.
DEFAULT_TARGET = 1.00

MAX_TRADES_PER_DAY = 2

ROLLING_TRADES = 64

MIN_WIN_RATE = float(
    os.getenv("MIN_WIN_RATE", "90")
)

# Require the full 64 completed trades.
REQUIRE_FULL_64 = (
    os.getenv("REQUIRE_FULL_64", "true")
    .strip()
    .lower()
    == "true"
)

# Historical calendar range used to FIND 64 completed setups.
HISTORY_DAYS = int(
    os.getenv("HISTORY_DAYS", "120")
)


# ============================================================
# MORNING SCAN
# ============================================================

SCAN_HOUR = int(
    os.getenv("SCAN_HOUR", "4")
)

SCAN_MINUTE = int(
    os.getenv("SCAN_MINUTE", "0")
)

# How many symbols per batch request.
BATCH_SIZE = int(
    os.getenv("BATCH_SIZE", "100")
)

# Small pause to avoid hammering API.
REQUEST_PAUSE = float(
    os.getenv("REQUEST_PAUSE", "0.30")
)


# ============================================================
# STATE
# ============================================================

lock = threading.Lock()

STATE = {
    "status": "STARTING",
    "scan_date": None,
    "scan_started": None,
    "scan_finished": None,
    "universe_count": 0,
    "scanned_count": 0,
    "qualified_count": 0,
    "qualified": [],
    "last_error": None,
}


# ============================================================
# API
# ============================================================

def headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }


def api_request(
    method,
    path,
    base=TRADING_BASE_URL,
    params=None,
    timeout=45,
):
    response = requests.request(
        method,
        f"{base}{path}",
        headers=headers(),
        params=params,
        timeout=timeout,
    )

    if not response.ok:
        raise RuntimeError(
            f"{method} {path} -> "
            f"{response.status_code}: "
            f"{response.text[:500]}"
        )

    if not response.text:
        return {}

    return response.json()


# ============================================================
# ACCOUNT TEST
# ============================================================

def get_account():
    return api_request(
        "GET",
        "/v2/account",
    )


# ============================================================
# GET FULL ELIGIBLE STOCK UNIVERSE
# ============================================================

def get_stock_universe():

    logging.info(
        "Downloading Alpaca asset universe..."
    )

    assets = api_request(
        "GET",
        "/v2/assets",
        params={
            "status": "active",
            "asset_class": "us_equity",
        },
    )

    symbols = []

    for asset in assets:

        symbol = str(
            asset.get("symbol", "")
        ).upper().strip()

        if not symbol:
            continue

        # Must be tradable
        if not asset.get("tradable", False):
            continue

        # We eventually want options,
        # so keep optionable names.
        if not asset.get("options_enabled", False):
            continue

        # Skip weird symbols that are
        # generally not useful for this scanner.
        if "/" in symbol:
            continue

        symbols.append(symbol)

    symbols = sorted(
        list(set(symbols))
    )

    logging.info(
        "Eligible universe: %s stocks",
        len(symbols),
    )

    return symbols


# ============================================================
# HISTORICAL MULTI-SYMBOL BARS
# ============================================================

def fetch_batch_bars(
    symbols,
    days,
):

    if not symbols:
        return {}

    end = datetime.now(UTC)

    start = (
        end -
        timedelta(days=days)
    )

    params = {
        "symbols": ",".join(symbols),
        "timeframe": TIMEFRAME,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": 10000,
        "adjustment": "raw",
        "feed": DATA_FEED,
        "sort": "asc",
    }

    all_bars = {
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
            "GET",
            "/v2/stocks/bars",
            base=DATA_BASE_URL,
            params=params,
        )

        bars_by_symbol = result.get(
            "bars",
            {}
        )

        for symbol, bars in bars_by_symbol.items():

            if symbol not in all_bars:
                all_bars[symbol] = []

            all_bars[symbol].extend(
                bars
            )

        page_token = result.get(
            "next_page_token"
        )

        if not page_token:
            break

        time.sleep(REQUEST_PAUSE)

    return all_bars


# ============================================================
# CONVERT BARS TO DATAFRAME
# ============================================================

def bars_to_dataframe(bars):

    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)

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
        utc=True,
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
            errors="coerce",
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
            adjust=False,
        )
        .mean()
    )

    df["ema9"] = (
        df["close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False,
        )
        .mean()
    )

    df["ema30"] = (
        df["close"]
        .ewm(
            span=EMA_TREND,
            adjust=False,
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
        index=df.index,
    )

    price_volume = (
        typical *
        df["volume"]
    )

    cumulative_pv = (
        price_volume
        .groupby(dates)
        .cumsum()
    )

    cumulative_volume = (
        df["volume"]
        .groupby(dates)
        .cumsum()
        .replace(0, float("nan"))
    )

    df["vwap"] = (
        cumulative_pv /
        cumulative_volume
    )

    return df


# ============================================================
# TARGET
# ============================================================

def target_move(symbol):

    if symbol == "IWM":
        return IWM_TARGET

    if symbol == "SPY":
        return SPY_TARGET

    return DEFAULT_TARGET


# ============================================================
# PREMARKET LEVEL
# ============================================================

def get_pm_levels(day_df):

    pm = day_df[
        (day_df.index.time >= PM_START) &
        (day_df.index.time < PM_END)
    ]

    if pm.empty:
        return None

    return {
        "high": float(
            pm["high"].max()
        ),
        "low": float(
            pm["low"].min()
        ),
    }


# ============================================================
# BACKTEST ONE STOCK
#
# Mirrors your Pine:
#
# CALL:
# close crosses above PM high
# EMA5 > EMA9 > EMA30
# close > VWAP
#
# PUT:
# close crosses below PM low
# EMA5 < EMA9 < EMA30
# close < VWAP
#
# TP checked before EMA9 exit.
# Positive EMA9 exit = win.
# Negative/zero exit = loss.
# ============================================================

def calculate_trades(
    symbol,
    raw_df,
):

    if raw_df.empty:
        return []

    df = add_indicators(
        raw_df
    )

    completed = []

    today = datetime.now(
        NY
    ).date()

    days = sorted(
        set(df.index.date)
    )

    target = target_move(
        symbol
    )

    for day in days:

        # Don't contaminate morning qualification
        # with today's unfinished session.
        if day >= today:
            continue

        day_df = df[
            df.index.date == day
        ]

        levels = get_pm_levels(
            day_df
        )

        if not levels:
            continue

        pm_high = levels["high"]
        pm_low = levels["low"]

        rth = day_df[
            (day_df.index.time >= RTH_START) &
            (day_df.index.time < RTH_END)
        ]

        if len(rth) < 2:
            continue

        in_trade = False
        direction = None

        entry_price = None
        target_price = None

        trades_today = 0

        previous_close = None

        for timestamp, row in rth.iterrows():

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
            # EXIT EXISTING TRADE
            # ====================================

            if in_trade:

                # ----------------------------
                # CALL
                # ----------------------------

                if direction == "CALL":

                    # Pine TP has priority
                    if high >= target_price:

                        completed.append({
                            "timestamp": timestamp,
                            "direction": "CALL",
                            "result": "WIN",
                            "move": target_price - entry_price,
                        })

                        in_trade = False
                        direction = None

                    elif close <= ema9:

                        move = (
                            close -
                            entry_price
                        )

                        completed.append({
                            "timestamp": timestamp,
                            "direction": "CALL",
                            "result": (
                                "WIN"
                                if move > 0
                                else "LOSS"
                            ),
                            "move": move,
                        })

                        in_trade = False
                        direction = None

                # ----------------------------
                # PUT
                # ----------------------------

                else:

                    if low <= target_price:

                        completed.append({
                            "timestamp": timestamp,
                            "direction": "PUT",
                            "result": "WIN",
                            "move": entry_price - target_price,
                        })

                        in_trade = False
                        direction = None

                    elif close >= ema9:

                        move = (
                            entry_price -
                            close
                        )

                        completed.append({
                            "timestamp": timestamp,
                            "direction": "PUT",
                            "result": (
                                "WIN"
                                if move > 0
                                else "LOSS"
                            ),
                            "move": move,
                        })

                        in_trade = False
                        direction = None

            # ====================================
            # ENTRY
            # ====================================

            if (
                not in_trade
                and trades_today < MAX_TRADES_PER_DAY
                and previous_close is not None
            ):

                bull_trend = (
                    ema5 > ema9
                    and ema9 > ema30
                )

                bear_trend = (
                    ema5 < ema9
                    and ema9 < ema30
                )

                call_signal = (
                    close > pm_high
                    and previous_close <= pm_high
                    and bull_trend
                    and close > vwap
                )

                put_signal = (
                    close < pm_low
                    and previous_close >= pm_low
                    and bear_trend
                    and close < vwap
                )

                if call_signal:

                    in_trade = True
                    direction = "CALL"

                    entry_price = close

                    target_price = (
                        close +
                        target
                    )

                    trades_today += 1

                elif put_signal:

                    in_trade = True
                    direction = "PUT"

                    entry_price = close

                    target_price = (
                        close -
                        target
                    )

                    trades_today += 1

            previous_close = close

    return completed


# ============================================================
# EXACT ROLLING LAST 64
# ============================================================

def rolling_64_stats(
    symbol,
    raw_df,
):

    trades = calculate_trades(
        symbol,
        raw_df,
    )

    if REQUIRE_FULL_64:

        if len(trades) < ROLLING_TRADES:
            return None

    if not trades:
        return None

    recent = trades[
        -ROLLING_TRADES:
    ]

    total = len(recent)

    wins = sum(
        1
        for trade in recent
        if trade["result"] == "WIN"
    )

    losses = (
        total -
        wins
    )

    calls = [
        trade
        for trade in recent
        if trade["direction"] == "CALL"
    ]

    puts = [
        trade
        for trade in recent
        if trade["direction"] == "PUT"
    ]

    call_wins = sum(
        1
        for trade in calls
        if trade["result"] == "WIN"
    )

    put_wins = sum(
        1
        for trade in puts
        if trade["result"] == "WIN"
    )

    overall_rate = (
        wins * 100.0 / total
    )

    call_rate = (
        call_wins * 100.0 / len(calls)
        if calls
        else 0.0
    )

    put_rate = (
        put_wins * 100.0 / len(puts)
        if puts
        else 0.0
    )

    net_move = sum(
        float(
            trade["move"]
        )
        for trade in recent
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
        "wins": wins,
        "losses": losses,
        "trades": total,
        "calls": len(calls),
        "puts": len(puts),
        "net_move": round(
            net_move,
            2
        ),
    }


# ============================================================
# SCAN ONE BATCH
# ============================================================

def process_batch(
    symbols,
):

    qualified = []

    bars_by_symbol = fetch_batch_bars(
        symbols,
        HISTORY_DAYS,
    )

    for symbol in symbols:

        try:

            bars = bars_by_symbol.get(
                symbol,
                []
            )

            df = bars_to_dataframe(
                bars
            )

            stats = rolling_64_stats(
                symbol,
                df,
            )

            with lock:
                STATE["scanned_count"] += 1

            if not stats:
                continue

            logging.info(
                "%s | %.1f%% | %s/%s",
                symbol,
                stats["overall"],
                stats["wins"],
                stats["trades"],
            )

            if (
                stats["overall"]
                >= MIN_WIN_RATE
            ):

                qualified.append(
                    stats
                )

                logging.info(
                    "QUALIFIED %s | %.1f%%",
                    symbol,
                    stats["overall"],
                )

        except Exception as exc:

            logging.warning(
                "%s failed: %s",
                symbol,
                exc,
            )

    return qualified


# ============================================================
# FULL MARKET SCAN
# ============================================================

def run_full_scan():

    now = datetime.now(NY)

    with lock:

        STATE["status"] = (
            "SCANNING ENTIRE ELIGIBLE MARKET"
        )

        STATE["scan_started"] = (
            now.isoformat()
        )

        STATE["scan_finished"] = None
        STATE["scanned_count"] = 0
        STATE["qualified_count"] = 0
        STATE["qualified"] = []
        STATE["last_error"] = None

    try:

        universe = get_stock_universe()

        with lock:
            STATE["universe_count"] = len(
                universe
            )

        all_qualified = []

        batches = [
            universe[i:i + BATCH_SIZE]
            for i in range(
                0,
                len(universe),
                BATCH_SIZE,
            )
        ]

        logging.info(
            "Starting %s batches...",
            len(batches),
        )

        for batch_number, batch in enumerate(
            batches,
            start=1,
        ):

            logging.info(
                "================================="
            )

            logging.info(
                "BATCH %s / %s",
                batch_number,
                len(batches),
            )

            logging.info(
                "Symbols: %s",
                len(batch),
            )

            batch_qualified = (
                process_batch(
                    batch
                )
            )

            all_qualified.extend(
                batch_qualified
            )

            all_qualified = sorted(
                all_qualified,
                key=lambda item:
                    (
                        item["overall"],
                        item["trades"],
                    ),
                reverse=True,
            )

            with lock:

                STATE["qualified"] = (
                    all_qualified.copy()
                )

                STATE["qualified_count"] = len(
                    all_qualified
                )

            time.sleep(
                REQUEST_PAUSE
            )

        finished = datetime.now(
            NY
        )

        with lock:

            STATE["scan_date"] = (
                now.date().isoformat()
            )

            STATE["scan_finished"] = (
                finished.isoformat()
            )

            STATE["status"] = (
                "TODAY'S LIST LOCKED"
            )

        logging.info(
            "================================="
        )

        logging.info(
            "FULL MARKET SCAN FINISHED"
        )

        logging.info(
            "QUALIFIED >= %.1f%%: %s",
            MIN_WIN_RATE,
            len(all_qualified),
        )

        for stock in all_qualified:

            logging.info(
                "%s | %.1f%% | "
                "%sW %sL | "
                "CALL %.1f%% | "
                "PUT %.1f%%",
                stock["symbol"],
                stock["overall"],
                stock["wins"],
                stock["losses"],
                stock["call_rate"],
                stock["put_rate"],
            )

    except Exception as exc:

        logging.exception(
            "FULL SCAN FAILED"
        )

        with lock:

            STATE["status"] = "ERROR"

            STATE["last_error"] = str(
                exc
            )


# ============================================================
# DAILY SCHEDULER
# ============================================================

def scan_is_due():

    now = datetime.now(NY)

    # No weekend scan
    if now.weekday() >= 5:
        return False

    today = (
        now.date()
        .isoformat()
    )

    with lock:

        if STATE["scan_date"] == today:
            return False

        if STATE["status"].startswith(
            "SCANNING"
        ):
            return False

    scheduled = dt_time(
        SCAN_HOUR,
        SCAN_MINUTE,
    )

    return (
        now.time() >= scheduled
        and now.time() < RTH_START
    )


def scheduler():

    logging.info(
        "Scanner scheduler started."
    )

    logging.info(
        "Daily scan begins at %02d:%02d ET",
        SCAN_HOUR,
        SCAN_MINUTE,
    )

    while True:

        try:

            if scan_is_due():

                run_full_scan()

        except Exception as exc:

            logging.exception(
                "Scheduler error"
            )

            with lock:
                STATE["last_error"] = str(
                    exc
                )

        time.sleep(30)


# ============================================================
# PHONE DASHBOARD
# ============================================================

WATCHLIST_HTML = """
<!doctype html>
<html>
<head>
<meta name="viewport"
      content="width=device-width, initial-scale=1">

<title>90% Stock Scanner</title>

<style>

body {
    background: #0b0e11;
    color: white;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        Arial,
        sans-serif;
    margin: 0;
    padding: 16px;
}

.header {
    margin-bottom: 18px;
}

h1 {
    margin: 0 0 6px 0;
    font-size: 27px;
}

.subtitle {
    color: #aab2bd;
    font-size: 14px;
}

.stats {
    display: grid;
    grid-template-columns:
        repeat(3, 1fr);
    gap: 8px;
    margin: 16px 0;
}

.card {
    background: #171b20;
    border-radius: 12px;
    padding: 12px 8px;
    text-align: center;
}

.card .number {
    font-size: 22px;
    font-weight: 700;
}

.card .label {
    color: #9aa4ae;
    font-size: 11px;
    margin-top: 3px;
}

.status {
    background: #171b20;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 15px;
}

.stock {
    background: #171b20;
    border-radius: 14px;
    margin-bottom: 10px;
    padding: 14px;
}

.stock-top {
    display: flex;
    justify-content:
        space-between;
    align-items: center;
}

.symbol {
    font-size: 22px;
    font-weight: 800;
}

.rate {
    color: #2ecc71;
    font-size: 22px;
    font-weight: 800;
}

.details {
    display: grid;
    grid-template-columns:
        repeat(3, 1fr);
    gap: 6px;
    margin-top: 12px;
}

.detail {
    background: #0f1317;
    border-radius: 8px;
    padding: 8px;
    text-align: center;
}

.detail-name {
    color: #8f9aa5;
    font-size: 10px;
}

.detail-value {
    font-size: 14px;
    font-weight: 700;
    margin-top: 3px;
}

.good {
    color: #2ecc71;
}

.error {
    color: #ff5b5b;
}

.empty {
    background: #171b20;
    border-radius: 14px;
    padding: 25px;
    text-align: center;
    color: #aab2bd;
}

</style>

</head>

<body>

<div class="header">

<h1>90% QUALIFYING STOCKS</h1>

<div class="subtitle">
Rolling last 64 completed trades
</div>

</div>


<div class="stats">

<div class="card">
<div class="number">
{{ universe_count }}
</div>
<div class="label">
ELIGIBLE
</div>
</div>

<div class="card">
<div class="number">
{{ scanned_count }}
</div>
<div class="label">
SCANNED
</div>
</div>

<div class="card">
<div class="number good">
{{ qualified_count }}
</div>
<div class="label">
QUALIFIED
</div>
</div>

</div>


<div class="status">

<strong>Status:</strong>
{{ status }}

<br>

<strong>Minimum:</strong>
{{ minimum }}%

<br>

<strong>Scan date:</strong>
{{ scan_date or "Not finished yet" }}

{% if last_error %}

<br>

<span class="error">
{{ last_error }}
</span>

{% endif %}

</div>


{% if qualified %}

{% for stock in qualified %}

<div class="stock">

<div class="stock-top">

<div class="symbol">
{{ stock.symbol }}
</div>

<div class="rate">
{{ stock.overall }}%
</div>

</div>


<div class="details">

<div class="detail">

<div class="detail-name">
RECORD
</div>

<div class="detail-value">
{{ stock.wins }}W /
{{ stock.losses }}L
</div>

</div>


<div class="detail">

<div class="detail-name">
CALL
</div>

<div class="detail-value">
{{ stock.call_rate }}%
</div>

</div>


<div class="detail">

<div class="detail-name">
PUT
</div>

<div class="detail-value">
{{ stock.put_rate }}%
</div>

</div>

</div>

</div>

{% endfor %}

{% else %}

<div class="empty">

No ≥{{ minimum }}% stocks
are available yet.

<br><br>

If the scanner is running,
refresh this page later.

</div>

{% endif %}


</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "bot": "Full Market Rolling 64 Scanner",
        "status": STATE["status"],
        "watchlist": "/watchlist",
        "api": "/api/watchlist",
        "minimum_win_rate": MIN_WIN_RATE,
        "rolling_trades": ROLLING_TRADES,
    })


@app.route("/watchlist")
def watchlist():

    with lock:

        snapshot = {
            "status": STATE["status"],
            "scan_date": STATE["scan_date"],
            "universe_count": STATE[
                "universe_count"
            ],
            "scanned_count": STATE[
                "scanned_count"
            ],
            "qualified_count": STATE[
                "qualified_count"
            ],
            "qualified": STATE[
                "qualified"
            ].copy(),
            "last_error": STATE[
                "last_error"
            ],
        }

    return render_template_string(
        WATCHLIST_HTML,
        minimum=MIN_WIN_RATE,
        **snapshot,
    )


@app.route("/api/watchlist")
def api_watchlist():

    with lock:

        return jsonify({
            "status": STATE["status"],
            "scan_date": STATE[
                "scan_date"
            ],
            "scan_started": STATE[
                "scan_started"
            ],
            "scan_finished": STATE[
                "scan_finished"
            ],
            "universe_count": STATE[
                "universe_count"
            ],
            "scanned_count": STATE[
                "scanned_count"
            ],
            "minimum_win_rate": (
                MIN_WIN_RATE
            ),
            "rolling_trades": (
                ROLLING_TRADES
            ),
            "qualified_count": STATE[
                "qualified_count"
            ],
            "qualified": STATE[
                "qualified"
            ],
            "error": STATE[
                "last_error"
            ],
        })


@app.route("/health")
def health():

    try:

        account = get_account()

        return jsonify({
            "status": "healthy",
            "alpaca_connected": True,
            "account_status": (
                account.get("status")
            ),
            "scanner": STATE[
                "status"
            ],
        })

    except Exception as exc:

        return jsonify({
            "status": "error",
            "alpaca_connected": False,
            "error": str(exc),
        }), 500


# ============================================================
# START SCANNER
# ============================================================

if RUN_SCANNER:

    scanner_thread = threading.Thread(
        target=scheduler,
        daemon=True,
    )

    scanner_thread.start()


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "10000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
    )