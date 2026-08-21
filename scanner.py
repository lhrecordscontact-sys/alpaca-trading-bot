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

# Your Pine uses SPY target for symbols
# other than IWM too.
DEFAULT_TARGET = 1.00

MAX_TRADES_PER_DAY = 2

MIN_WIN_RATE = float(
    os.getenv(
        "MIN_WIN_RATE",
        "90"
    )
)

# Pine itself allows even 1 completed CALL
# to produce a percentage.
# Leave at 1 for Pine-style behavior.
MIN_SIDE_TRADES = int(
    os.getenv(
        "MIN_SIDE_TRADES",
        "1"
    )
)

# Amount of historical data the Python scanner
# downloads for each stock.
HISTORY_DAYS = int(
    os.getenv(
        "HISTORY_DAYS",
        "120"
    )
)

# Symbols downloaded in groups instead of
# one request per ticker.
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

# Automatically rescan.
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

        # Only active/tradable stocks.
        if not asset.get(
            "tradable",
            False
        ):
            continue

        # Skip malformed / unusual slash symbols.
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
# TARGET
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

    df[
        "ema5"
    ] = (
        df["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False
        )
        .mean()
    )

    df[
        "ema9"
    ] = (
        df["close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False
        )
        .mean()
    )

    df[
        "ema30"
    ] = (
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

    df[
        "vwap"
    ] = (
        cumulative_pv /
        cumulative_volume
    )

    return df


# ============================================================
# PINE-STYLE STRATEGY CALCULATION
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

    current_signal = "WAITING"

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
        # PREMARKET
        # ====================================

        in_pm = (
            current_time >= PREMARKET_START
            and current_time < PREMARKET_END
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

        # ====================================
        # RTH
        # ====================================

        in_rth = (
            current_time >= RTH_START
            and current_time < RTH_END
        )

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
        # BREAKOUTS
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
        # ====================================

        can_trade = (
            not in_trade
            and trades_today <
            MAX_TRADES_PER_DAY
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

            current_signal = (
                "CALL SIGNAL"
            )

        elif short_entry:

            in_trade = True
            long_trade = False

            entry_price = close

            target_price = (
                close -
                target
            )

            trades_today += 1

            current_signal = (
                "PUT SIGNAL"
            )

        # ====================================
        # EXITS
        # ====================================

        if in_trade:

            # --------------------------------
            # CALL
            # --------------------------------

            if long_trade:

                if high >= target_price:

                    move = (
                        target_price -
                        entry_price
                    )

                    wins += 1
                    total_trades += 1

                    call_wins += 1
                    call_trades += 1

                    total_move += (
                        move
                    )

                    in_trade = False

                elif close <= ema9:

                    move = (
                        close -
                        entry_price
                    )

                    if move > 0:

                        wins += 1
                        call_wins += 1

                    else:

                        losses += 1
                        call_losses += 1

                    total_trades += 1
                    call_trades += 1

                    total_move += (
                        move
                    )

                    in_trade = False

            # --------------------------------
            # PUT
            # --------------------------------

            else:

                if low <= target_price:

                    move = (
                        entry_price -
                        target_price
                    )

                    wins += 1
                    total_trades += 1

                    put_wins += 1
                    put_trades += 1

                    total_move += (
                        move
                    )

                    in_trade = False

                elif close >= ema9:

                    move = (
                        entry_price -
                        close
                    )

                    if move > 0:

                        wins += 1
                        put_wins += 1

                    else:

                        losses += 1
                        put_losses += 1

                    total_trades += 1
                    put_trades += 1

                    total_move += (
                        move
                    )

                    in_trade = False

        previous_close = close
        previous_date = current_date

    # ========================================
    # PERCENTAGES
    # ========================================

    overall_rate = (
        wins * 100.0 /
        total_trades
        if total_trades
        else 0.0
    )

    call_rate = (
        call_wins * 100.0 /
        call_trades
        if call_trades
        else 0.0
    )

    put_rate = (
        put_wins * 100.0 /
        put_trades
        if put_trades
        else 0.0
    )

    # ========================================
    # IMPORTANT:
    # EACH SIDE MUST QUALIFY BY ITSELF
    # ========================================

    call_qualified = (
        call_trades >=
        MIN_SIDE_TRADES
        and call_rate >=
        MIN_WIN_RATE
    )

    put_qualified = (
        put_trades >=
        MIN_SIDE_TRADES
        and put_rate >=
        MIN_WIN_RATE
    )

    if (
        call_qualified
        and put_qualified
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

    if in_trade:

        current_signal = (
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

        "call_qualified":
            call_qualified,

        "put_qualified":
            put_qualified,

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
                "%s | CALL %.1f%% | "
                "PUT %.1f%% | %s",
                symbol,
                stats[
                    "call_rate"
                ],
                stats[
                    "put_rate"
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
            "Scanning %s stocks "
            "in %s batches",
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

            # Best side first.
            all_results.sort(
                key=lambda stock:
                    max(
                        stock[
                            "call_rate"
                        ],
                        stock[
                            "put_rate"
                        ],
                    ),
                reverse=True
            )

            qualified = [

                stock

                for stock
                in all_results

                if (
                    stock[
                        "call_qualified"
                    ]
                    or
                    stock[
                        "put_qualified"
                    ]
                )
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
            "90%%+ qualifying: %s",
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
90% Market Scanner
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
    font-size: 25px;
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

.rates {
    display: grid;
    grid-template-columns:
        1fr 1fr;
    gap: 8px;
    margin-top: 12px;
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
    font-size: 22px;
    font-weight: 900;
    margin-top: 3px;
}

.record {
    margin-top: 10px;
    font-size: 12px;
    color: #aaa;
    line-height: 1.5;
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
90% FULL MARKET SCANNER
</h1>

<div class="subtitle">
4-minute CALL / PUT strategy
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

<strong>90%+ stocks:</strong>
{{ qualified_count }}

<br>

<strong>Minimum:</strong>
{{ minimum }}%

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


<div class="rates">

<div class="rate">

<div class="label">
CALL WIN RATE
</div>

<div class="number
{% if stock.call_qualified %}
call
{% endif %}
">

{{ stock.call_rate }}%

</div>

</div>


<div class="rate">

<div class="label">
PUT WIN RATE
</div>

<div class="number
{% if stock.put_qualified %}
put
{% endif %}
">

{{ stock.put_rate }}%

</div>

</div>

</div>


<div class="record">

CALL:
{{ stock.call_wins }}W /
{{ stock.call_losses }}L
—
{{ stock.call_trades }}
trades

<br>

PUT:
{{ stock.put_wins }}W /
{{ stock.put_losses }}L
—
{{ stock.put_trades }}
trades

<br>

OVERALL:
{{ stock.overall }}%

</div>


<div class="status">

STATUS:
{{ stock.signal }}

</div>

</div>

{% endfor %}

{% else %}

<div class="empty">

No 90%+ CALL or PUT stocks
found yet.

<br><br>

If the market scan is still running,
this page will update as batches finish.

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
            "Full Market 90% CALL PUT Scanner",

        "timeframe":
            TIMEFRAME,

        "minimum":
            MIN_WIN_RATE,

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
            "full market scan started"
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