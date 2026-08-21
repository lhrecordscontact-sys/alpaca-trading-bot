import os
import time
import threading
import logging
import math
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from flask import Flask, jsonify


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
# ALPACA PAPER TRADING
# ============================================================

ALPACA_API_KEY = os.getenv(
    "ALPACA_API_KEY",
    ""
).strip()

ALPACA_SECRET_KEY = os.getenv(
    "ALPACA_SECRET_KEY",
    ""
).strip()

# PAPER ONLY
TRADING_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

DATA_FEED = os.getenv(
    "DATA_FEED",
    "iex"
).strip().lower()

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json",
}


# ============================================================
# SCANNER CONNECTION
# ============================================================

SCANNER_URL = os.getenv(
    "SCANNER_URL",
    "https://nine0-percent-scanner.onrender.com/api/watchlist"
).strip()

MIN_SCANNER_WIN_RATE = float(
    os.getenv(
        "MIN_SCANNER_WIN_RATE",
        "90"
    )
)

REQUIRED_SCANNER_TRADES = int(
    os.getenv(
        "REQUIRED_SCANNER_TRADES",
        "64"
    )
)


# ============================================================
# BOT SETTINGS
# ============================================================

RUN_BOT_LOOP = (
    os.getenv(
        "RUN_BOT_LOOP",
        "true"
    ).strip().lower()
    == "true"
)

# Leave true only for PAPER trading.
AUTO_TRADE = (
    os.getenv(
        "AUTO_TRADE",
        "true"
    ).strip().lower()
    == "true"
)

LOOP_SECONDS = int(
    os.getenv(
        "LOOP_SECONDS",
        "20"
    )
)

TIMEFRAME = "4Min"

EMA_FAST = 5
EMA_SLOW = 9
EMA_TREND = 30

PREMARKET_START = dt_time(4, 0)
PREMARKET_END = dt_time(9, 30)

RTH_START = dt_time(9, 30)

LAST_ENTRY = dt_time(14, 45)
FORCE_EXIT = dt_time(15, 15)

MAX_OPEN_POSITIONS = int(
    os.getenv(
        "MAX_OPEN_POSITIONS",
        "3"
    )
)

MAX_NEW_TRADES_PER_CYCLE = int(
    os.getenv(
        "MAX_NEW_TRADES_PER_CYCLE",
        "1"
    )
)

MAX_TRADES_PER_SYMBOL_DAY = int(
    os.getenv(
        "MAX_TRADES_PER_SYMBOL_DAY",
        "2"
    )
)

# Dollar budget used to decide contract quantity.
POSITION_DOLLARS = float(
    os.getenv(
        "POSITION_DOLLARS",
        "500"
    )
)

# Option premium management.
STOP_LOSS_PCT = float(
    os.getenv(
        "STOP_LOSS_PCT",
        "0.20"
    )
)

TAKE_PROFIT_PCT = float(
    os.getenv(
        "TAKE_PROFIT_PCT",
        "0.30"
    )
)

RUNNER_TRAIL_PCT = float(
    os.getenv(
        "RUNNER_TRAIL_PCT",
        "0.15"
    )
)


# ============================================================
# STATE
# ============================================================

lock = threading.Lock()

STATE = {
    "status": "STARTING",
    "last_cycle": None,
    "scanner_status": None,
    "watchlist": [],
    "watchlist_count": 0,
    "last_signal": None,
    "last_order": None,
    "last_error": None,
    "auto_trade": AUTO_TRADE,
}

# Prevent duplicate trades from the same 4-minute signal.
processed_signals = set()

# Count entries by underlying/day.
daily_trade_counts = {}

# Track runner high water marks by option symbol.
runner_highs = {}

# Remember positions that already took first profit.
tp_taken = set()


# ============================================================
# BASIC REQUEST HELPERS
# ============================================================

def alpaca_request(
    method,
    path,
    base=TRADING_BASE_URL,
    params=None,
    json_data=None,
    timeout=30,
):

    response = requests.request(
        method=method,
        url=f"{base}{path}",
        headers=HEADERS,
        params=params,
        json=json_data,
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
# ACCOUNT
# ============================================================

def get_account():

    return alpaca_request(
        "GET",
        "/v2/account",
    )


def get_clock():

    return alpaca_request(
        "GET",
        "/v2/clock",
    )


# ============================================================
# SCANNER WATCHLIST
# ============================================================

def get_scanner_watchlist():

    response = requests.get(
        SCANNER_URL,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    scanner_status = payload.get(
        "status"
    )

    qualified = payload.get(
        "qualified",
        []
    )

    good = []

    for item in qualified:

        symbol = str(
            item.get(
                "symbol",
                ""
            )
        ).upper().strip()

        if not symbol:
            continue

        rate = float(
            item.get(
                "rolling_win_rate",
                item.get(
                    "overall",
                    0
                )
            )
            or 0
        )

        trades = int(
            item.get(
                "rolling_trades",
                item.get(
                    "trades",
                    item.get(
                        "total_trades",
                        0
                    )
                )
            )
            or 0
        )

        qualification = str(
            item.get(
                "qualification",
                "CALL + PUT"
            )
        ).upper()

        # HARD GATE:
        # must have a full 64 AND >=90%.
        if trades < REQUIRED_SCANNER_TRADES:
            continue

        if rate < MIN_SCANNER_WIN_RATE:
            continue

        good.append({
            "symbol": symbol,
            "win_rate": rate,
            "trades": trades,
            "qualification": qualification,
        })

    good.sort(
        key=lambda x: x[
            "win_rate"
        ],
        reverse=True,
    )

    with lock:

        STATE[
            "scanner_status"
        ] = scanner_status

        STATE[
            "watchlist"
        ] = good.copy()

        STATE[
            "watchlist_count"
        ] = len(
            good
        )

    return good


# ============================================================
# STOCK BAR DATA
# ============================================================

def get_today_bars(symbol):

    now = datetime.now(
        UTC
    )

    # Enough data to include today's premarket.
    start = (
        now -
        timedelta(
            hours=18
        )
    )

    result = alpaca_request(
        "GET",
        f"/v2/stocks/{symbol}/bars",
        base=DATA_BASE_URL,
        params={
            "timeframe": TIMEFRAME,
            "start": start.isoformat(),
            "end": now.isoformat(),
            "adjustment": "raw",
            "feed": DATA_FEED,
            "sort": "asc",
            "limit": 1000,
        },
    )

    bars = result.get(
        "bars",
        []
    )

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
        df[
            "timestamp"
        ],
        utc=True,
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
        .groupby(
            dates
        )
        .cumsum()
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
# CURRENT-DAY SIGNAL
# ============================================================

def evaluate_signal(
    symbol,
    qualification
):

    df = get_today_bars(
        symbol
    )

    if df.empty:
        return None

    df = add_indicators(
        df
    )

    today = datetime.now(
        NY
    ).date()

    today_df = df[
        df.index.date ==
        today
    ]

    if today_df.empty:
        return None

    premarket = today_df[
        (
            today_df.index.time >=
            PREMARKET_START
        )
        &
        (
            today_df.index.time <
            PREMARKET_END
        )
    ]

    if premarket.empty:
        return None

    pm_high = float(
        premarket[
            "high"
        ].max()
    )

    pm_low = float(
        premarket[
            "low"
        ].min()
    )

    rth = today_df[
        today_df.index.time >=
        RTH_START
    ]

    # Need previous + latest closed bar.
    if len(rth) < 2:
        return None

    previous = rth.iloc[
        -2
    ]

    current = rth.iloc[
        -1
    ]

    previous_time = rth.index[
        -2
    ]

    current_time = rth.index[
        -1
    ]

    previous_close = float(
        previous[
            "close"
        ]
    )

    close = float(
        current[
            "close"
        ]
    )

    ema5 = float(
        current[
            "ema5"
        ]
    )

    ema9 = float(
        current[
            "ema9"
        ]
    )

    ema30 = float(
        current[
            "ema30"
        ]
    )

    vwap = float(
        current[
            "vwap"
        ]
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

    call_signal = (
        close > pm_high
        and
        previous_close <= pm_high
        and
        bull_trend
        and
        close > vwap
    )

    put_signal = (
        close < pm_low
        and
        previous_close >= pm_low
        and
        bear_trend
        and
        close < vwap
    )

    allowed = (
        qualification
        or
        "CALL + PUT"
    ).upper()

    if (
        call_signal
        and
        allowed in (
            "CALL",
            "CALL + PUT",
        )
    ):

        return {
            "symbol": symbol,
            "direction": "CALL",
            "bar_time": (
                current_time.isoformat()
            ),
            "close": close,
            "pm_high": pm_high,
            "pm_low": pm_low,
            "ema5": ema5,
            "ema9": ema9,
            "ema30": ema30,
            "vwap": vwap,
        }

    if (
        put_signal
        and
        allowed in (
            "PUT",
            "CALL + PUT",
        )
    ):

        return {
            "symbol": symbol,
            "direction": "PUT",
            "bar_time": (
                current_time.isoformat()
            ),
            "close": close,
            "pm_high": pm_high,
            "pm_low": pm_low,
            "ema5": ema5,
            "ema9": ema9,
            "ema30": ema30,
            "vwap": vwap,
        }

    return None


# ============================================================
# POSITIONS
# ============================================================

def get_positions():

    result = alpaca_request(
        "GET",
        "/v2/positions",
    )

    if isinstance(
        result,
        list
    ):
        return result

    return []


def open_position_count():

    return len(
        get_positions()
    )


def underlying_already_open(
    underlying
):

    underlying = (
        underlying.upper()
    )

    for position in get_positions():

        symbol = str(
            position.get(
                "symbol",
                ""
            )
        ).upper()

        # OCC options begin with
        # the underlying symbol.
        if symbol.startswith(
            underlying
        ):
            return True

    return False


# ============================================================
# OPTION CONTRACTS
# ============================================================

def get_same_day_option_contracts(
    underlying,
    direction,
):

    today = datetime.now(
        NY
    ).date().isoformat()

    option_type = (
        "call"
        if direction == "CALL"
        else "put"
    )

    result = alpaca_request(
        "GET",
        "/v2/options/contracts",
        params={
            "underlying_symbols":
                underlying,
            "expiration_date":
                today,
            "type":
                option_type,
            "status":
                "active",
            "limit":
                1000,
        },
    )

    return result.get(
        "option_contracts",
        []
    )


def choose_atm_contract(
    underlying,
    direction,
    stock_price,
):

    contracts = (
        get_same_day_option_contracts(
            underlying,
            direction,
        )
    )

    if not contracts:

        logging.info(
            "%s has no 0DTE %s contracts",
            underlying,
            direction,
        )

        return None

    valid = []

    for contract in contracts:

        symbol = contract.get(
            "symbol"
        )

        strike = contract.get(
            "strike_price"
        )

        if not symbol:
            continue

        try:
            strike = float(
                strike
            )
        except (
            TypeError,
            ValueError
        ):
            continue

        valid.append({
            "symbol": symbol,
            "strike": strike,
            "distance": abs(
                strike -
                stock_price
            ),
        })

    if not valid:
        return None

    valid.sort(
        key=lambda x: (
            x["distance"],
            x["strike"],
        )
    )

    return valid[0]


# ============================================================
# OPTION QUOTE / PRICE
# ============================================================

def get_option_snapshot(
    option_symbol
):

    result = alpaca_request(
        "GET",
        f"/v1beta1/options/snapshots/{option_symbol}",
        base=DATA_BASE_URL,
    )

    return result


def option_estimated_price(
    option_symbol
):

    try:

        snapshot = (
            get_option_snapshot(
                option_symbol
            )
        )

        quote = snapshot.get(
            "latestQuote",
            snapshot.get(
                "latest_quote",
                {}
            )
        ) or {}

        ask = quote.get(
            "ap",
            quote.get(
                "ask_price"
            )
        )

        bid = quote.get(
            "bp",
            quote.get(
                "bid_price"
            )
        )

        ask = (
            float(ask)
            if ask is not None
            else 0.0
        )

        bid = (
            float(bid)
            if bid is not None
            else 0.0
        )

        if ask > 0 and bid > 0:

            return (
                ask +
                bid
            ) / 2.0

        if ask > 0:
            return ask

        if bid > 0:
            return bid

    except Exception as exc:

        logging.warning(
            "Option quote failed %s: %s",
            option_symbol,
            exc,
        )

    return None


# ============================================================
# ORDER QUANTITY
# ============================================================

def option_quantity(
    option_symbol
):

    premium = (
        option_estimated_price(
            option_symbol
        )
    )

    if (
        premium is None
        or
        premium <= 0
    ):

        # Safe fallback:
        # one paper contract.
        return 1

    contract_cost = (
        premium *
        100.0
    )

    qty = math.floor(
        POSITION_DOLLARS /
        contract_cost
    )

    return max(
        1,
        qty,
    )


# ============================================================
# SUBMIT OPTION ORDER
# ============================================================

def submit_option_order(
    option_symbol,
    side,
    qty,
):

    if not AUTO_TRADE:

        logging.info(
            "AUTO_TRADE OFF | "
            "%s %s x%s",
            side,
            option_symbol,
            qty,
        )

        return {
            "paper_preview": True,
            "symbol": option_symbol,
            "side": side,
            "qty": qty,
        }

    order = alpaca_request(
        "POST",
        "/v2/orders",
        json_data={
            "symbol":
                option_symbol,

            "qty":
                str(
                    int(qty)
                ),

            "side":
                side.lower(),

            "type":
                "market",

            "time_in_force":
                "day",
        },
    )

    with lock:

        STATE[
            "last_order"
        ] = order

    return order


# ============================================================
# CLOSE OPTION POSITION
# ============================================================

def close_option_position(
    option_symbol,
    qty=None,
):

    if not AUTO_TRADE:

        logging.info(
            "AUTO_TRADE OFF | "
            "would close %s",
            option_symbol,
        )

        return {}

    if qty is None:

        return alpaca_request(
            "DELETE",
            f"/v2/positions/{option_symbol}",
        )

    return submit_option_order(
        option_symbol=option_symbol,
        side="sell",
        qty=qty,
    )


# ============================================================
# POSITION MANAGEMENT
# ============================================================

def manage_positions():

    now = datetime.now(
        NY
    )

    positions = get_positions()

    for position in positions:

        option_symbol = str(
            position.get(
                "symbol",
                ""
            )
        )

        try:

            qty = abs(
                int(
                    float(
                        position.get(
                            "qty",
                            0
                        )
                    )
                )
            )

        except Exception:

            qty = 0

        if qty <= 0:
            continue

        try:

            avg_price = float(
                position.get(
                    "avg_entry_price",
                    0
                )
                or 0
            )

            current_price = float(
                position.get(
                    "current_price",
                    0
                )
                or 0
            )

        except Exception:
            continue

        if (
            avg_price <= 0
            or
            current_price <= 0
        ):
            continue

        pnl_pct = (
            current_price -
            avg_price
        ) / avg_price

        # ----------------------------------------
        # FORCE EXIT
        # ----------------------------------------

        if now.time() >= FORCE_EXIT:

            logging.info(
                "FORCE EXIT %s",
                option_symbol,
            )

            close_option_position(
                option_symbol
            )

            tp_taken.discard(
                option_symbol
            )

            runner_highs.pop(
                option_symbol,
                None
            )

            continue

        # ----------------------------------------
        # STOP LOSS
        # ----------------------------------------

        if pnl_pct <= (
            -STOP_LOSS_PCT
        ):

            logging.info(
                "STOP LOSS %s | %.1f%%",
                option_symbol,
                pnl_pct * 100,
            )

            close_option_position(
                option_symbol
            )

            tp_taken.discard(
                option_symbol
            )

            runner_highs.pop(
                option_symbol,
                None
            )

            continue

        # ----------------------------------------
        # FIRST TAKE PROFIT
        # ----------------------------------------

        if (
            option_symbol
            not in tp_taken
            and
            pnl_pct >=
            TAKE_PROFIT_PCT
        ):

            # If we have at least 2,
            # sell half and let the rest run.
            if qty >= 2:

                sell_qty = max(
                    1,
                    qty // 2
                )

                logging.info(
                    "TAKE PROFIT %s | "
                    "selling %s of %s",
                    option_symbol,
                    sell_qty,
                    qty,
                )

                close_option_position(
                    option_symbol,
                    qty=sell_qty,
                )

                tp_taken.add(
                    option_symbol
                )

                runner_highs[
                    option_symbol
                ] = current_price

            else:

                # Only one contract:
                # take the full profit.
                logging.info(
                    "TAKE PROFIT %s | "
                    "closing 1 contract",
                    option_symbol,
                )

                close_option_position(
                    option_symbol
                )

                tp_taken.discard(
                    option_symbol
                )

                runner_highs.pop(
                    option_symbol,
                    None
                )

            continue

        # ----------------------------------------
        # RUNNER TRAILING EXIT
        # ----------------------------------------

        if (
            option_symbol
            in tp_taken
        ):

            old_high = runner_highs.get(
                option_symbol,
                current_price,
            )

            high_water = max(
                old_high,
                current_price,
            )

            runner_highs[
                option_symbol
            ] = high_water

            trailing_floor = (
                high_water *
                (
                    1.0 -
                    RUNNER_TRAIL_PCT
                )
            )

            if (
                current_price <=
                trailing_floor
            ):

                logging.info(
                    "RUNNER TRAIL EXIT "
                    "%s | high %.2f | "
                    "now %.2f",
                    option_symbol,
                    high_water,
                    current_price,
                )

                close_option_position(
                    option_symbol
                )

                tp_taken.discard(
                    option_symbol
                )

                runner_highs.pop(
                    option_symbol,
                    None
                )


# ============================================================
# DAILY COUNT
# ============================================================

def today_trade_key(
    symbol
):

    today = datetime.now(
        NY
    ).date().isoformat()

    return (
        today,
        symbol.upper(),
    )


def can_trade_symbol_today(
    symbol
):

    key = today_trade_key(
        symbol
    )

    count = daily_trade_counts.get(
        key,
        0
    )

    return (
        count <
        MAX_TRADES_PER_SYMBOL_DAY
    )


def register_symbol_trade(
    symbol
):

    key = today_trade_key(
        symbol
    )

    daily_trade_counts[
        key
    ] = (
        daily_trade_counts.get(
            key,
            0
        )
        +
        1
    )


# ============================================================
# TRADE ONE SIGNAL
# ============================================================

def trade_signal(
    signal,
    scanner_item
):

    underlying = signal[
        "symbol"
    ]

    direction = signal[
        "direction"
    ]

    signal_key = (
        underlying,
        direction,
        signal[
            "bar_time"
        ],
    )

    if signal_key in processed_signals:

        return False

    processed_signals.add(
        signal_key
    )

    if not can_trade_symbol_today(
        underlying
    ):

        logging.info(
            "%s daily trade limit reached",
            underlying,
        )

        return False

    if underlying_already_open(
        underlying
    ):

        logging.info(
            "%s already has open option",
            underlying,
        )

        return False

    if (
        open_position_count()
        >= MAX_OPEN_POSITIONS
    ):

        logging.info(
            "Maximum positions reached"
        )

        return False

    contract = choose_atm_contract(
        underlying=underlying,
        direction=direction,
        stock_price=signal[
            "close"
        ],
    )

    if not contract:

        logging.info(
            "%s %s signal but "
            "no same-day option contract",
            underlying,
            direction,
        )

        return False

    option_symbol = contract[
        "symbol"
    ]

    qty = option_quantity(
        option_symbol
    )

    logging.info(
        "ENTRY %s %s | "
        "scanner %.1f%% | "
        "underlying %.2f | "
        "strike %.2f | "
        "%s x%s",
        underlying,
        direction,
        scanner_item[
            "win_rate"
        ],
        signal[
            "close"
        ],
        contract[
            "strike"
        ],
        option_symbol,
        qty,
    )

    order = submit_option_order(
        option_symbol=
            option_symbol,
        side="buy",
        qty=qty,
    )

    register_symbol_trade(
        underlying
    )

    with lock:

        STATE[
            "last_signal"
        ] = {
            **signal,
            "scanner_win_rate":
                scanner_item[
                    "win_rate"
                ],
            "scanner_trades":
                scanner_item[
                    "trades"
                ],
            "option_symbol":
                option_symbol,
            "strike":
                contract[
                    "strike"
                ],
            "qty":
                qty,
            "order":
                order,
        }

    return True


# ============================================================
# MAIN TRADING CYCLE
# ============================================================

def run_trading_cycle():

    now = datetime.now(
        NY
    )

    with lock:

        STATE[
            "last_cycle"
        ] = now.isoformat()

    # Weekends
    if now.weekday() >= 5:

        with lock:
            STATE[
                "status"
            ] = "WEEKEND"

        return

    # Always manage existing positions.
    manage_positions()

    # No new entries outside entry window.
    if (
        now.time() <
        RTH_START
    ):

        with lock:

            STATE[
                "status"
            ] = "WAITING FOR MARKET"

        return

    if (
        now.time() >=
        LAST_ENTRY
    ):

        with lock:

            STATE[
                "status"
            ] = "ENTRY WINDOW CLOSED"

        return

    # ----------------------------------------
    # GET QUALIFIED SCANNER LIST
    # ----------------------------------------

    watchlist = (
        get_scanner_watchlist()
    )

    if not watchlist:

        with lock:

            STATE[
                "status"
            ] = "NO 90% QUALIFIERS"

        logging.info(
            "Scanner returned "
            "no full-64 90%%+ stocks"
        )

        return

    with lock:

        STATE[
            "status"
        ] = "WATCHING QUALIFIED STOCKS"

    new_trades = 0

    for item in watchlist:

        if (
            new_trades >=
            MAX_NEW_TRADES_PER_CYCLE
        ):
            break

        symbol = item[
            "symbol"
        ]

        try:

            signal = evaluate_signal(
                symbol=symbol,
                qualification=item[
                    "qualification"
                ],
            )

            if not signal:
                continue

            logging.info(
                "CONFIRMED SIGNAL | "
                "%s | %s | %.1f%%",
                symbol,
                signal[
                    "direction"
                ],
                item[
                    "win_rate"
                ],
            )

            traded = trade_signal(
                signal,
                item,
            )

            if traded:

                new_trades += 1

        except Exception as exc:

            logging.exception(
                "%s cycle error: %s",
                symbol,
                exc,
            )


# ============================================================
# BOT LOOP
# ============================================================

def bot_loop():

    logging.info(
        "===================================="
    )

    logging.info(
        "90%% ROLLING-64 PAPER TRADING BOT"
    )

    logging.info(
        "Scanner: %s",
        SCANNER_URL
    )

    logging.info(
        "AUTO_TRADE: %s",
        AUTO_TRADE
    )

    logging.info(
        "===================================="
    )

    time.sleep(
        3
    )

    while True:

        try:

            run_trading_cycle()

            with lock:

                STATE[
                    "last_error"
                ] = None

        except Exception as exc:

            logging.exception(
                "BOT LOOP ERROR"
            )

            with lock:

                STATE[
                    "status"
                ] = "ERROR"

                STATE[
                    "last_error"
                ] = str(
                    exc
                )

        time.sleep(
            LOOP_SECONDS
        )


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():

    with lock:

        return jsonify({

            "bot":
                "90% Rolling-64 "
                "0DTE Paper Trading Bot",

            "status":
                STATE[
                    "status"
                ],

            "auto_trade":
                AUTO_TRADE,

            "scanner":
                SCANNER_URL,

            "scanner_status":
                STATE[
                    "scanner_status"
                ],

            "watchlist_count":
                STATE[
                    "watchlist_count"
                ],

            "watchlist":
                STATE[
                    "watchlist"
                ],

            "last_signal":
                STATE[
                    "last_signal"
                ],

            "last_cycle":
                STATE[
                    "last_cycle"
                ],

            "last_error":
                STATE[
                    "last_error"
                ],

            "health":
                "/health",

            "positions":
                "/positions",
        })


@app.route("/watchlist")
def watchlist():

    try:

        qualified = (
            get_scanner_watchlist()
        )

        return jsonify({

            "count":
                len(
                    qualified
                ),

            "required_win_rate":
                MIN_SCANNER_WIN_RATE,

            "required_trades":
                REQUIRED_SCANNER_TRADES,

            "qualified":
                qualified,
        })

    except Exception as exc:

        return jsonify({
            "error":
                str(
                    exc
                )
        }), 500


@app.route("/positions")
def positions():

    try:

        return jsonify({
            "positions":
                get_positions()
        })

    except Exception as exc:

        return jsonify({
            "error":
                str(
                    exc
                )
        }), 500


@app.route("/health")
def health():

    try:

        account = get_account()

        scanner_ok = False
        scanner_error = None

        try:

            watchlist = (
                get_scanner_watchlist()
            )

            scanner_ok = True

        except Exception as exc:

            watchlist = []

            scanner_error = str(
                exc
            )

        return jsonify({

            "status":
                "healthy",

            "alpaca_connected":
                True,

            "account_status":
                account.get(
                    "status"
                ),

            "paper_account":
                True,

            "auto_trade":
                AUTO_TRADE,

            "scanner_connected":
                scanner_ok,

            "scanner_error":
                scanner_error,

            "qualified_count":
                len(
                    watchlist
                ),

            "bot_status":
                STATE[
                    "status"
                ],
        })

    except Exception as exc:

        return jsonify({

            "status":
                "error",

            "alpaca_connected":
                False,

            "paper_account":
                True,

            "error":
                str(
                    exc
                ),

        }), 500


# ============================================================
# START BOT
# ============================================================

if RUN_BOT_LOOP:

    trading_thread = (
        threading.Thread(
            target=bot_loop,
            daemon=True,
        )
    )

    trading_thread.start()


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
        debug=False,
        threaded=True,
    )