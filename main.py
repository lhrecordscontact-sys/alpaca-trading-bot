import os
import time
import math
import threading
import logging
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
    format="%(asctime)s | %(levelname)s | %(message)s",
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

AUTO_TRADE = (
    os.getenv("AUTO_TRADE", "false")
    .strip()
    .lower()
    == "true"
)

RUN_BOT_LOOP = (
    os.getenv("RUN_BOT_LOOP", "true")
    .strip()
    .lower()
    == "true"
)


# ============================================================
# STRATEGY SETTINGS
# ============================================================

TIMEFRAME_MINUTES = 4

EMA_FAST = 5
EMA_SLOW = 9
EMA_TREND = 30

MIN_OVERALL_WIN_RATE = float(
    os.getenv("MIN_OVERALL_WIN_RATE", "90")
)

MIN_COMPLETED_TRADES = int(
    os.getenv("MIN_COMPLETED_TRADES", "20")
)

MAX_TRADES_PER_DAY = int(
    os.getenv("MAX_TRADES_PER_DAY", "2")
)

SPY_TARGET = float(
    os.getenv("SPY_TARGET", "1.00")
)

IWM_TARGET = float(
    os.getenv("IWM_TARGET", "0.50")
)

DEFAULT_TARGET = float(
    os.getenv("DEFAULT_TARGET", "1.00")
)


# ============================================================
# MORNING SCANNER SETTINGS
# ============================================================

# Build list at 8:45 AM ET
SCAN_HOUR = int(
    os.getenv("SCAN_HOUR", "8")
)

SCAN_MINUTE = int(
    os.getenv("SCAN_MINUTE", "45")
)

# Alpaca supports up to 100 most-active names.
MOST_ACTIVE_TOP = int(
    os.getenv("MOST_ACTIVE_TOP", "100")
)

# Number of names we'll actually backtest.
# Lower this if Render becomes slow.
MAX_SCAN_CANDIDATES = int(
    os.getenv("MAX_SCAN_CANDIDATES", "50")
)

BACKTEST_DAYS = int(
    os.getenv("BACKTEST_DAYS", "45")
)

SCAN_PAUSE_SECONDS = float(
    os.getenv("SCAN_PAUSE_SECONDS", "0.35")
)


# ============================================================
# EXECUTION SETTINGS
# ============================================================

POSITION_DOLLARS = float(
    os.getenv("POSITION_DOLLARS", "500")
)

MAX_OPEN_POSITIONS = int(
    os.getenv("MAX_OPEN_POSITIONS", "3")
)

MAX_NEW_TRADES_PER_CYCLE = int(
    os.getenv("MAX_NEW_TRADES_PER_CYCLE", "1")
)

LOOP_SECONDS = int(
    os.getenv("LOOP_SECONDS", "20")
)

LAST_ENTRY_TIME = dt_time(14, 45)
FORCE_EXIT_TIME = dt_time(15, 15)

PM_START = dt_time(4, 0)
PM_END = dt_time(9, 30)

RTH_START = dt_time(9, 30)
RTH_END = dt_time(16, 0)


# ============================================================
# BOT STATE
# ============================================================

state_lock = threading.Lock()

BOT_STATE = {
    "status": "starting",
    "scan_date": None,
    "scan_started": None,
    "scan_finished": None,
    "daily_watchlist": [],
    "qualified_details": {},
    "current_signals": {},
    "last_cycle": None,
    "last_error": None,
    "orders": [],
}

last_signal_bar = {}

active_trades = {}


# ============================================================
# API HELPERS
# ============================================================

def alpaca_headers():
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
    payload=None,
    timeout=30,
):
    url = f"{base}{path}"

    r = requests.request(
        method,
        url,
        headers=alpaca_headers(),
        params=params,
        json=payload,
        timeout=timeout,
    )

    if not r.ok:
        raise RuntimeError(
            f"{method} {path} -> "
            f"{r.status_code}: {r.text}"
        )

    if not r.text:
        return {}

    return r.json()


# ============================================================
# ACCOUNT / CLOCK
# ============================================================

def get_account():
    return api_request(
        "GET",
        "/v2/account",
    )


def get_clock():
    return api_request(
        "GET",
        "/v2/clock",
    )


def market_is_open():
    try:
        return bool(
            get_clock().get("is_open", False)
        )
    except Exception:
        return False


# ============================================================
# MOST ACTIVE STOCK DISCOVERY
# ============================================================

def get_most_active_stocks():

    result = api_request(
        "GET",
        "/v1beta1/screener/stocks/most-actives",
        base=DATA_BASE_URL,
        params={
            "by": "volume",
            "top": MOST_ACTIVE_TOP,
        },
    )

    rows = result.get(
        "most_actives",
        result.get("mostActives", [])
    )

    symbols = []

    for row in rows:

        symbol = str(
            row.get("symbol", "")
        ).upper().strip()

        if symbol and symbol not in symbols:
            symbols.append(symbol)

    return symbols


# ============================================================
# 0DTE AVAILABILITY
# ============================================================

def today_et():
    return datetime.now(NY).date().isoformat()


def get_option_contracts(
    underlying,
    option_type=None,
):
    params = {
        "underlying_symbols": underlying,
        "expiration_date": today_et(),
        "status": "active",
        "limit": 1000,
    }

    if option_type:
        params["type"] = option_type

    result = api_request(
        "GET",
        "/v2/options/contracts",
        params=params,
    )

    return result.get(
        "option_contracts",
        []
    )


def has_0dte_options(symbol):

    try:
        contracts = get_option_contracts(symbol)

        return len(contracts) > 0

    except Exception as exc:
        logging.warning(
            "%s option check failed: %s",
            symbol,
            exc,
        )

        return False


# ============================================================
# HISTORICAL BARS
# ============================================================

def fetch_bars(
    symbol,
    days,
):

    end = datetime.now(UTC)

    start = (
        end
        - timedelta(days=days)
    )

    params = {
        "timeframe": "4Min",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": 10000,
        "adjustment": "raw",
        "feed": DATA_FEED,
        "sort": "asc",
    }

    result = api_request(
        "GET",
        f"/v2/stocks/{symbol}/bars",
        base=DATA_BASE_URL,
        params=params,
    )

    bars = result.get(
        "bars",
        []
    )

    token = result.get(
        "next_page_token"
    )

    while token:

        params["page_token"] = token

        page = api_request(
            "GET",
            f"/v2/stocks/{symbol}/bars",
            base=DATA_BASE_URL,
            params=params,
        )

        bars.extend(
            page.get("bars", [])
        )

        token = page.get(
            "next_page_token"
        )

    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)

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
        ]
    )

    return df


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
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    dates = pd.Series(
        df.index.date,
        index=df.index,
    )

    pv = (
        typical
        * df["volume"]
    )

    cumulative_pv = (
        pv.groupby(dates).cumsum()
    )

    cumulative_volume = (
        df["volume"]
        .groupby(dates)
        .cumsum()
        .replace(0, float("nan"))
    )

    df["vwap"] = (
        cumulative_pv
        / cumulative_volume
    )

    return df


# ============================================================
# TARGET MOVE
# ============================================================

def target_for_symbol(symbol):

    if symbol == "IWM":
        return IWM_TARGET

    if symbol == "SPY":
        return SPY_TARGET

    return DEFAULT_TARGET


# ============================================================
# PREMARKET LEVELS BY DATE
# ============================================================

def calculate_pm_levels(df):

    levels = {}

    for day in sorted(
        set(df.index.date)
    ):

        d = df[
            df.index.date == day
        ]

        pm = d[
            (d.index.time >= PM_START)
            & (d.index.time < PM_END)
        ]

        if pm.empty:
            continue

        levels[day] = {
            "high": float(
                pm["high"].max()
            ),
            "low": float(
                pm["low"].min()
            ),
        }

    return levels


# ============================================================
# SAME PINE-STYLE STRATEGY BACKTEST
# ============================================================

def backtest_symbol(
    symbol,
    df,
):

    if df.empty:
        return None

    df = add_indicators(df)

    pm_levels = calculate_pm_levels(df)

    wins = 0
    losses = 0
    total_trades = 0

    call_wins = 0
    call_losses = 0
    call_trades = 0

    put_wins = 0
    put_losses = 0
    put_trades = 0

    target_move = target_for_symbol(
        symbol
    )

    # Do not use today when determining
    # today's qualifying watchlist.
    today = datetime.now(NY).date()

    days = sorted(
        set(df.index.date)
    )

    for day in days:

        if day >= today:
            continue

        if day not in pm_levels:
            continue

        pm_high = pm_levels[day]["high"]
        pm_low = pm_levels[day]["low"]

        day_df = df[
            df.index.date == day
        ]

        rth = day_df[
            (day_df.index.time >= RTH_START)
            & (day_df.index.time < RTH_END)
        ]

        if len(rth) < 2:
            continue

        in_trade = False
        long_trade = False

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

            # --------------------------------
            # MANAGE OPEN TRADE
            # --------------------------------

            if in_trade:

                if long_trade:

                    # Pine checks TP first
                    if high >= target_price:

                        wins += 1
                        total_trades += 1

                        call_wins += 1
                        call_trades += 1

                        in_trade = False

                    elif close <= ema9:

                        move = (
                            close
                            - entry_price
                        )

                        total_trades += 1
                        call_trades += 1

                        if move > 0:
                            wins += 1
                            call_wins += 1
                        else:
                            losses += 1
                            call_losses += 1

                        in_trade = False

                else:

                    if low <= target_price:

                        wins += 1
                        total_trades += 1

                        put_wins += 1
                        put_trades += 1

                        in_trade = False

                    elif close >= ema9:

                        move = (
                            entry_price
                            - close
                        )

                        total_trades += 1
                        put_trades += 1

                        if move > 0:
                            wins += 1
                            put_wins += 1
                        else:
                            losses += 1
                            put_losses += 1

                        in_trade = False

            # --------------------------------
            # NEW SIGNAL
            # --------------------------------

            if (
                not in_trade
                and trades_today
                < MAX_TRADES_PER_DAY
                and previous_close
                is not None
            ):

                bull_trend = (
                    ema5 > ema9
                    and ema9 > ema30
                )

                bear_trend = (
                    ema5 < ema9
                    and ema9 < ema30
                )

                long_break = (
                    close > pm_high
                    and previous_close <= pm_high
                    and bull_trend
                    and close > vwap
                )

                short_break = (
                    close < pm_low
                    and previous_close >= pm_low
                    and bear_trend
                    and close < vwap
                )

                if long_break:

                    in_trade = True
                    long_trade = True

                    entry_price = close

                    target_price = (
                        close
                        + target_move
                    )

                    trades_today += 1

                elif short_break:

                    in_trade = True
                    long_trade = False

                    entry_price = close

                    target_price = (
                        close
                        - target_move
                    )

                    trades_today += 1

            previous_close = close

    if total_trades == 0:
        return None

    win_rate = (
        wins
        * 100.0
        / total_trades
    )

    call_win_rate = (
        call_wins
        * 100.0
        / call_trades
        if call_trades > 0
        else 0
    )

    put_win_rate = (
        put_wins
        * 100.0
        / put_trades
        if put_trades > 0
        else 0
    )

    return {
        "symbol": symbol,
        "overall_win_rate": round(
            win_rate,
            1,
        ),
        "call_win_rate": round(
            call_win_rate,
            1,
        ),
        "put_win_rate": round(
            put_win_rate,
            1,
        ),
        "wins": wins,
        "losses": losses,
        "total_trades": total_trades,
        "call_trades": call_trades,
        "put_trades": put_trades,
    }


# ============================================================
# MORNING WATCHLIST SCAN
# ============================================================

def build_daily_watchlist():

    now = datetime.now(NY)

    logging.info(
        "======================================"
    )
    logging.info(
        "MORNING 90%% SCANNER STARTED"
    )
    logging.info(
        "Time: %s",
        now,
    )
    logging.info(
        "======================================"
    )

    with state_lock:
        BOT_STATE["scan_started"] = (
            now.isoformat()
        )

        BOT_STATE["status"] = (
            "morning_scan"
        )

    # ----------------------------------------
    # FIND ACTIVE STOCKS
    # ----------------------------------------

    try:

        symbols = get_most_active_stocks()

    except Exception as exc:

        logging.exception(
            "Most-active scanner failed"
        )

        with state_lock:
            BOT_STATE["last_error"] = str(
                exc
            )

        return

    symbols = symbols[
        :MAX_SCAN_CANDIDATES
    ]

    logging.info(
        "Candidates found: %s",
        len(symbols),
    )

    qualified = {}

    # ----------------------------------------
    # CHECK EACH STOCK
    # ----------------------------------------

    for number, symbol in enumerate(
        symbols,
        start=1,
    ):

        try:

            logging.info(
                "[%s/%s] Checking %s",
                number,
                len(symbols),
                symbol,
            )

            # Must have 0DTE option contracts
            if not has_0dte_options(
                symbol
            ):

                logging.info(
                    "%s SKIP - no 0DTE options",
                    symbol,
                )

                continue

            bars = fetch_bars(
                symbol,
                BACKTEST_DAYS,
            )

            stats = backtest_symbol(
                symbol,
                bars,
            )

            if not stats:
                continue

            logging.info(
                "%s | %.1f%% | trades=%s",
                symbol,
                stats[
                    "overall_win_rate"
                ],
                stats[
                    "total_trades"
                ],
            )

            # HARD 90% FILTER
            if (
                stats[
                    "overall_win_rate"
                ]
                < MIN_OVERALL_WIN_RATE
            ):
                continue

            if (
                stats[
                    "total_trades"
                ]
                < MIN_COMPLETED_TRADES
            ):
                continue

            qualified[symbol] = stats

            logging.info(
                "✅ QUALIFIED: %s %.1f%%",
                symbol,
                stats[
                    "overall_win_rate"
                ],
            )

        except Exception as exc:

            logging.warning(
                "%s scan error: %s",
                symbol,
                exc,
            )

        time.sleep(
            SCAN_PAUSE_SECONDS
        )

    # ----------------------------------------
    # SORT BEST FIRST
    # ----------------------------------------

    qualified = dict(
        sorted(
            qualified.items(),
            key=lambda x:
                x[1][
                    "overall_win_rate"
                ],
            reverse=True,
        )
    )

    watchlist = list(
        qualified.keys()
    )

    # ----------------------------------------
    # LOCK TODAY'S LIST
    # ----------------------------------------

    with state_lock:

        BOT_STATE[
            "daily_watchlist"
        ] = watchlist

        BOT_STATE[
            "qualified_details"
        ] = qualified

        BOT_STATE[
            "scan_date"
        ] = now.date().isoformat()

        BOT_STATE[
            "scan_finished"
        ] = (
            datetime.now(NY)
            .isoformat()
        )

        BOT_STATE[
            "status"
        ] = "watchlist_ready"

    logging.info(
        "======================================"
    )

    logging.info(
        "TODAY'S LOCKED 90%% WATCHLIST:"
    )

    if not watchlist:

        logging.info(
            "NO STOCKS QUALIFIED TODAY"
        )

    for symbol in watchlist:

        stats = qualified[
            symbol
        ]

        logging.info(
            "%s | Overall %.1f%% | "
            "CALL %.1f%% | PUT %.1f%% | "
            "Trades %s",
            symbol,
            stats[
                "overall_win_rate"
            ],
            stats[
                "call_win_rate"
            ],
            stats[
                "put_win_rate"
            ],
            stats[
                "total_trades"
            ],
        )

    logging.info(
        "======================================"
    )


# ============================================================
# TODAY'S PREMARKET LEVELS
# ============================================================

def today_pm_levels(df):

    if df.empty:
        return None

    today = datetime.now(NY).date()

    d = df[
        df.index.date == today
    ]

    pm = d[
        (d.index.time >= PM_START)
        & (d.index.time < PM_END)
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
# CURRENT SIGNAL
# ============================================================

def get_current_signal(
    symbol,
):

    df = fetch_bars(
        symbol,
        3,
    )

    if df.empty:
        return None

    df = add_indicators(
        df
    )

    pm = today_pm_levels(
        df
    )

    if not pm:
        return None

    today = datetime.now(NY).date()

    rth = df[
        (df.index.date == today)
        & (
            df.index.time
            >= RTH_START
        )
        & (
            df.index.time
            < RTH_END
        )
    ]

    if len(rth) < 2:
        return None

    previous = rth.iloc[-2]
    current = rth.iloc[-1]

    timestamp = rth.index[-1]

    close = float(
        current["close"]
    )

    previous_close = float(
        previous["close"]
    )

    ema5 = float(
        current["ema5"]
    )

    ema9 = float(
        current["ema9"]
    )

    ema30 = float(
        current["ema30"]
    )

    vwap = float(
        current["vwap"]
    )

    bull_trend = (
        ema5 > ema9
        and ema9 > ema30
    )

    bear_trend = (
        ema5 < ema9
        and ema9 < ema30
    )

    call_signal = (
        close > pm["high"]
        and previous_close
        <= pm["high"]
        and bull_trend
        and close > vwap
    )

    put_signal = (
        close < pm["low"]
        and previous_close
        >= pm["low"]
        and bear_trend
        and close < vwap
    )

    if call_signal:

        return {
            "symbol": symbol,
            "signal": "CALL",
            "price": close,
            "ema9": ema9,
            "pm_high": pm["high"],
            "pm_low": pm["low"],
            "bar": str(timestamp),
        }

    if put_signal:

        return {
            "symbol": symbol,
            "signal": "PUT",
            "price": close,
            "ema9": ema9,
            "pm_high": pm["high"],
            "pm_low": pm["low"],
            "bar": str(timestamp),
        }

    return None


# ============================================================
# POSITIONS
# ============================================================

def get_positions():

    try:
        return api_request(
            "GET",
            "/v2/positions",
        )

    except Exception:
        return []


def get_open_orders():

    try:

        return api_request(
            "GET",
            "/v2/orders",
            params={
                "status": "open",
                "limit": 500,
            },
        )

    except Exception:
        return []


def has_underlying_position(
    underlying,
):

    underlying = (
        underlying.upper()
    )

    for position in get_positions():

        symbol = str(
            position.get(
                "symbol",
                "",
            )
        ).upper()

        if symbol.startswith(
            underlying
        ):
            return True

    for order in get_open_orders():

        symbol = str(
            order.get(
                "symbol",
                "",
            )
        ).upper()

        if symbol.startswith(
            underlying
        ):
            return True

    return False


# ============================================================
# ATM 0DTE CONTRACT
# ============================================================

def choose_atm_contract(
    underlying,
    direction,
    underlying_price,
):

    option_type = (
        "call"
        if direction == "CALL"
        else "put"
    )

    contracts = get_option_contracts(
        underlying,
        option_type,
    )

    tradable = [
        c
        for c in contracts
        if c.get(
            "tradable",
            True,
        )
    ]

    if not tradable:

        raise RuntimeError(
            f"No tradable 0DTE "
            f"{direction} contracts "
            f"for {underlying}"
        )

    best = min(
        tradable,
        key=lambda c:
            abs(
                float(
                    c[
                        "strike_price"
                    ]
                )
                - underlying_price
            ),
    )

    return best


# ============================================================
# OPTION QUOTE
# ============================================================

def get_option_mid(
    symbol,
):

    result = api_request(
        "GET",
        "/v1beta1/options/quotes/latest",
        base=DATA_BASE_URL,
        params={
            "symbols": symbol,
        },
    )

    quote = (
        result
        .get("quotes", {})
        .get(symbol, {})
    )

    bid = float(
        quote.get("bp", 0)
        or 0
    )

    ask = float(
        quote.get("ap", 0)
        or 0
    )

    if bid > 0 and ask > 0:
        return (
            bid + ask
        ) / 2

    if ask > 0:
        return ask

    if bid > 0:
        return bid

    return None


# ============================================================
# BUY OPTION
# ============================================================

def submit_option_order(
    option_symbol,
    qty,
):

    payload = {
        "symbol": option_symbol,
        "qty": str(
            int(qty)
        ),
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
    }

    if not AUTO_TRADE:

        logging.info(
            "SIMULATED ORDER: %s",
            payload,
        )

        return {
            "simulated": True,
            "order": payload,
        }

    return api_request(
        "POST",
        "/v2/orders",
        payload=payload,
    )


# ============================================================
# EXECUTE QUALIFIED SIGNAL
# ============================================================

def execute_signal(
    symbol,
    signal,
):

    # ----------------------------------------
    # FINAL WATCHLIST CHECK
    # ----------------------------------------

    with state_lock:

        watchlist = BOT_STATE[
            "daily_watchlist"
        ].copy()

        stats = BOT_STATE[
            "qualified_details"
        ].get(symbol)

    if symbol not in watchlist:

        logging.warning(
            "%s BLOCKED - "
            "not on today's locked list",
            symbol,
        )

        return None

    if not stats:
        return None

    if (
        stats["overall_win_rate"]
        < MIN_OVERALL_WIN_RATE
    ):

        logging.warning(
            "%s BLOCKED - "
            "win rate fell below 90",
            symbol,
        )

        return None

    if has_underlying_position(
        symbol
    ):

        logging.info(
            "%s SKIP - "
            "already has position/order",
            symbol,
        )

        return None

    positions = get_positions()

    if (
        len(positions)
        >= MAX_OPEN_POSITIONS
    ):

        logging.info(
            "Max positions reached"
        )

        return None

    # ----------------------------------------
    # FIND ATM 0DTE
    # ----------------------------------------

    contract = choose_atm_contract(
        symbol,
        signal["signal"],
        signal["price"],
    )

    option_symbol = contract[
        "symbol"
    ]

    mid = get_option_mid(
        option_symbol
    )

    if not mid or mid <= 0:

        logging.warning(
            "%s no option price",
            option_symbol,
        )

        return None

    contract_cost = (
        mid
        * 100
    )

    qty = math.floor(
        POSITION_DOLLARS
        / contract_cost
    )

    if qty < 1:

        logging.warning(
            "%s costs $%.2f per "
            "contract - exceeds "
            "position budget $%.2f",
            option_symbol,
            contract_cost,
            POSITION_DOLLARS,
        )

        return None

    # ----------------------------------------
    # ORDER
    # ----------------------------------------

    result = submit_option_order(
        option_symbol,
        qty,
    )

    target_move = target_for_symbol(
        symbol
    )

    active_trades[symbol] = {
        "underlying": symbol,
        "option_symbol": option_symbol,
        "direction": signal["signal"],
        "entry_underlying": signal[
            "price"
        ],
        "target_underlying": (
            signal["price"]
            + target_move
            if signal["signal"]
            == "CALL"
            else signal["price"]
            - target_move
        ),
        "qty": qty,
        "win_rate": stats[
            "overall_win_rate"
        ],
        "entered": datetime.now(
            NY
        ).isoformat(),
    }

    order_record = {
        "time": datetime.now(
            NY
        ).isoformat(),
        "symbol": symbol,
        "direction": signal[
            "signal"
        ],
        "win_rate": stats[
            "overall_win_rate"
        ],
        "option": option_symbol,
        "qty": qty,
        "option_mid": mid,
        "result": result,
    }

    with state_lock:

        BOT_STATE[
            "orders"
        ].append(
            order_record
        )

        BOT_STATE[
            "orders"
        ] = BOT_STATE[
            "orders"
        ][-50:]

    logging.info(
        "✅ ORDER %s | %s | "
        "%.1f%% | %s x%s",
        symbol,
        signal["signal"],
        stats[
            "overall_win_rate"
        ],
        option_symbol,
        qty,
    )

    return result


# ============================================================
# MONITOR LOCKED WATCHLIST
# ============================================================

def monitor_watchlist():

    with state_lock:

        watchlist = BOT_STATE[
            "daily_watchlist"
        ].copy()

    if not watchlist:
        return

    new_trades = 0

    for symbol in watchlist:

        if (
            new_trades
            >= MAX_NEW_TRADES_PER_CYCLE
        ):
            break

        try:

            signal = get_current_signal(
                symbol
            )

            if not signal:
                continue

            signal_key = (
                f"{symbol}:"
                f"{signal['bar']}:"
                f"{signal['signal']}"
            )

            if (
                last_signal_bar.get(
                    symbol
                )
                == signal_key
            ):
                continue

            last_signal_bar[
                symbol
            ] = signal_key

            with state_lock:

                BOT_STATE[
                    "current_signals"
                ][symbol] = signal

            logging.info(
                "SIGNAL %s %s",
                symbol,
                signal["signal"],
            )

            result = execute_signal(
                symbol,
                signal,
            )

            if result is not None:
                new_trades += 1

        except Exception as exc:

            logging.warning(
                "%s monitor error: %s",
                symbol,
                exc,
            )

        time.sleep(0.15)


# ============================================================
# DETERMINE WHETHER MORNING SCAN IS DUE
# ============================================================

def morning_scan_due():

    now = datetime.now(NY)

    if now.weekday() >= 5:
        return False

    today = (
        now.date()
        .isoformat()
    )

    with state_lock:

        already_scanned = (
            BOT_STATE[
                "scan_date"
            ]
            == today
        )

    if already_scanned:
        return False

    scan_time = dt_time(
        SCAN_HOUR,
        SCAN_MINUTE,
    )

    # Once 8:45 arrives, build the list.
    # If Render starts late, it still builds
    # today's list once.
    return (
        now.time()
        >= scan_time
    )


# ============================================================
# MAIN LOOP
# ============================================================

def bot_loop():

    logging.info(
        "======================================"
    )

    logging.info(
        "90%% PREMARKET WATCHLIST BOT STARTED"
    )

    logging.info(
        "AUTO_TRADE = %s",
        AUTO_TRADE,
    )

    logging.info(
        "Morning scan = %02d:%02d ET",
        SCAN_HOUR,
        SCAN_MINUTE,
    )

    logging.info(
        "Minimum win rate = %.1f%%",
        MIN_OVERALL_WIN_RATE,
    )

    logging.info(
        "======================================"
    )

    while True:

        try:

            now = datetime.now(NY)

            with state_lock:

                BOT_STATE[
                    "last_cycle"
                ] = now.isoformat()

                BOT_STATE[
                    "last_error"
                ] = None

            # --------------------------------
            # MORNING SCAN
            # --------------------------------

            if morning_scan_due():

                build_daily_watchlist()

            # --------------------------------
            # MARKET HOURS
            # --------------------------------

            if (
                now.weekday() < 5
                and now.time()
                >= RTH_START
                and now.time()
                <= LAST_ENTRY_TIME
            ):

                with state_lock:

                    today_ready = (
                        BOT_STATE[
                            "scan_date"
                        ]
                        == now.date()
                        .isoformat()
                    )

                # Only trade after today's
                # morning list exists.
                if today_ready:

                    monitor_watchlist()

            # --------------------------------
            # AFTER HOURS
            # --------------------------------

            if (
                now.time()
                > RTH_END
            ):

                with state_lock:
                    BOT_STATE[
                        "status"
                    ] = "market_closed"

        except Exception as exc:

            logging.exception(
                "BOT LOOP ERROR"
            )

            with state_lock:

                BOT_STATE[
                    "last_error"
                ] = str(exc)

        time.sleep(
            LOOP_SECONDS
        )


# ============================================================
# WEB ROUTES
# ============================================================

@app.route("/")
def home():

    with state_lock:

        return jsonify({
            "bot": (
                "90% Premarket "
                "Locked Watchlist Bot"
            ),
            "status": BOT_STATE[
                "status"
            ],
            "auto_trade": AUTO_TRADE,
            "minimum_win_rate": (
                MIN_OVERALL_WIN_RATE
            ),
            "scan_time_et": (
                f"{SCAN_HOUR:02d}:"
                f"{SCAN_MINUTE:02d}"
            ),
            "scan_date": BOT_STATE[
                "scan_date"
            ],
            "watchlist_count": len(
                BOT_STATE[
                    "daily_watchlist"
                ]
            ),
            "daily_watchlist": (
                BOT_STATE[
                    "daily_watchlist"
                ]
            ),
        })


@app.route("/watchlist")
def watchlist():

    with state_lock:

        return jsonify({
            "scan_date": BOT_STATE[
                "scan_date"
            ],
            "scan_started": BOT_STATE[
                "scan_started"
            ],
            "scan_finished": BOT_STATE[
                "scan_finished"
            ],
            "locked": True,
            "minimum_win_rate": (
                MIN_OVERALL_WIN_RATE
            ),
            "symbols": BOT_STATE[
                "daily_watchlist"
            ],
            "details": BOT_STATE[
                "qualified_details"
            ],
        })


@app.route("/signals")
def signals():

    with state_lock:

        return jsonify(
            BOT_STATE[
                "current_signals"
            ]
        )


@app.route("/orders")
def orders():

    with state_lock:

        return jsonify(
            BOT_STATE[
                "orders"
            ]
        )


@app.route("/health")
def health():

    try:

        account = get_account()

        return jsonify({
            "status": "healthy",
            "alpaca": True,
            "account_status": (
                account.get(
                    "status"
                )
            ),
            "auto_trade": (
                AUTO_TRADE
            ),
        })

    except Exception as exc:

        return jsonify({
            "status": "error",
            "alpaca": False,
            "error": str(exc),
        }), 500


# ============================================================
# START BOT THREAD
# ============================================================

if RUN_BOT_LOOP:

    thread = threading.Thread(
        target=bot_loop,
        daemon=True,
    )

    thread.start()


# ============================================================
# FLASK START
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