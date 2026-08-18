import os
import time
import threading
import requests
import pandas as pd
import numpy as np

from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from flask import Flask, jsonify


# ============================================================
# APP / CONFIG
# ============================================================

app = Flask(__name__)
NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

TRADING_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

TIMEFRAME_MINUTES = 4
DATA_FEED = os.getenv("DATA_FEED", "iex").strip().lower()
OPTION_FEED = os.getenv("OPTION_FEED", "opra").strip().lower()

AUTO_TRADE = os.getenv("AUTO_TRADE", "false").strip().lower() == "true"
RUN_BOT_LOOP = os.getenv("RUN_BOT_LOOP", "true").strip().lower() == "true"

MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
MAX_NEW_TRADES_PER_CYCLE = int(os.getenv("MAX_NEW_TRADES_PER_CYCLE", "1"))
POSITION_DOLLARS = float(os.getenv("POSITION_DOLLARS", "500"))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.20"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "0.30"))
TAKE_PROFIT_FRACTION = float(os.getenv("TAKE_PROFIT_FRACTION", "0.50"))
RUNNER_TRAIL_PERCENT = float(os.getenv("RUNNER_TRAIL_PERCENT", "0.15"))

LAST_ENTRY_TIME = dt_time(14, 45)
FORCE_EXIT_TIME = dt_time(15, 15)

EMA_FAST = 5
EMA_SLOW = 9
EMA_TREND = 30

ATR_LENGTH = 14
RETEST_ATR_TOLERANCE = 0.20
MAX_RETEST_BARS = 8

PREMARKET_START = dt_time(4, 0)
PREMARKET_END = dt_time(9, 30)

RTH_START = dt_time(9, 30)
RTH_END = dt_time(16, 0)

# ============================================================
# STRICT STATS RULES
# ============================================================

# The bot MUST have exactly 64 completed historical option trades
# before a ticker can become READY.
ROLLING_TRADE_COUNT = 64
MIN_STATS_TRADES = 64

# 80% minimum win rate.
MIN_WIN_RATE = float(os.getenv("MIN_WIN_RATE", "0.80"))

SCAN_LIMIT = int(os.getenv("SCAN_LIMIT", "150"))
LOOP_SECONDS = int(os.getenv("LOOP_SECONDS", "45"))

PREMARKET_REFRESH_SECONDS = int(
    os.getenv("PREMARKET_REFRESH_SECONDS", "600")
)

PREMARKET_WATCHLIST_SIZE = int(
    os.getenv("PREMARKET_WATCHLIST_SIZE", "25")
)

# Use more history so the bot has a better chance of finding
# 64 completed qualifying signals.
BACKTEST_LOOKBACK_DAYS = int(
    os.getenv("BACKTEST_LOOKBACK_DAYS", "180")
)

PRIORITY_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "AAPL",
    "NVDA",
    "TSLA",
    "AMD",
    "AMZN",
    "META",
    "MSFT",
    "GOOGL",
    "NFLX",
    "AVGO",
    "PLTR",
    "COIN",
    "MSTR",
    "AMAT",
]


# ============================================================
# CREDENTIALS
# ============================================================

def clean_credential(value):

    if value is None:
        return ""

    value = str(value).strip()

    return (
        value
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
        .replace("\xa0", "")
        .encode(
            "ascii",
            errors="ignore",
        )
        .decode("ascii")
        .strip()
    )


ALPACA_API_KEY = clean_credential(
    os.getenv("ALPACA_API_KEY", "")
)

ALPACA_SECRET_KEY = clean_credential(
    os.getenv("ALPACA_SECRET_KEY", "")
)

HEADERS = {
    "APCA-API-KEY-ID":
        ALPACA_API_KEY,

    "APCA-API-SECRET-KEY":
        ALPACA_SECRET_KEY,

    "Content-Type":
        "application/json",
}


# ============================================================
# STATE
# ============================================================

bot_state = {

    "running":
        False,

    "credentials_ok":
        False,

    "market_open":
        False,

    "last_cycle":
        None,

    "last_scan":
        None,

    "stocks_scanned_this_cycle":
        0,

    "signals":
        [],

    "premarket_watchlist":
        [],

    "premarket_last_run":
        None,

    "stats_by_symbol":
        {},

    "backtest_last_run":
        None,

    "errors":
        [],
}

managed_positions = {}

_universe_cache = {
    "symbols":
        [],

    "loaded_at":
        None,
}

_daily_state = {
    "stats_date":
        None,

    "premarket_epoch":
        0,
}

_contract_cache = {}


# ============================================================
# HELPERS
# ============================================================

def now_et():

    return datetime.now(NY)


def safe_text(value):

    try:

        return (
            str(value)
            .encode(
                "ascii",
                errors="replace",
            )
            .decode("ascii")
        )

    except Exception:

        return "Unknown error"


def log(message):

    stamp = now_et().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        f"[{stamp} ET] "
        f"{safe_text(message)}",
        flush=True,
    )


def add_error(message):

    message = safe_text(
        message
    )

    bot_state[
        "errors"
    ].append(
        message
    )

    bot_state[
        "errors"
    ] = bot_state[
        "errors"
    ][-25:]

    log(
        f"ERROR: {message}"
    )


def in_time_window(
    value,
    start,
    end,
):

    return (
        start
        <= value
        < end
    )


# ============================================================
# ALPACA REQUESTS
# ============================================================

def alpaca_get(
    path,
    params=None,
    data_api=False,
):

    base = (
        DATA_BASE_URL
        if data_api
        else TRADING_BASE_URL
    )

    response = requests.get(

        f"{base}{path}",

        headers=HEADERS,

        params=params,

        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def alpaca_post(
    path,
    payload,
):

    response = requests.post(

        f"{TRADING_BASE_URL}{path}",

        headers=HEADERS,

        json=payload,

        timeout=30,
    )

    if not response.ok:

        raise RuntimeError(

            f"Alpaca POST {path} "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    if response.text:

        return response.json()

    return {}


# ============================================================
# VERIFY ACCOUNT
# ============================================================

def verify_credentials():

    if (
        not ALPACA_API_KEY
        or not ALPACA_SECRET_KEY
    ):

        bot_state[
            "credentials_ok"
        ] = False

        add_error(
            "Alpaca credentials are missing."
        )

        return False

    try:

        account = alpaca_get(
            "/v2/account"
        )

        bot_state[
            "credentials_ok"
        ] = True

        log(
            "ALPACA PAPER CONNECTED | "
            f'equity=${account.get("equity")}'
        )

        return True

    except Exception as e:

        bot_state[
            "credentials_ok"
        ] = False

        add_error(
            "Credential verification failed: "
            f"{safe_text(e)}"
        )

        return False


# ============================================================
# MARKET CLOCK
# ============================================================

def market_is_open():

    try:

        clock = alpaca_get(
            "/v2/clock"
        )

        bot_state[
            "market_open"
        ] = bool(
            clock.get(
                "is_open",
                False,
            )
        )

        return bot_state[
            "market_open"
        ]

    except Exception as e:

        bot_state[
            "market_open"
        ] = False

        add_error(
            "Market clock error: "
            f"{safe_text(e)}"
        )

        return False


# ============================================================
# STOCK UNIVERSE
# ============================================================

def get_stock_universe(
    force=False,
):

    loaded_at = (
        _universe_cache[
            "loaded_at"
        ]
    )

    if (

        not force

        and _universe_cache[
            "symbols"
        ]

        and loaded_at

        and (
            now_et()
            - loaded_at
        ).total_seconds()
        < 6 * 3600

    ):

        return (
            _universe_cache[
                "symbols"
            ]
        )

    symbols = []

    try:

        assets = alpaca_get(

            "/v2/assets",

            params={
                "status":
                    "active",

                "asset_class":
                    "us_equity",
            },
        )

        for asset in assets:

            symbol = asset.get(
                "symbol"
            )

            if (

                symbol

                and asset.get(
                    "tradable",
                    False,
                )

                and "." not in symbol

            ):

                symbols.append(
                    symbol
                )

    except Exception as e:

        add_error(
            "Universe load error: "
            f"{safe_text(e)}"
        )

    symbols = list(

        dict.fromkeys(

            PRIORITY_SYMBOLS
            + symbols

        )
    )

    _universe_cache[
        "symbols"
    ] = symbols

    _universe_cache[
        "loaded_at"
    ] = now_et()

    return symbols


# ============================================================
# STOCK DATA
# ============================================================

def bars_to_df(bars):

    if not bars:

        return None

    df = pd.DataFrame(
        bars
    )

    if df.empty:

        return None

    df[
        "timestamp"
    ] = (

        pd.to_datetime(
            df["t"],
            utc=True,
        )

        .dt

        .tz_convert(NY)
    )

    df[
        "open"
    ] = pd.to_numeric(
        df["o"],
        errors="coerce",
    )

    df[
        "high"
    ] = pd.to_numeric(
        df["h"],
        errors="coerce",
    )

    df[
        "low"
    ] = pd.to_numeric(
        df["l"],
        errors="coerce",
    )

    df[
        "close"
    ] = pd.to_numeric(
        df["c"],
        errors="coerce",
    )

    df[
        "volume"
    ] = pd.to_numeric(
        df["v"],
        errors="coerce",
    )

    return (

        df[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]

        .dropna()

        .sort_values(
            "timestamp"
        )

        .reset_index(
            drop=True
        )
    )


def get_recent_bars(
    symbol,
    limit=1000,
):

    try:

        data = alpaca_get(

            f"/v2/stocks/"
            f"{symbol}/bars",

            params={
                "timeframe":
                    f"{TIMEFRAME_MINUTES}Min",

                "limit":
                    limit,

                "adjustment":
                    "raw",

                "feed":
                    DATA_FEED,
            },

            data_api=True,
        )

        return bars_to_df(

            data.get(
                "bars",
                [],
            )
        )

    except Exception:

        return None


def get_historical_bars(
    symbol,
    days=180,
):

    end = (
        now_et()
        .astimezone(UTC)
    )

    start = (
        end
        - timedelta(
            days=days
        )
    )

    all_bars = []

    page_token = None

    for _ in range(30):

        params = {

            "timeframe":
                f"{TIMEFRAME_MINUTES}Min",

            "start":
                start.isoformat(),

            "end":
                end.isoformat(),

            "limit":
                10000,

            "adjustment":
                "raw",

            "feed":
                DATA_FEED,
        }

        if page_token:

            params[
                "page_token"
            ] = page_token

        try:

            data = alpaca_get(

                f"/v2/stocks/"
                f"{symbol}/bars",

                params=params,

                data_api=True,
            )

        except Exception:

            break

        all_bars.extend(

            data.get(
                "bars",
                [],
            )
        )

        page_token = (
            data.get(
                "next_page_token"
            )
        )

        if not page_token:

            break

    return bars_to_df(
        all_bars
    )


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    if (
        df is None
        or len(df) < 40
    ):

        return None

    df = (

        df.copy()

        .sort_values(
            "timestamp"
        )

        .reset_index(
            drop=True
        )
    )

    df[
        "ema5"
    ] = (

        df["close"]

        .ewm(
            span=EMA_FAST,
            adjust=False,
        )

        .mean()
    )

    df[
        "ema9"
    ] = (

        df["close"]

        .ewm(
            span=EMA_SLOW,
            adjust=False,
        )

        .mean()
    )

    df[
        "ema30"
    ] = (

        df["close"]

        .ewm(
            span=EMA_TREND,
            adjust=False,
        )

        .mean()
    )

    previous_close = (
        df["close"]
        .shift(1)
    )

    true_range = pd.concat(

        [

            df["high"]
            - df["low"],

            (
                df["high"]
                - previous_close
            ).abs(),

            (
                df["low"]
                - previous_close
            ).abs(),
        ],

        axis=1,

    ).max(
        axis=1
    )

    df[
        "atr"
    ] = (

        true_range

        .rolling(
            ATR_LENGTH
        )

        .mean()
    )

    session_date = (

        df["timestamp"]
        .dt
        .date
    )

    typical_price = (

        df["high"]
        + df["low"]
        + df["close"]

    ) / 3.0

    price_volume = (

        typical_price
        * df["volume"]
    )

    df[
        "vwap"
    ] = (

        price_volume

        .groupby(
            session_date
        )

        .cumsum()

        /

        df["volume"]

        .groupby(
            session_date
        )

        .cumsum()

        .replace(
            0,
            np.nan,
        )
    )

    df[
        "pm_high"
    ] = np.nan

    df[
        "pm_low"
    ] = np.nan

    for _, indexes in (

        df.groupby(
            session_date
        ).groups.items()

    ):

        indexes = list(
            indexes
        )

        rows = df.loc[
            indexes
        ]

        times = (
            rows[
                "timestamp"
            ]
            .dt
            .time
        )

        premarket = rows[

            (
                times
                >= PREMARKET_START
            )

            &

            (
                times
                < PREMARKET_END
            )
        ]

        if premarket.empty:

            continue

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

        df.loc[
            indexes,
            "pm_high",
        ] = pm_high

        df.loc[
            indexes,
            "pm_low",
        ] = pm_low

    return df


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signals(
    df,
    symbol,
):

    df = calculate_indicators(
        df
    )

    if df is None:

        return []

    signals = []

    current_day = None

    long_break_index = None
    short_break_index = None

    long_used = False
    short_used = False

    for i in range(
        1,
        len(df),
    ):

        row = df.iloc[i]

        previous = df.iloc[
            i - 1
        ]

        timestamp = row[
            "timestamp"
        ]

        day = timestamp.date()

        current_time = (
            timestamp.time()
        )

        if day != current_day:

            current_day = day

            long_break_index = None
            short_break_index = None

            long_used = False
            short_used = False

        if not in_time_window(
            current_time,
            RTH_START,
            RTH_END,
        ):

            continue

        pm_high = row[
            "pm_high"
        ]

        pm_low = row[
            "pm_low"
        ]

        if (

            pd.isna(
                pm_high
            )

            or pd.isna(
                pm_low
            )

            or pd.isna(
                row["atr"]
            )

        ):

            continue

        bullish_trend = (

            row["ema5"]
            > row["ema9"]
            > row["ema30"]

            and row["close"]
            > row["vwap"]

            and row["close"]
            > row["ema30"]
        )

        bearish_trend = (

            row["ema5"]
            < row["ema9"]
            < row["ema30"]

            and row["close"]
            < row["vwap"]

            and row["close"]
            < row["ema30"]
        )

        # ================================================
        # BREAK ABOVE PREMARKET HIGH
        # ================================================

        if (

            not long_used

            and row["close"]
            > pm_high

            and previous[
                "close"
            ] <= pm_high

        ):

            long_break_index = i

        # ================================================
        # BREAK BELOW PREMARKET LOW
        # ================================================

        if (

            not short_used

            and row["close"]
            < pm_low

            and previous[
                "close"
            ] >= pm_low

        ):

            short_break_index = i

        # ================================================
        # RETEST WINDOW
        # ================================================

        if (

            long_break_index
            is not None

            and (
                i
                - long_break_index
            )
            > MAX_RETEST_BARS

        ):

            long_break_index = None

        if (

            short_break_index
            is not None

            and (
                i
                - short_break_index
            )
            > MAX_RETEST_BARS

        ):

            short_break_index = None

        tolerance = (

            float(
                row["atr"]
            )

            * RETEST_ATR_TOLERANCE
        )

        # ================================================
        # CALL
        # ================================================

        if (

            not long_used

            and long_break_index
            is not None

            and i
            > long_break_index

        ):

            retest_touched = (

                row["low"]
                <= (
                    pm_high
                    + tolerance
                )
            )

            retest_held = (

                row["close"]
                >= pm_high
            )

            bullish_confirmation = (

                row["close"]
                > row["open"]
            )

            if (

                retest_touched

                and retest_held

                and bullish_confirmation

                and bullish_trend

            ):

                signals.append({

                    "symbol":
                        symbol,

                    "side":
                        "CALL",

                    "label":
                        "BUY",

                    "timestamp":
                        timestamp,

                    "entry":
                        float(
                            row[
                                "close"
                            ]
                        ),

                    "pm_level":
                        float(
                            pm_high
                        ),

                    "bar_index":
                        i,
                })

                long_used = True

                long_break_index = None

        # ================================================
        # PUT
        # ================================================

        if (

            not short_used

            and short_break_index
            is not None

            and i
            > short_break_index

        ):

            retest_touched = (

                row["high"]
                >= (
                    pm_low
                    - tolerance
                )
            )

            retest_held = (

                row["close"]
                <= pm_low
            )

            bearish_confirmation = (

                row["close"]
                < row["open"]
            )

            if (

                retest_touched

                and retest_held

                and bearish_confirmation

                and bearish_trend

            ):

                signals.append({

                    "symbol":
                        symbol,

                    "side":
                        "PUT",

                    "label":
                        "SELL",

                    "timestamp":
                        timestamp,

                    "entry":
                        float(
                            row[
                                "close"
                            ]
                        ),

                    "pm_level":
                        float(
                            pm_low
                        ),

                    "bar_index":
                        i,
                })

                short_used = True

                short_break_index = None

    return signals


# ============================================================
# HISTORICAL 0DTE CONTRACTS
# ============================================================

def get_contracts_for_date(
    symbol,
    side,
    expiration_date,
):

    cache_key = (
        symbol,
        side,
        expiration_date,
    )

    if cache_key in _contract_cache:

        return _contract_cache[
            cache_key
        ]

    option_type = (

        "call"

        if side
        == "CALL"

        else "put"
    )

    contracts = []

    # Expired historical contracts are normally inactive.
    # We also try active for completeness.
    for status in (
        "inactive",
        "active",
    ):

        try:

            data = alpaca_get(

                "/v2/options/contracts",

                params={
                    "underlying_symbols":
                        symbol,

                    "expiration_date":
                        expiration_date,

                    "type":
                        option_type,

                    "status":
                        status,

                    "limit":
                        10000,
                },
            )

            contracts.extend(

                data.get(
                    "option_contracts",
                    [],
                )
            )

        except Exception:

            pass

    unique = {}

    for contract in contracts:

        option_symbol = (
            contract.get(
                "symbol"
            )
        )

        if option_symbol:

            unique[
                option_symbol
            ] = contract

    result = list(
        unique.values()
    )

    _contract_cache[
        cache_key
    ] = result

    return result


def choose_historical_atm_contract(
    symbol,
    side,
    stock_price,
    trade_date,
):

    date_string = (
        trade_date.strftime(
            "%Y-%m-%d"
        )
    )

    contracts = (
        get_contracts_for_date(

            symbol,

            side,

            date_string,
        )
    )

    choices = []

    for contract in contracts:

        try:

            strike = float(

                contract.get(
                    "strike_price",
                    0,
                )
            )

            option_symbol = (
                contract.get(
                    "symbol"
                )
            )

            if option_symbol:

                choices.append(

                    (

                        abs(
                            strike
                            - stock_price
                        ),

                        strike,

                        option_symbol,
                    )
                )

        except Exception:

            continue

    if not choices:

        return None

    choices.sort(
        key=lambda item:
            item[0]
    )

    (
        _,
        strike,
        option_symbol,

    ) = choices[0]

    return {

        "symbol":
            option_symbol,

        "strike":
            strike,

        "underlying":
            symbol,

        "side":
            side,
    }


# ============================================================
# HISTORICAL OPTION BARS
# ============================================================

def option_bars_to_df(
    data,
    option_symbol,
):

    bars_root = (
        data.get(
            "bars",
            {},
        )
    )

    if isinstance(
        bars_root,
        dict,
    ):

        bars = (
            bars_root.get(
                option_symbol,
                [],
            )
        )

    elif isinstance(
        bars_root,
        list,
    ):

        bars = bars_root

    else:

        bars = []

    if not bars:

        return None

    df = pd.DataFrame(
        bars
    )

    if df.empty:

        return None

    df[
        "timestamp"
    ] = (

        pd.to_datetime(
            df["t"],
            utc=True,
        )

        .dt

        .tz_convert(NY)
    )

    df[
        "open"
    ] = pd.to_numeric(
        df["o"],
        errors="coerce",
    )

    df[
        "high"
    ] = pd.to_numeric(
        df["h"],
        errors="coerce",
    )

    df[
        "low"
    ] = pd.to_numeric(
        df["l"],
        errors="coerce",
    )

    df[
        "close"
    ] = pd.to_numeric(
        df["c"],
        errors="coerce",
    )

    return (

        df[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
            ]
        ]

        .dropna()

        .sort_values(
            "timestamp"
        )

        .reset_index(
            drop=True
        )
    )


def get_historical_option_bars(
    option_symbol,
    entry_time,
):

    start_et = entry_time

    end_et = datetime.combine(

        entry_time.date(),

        FORCE_EXIT_TIME,

        tzinfo=NY,
    )

    if start_et >= end_et:

        return None

    try:

        data = alpaca_get(

            "/v1beta1/options/bars",

            params={
                "symbols":
                    option_symbol,

                "timeframe":
                    "1Min",

                "start":
                    start_et
                    .astimezone(UTC)
                    .isoformat(),

                "end":
                    end_et
                    .astimezone(UTC)
                    .isoformat(),

                "limit":
                    10000,

                "feed":
                    OPTION_FEED,
            },

            data_api=True,
        )

        return option_bars_to_df(

            data,

            option_symbol,
        )

    except Exception:

        return None


# ============================================================
# HISTORICAL OPTION TRADE
# ============================================================

def evaluate_historical_option_trade(
    signal,
):

    contract = (
        choose_historical_atm_contract(

            signal[
                "symbol"
            ],

            signal[
                "side"
            ],

            signal[
                "entry"
            ],

            signal[
                "timestamp"
            ].date(),
        )
    )

    if not contract:

        return None

    option_df = (
        get_historical_option_bars(

            contract[
                "symbol"
            ],

            signal[
                "timestamp"
            ],
        )
    )

    if (
        option_df is None
        or option_df.empty
    ):

        return None

    option_df = option_df[

        option_df[
            "timestamp"
        ]

        >= signal[
            "timestamp"
        ]

    ].reset_index(
        drop=True
    )

    if option_df.empty:

        return None

    # Use first historical option close available at/after
    # the stock confirmation signal.
    entry_price = float(

        option_df.iloc[
            0
        ][
            "close"
        ]
    )

    if entry_price <= 0:

        return None

    stop_price = (

        entry_price

        * (
            1.0
            - STOP_LOSS_PERCENT
        )
    )

    tp_price = (

        entry_price

        * (
            1.0
            + TAKE_PROFIT_PERCENT
        )
    )

    tp_hit = False

    runner_high = None

    exit_price = entry_price

    result = None

    exit_reason = "END"

    for _, row in (
        option_df.iterrows()
    ):

        high = float(
            row[
                "high"
            ]
        )

        low = float(
            row[
                "low"
            ]
        )

        close = float(
            row[
                "close"
            ]
        )

        # Conservative rule:
        # if stop and TP both fall inside the same one-minute
        # candle before TP, treat stop as happening first.
        if not tp_hit:

            if low <= stop_price:

                exit_price = stop_price

                result = "LOSS"

                exit_reason = "STOP"

                break

            if high >= tp_price:

                tp_hit = True

                runner_high = max(

                    tp_price,

                    high,
                )

                exit_price = tp_price

                result = "WIN"

                exit_reason = "TP"

        else:

            runner_high = max(

                float(
                    runner_high
                ),

                high,
            )

            trailing_price = (

                runner_high

                * (
                    1.0
                    - RUNNER_TRAIL_PERCENT
                )
            )

            if low <= trailing_price:

                exit_price = trailing_price

                exit_reason = (
                    "TP+RUNNER"
                )

                break

        if (
            row[
                "timestamp"
            ].time()
            >= FORCE_EXIT_TIME
        ):

            exit_price = close

            if result is None:

                result = (

                    "WIN"

                    if close
                    > entry_price

                    else "LOSS"
                )

                exit_reason = "TIME"

            break

    if result is None:

        exit_price = float(

            option_df.iloc[
                -1
            ][
                "close"
            ]
        )

        result = (

            "WIN"

            if exit_price
            > entry_price

            else "LOSS"
        )

        exit_reason = "END"

    return {

        "symbol":
            signal[
                "symbol"
            ],

        "side":
            signal[
                "side"
            ],

        "signal_time":
            signal[
                "timestamp"
            ],

        "stock_entry":
            round(
                signal[
                    "entry"
                ],
                4,
            ),

        "option_symbol":
            contract[
                "symbol"
            ],

        "strike":
            contract[
                "strike"
            ],

        "option_entry":
            round(
                entry_price,
                4,
            ),

        "option_exit":
            round(
                exit_price,
                4,
            ),

        "result":
            result,

        "exit_reason":
            exit_reason,
    }


# ============================================================
# STATS
# ============================================================

def empty_stats(
    symbol,
    reason="WAITING",
):

    return {

        "symbol":
            symbol,

        "overall_win_rate":
            0.0,

        "call_win_rate":
            0.0,

        "put_win_rate":
            0.0,

        "call_w_l":
            "0 / 0",

        "put_w_l":
            "0 / 0",

        "call_trades":
            0,

        "put_trades":
            0,

        "wins":
            0,

        "losses":
            0,

        "total_trades":
            0,

        "status":
            "WAITING",

        "qualified":
            False,

        "call_qualified":
            False,

        "put_qualified":
            False,

        "minimum_win_rate":
            round(
                MIN_WIN_RATE
                * 100,
                1,
            ),

        "required_completed_trades":
            ROLLING_TRADE_COUNT,

        "reason":
            reason,

        "trades":
            [],
    }


# ============================================================
# BUILD EXACT 64-TRADE STATS
# ============================================================

def build_symbol_stats(
    symbol,
):

    stock_df = get_historical_bars(

        symbol,

        BACKTEST_LOOKBACK_DAYS,
    )

    if stock_df is None:

        return empty_stats(
            symbol,
            "NO_STOCK_HISTORY",
        )

    signals = generate_signals(

        stock_df,

        symbol,
    )

    if not signals:

        return empty_stats(
            symbol,
            "NO_SIGNALS",
        )

    completed = []

    # Work newest -> oldest.
    # Stop immediately once 64 valid historical
    # option trades have been collected.
    for signal in reversed(
        signals
    ):

        if (
            len(completed)
            >= ROLLING_TRADE_COUNT
        ):

            break

        # Never include today's incomplete session
        # in historical qualification.
        if (

            signal[
                "timestamp"
            ].date()

            >= now_et().date()

        ):

            continue

        trade = (
            evaluate_historical_option_trade(
                signal
            )
        )

        if trade:

            completed.append(
                trade
            )

        # Helps reduce burst API pressure.
        time.sleep(
            0.03
        )

    completed.reverse()

    # ========================================================
    # STRICT: MUST HAVE ALL 64
    # ========================================================

    if (
        len(completed)
        != ROLLING_TRADE_COUNT
    ):

        stats = empty_stats(

            symbol,

            (
                "NEED_64_OPTION_TRADES_"
                f"HAVE_{len(completed)}"
            ),
        )

        stats[
            "total_trades"
        ] = len(
            completed
        )

        stats[
            "trades"
        ] = [

            {

                **trade,

                "signal_time":
                    trade[
                        "signal_time"
                    ].isoformat(),
            }

            for trade
            in completed
        ]

        return stats

    call_trades = [

        trade

        for trade
        in completed

        if trade[
            "side"
        ] == "CALL"
    ]

    put_trades = [

        trade

        for trade
        in completed

        if trade[
            "side"
        ] == "PUT"
    ]

    call_wins = sum(

        trade[
            "result"
        ] == "WIN"

        for trade
        in call_trades
    )

    put_wins = sum(

        trade[
            "result"
        ] == "WIN"

        for trade
        in put_trades
    )

    call_losses = (

        len(
            call_trades
        )

        - call_wins
    )

    put_losses = (

        len(
            put_trades
        )

        - put_wins
    )

    wins = (

        call_wins
        + put_wins
    )

    losses = (

        call_losses
        + put_losses
    )

    total = len(
        completed
    )

    overall_rate = (

        wins
        / total
    )

    call_rate = (

        call_wins
        / len(
            call_trades
        )

        if call_trades

        else 0.0
    )

    put_rate = (

        put_wins
        / len(
            put_trades
        )

        if put_trades

        else 0.0
    )

    # ========================================================
    # 80% OVERALL REQUIRED
    # ========================================================

    qualified = (

        total
        == ROLLING_TRADE_COUNT

        and overall_rate
        >= MIN_WIN_RATE
    )

    # ========================================================
    # CALL SIDE ALSO MUST BE >= 80%
    # ========================================================

    call_qualified = (

        qualified

        and len(
            call_trades
        ) > 0

        and call_rate
        >= MIN_WIN_RATE
    )

    # ========================================================
    # PUT SIDE ALSO MUST BE >= 80%
    # ========================================================

    put_qualified = (

        qualified

        and len(
            put_trades
        ) > 0

        and put_rate
        >= MIN_WIN_RATE
    )

    return {

        "symbol":
            symbol,

        "overall_win_rate":
            round(
                overall_rate
                * 100,
                1,
            ),

        "call_win_rate":
            round(
                call_rate
                * 100,
                1,
            ),

        "put_win_rate":
            round(
                put_rate
                * 100,
                1,
            ),

        "call_w_l":
            f"{call_wins} / "
            f"{call_losses}",

        "put_w_l":
            f"{put_wins} / "
            f"{put_losses}",

        "call_trades":
            len(
                call_trades
            ),

        "put_trades":
            len(
                put_trades
            ),

        "wins":
            wins,

        "losses":
            losses,

        "total_trades":
            total,

        "status":
            (
                "READY"

                if qualified

                else "WAITING"
            ),

        "qualified":
            qualified,

        "call_qualified":
            call_qualified,

        "put_qualified":
            put_qualified,

        "minimum_win_rate":
            round(
                MIN_WIN_RATE
                * 100,
                1,
            ),

        "required_completed_trades":
            ROLLING_TRADE_COUNT,

        "reason":
            (
                "QUALIFIED"

                if qualified

                else "WIN_RATE_BELOW_80"
            ),

        "trades": [

            {

                **trade,

                "signal_time":
                    trade[
                        "signal_time"
                    ].isoformat(),
            }

            for trade
            in completed
        ],
    }


# ============================================================
# REFRESH STATS
# ============================================================

def refresh_priority_stats():

    results = {}

    for symbol in PRIORITY_SYMBOLS:

        try:

            stats = (
                build_symbol_stats(
                    symbol
                )
            )

            results[
                symbol
            ] = stats

            log(

                f'STATS {symbol}: '

                f'{stats["wins"]}/'
                f'{stats["total_trades"]} '
                f'wins | '

                f'overall='
                f'{stats["overall_win_rate"]:.1f}% | '

                f'CALL='
                f'{stats["call_win_rate"]:.1f}% | '

                f'PUT='
                f'{stats["put_win_rate"]:.1f}% | '

                f'{stats["status"]} | '

                f'{stats["reason"]}'
            )

        except Exception as e:

            add_error(

                f"Stats {symbol}: "
                f"{safe_text(e)}"
            )

            results[
                symbol
            ] = empty_stats(

                symbol,

                "BACKTEST_ERROR",
            )

    bot_state[
        "stats_by_symbol"
    ] = results

    bot_state[
        "backtest_last_run"
    ] = now_et().isoformat()

    return results


# ============================================================
# PREMARKET WATCHLIST
# ============================================================

def build_premarket_watchlist():

    rows = []

    for (
        symbol,
        stats,
    ) in bot_state[
        "stats_by_symbol"
    ].items():

        if not stats.get(
            "qualified"
        ):

            continue

        if (
            stats.get(
                "total_trades"
            )
            != ROLLING_TRADE_COUNT
        ):

            continue

        df = calculate_indicators(

            get_recent_bars(

                symbol,

                limit=1000,
            )
        )

        if (
            df is None
            or df.empty
        ):

            continue

        last = df.iloc[
            -1
        ]

        rows.append({

            "symbol":
                symbol,

            "overall_win_rate":
                stats[
                    "overall_win_rate"
                ],

            "call_win_rate":
                stats[
                    "call_win_rate"
                ],

            "put_win_rate":
                stats[
                    "put_win_rate"
                ],

            "total_trades":
                stats[
                    "total_trades"
                ],

            "price":
                round(
                    float(
                        last[
                            "close"
                        ]
                    ),
                    2,
                ),

            "pm_high":
                (
                    None

                    if pd.isna(
                        last[
                            "pm_high"
                        ]
                    )

                    else round(
                        float(
                            last[
                                "pm_high"
                            ]
                        ),
                        2,
                    )
                ),

            "pm_low":
                (
                    None

                    if pd.isna(
                        last[
                            "pm_low"
                        ]
                    )

                    else round(
                        float(
                            last[
                                "pm_low"
                            ]
                        ),
                        2,
                    )
                ),
        })

    rows.sort(

        key=lambda item: (

            item[
                "overall_win_rate"
            ],

            item[
                "total_trades"
            ],
        ),

        reverse=True,
    )

    bot_state[
        "premarket_watchlist"
    ] = rows[
        :PREMARKET_WATCHLIST_SIZE
    ]

    bot_state[
        "premarket_last_run"
    ] = now_et().isoformat()

    log(

        "PREMARKET WATCHLIST READY: "

        f'{len(bot_state["premarket_watchlist"])} '

        "qualified symbols"
    )

    return bot_state[
        "premarket_watchlist"
    ]


# ============================================================
# LIVE SIGNAL
# ============================================================

def latest_live_signal(
    symbol,
):

    stats = (

        bot_state[
            "stats_by_symbol"
        ].get(
            symbol
        )
    )

    if (

        not stats

        or not stats.get(
            "qualified"
        )

    ):

        return None

    # Absolutely require exactly 64.
    if (
        stats.get(
            "total_trades"
        )
        != ROLLING_TRADE_COUNT
    ):

        return None

    df = get_recent_bars(

        symbol,

        limit=1000,
    )

    if df is None:

        return None

    signals = generate_signals(

        df,

        symbol,
    )

    if not signals:

        return None

    signal = signals[
        -1
    ]

    age_seconds = (

        now_et()

        - signal[
            "timestamp"
        ]

    ).total_seconds()

    # Only accept very recent 4-minute signals.
    if (

        age_seconds < 0

        or age_seconds
        > (
            TIMEFRAME_MINUTES
            * 60
            * 2
        )

    ):

        return None

    side = signal[
        "side"
    ]

    if (

        side == "CALL"

        and not stats.get(
            "call_qualified"
        )

    ):

        return None

    if (

        side == "PUT"

        and not stats.get(
            "put_qualified"
        )

    ):

        return None

    side_rate = (

        stats[
            "call_win_rate"
        ]

        if side
        == "CALL"

        else stats[
            "put_win_rate"
        ]
    )

    return {

        "symbol":
            symbol,

        "side":
            side,

        "label":
            (
                "BUY"

                if side
                == "CALL"

                else "SELL"
            ),

        "entry":
            round(
                signal[
                    "entry"
                ],
                2,
            ),

        "timestamp":
            signal[
                "timestamp"
            ].isoformat(),

        "overall_win_rate":
            stats[
                "overall_win_rate"
            ],

        "side_win_rate":
            side_rate,

        "total_trades":
            stats[
                "total_trades"
            ],

        "pm_level":
            round(
                signal[
                    "pm_level"
                ],
                2,
            ),
    }


# ============================================================
# LIVE SCANNER
# ============================================================

def scan_market():

    watch = [

        item[
            "symbol"
        ]

        for item
        in bot_state[
            "premarket_watchlist"
        ]
    ]

    universe = (
        get_stock_universe()
    )

    qualified = [

        symbol

        for (
            symbol,
            stats,
        ) in bot_state[
            "stats_by_symbol"
        ].items()

        if stats.get(
            "qualified"
        )

        and stats.get(
            "total_trades"
        )
        == ROLLING_TRADE_COUNT
    ]

    symbols = list(

        dict.fromkeys(

            PRIORITY_SYMBOLS

            + watch

            + qualified

            + universe
        )

    )[:SCAN_LIMIT]

    signals = []

    scanned = 0

    for symbol in symbols:

        scanned += 1

        # Unknown symbols do NOT get to trade.
        # They must first have a completed 64-trade backtest.
        if (
            symbol
            not in bot_state[
                "stats_by_symbol"
            ]
        ):

            continue

        try:

            signal = (
                latest_live_signal(
                    symbol
                )
            )

            if signal:

                signals.append(
                    signal
                )

        except Exception:

            continue

    signals.sort(

        key=lambda item: (

            item[
                "side_win_rate"
            ],

            item[
                "overall_win_rate"
            ],
        ),

        reverse=True,
    )

    bot_state[
        "signals"
    ] = signals

    bot_state[
        "stocks_scanned_this_cycle"
    ] = scanned

    bot_state[
        "last_scan"
    ] = now_et().isoformat()

    return signals


# ============================================================
# LIVE 0DTE OPTIONS
# ============================================================

def get_live_0dte_contracts(
    symbol,
    side,
):

    option_type = (

        "call"

        if side
        == "CALL"

        else "put"
    )

    try:

        data = alpaca_get(

            "/v2/options/contracts",

            params={
                "underlying_symbols":
                    symbol,

                "expiration_date":
                    now_et().strftime(
                        "%Y-%m-%d"
                    ),

                "type":
                    option_type,

                "status":
                    "active",

                "limit":
                    10000,
            },
        )

        return data.get(
            "option_contracts",
            [],
        )

    except Exception as e:

        add_error(

            f"{symbol} "
            "option chain error: "
            f"{safe_text(e)}"
        )

        return []


def choose_live_atm_0dte(
    symbol,
    side,
    stock_price,
):

    contracts = (
        get_live_0dte_contracts(

            symbol,

            side,
        )
    )

    choices = []

    for contract in contracts:

        try:

            strike = float(

                contract.get(
                    "strike_price",
                    0,
                )
            )

            option_symbol = (
                contract.get(
                    "symbol"
                )
            )

            if option_symbol:

                choices.append(

                    (

                        abs(
                            strike
                            - stock_price
                        ),

                        strike,

                        option_symbol,
                    )
                )

        except Exception:

            continue

    if not choices:

        return None

    choices.sort(
        key=lambda item:
            item[0]
    )

    (
        _,
        strike,
        option_symbol,

    ) = choices[0]

    return {

        "symbol":
            option_symbol,

        "strike":
            strike,

        "underlying":
            symbol,

        "side":
            side,
    }


# ============================================================
# LIVE OPTION QUOTE
# ============================================================

def get_option_quote(
    option_symbol,
):

    try:

        data = alpaca_get(

            "/v1beta1/options/"
            "quotes/latest",

            params={
                "symbols":
                    option_symbol,
            },

            data_api=True,
        )

        quote = (

            data.get(
                "quotes",
                {},
            )

            .get(
                option_symbol
            )
        )

        if not quote:

            return None

        bid = float(

            quote.get(
                "bp",
                0,
            )

            or 0
        )

        ask = float(

            quote.get(
                "ap",
                0,
            )

            or 0
        )

        if (
            bid <= 0
            and ask <= 0
        ):

            return None

        mid = (

            (
                bid
                + ask
            ) / 2

            if (
                bid > 0
                and ask > 0
            )

            else max(
                bid,
                ask,
            )
        )

        return {

            "bid":
                bid,

            "ask":
                ask,

            "mid":
                mid,
        }

    except Exception:

        return None


# ============================================================
# POSITIONS
# ============================================================

def get_positions():

    try:

        return alpaca_get(
            "/v2/positions"
        )

    except Exception as e:

        add_error(
            "Position error: "
            f"{safe_text(e)}"
        )

        return []


def open_option_positions():

    return [

        position

        for position
        in get_positions()

        if (
            "option"
            in str(
                position.get(
                    "asset_class",
                    "",
                )
            ).lower()
        )
    ]


# ============================================================
# ORDERS
# ============================================================

def submit_option_order(
    option_symbol,
    quantity,
    side,
):

    payload = {

        "symbol":
            option_symbol,

        "qty":
            str(
                quantity
            ),

        "side":
            side,

        "type":
            "market",

        "time_in_force":
            "day",
    }

    if not AUTO_TRADE:

        log(

            f"SIGNAL ONLY: "

            f"{side.upper()} "

            f"{quantity} "

            f"{option_symbol}"
        )

        return {

            "signal_only":
                True,

            **payload,
        }

    return alpaca_post(

        "/v2/orders",

        payload,
    )


# ============================================================
# ENTRY
# ============================================================

def enter_signal(signal):

    if (
        now_et().time()
        >= LAST_ENTRY_TIME
    ):

        return False

    if (

        len(
            open_option_positions()
        )

        >= MAX_OPEN_POSITIONS

    ):

        return False

    underlying = (
        signal[
            "symbol"
        ]
    )

    if any(

        trade.get(
            "underlying"
        ) == underlying

        for trade
        in managed_positions.values()

    ):

        return False

    stats = (

        bot_state[
            "stats_by_symbol"
        ].get(
            underlying,
            {},
        )
    )

    # ========================================================
    # FINAL 64-TRADE / 80% CHECK
    # ========================================================

    if (

        stats.get(
            "total_trades"
        )
        != ROLLING_TRADE_COUNT

        or stats.get(
            "overall_win_rate",
            0,
        )
        < (
            MIN_WIN_RATE
            * 100
        )

    ):

        return False

    if (

        signal[
            "side"
        ] == "CALL"

        and stats.get(
            "call_win_rate",
            0,
        )
        < (
            MIN_WIN_RATE
            * 100
        )

    ):

        return False

    if (

        signal[
            "side"
        ] == "PUT"

        and stats.get(
            "put_win_rate",
            0,
        )
        < (
            MIN_WIN_RATE
            * 100
        )

    ):

        return False

    contract = (
        choose_live_atm_0dte(

            underlying,

            signal[
                "side"
            ],

            float(
                signal[
                    "entry"
                ]
            ),
        )
    )

    if not contract:

        return False

    quote = get_option_quote(

        contract[
            "symbol"
        ]
    )

    if (

        not quote

        or quote[
            "mid"
        ] <= 0

    ):

        return False

    premium = float(
        quote[
            "mid"
        ]
    )

    contract_cost = (

        premium
        * 100
    )

    if (
        contract_cost
        > POSITION_DOLLARS
    ):

        log(

            f'SKIP '
            f'{contract["symbol"]}: '

            f'contract cost '
            f'${contract_cost:.2f} '

            f'> '
            f'${POSITION_DOLLARS:.2f}'
        )

        return False

    quantity = int(

        POSITION_DOLLARS

        // contract_cost
    )

    if quantity < 1:

        return False

    order = (
        submit_option_order(

            contract[
                "symbol"
            ],

            quantity,

            "buy",
        )
    )

    managed_positions[
        contract[
            "symbol"
        ]
    ] = {

        "underlying":
            underlying,

        "direction":
            signal[
                "side"
            ],

        "entry_price":
            premium,

        "quantity":
            quantity,

        "tp_hit":
            False,

        "highest_after_tp":
            premium,

        "entry_time":
            now_et().isoformat(),

        "signal_win_rate":
            signal[
                "side_win_rate"
            ],
    }

    log(

        f'ENTRY '
        f'{underlying} '

        f'{signal["side"]} | '

        f'{contract["symbol"]} | '

        f'qty={quantity} | '

        f'64-trade side win rate='
        f'{signal["side_win_rate"]:.1f}%'
    )

    return order


# ============================================================
# CLOSE
# ============================================================

def close_managed(
    option_symbol,
    reason,
):

    trade = (
        managed_positions.get(
            option_symbol
        )
    )

    if not trade:

        return

    quantity = int(

        trade.get(
            "quantity",
            0,
        )
    )

    if quantity > 0:

        submit_option_order(

            option_symbol,

            quantity,

            "sell",
        )

    log(

        f"EXIT "
        f"{option_symbol} | "
        f"{reason}"
    )

    managed_positions.pop(

        option_symbol,

        None,
    )


# ============================================================
# POSITION MANAGEMENT
# ============================================================

def manage_positions():

    for option_symbol in list(

        managed_positions.keys()

    ):

        trade = (

            managed_positions.get(
                option_symbol
            )
        )

        if not trade:

            continue

        # ================================================
        # 3:15 FORCE EXIT
        # ================================================

        if (
            now_et().time()
            >= FORCE_EXIT_TIME
        ):

            close_managed(

                option_symbol,

                "3:15 PM 0DTE FORCE EXIT",
            )

            continue

        quote = get_option_quote(
            option_symbol
        )

        if not quote:

            continue

        current = float(
            quote[
                "mid"
            ]
        )

        entry = float(
            trade[
                "entry_price"
            ]
        )

        quantity = int(
            trade[
                "quantity"
            ]
        )

        if (
            entry <= 0
            or quantity <= 0
        ):

            continue

        pnl_percent = (

            current
            - entry

        ) / entry

        # ================================================
        # 20% HARD STOP
        # ================================================

        if (

            not trade[
                "tp_hit"
            ]

            and pnl_percent
            <= -STOP_LOSS_PERCENT

        ):

            close_managed(

                option_symbol,

                f"HARD STOP "
                f"{pnl_percent:.1%}",
            )

            continue

        # ================================================
        # 30% TP
        # ================================================

        if (

            not trade[
                "tp_hit"
            ]

            and pnl_percent
            >= TAKE_PROFIT_PERCENT

        ):

            if quantity > 1:

                sell_quantity = max(

                    1,

                    int(
                        quantity
                        * TAKE_PROFIT_FRACTION
                    ),
                )

                sell_quantity = min(

                    sell_quantity,

                    quantity - 1,
                )

                if sell_quantity > 0:

                    submit_option_order(

                        option_symbol,

                        sell_quantity,

                        "sell",
                    )

                    trade[
                        "quantity"
                    ] -= sell_quantity

            trade[
                "tp_hit"
            ] = True

            trade[
                "highest_after_tp"
            ] = current

            log(

                f"TP HIT "
                f"{option_symbol} "

                f"{pnl_percent:.1%} | "

                f'runner qty='
                f'{trade["quantity"]}'
            )

            continue

        # ================================================
        # RUNNER
        # ================================================

        if trade[
            "tp_hit"
        ]:

            trade[
                "highest_after_tp"
            ] = max(

                float(
                    trade[
                        "highest_after_tp"
                    ]
                ),

                current,
            )

            trailing_price = (

                float(
                    trade[
                        "highest_after_tp"
                    ]
                )

                * (
                    1
                    - RUNNER_TRAIL_PERCENT
                )
            )

            if (
                current
                <= trailing_price
            ):

                close_managed(

                    option_symbol,

                    "RUNNER TRAIL",
                )


# ============================================================
# DAILY PREP
# ============================================================

def maybe_run_daily_prep():

    current = now_et()

    today = (
        current.date()
        .isoformat()
    )

    current_time = (
        current.time()
    )

    # ================================================
    # RUN OPTION BACKTEST BEFORE MARKET
    # ================================================

    if (

        _daily_state[
            "stats_date"
        ] != today

        and current_time
        < dt_time(
            9,
            25,
        )

    ):

        refresh_priority_stats()

        _daily_state[
            "stats_date"
        ] = today

    # ================================================
    # PREMARKET WATCHLIST
    # ================================================

    if (

        dt_time(
            7,
            0,
        )

        <= current_time

        < dt_time(
            9,
            30,
        )

    ):

        if (

            time.time()

            - _daily_state[
                "premarket_epoch"
            ]

            >= PREMARKET_REFRESH_SECONDS

        ):

            build_premarket_watchlist()

            _daily_state[
                "premarket_epoch"
            ] = time.time()


# ============================================================
# BOT CYCLE
# ============================================================

def bot_cycle():

    bot_state[
        "last_cycle"
    ] = now_et().isoformat()

    if not bot_state[
        "credentials_ok"
    ]:

        if not verify_credentials():

            return

    maybe_run_daily_prep()

    if not market_is_open():

        return

    manage_positions()

    signals = scan_market()

    entered = 0

    for signal in signals:

        if (
            entered
            >= MAX_NEW_TRADES_PER_CYCLE
        ):

            break

        try:

            if enter_signal(
                signal
            ):

                entered += 1

        except Exception as e:

            add_error(

                f'Entry '
                f'{signal.get("symbol")}: '
                f'{safe_text(e)}'
            )


# ============================================================
# BOT LOOP
# ============================================================

def bot_loop():

    bot_state[
        "running"
    ] = True

    log(
        "STRICT 64-TRADE "
        "80% 0DTE BOT STARTED"
    )

    while True:

        try:

            bot_cycle()

        except Exception as e:

            add_error(

                "Bot cycle error: "
                f"{safe_text(e)}"
            )

        time.sleep(
            LOOP_SECONDS
        )


def start_background_bot():

    if not RUN_BOT_LOOP:

        log(
            "RUN_BOT_LOOP disabled."
        )

        return

    thread = threading.Thread(

        target=bot_loop,

        daemon=True,
    )

    thread.start()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    account = {}

    try:

        account = alpaca_get(
            "/v2/account"
        )

    except Exception:

        pass

    return jsonify({

        "status":
            "online",

        "bot":
            "Strict 64-Trade "
            "80% Premarket Retest "
            "0DTE Paper Bot",

        "paper_trading":
            True,

        "credentials_ok":
            bot_state[
                "credentials_ok"
            ],

        "auto_trade":
            AUTO_TRADE,

        "run_bot_loop":
            RUN_BOT_LOOP,

        "market_open":
            bot_state[
                "market_open"
            ],

        "timeframe":
            "4m",

        "signal_model":
            "Premarket breakout -> "
            "retest -> confirmation -> "
            "EMA5/9/30 + VWAP",

        "backtest_model":
            "Historical ATM 0DTE option "
            "premium using Alpaca "
            "option bars",

        "required_completed_trades":
            ROLLING_TRADE_COUNT,

        "minimum_win_rate":
            f"{MIN_WIN_RATE:.0%}",

        "stocks_loaded":
            len(
                get_stock_universe()
            ),

        "stocks_scanned_this_cycle":
            bot_state[
                "stocks_scanned_this_cycle"
            ],

        "account_equity":
            account.get(
                "equity"
            ),

        "buying_power":
            account.get(
                "buying_power"
            ),

        "stop_loss":
            f"{STOP_LOSS_PERCENT:.0%}",

        "take_profit":
            f"{TAKE_PROFIT_PERCENT:.0%}",

        "runner_trail":
            f"{RUNNER_TRAIL_PERCENT:.0%}",

        "entry_cutoff":
            "2:45 PM ET",

        "force_exit":
            "3:15 PM ET",

        "errors":
            bot_state[
                "errors"
            ],
    })


# ============================================================
# STATUS
# ============================================================

@app.route("/status")
def status():

    return jsonify({

        **bot_state,

        "managed_positions":
            managed_positions,
    })


# ============================================================
# SYMBOL STATS
# ============================================================

@app.route(
    "/stats/<symbol>"
)
def symbol_stats(symbol):

    symbol = (
        symbol
        .upper()
        .strip()
    )

    try:

        stats = (
            build_symbol_stats(
                symbol
            )
        )

        bot_state[
            "stats_by_symbol"
        ][symbol] = stats

        return jsonify(
            stats
        )

    except Exception as e:

        return jsonify({

            "error":
                safe_text(e)

        }), 500


# ============================================================
# BACKTEST
# ============================================================

@app.route("/backtest")
def backtest():

    try:

        return jsonify(
            refresh_priority_stats()
        )

    except Exception as e:

        return jsonify({

            "error":
                safe_text(e)

        }), 500


# ============================================================
# PREMARKET
# ============================================================

@app.route("/premarket")
def premarket():

    try:

        return jsonify(
            build_premarket_watchlist()
        )

    except Exception as e:

        return jsonify({

            "error":
                safe_text(e)

        }), 500


# ============================================================
# SCAN
# ============================================================

@app.route("/scan")
def scan():

    try:

        return jsonify(
            scan_market()
        )

    except Exception as e:

        return jsonify({

            "error":
                safe_text(e)

        }), 500


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "healthy",

        "bot_running":
            bot_state[
                "running"
            ],

        "time":
            now_et().isoformat(),
    })


# ============================================================
# START
# ============================================================

verify_credentials()

start_background_bot()


if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "10000",
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False,
    )