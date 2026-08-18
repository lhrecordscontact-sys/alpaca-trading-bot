import os
import time
import math
import threading
import requests
import pandas as pd
import numpy as np

from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from flask import Flask, jsonify


# ============================================================
# CONFIG
# ============================================================

app = Flask(__name__)

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

TRADING_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

TIMEFRAME_MINUTES = 4

# Stock data
DATA_FEED = os.getenv(
    "DATA_FEED",
    "iex",
).strip().lower()

# Free Alpaca option data feed.
# Change to "opra" only if your account has OPRA access.
OPTION_FEED = os.getenv(
    "OPTION_FEED",
    "indicative",
).strip().lower()

AUTO_TRADE = (
    os.getenv(
        "AUTO_TRADE",
        "false",
    ).strip().lower()
    == "true"
)

RUN_BOT_LOOP = (
    os.getenv(
        "RUN_BOT_LOOP",
        "true",
    ).strip().lower()
    == "true"
)

LOOP_SECONDS = int(
    os.getenv(
        "LOOP_SECONDS",
        "45",
    )
)

SCAN_LIMIT = int(
    os.getenv(
        "SCAN_LIMIT",
        "25",
    )
)


# ============================================================
# RISK
# ============================================================

POSITION_DOLLARS = float(
    os.getenv(
        "POSITION_DOLLARS",
        "500",
    )
)

MAX_OPEN_POSITIONS = int(
    os.getenv(
        "MAX_OPEN_POSITIONS",
        "3",
    )
)

MAX_NEW_TRADES_PER_CYCLE = int(
    os.getenv(
        "MAX_NEW_TRADES_PER_CYCLE",
        "1",
    )
)

STOP_LOSS_PERCENT = float(
    os.getenv(
        "STOP_LOSS_PERCENT",
        "0.20",
    )
)

TAKE_PROFIT_PERCENT = float(
    os.getenv(
        "TAKE_PROFIT_PERCENT",
        "0.30",
    )
)

TAKE_PROFIT_FRACTION = float(
    os.getenv(
        "TAKE_PROFIT_FRACTION",
        "0.50",
    )
)

RUNNER_TRAIL_PERCENT = float(
    os.getenv(
        "RUNNER_TRAIL_PERCENT",
        "0.15",
    )
)


# ============================================================
# STRATEGY
# ============================================================

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

LAST_ENTRY_TIME = dt_time(14, 45)

# Force close 0DTE positions well before expiration handling.
FORCE_EXIT_TIME = dt_time(15, 15)


# ============================================================
# SYMBOLS
# ============================================================

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
]


# ============================================================
# CREDENTIAL CLEANING
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
    os.getenv(
        "ALPACA_API_KEY",
        "",
    )
)

ALPACA_SECRET_KEY = clean_credential(
    os.getenv(
        "ALPACA_SECRET_KEY",
        "",
    )
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

state_lock = threading.Lock()

bot_state = {

    "running":
        False,

    "credentials_ok":
        False,

    "options_level":
        0,

    "market_open":
        False,

    "last_cycle":
        None,

    "last_scan":
        None,

    "stocks_scanned":
        0,

    "signals":
        [],

    "managed_positions":
        {},

    "errors":
        [],
}


# Positions opened by THIS running bot.
managed_positions = {}

universe_cache = {

    "symbols":
        [],

    "loaded_at":
        None,
}


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


def to_float(
    value,
    default=None,
):

    try:

        if value is None:
            return default

        return float(value)

    except Exception:

        return default


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

    text = safe_text(message)

    with state_lock:

        bot_state[
            "errors"
        ].append(text)

        bot_state[
            "errors"
        ] = (
            bot_state[
                "errors"
            ][-50:]
        )

    log(
        f"ERROR: {text}"
    )


# ============================================================
# ALPACA HTTP
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

    url = (
        f"{base}{path}"
    )

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=30,
        )

    except requests.RequestException as e:

        raise RuntimeError(
            f"GET {url} "
            f"network error: "
            f"{safe_text(e)}"
        ) from e

    if not response.ok:

        body = (
            safe_text(
                response.text
            )[:1200]
        )

        raise RuntimeError(
            f"GET {path} "
            f"HTTP "
            f"{response.status_code} | "
            f"url="
            f"{safe_text(response.url)} | "
            f"body={body}"
        )

    try:

        return response.json()

    except Exception as e:

        raise RuntimeError(
            f"GET {path} "
            f"invalid JSON: "
            f"{safe_text(e)}"
        ) from e


def alpaca_post(
    path,
    payload,
):

    url = (
        f"{TRADING_BASE_URL}"
        f"{path}"
    )

    try:

        response = requests.post(
            url,
            headers=HEADERS,
            json=payload,
            timeout=30,
        )

    except requests.RequestException as e:

        raise RuntimeError(
            f"POST {url} "
            f"network error: "
            f"{safe_text(e)}"
        ) from e

    if not response.ok:

        body = (
            safe_text(
                response.text
            )[:1200]
        )

        raise RuntimeError(
            f"POST {path} "
            f"HTTP "
            f"{response.status_code} | "
            f"body={body}"
        )

    try:

        return response.json()

    except Exception:

        return {}


# ============================================================
# ACCOUNT
# ============================================================

def verify_credentials():

    if (
        not ALPACA_API_KEY
        or not ALPACA_SECRET_KEY
    ):

        add_error(
            "Alpaca credentials "
            "are missing."
        )

        return False

    try:

        account = alpaca_get(
            "/v2/account"
        )

        options_level = int(
            account.get(
                "options_trading_level"
            )
            or account.get(
                "options_approved_level"
            )
            or 0
        )

        with state_lock:

            bot_state[
                "credentials_ok"
            ] = True

            bot_state[
                "options_level"
            ] = options_level

        log(
            "ALPACA PAPER CONNECTED | "
            f"equity=$"
            f"{account.get('equity')} | "
            f"options_level="
            f"{options_level}"
        )

        if options_level < 2:

            add_error(
                "OPTIONS LEVEL BELOW 2 | "
                "long calls/puts "
                "cannot be opened"
            )

        return True

    except Exception as e:

        with state_lock:

            bot_state[
                "credentials_ok"
            ] = False

        add_error(
            "Credential verification "
            f"failed: {safe_text(e)}"
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

        opened = bool(
            clock.get(
                "is_open",
                False,
            )
        )

        with state_lock:

            bot_state[
                "market_open"
            ] = opened

        return opened

    except Exception as e:

        with state_lock:

            bot_state[
                "market_open"
            ] = False

        add_error(
            "MARKET CLOCK ERROR | "
            f"{safe_text(e)}"
        )

        return False


# ============================================================
# STOCK UNIVERSE
# ============================================================

def get_stock_universe():

    loaded_at = universe_cache[
        "loaded_at"
    ]

    if (
        universe_cache[
            "symbols"
        ]
        and loaded_at
        and (
            now_et()
            - loaded_at
        ).total_seconds()
        < 21600
    ):

        return universe_cache[
            "symbols"
        ]

    symbols = list(
        PRIORITY_SYMBOLS
    )

    try:

        assets = alpaca_get(
            "/v2/assets",
            params={
                "status":
                    "active",

                "asset_class":
                    "us_equity",

                "attributes":
                    "options_enabled",
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
                and "."
                not in symbol
            ):

                symbols.append(
                    symbol
                )

    except Exception as e:

        add_error(
            "UNIVERSE ERROR | "
            f"{safe_text(e)}"
        )

    symbols = list(
        dict.fromkeys(
            symbols
        )
    )

    universe_cache[
        "symbols"
    ] = symbols

    universe_cache[
        "loaded_at"
    ] = now_et()

    return symbols


# ============================================================
# STOCK BARS
# ============================================================

def bars_to_df(bars):

    if not bars:

        return None

    df = pd.DataFrame(
        bars
    )

    if df.empty:

        return None

    required = {
        "t",
        "o",
        "h",
        "l",
        "c",
        "v",
    }

    if not required.issubset(
        df.columns
    ):

        raise RuntimeError(
            "Unexpected bar fields: "
            f"{list(df.columns)}"
        )

    df[
        "timestamp"
    ] = (
        pd.to_datetime(
            df["t"],
            utc=True,
        )
        .dt.tz_convert(NY)
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


# ============================================================
# RECENT STOCK BARS
# ============================================================

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
                    f"{TIMEFRAME_MINUTES}"
                    f"Min",

                "limit":
                    limit,

                "adjustment":
                    "raw",

                "feed":
                    DATA_FEED,
            },

            data_api=True,
        )

        bars = data.get(
            "bars",
            [],
        )

        if not bars:

            add_error(
                "RECENT DATA EMPTY "
                f"{symbol} | "
                f"feed={DATA_FEED}"
            )

            return None

        return bars_to_df(
            bars
        )

    except Exception as e:

        add_error(
            "RECENT DATA ERROR "
            f"{symbol} | "
            f"feed={DATA_FEED} | "
            f"{safe_text(e)}"
        )

        return None


# ============================================================
# HISTORICAL STOCK BARS
# ============================================================

def get_historical_bars(
    symbol,
    days=30,
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

    for page_number in range(
        1,
        31,
    ):

        params = {

            "timeframe":
                f"{TIMEFRAME_MINUTES}"
                f"Min",

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

            "sort":
                "asc",
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

        except Exception as e:

            add_error(
                "HISTORICAL DATA ERROR "
                f"{symbol} | "
                f"feed={DATA_FEED} | "
                f"page={page_number} | "
                f"{safe_text(e)}"
            )

            break

        bars = data.get(
            "bars",
            [],
        )

        if (
            not bars
            and not all_bars
        ):

            add_error(
                "HISTORICAL DATA EMPTY "
                f"{symbol} | "
                f"feed={DATA_FEED}"
            )

            break

        all_bars.extend(
            bars
        )

        page_token = data.get(
            "next_page_token"
        )

        if not page_token:

            break

    if not all_bars:

        return None

    df = bars_to_df(
        all_bars
    )

    if df is not None:

        log(
            f"HISTORY {symbol}: "
            f"{len(df)} bars loaded | "
            f"feed={DATA_FEED}"
        )

    return df


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
        df[
            "close"
        ]
        .ewm(
            span=EMA_FAST,
            adjust=False,
        )
        .mean()
    )

    df[
        "ema9"
    ] = (
        df[
            "close"
        ]
        .ewm(
            span=EMA_SLOW,
            adjust=False,
        )
        .mean()
    )

    df[
        "ema30"
    ] = (
        df[
            "close"
        ]
        .ewm(
            span=EMA_TREND,
            adjust=False,
        )
        .mean()
    )

    previous_close = (
        df[
            "close"
        ].shift(1)
    )

    true_range = pd.concat(

        [

            df[
                "high"
            ]
            - df[
                "low"
            ],

            (
                df[
                    "high"
                ]
                - previous_close
            ).abs(),

            (
                df[
                    "low"
                ]
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
        df[
            "timestamp"
        ].dt.date
    )

    typical_price = (

        df[
            "high"
        ]

        + df[
            "low"
        ]

        + df[
            "close"
        ]

    ) / 3.0

    price_volume = (
        typical_price
        * df[
            "volume"
        ]
    )

    cumulative_volume = (

        df[
            "volume"
        ]

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
        "vwap"
    ] = (

        price_volume
        .groupby(
            session_date
        )
        .cumsum()

        / cumulative_volume
    )

    df[
        "pm_high"
    ] = np.nan

    df[
        "pm_low"
    ] = np.nan

    groups = df.groupby(
        session_date
    ).groups

    for indexes in groups.values():

        indexes = list(
            indexes
        )

        rows = df.loc[
            indexes
        ]

        times = rows[
            "timestamp"
        ].dt.time

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

        if not (
            RTH_START
            <= current_time
            < RTH_END
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
                row[
                    "atr"
                ]
            )
            or pd.isna(
                row[
                    "vwap"
                ]
            )
        ):

            continue

        bullish = (

            row[
                "ema5"
            ]
            > row[
                "ema9"
            ]
            > row[
                "ema30"
            ]

            and row[
                "close"
            ]
            > row[
                "vwap"
            ]

            and row[
                "close"
            ]
            > row[
                "ema30"
            ]
        )

        bearish = (

            row[
                "ema5"
            ]
            < row[
                "ema9"
            ]
            < row[
                "ema30"
            ]

            and row[
                "close"
            ]
            < row[
                "vwap"
            ]

            and row[
                "close"
            ]
            < row[
                "ema30"
            ]
        )

        if (

            not long_used

            and row[
                "close"
            ]
            > pm_high

            and previous[
                "close"
            ]
            <= pm_high
        ):

            long_break_index = i

        if (

            not short_used

            and row[
                "close"
            ]
            < pm_low

            and previous[
                "close"
            ]
            >= pm_low
        ):

            short_break_index = i

        if (

            long_break_index
            is not None

            and i
            - long_break_index
            > MAX_RETEST_BARS
        ):

            long_break_index = None

        if (

            short_break_index
            is not None

            and i
            - short_break_index
            > MAX_RETEST_BARS
        ):

            short_break_index = None

        tolerance = (

            float(
                row[
                    "atr"
                ]
            )

            * RETEST_ATR_TOLERANCE
        )

        # CALL
        if (

            long_break_index
            is not None

            and i
            > long_break_index

            and not long_used

            and bullish

            and row[
                "low"
            ]
            <= (
                pm_high
                + tolerance
            )

            and row[
                "close"
            ]
            > pm_high
        ):

            signals.append(
                {

                    "symbol":
                        symbol,

                    "side":
                        "CALL",

                    "timestamp":
                        timestamp,

                    "underlying_entry":
                        float(
                            row[
                                "close"
                            ]
                        ),

                    "pm_level":
                        float(
                            pm_high
                        ),
                }
            )

            long_used = True

            long_break_index = None

        # PUT
        if (

            short_break_index
            is not None

            and i
            > short_break_index

            and not short_used

            and bearish

            and row[
                "high"
            ]
            >= (
                pm_low
                - tolerance
            )

            and row[
                "close"
            ]
            < pm_low
        ):

            signals.append(
                {

                    "symbol":
                        symbol,

                    "side":
                        "PUT",

                    "timestamp":
                        timestamp,

                    "underlying_entry":
                        float(
                            row[
                                "close"
                            ]
                        ),

                    "pm_level":
                        float(
                            pm_low
                        ),
                }
            )

            short_used = True

            short_break_index = None

    return signals


# ============================================================
# LATEST SIGNAL
# ============================================================

def latest_live_signal(
    symbol,
):

    df = get_recent_bars(
        symbol,
        limit=1000,
    )

    if (
        df is None
        or len(df) < 50
    ):

        return None

    signals = generate_signals(
        df,
        symbol,
    )

    if not signals:

        return None

    signal = signals[-1]

    age = (
        now_et()
        - signal[
            "timestamp"
        ]
    )

    max_age = (
        TIMEFRAME_MINUTES
        * 60
        * 2
    )

    if (
        age.total_seconds()
        > max_age
    ):

        return None

    return signal


# ============================================================
# GET 0DTE CONTRACTS
# ============================================================

def get_0dte_contracts(
    underlying,
    side,
):

    today = (
        now_et()
        .date()
        .isoformat()
    )

    option_type = (
        "call"
        if side == "CALL"
        else "put"
    )

    contracts = []

    page_token = None

    for _ in range(10):

        params = {

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
        }

        if page_token:

            params[
                "page_token"
            ] = page_token

        data = alpaca_get(

            "/v2/options/contracts",

            params=params,

            data_api=False,
        )

        contracts.extend(
            data.get(
                "option_contracts",
                [],
            )
        )

        page_token = data.get(
            "next_page_token"
        )

        if not page_token:

            break

    return [

        contract

        for contract
        in contracts

        if contract.get(
            "tradable",
            True,
        )
    ]


# ============================================================
# OPTION QUOTE
# ============================================================

def get_option_quote(
    option_symbol,
):

    data = alpaca_get(

        "/v1beta1/options/"
        "quotes/latest",

        params={

            "symbols":
                option_symbol,

            "feed":
                OPTION_FEED,
        },

        data_api=True,
    )

    quotes = data.get(
        "quotes",
        {},
    )

    quote = quotes.get(
        option_symbol
    )

    if not quote:

        return None

    bid = to_float(
        quote.get(
            "bp"
        )
    )

    ask = to_float(
        quote.get(
            "ap"
        )
    )

    if (
        bid is None
        or ask is None
    ):

        return None

    if (
        bid <= 0
        or ask <= 0
        or ask < bid
    ):

        return None

    mid = (
        bid
        + ask
    ) / 2.0

    return {

        "bid":
            bid,

        "ask":
            ask,

        "mid":
            mid,

        "spread":
            ask - bid,
    }


# ============================================================
# SELECT ATM 0DTE
# ============================================================

def select_0dte_contract(
    signal,
):

    underlying = signal[
        "symbol"
    ]

    side = signal[
        "side"
    ]

    stock_price = float(
        signal[
            "underlying_entry"
        ]
    )

    contracts = get_0dte_contracts(
        underlying,
        side,
    )

    if not contracts:

        add_error(
            "NO 0DTE CONTRACTS | "
            f"{underlying} {side}"
        )

        return None

    candidates = []

    for contract in contracts:

        strike = to_float(
            contract.get(
                "strike_price"
            )
        )

        option_symbol = (
            contract.get(
                "symbol"
            )
        )

        if (
            strike is None
            or not option_symbol
        ):

            continue

        distance = abs(
            strike
            - stock_price
        )

        candidates.append(
            (
                distance,
                strike,
                option_symbol,
                contract,
            )
        )

    candidates.sort(
        key=lambda item:
            item[0]
    )

    best = None

    # Look at nearest strikes.
    for (
        distance,
        strike,
        option_symbol,
        contract,
    ) in candidates[:12]:

        try:

            quote = (
                get_option_quote(
                    option_symbol
                )
            )

            if not quote:

                continue

            mid = quote[
                "mid"
            ]

            if mid <= 0:

                continue

            spread_percent = (

                quote[
                    "spread"
                ]

                / mid
            )

            # Skip very wide options.
            if (
                spread_percent
                > 0.40
            ):

                continue

            candidate = {

                "symbol":
                    option_symbol,

                "strike":
                    strike,

                "expiration":
                    contract.get(
                        "expiration_date"
                    ),

                "bid":
                    quote[
                        "bid"
                    ],

                "ask":
                    quote[
                        "ask"
                    ],

                "mid":
                    quote[
                        "mid"
                    ],

                "distance":
                    distance,

                "spread_percent":
                    spread_percent,
            }

            if best is None:

                best = candidate

                continue

            old_score = (
                best[
                    "distance"
                ],
                best[
                    "spread_percent"
                ],
            )

            new_score = (
                candidate[
                    "distance"
                ],
                candidate[
                    "spread_percent"
                ],
            )

            if (
                new_score
                < old_score
            ):

                best = candidate

        except Exception as e:

            add_error(
                "OPTION QUOTE ERROR | "
                f"{option_symbol} | "
                f"{safe_text(e)}"
            )

    if best is None:

        add_error(
            "NO USABLE 0DTE QUOTE | "
            f"{underlying} {side}"
        )

    return best


# ============================================================
# POSITIONS
# ============================================================

def get_open_positions():

    try:

        return alpaca_get(
            "/v2/positions"
        )

    except Exception as e:

        add_error(
            "POSITIONS ERROR | "
            f"{safe_text(e)}"
        )

        return []


def count_open_option_positions():

    positions = (
        get_open_positions()
    )

    count = 0

    for position in positions:

        asset_class = str(
            position.get(
                "asset_class",
                "",
            )
        ).lower()

        if (
            "option"
            in asset_class
        ):

            count += 1

    return count


# ============================================================
# OPTION ORDERS
# ============================================================

def submit_option_order(
    option_symbol,
    qty,
    side,
    intent,
):

    payload = {

        "symbol":
            option_symbol,

        "qty":
            str(
                int(qty)
            ),

        "side":
            side,

        "type":
            "market",

        "time_in_force":
            "day",

        "position_intent":
            intent,
    }

    return alpaca_post(
        "/v2/orders",
        payload,
    )


def get_order(
    order_id,
):

    return alpaca_get(
        f"/v2/orders/"
        f"{order_id}"
    )


def wait_for_fill(
    order_id,
    timeout=20,
):

    deadline = (
        time.time()
        + timeout
    )

    last = None

    while (
        time.time()
        < deadline
    ):

        last = get_order(
            order_id
        )

        status = str(
            last.get(
                "status",
                "",
            )
        ).lower()

        if status == "filled":

            return last

        if status in {

            "canceled",
            "expired",
            "rejected",
            "suspended",

        }:

            return last

        time.sleep(1)

    return last


# ============================================================
# CLOSE OPTION
# ============================================================

def close_option(
    option_symbol,
    qty,
    reason,
):

    qty = int(
        qty
    )

    if qty <= 0:

        return None

    try:

        order = (
            submit_option_order(

                option_symbol,

                qty,

                "sell",

                "sell_to_close",
            )
        )

        log(
            "EXIT SUBMITTED | "
            f"{option_symbol} | "
            f"qty={qty} | "
            f"reason={reason}"
        )

        return order

    except Exception as e:

        add_error(
            "EXIT ERROR | "
            f"{option_symbol} | "
            f"{reason} | "
            f"{safe_text(e)}"
        )

        return None


# ============================================================
# SYNC MANAGED POSITIONS
# ============================================================

def sync_managed_positions():

    with state_lock:

        bot_state[
            "managed_positions"
        ] = {

            symbol:
                dict(position)

            for (
                symbol,
                position,
            )

            in managed_positions.items()
        }


# ============================================================
# OPEN TRADE
# ============================================================

def open_signal_trade(
    signal,
):

    if not AUTO_TRADE:

        log(
            "SIGNAL ONLY | "
            f"{signal['symbol']} "
            f"{signal['side']} | "
            "AUTO_TRADE=False"
        )

        return False

    if (
        bot_state[
            "options_level"
        ]
        < 2
    ):

        add_error(
            "TRADE BLOCKED | "
            "options level < 2"
        )

        return False

    if (
        now_et().time()
        >= LAST_ENTRY_TIME
    ):

        return False

    if (
        count_open_option_positions()
        >= MAX_OPEN_POSITIONS
    ):

        log(
            "MAX OPEN POSITIONS "
            "REACHED"
        )

        return False

    # Don't duplicate same underlying.
    for position in (
        managed_positions.values()
    ):

        if (

            position.get(
                "underlying"
            )
            == signal[
                "symbol"
            ]

            and position.get(
                "remaining_qty",
                0,
            )
            > 0
        ):

            return False

    try:

        contract = (
            select_0dte_contract(
                signal
            )
        )

        if not contract:

            return False

        estimated_cost = (

            contract[
                "ask"
            ]

            * 100
        )

        if estimated_cost <= 0:

            return False

        qty = int(
            math.floor(

                POSITION_DOLLARS

                / estimated_cost
            )
        )

        if qty < 1:

            log(
                "CONTRACT TOO EXPENSIVE | "
                f"{contract['symbol']} | "
                f"estimated_cost="
                f"${estimated_cost:.2f} | "
                f"position_budget="
                f"${POSITION_DOLLARS:.2f}"
            )

            return False

        # Hard safety cap.
        qty = min(
            qty,
            10,
        )

        log(
            "ENTRY SELECTED | "
            f"{signal['symbol']} "
            f"{signal['side']} | "
            f"{contract['symbol']} | "
            f"strike="
            f"{contract['strike']} | "
            f"bid="
            f"{contract['bid']:.2f} | "
            f"ask="
            f"{contract['ask']:.2f} | "
            f"qty={qty}"
        )

        order = submit_option_order(

            contract[
                "symbol"
            ],

            qty,

            "buy",

            "buy_to_open",
        )

        order_id = (
            order.get(
                "id"
            )
        )

        if not order_id:

            add_error(
                "ENTRY ORDER "
                "MISSING ID"
            )

            return False

        filled = wait_for_fill(
            order_id
        )

        if not filled:

            add_error(
                "ENTRY FILL "
                "UNKNOWN | "
                f"{contract['symbol']}"
            )

            return False

        status = str(
            filled.get(
                "status",
                "",
            )
        ).lower()

        if status != "filled":

            add_error(
                "ENTRY NOT FILLED | "
                f"{contract['symbol']} | "
                f"status={status}"
            )

            return False

        fill_price = to_float(
            filled.get(
                "filled_avg_price"
            )
        )

        filled_qty = int(
            float(
                filled.get(
                    "filled_qty"
                )
                or qty
            )
        )

        if (
            fill_price is None
            or fill_price <= 0
        ):

            add_error(
                "INVALID FILL PRICE | "
                f"{contract['symbol']}"
            )

            return False

        managed_positions[
            contract[
                "symbol"
            ]
        ] = {

            "option_symbol":
                contract[
                    "symbol"
                ],

            "underlying":
                signal[
                    "symbol"
                ],

            "signal_side":
                signal[
                    "side"
                ],

            "entry_price":
                fill_price,

            "original_qty":
                filled_qty,

            "remaining_qty":
                filled_qty,

            "tp_done":
                False,

            "peak_price":
                fill_price,

            "opened_at":
                now_et()
                .isoformat(),

            "entry_order_id":
                order_id,
        }

        log(
            "ENTRY FILLED | "
            f"{contract['symbol']} | "
            f"qty={filled_qty} | "
            f"price="
            f"{fill_price:.2f} | "
            f"stop="
            f"{fill_price * 0.80:.2f} | "
            f"TP="
            f"{fill_price * 1.30:.2f}"
        )

        sync_managed_positions()

        return True

    except Exception as e:

        add_error(
            "ENTRY ERROR | "
            f"{signal['symbol']} "
            f"{signal['side']} | "
            f"{safe_text(e)}"
        )

        return False


# ============================================================
# MANAGE ONE POSITION
# ============================================================

def manage_one_position(
    option_symbol,
    position,
):

    remaining_qty = int(
        position.get(
            "remaining_qty",
            0,
        )
    )

    if remaining_qty <= 0:

        return

    quote = get_option_quote(
        option_symbol
    )

    if not quote:

        return

    mark = quote[
        "mid"
    ]

    entry = float(
        position[
            "entry_price"
        ]
    )

    if (
        mark <= 0
        or entry <= 0
    ):

        return

    old_peak = float(
        position.get(
            "peak_price",
            entry,
        )
    )

    position[
        "peak_price"
    ] = max(
        old_peak,
        mark,
    )

    stop_price = (

        entry

        * (
            1
            - STOP_LOSS_PERCENT
        )
    )

    tp_price = (

        entry

        * (
            1
            + TAKE_PROFIT_PERCENT
        )
    )


    # ========================================================
    # FORCE EXIT
    # ========================================================

    if (
        now_et().time()
        >= FORCE_EXIT_TIME
    ):

        order = close_option(

            option_symbol,

            remaining_qty,

            "FORCE_EXIT_0DTE",
        )

        if order:

            position[
                "remaining_qty"
            ] = 0

        return


    # ========================================================
    # STOP LOSS
    # ========================================================

    if (

        not position[
            "tp_done"
        ]

        and mark
        <= stop_price
    ):

        order = close_option(

            option_symbol,

            remaining_qty,

            "STOP_LOSS",
        )

        if order:

            position[
                "remaining_qty"
            ] = 0

        return


    # ========================================================
    # TAKE PROFIT
    # ========================================================

    if (

        not position[
            "tp_done"
        ]

        and mark
        >= tp_price
    ):

        original_qty = int(
            position[
                "original_qty"
            ]
        )

        # Need at least two contracts
        # to trim and leave a runner.
        if original_qty >= 2:

            trim_qty = int(
                math.floor(

                    original_qty

                    * TAKE_PROFIT_FRACTION
                )
            )

            trim_qty = max(
                trim_qty,
                1,
            )

            trim_qty = min(

                trim_qty,

                remaining_qty - 1,
            )

        else:

            trim_qty = (
                remaining_qty
            )

        if trim_qty > 0:

            order = close_option(

                option_symbol,

                trim_qty,

                "TAKE_PROFIT",
            )

            if order:

                position[
                    "remaining_qty"
                ] -= trim_qty

                position[
                    "tp_done"
                ] = True

                position[
                    "peak_price"
                ] = mark

                log(
                    "TP HIT | "
                    f"{option_symbol} | "
                    f"mark="
                    f"{mark:.2f} | "
                    f"sold="
                    f"{trim_qty} | "
                    f"runner="
                    f"{position['remaining_qty']}"
                )

        return


    # ========================================================
    # RUNNER
    # ========================================================

    if (

        position[
            "tp_done"
        ]

        and position[
            "remaining_qty"
        ]
        > 0
    ):

        peak = float(
            position[
                "peak_price"
            ]
        )

        trail_price = (

            peak

            * (
                1
                - RUNNER_TRAIL_PERCENT
            )
        )

        if (
            mark
            <= trail_price
        ):

            order = close_option(

                option_symbol,

                int(
                    position[
                        "remaining_qty"
                    ]
                ),

                "RUNNER_TRAIL",
            )

            if order:

                position[
                    "remaining_qty"
                ] = 0

                log(
                    "RUNNER EXIT | "
                    f"{option_symbol} | "
                    f"mark="
                    f"{mark:.2f} | "
                    f"peak="
                    f"{peak:.2f}"
                )


# ============================================================
# MANAGE ALL POSITIONS
# ============================================================

def manage_positions():

    symbols = list(
        managed_positions.keys()
    )

    for option_symbol in symbols:

        position = (
            managed_positions[
                option_symbol
            ]
        )

        try:

            manage_one_position(
                option_symbol,
                position,
            )

        except Exception as e:

            add_error(
                "MANAGE ERROR | "
                f"{option_symbol} | "
                f"{safe_text(e)}"
            )

    # Remove closed items.
    for option_symbol in list(
        managed_positions.keys()
    ):

        remaining = int(

            managed_positions[
                option_symbol
            ].get(
                "remaining_qty",
                0,
            )
        )

        if remaining <= 0:

            del managed_positions[
                option_symbol
            ]

    sync_managed_positions()


# ============================================================
# SCAN CYCLE
# ============================================================

def run_scan_cycle():

    with state_lock:

        bot_state[
            "last_cycle"
        ] = now_et().isoformat()

        bot_state[
            "stocks_scanned"
        ] = 0

    universe = (
        get_stock_universe()
    )

    symbols = (
        universe[
            :SCAN_LIMIT
        ]
    )

    found = []

    trades_opened = 0

    for symbol in symbols:

        try:

            signal = (
                latest_live_signal(
                    symbol
                )
            )

            with state_lock:

                bot_state[
                    "stocks_scanned"
                ] += 1

            if not signal:

                continue

            found.append(
                {

                    "symbol":
                        signal[
                            "symbol"
                        ],

                    "side":
                        signal[
                            "side"
                        ],

                    "timestamp":
                        signal[
                            "timestamp"
                        ].isoformat(),

                    "price":
                        signal[
                            "underlying_entry"
                        ],

                    "premarket_level":
                        signal[
                            "pm_level"
                        ],
                }
            )

            log(
                "SIGNAL | "
                f"{signal['symbol']} "
                f"{signal['side']} | "
                f"price="
                f"{signal['underlying_entry']:.2f}"
            )

            if (

                AUTO_TRADE

                and trades_opened
                < MAX_NEW_TRADES_PER_CYCLE
            ):

                opened = (
                    open_signal_trade(
                        signal
                    )
                )

                if opened:

                    trades_opened += 1

        except Exception as e:

            add_error(
                "SCAN ERROR | "
                f"{symbol} | "
                f"{safe_text(e)}"
            )

    with state_lock:

        bot_state[
            "signals"
        ] = found[-50:]

        bot_state[
            "last_scan"
        ] = now_et().isoformat()


# ============================================================
# BOT LOOP
# ============================================================

def bot_loop():

    with state_lock:

        bot_state[
            "running"
        ] = True

    while True:

        try:

            opened = (
                market_is_open()
            )

            if opened:

                # Manage exits first.
                manage_positions()

                current_time = (
                    now_et().time()
                )

                # Only scan during normal
                # market hours before cutoff.
                if (

                    current_time
                    >= RTH_START

                    and current_time
                    < LAST_ENTRY_TIME
                ):

                    run_scan_cycle()

            else:

                with state_lock:

                    bot_state[
                        "last_cycle"
                    ] = now_et().isoformat()

        except Exception as e:

            add_error(
                "BOT LOOP ERROR | "
                f"{safe_text(e)}"
            )

        time.sleep(
            LOOP_SECONDS
        )


# ============================================================
# ROUTE: HOME
# ============================================================

@app.route("/")
def home():

    with state_lock:

        state = dict(
            bot_state
        )

    return jsonify(
        {

            "service":
                "alpaca-0dte-paper-bot",

            "paper":
                True,

            "running":
                state[
                    "running"
                ],

            "credentials_ok":
                state[
                    "credentials_ok"
                ],

            "options_level":
                state[
                    "options_level"
                ],

            "market_open":
                state[
                    "market_open"
                ],

            "auto_trade":
                AUTO_TRADE,

            "stock_feed":
                DATA_FEED,

            "option_feed":
                OPTION_FEED,

            "timeframe":
                f"{TIMEFRAME_MINUTES}Min",

            "position_dollars":
                POSITION_DOLLARS,

            "stop_loss":
                STOP_LOSS_PERCENT,

            "take_profit":
                TAKE_PROFIT_PERCENT,

            "take_profit_fraction":
                TAKE_PROFIT_FRACTION,

            "runner_trail":
                RUNNER_TRAIL_PERCENT,

            "last_entry_time":
                "14:45 ET",

            "force_exit_time":
                "15:15 ET",

            "stocks_scanned":
                state[
                    "stocks_scanned"
                ],

            "last_cycle":
                state[
                    "last_cycle"
                ],

            "last_scan":
                state[
                    "last_scan"
                ],

            "signals":
                state[
                    "signals"
                ],

            "managed_positions":
                state[
                    "managed_positions"
                ],

            "errors":
                state[
                    "errors"
                ],
        }
    )


# ============================================================
# ROUTE: HEALTH
# ============================================================

@app.route("/health")
def health():

    return jsonify(
        {

            "ok":
                True,

            "credentials_ok":
                bot_state[
                    "credentials_ok"
                ],

            "running":
                bot_state[
                    "running"
                ],

            "market_open":
                bot_state[
                    "market_open"
                ],
        }
    )


# ============================================================
# ROUTE: HISTORY TEST
# ============================================================

@app.route(
    "/history-test/<symbol>"
)
def history_test(
    symbol,
):

    symbol = (
        symbol
        .upper()
        .strip()
    )

    df = get_historical_bars(
        symbol,
        days=30,
    )

    if (
        df is None
        or df.empty
    ):

        return jsonify(
            {

                "ok":
                    False,

                "symbol":
                    symbol,

                "feed":
                    DATA_FEED,

                "errors":
                    bot_state[
                        "errors"
                    ][-10:],
            }
        ), 500

    return jsonify(
        {

            "ok":
                True,

            "symbol":
                symbol,

            "feed":
                DATA_FEED,

            "bars":
                len(df),

            "first":
                df.iloc[
                    0
                ][
                    "timestamp"
                ].isoformat(),

            "last":
                df.iloc[
                    -1
                ][
                    "timestamp"
                ].isoformat(),
        }
    )


# ============================================================
# ROUTE: OPTION TEST
# ============================================================

@app.route(
    "/option-test/"
    "<underlying>/<side>"
)
def option_test(
    underlying,
    side,
):

    underlying = (
        underlying
        .upper()
        .strip()
    )

    side = (
        side
        .upper()
        .strip()
    )

    if side not in {
        "CALL",
        "PUT",
    }:

        return jsonify(
            {

                "ok":
                    False,

                "error":
                    "side must be "
                    "CALL or PUT",
            }
        ), 400

    df = get_recent_bars(
        underlying,
        limit=100,
    )

    if (
        df is None
        or df.empty
    ):

        return jsonify(
            {

                "ok":
                    False,

                "error":
                    "No stock data",
            }
        ), 500

    signal = {

        "symbol":
            underlying,

        "side":
            side,

        "timestamp":
            now_et(),

        "underlying_entry":
            float(
                df.iloc[
                    -1
                ][
                    "close"
                ]
            ),
    }

    try:

        contract = (
            select_0dte_contract(
                signal
            )
        )

        return jsonify(
            {

                "ok":
                    contract
                    is not None,

                "underlying":
                    underlying,

                "side":
                    side,

                "option_feed":
                    OPTION_FEED,

                "contract":
                    contract,

                "errors":
                    bot_state[
                        "errors"
                    ][-10:],
            }
        )

    except Exception as e:

        return jsonify(
            {

                "ok":
                    False,

                "error":
                    safe_text(e),

                "errors":
                    bot_state[
                        "errors"
                    ][-10:],
            }
        ), 500


# ============================================================
# START BACKGROUND THREAD
# ============================================================

def start_background_thread():

    thread = threading.Thread(
        target=bot_loop,
        daemon=True,
    )

    thread.start()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if verify_credentials():

        log(
            "0DTE PAPER BOT STARTED | "
            f"DATA_FEED="
            f"{DATA_FEED} | "
            f"OPTION_FEED="
            f"{OPTION_FEED} | "
            f"AUTO_TRADE="
            f"{AUTO_TRADE}"
        )

        # Confirm historical
        # stock data still works.
        try:

            test_history = (
                get_historical_bars(
                    "SPY",
                    days=5,
                )
            )

            if (
                test_history
                is not None
                and not
                test_history.empty
            ):

                log(
                    "SPY HISTORY TEST PASSED | "
                    f"{len(test_history)} "
                    "bars"
                )

        except Exception as e:

            add_error(
                "STARTUP HISTORY TEST "
                f"ERROR | "
                f"{safe_text(e)}"
            )

        if RUN_BOT_LOOP:

            start_background_thread()

    else:

        log(
            "BOT STARTED WITHOUT "
            "VALID ALPACA CREDENTIALS"
        )

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
        use_reloader=False,
    )