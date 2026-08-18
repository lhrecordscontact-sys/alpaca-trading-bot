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
# APP
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


# ============================================================
# ACCOUNT / RISK
# ============================================================

MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
MAX_NEW_TRADES_PER_CYCLE = int(
    os.getenv("MAX_NEW_TRADES_PER_CYCLE", "1")
)

POSITION_DOLLARS = float(
    os.getenv("POSITION_DOLLARS", "500")
)

STOP_LOSS_PERCENT = float(
    os.getenv("STOP_LOSS_PERCENT", "0.20")
)

TAKE_PROFIT_PERCENT = float(
    os.getenv("TAKE_PROFIT_PERCENT", "0.30")
)

TAKE_PROFIT_FRACTION = float(
    os.getenv("TAKE_PROFIT_FRACTION", "0.50")
)

RUNNER_TRAIL_PERCENT = float(
    os.getenv("RUNNER_TRAIL_PERCENT", "0.15")
)

LAST_ENTRY_TIME = dt_time(14, 45)
FORCE_EXIT_TIME = dt_time(15, 15)


# ============================================================
# TRADINGVIEW STYLE SIGNAL SETTINGS
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


# ============================================================
# STRICT 64-TRADE / 80% RULES
# ============================================================

ROLLING_TRADE_COUNT = 64
MIN_STATS_TRADES = 64

MIN_WIN_RATE = float(
    os.getenv("MIN_WIN_RATE", "0.80")
)

BACKTEST_LOOKBACK_DAYS = int(
    os.getenv("BACKTEST_LOOKBACK_DAYS", "180")
)

# Prevent backtester from appearing frozen forever.
MAX_SIGNAL_ATTEMPTS_PER_SYMBOL = int(
    os.getenv("MAX_SIGNAL_ATTEMPTS_PER_SYMBOL", "140")
)

BACKTEST_REQUEST_PAUSE = float(
    os.getenv("BACKTEST_REQUEST_PAUSE", "0.04")
)


# ============================================================
# SCANNER
# ============================================================

SCAN_LIMIT = int(
    os.getenv("SCAN_LIMIT", "150")
)

LOOP_SECONDS = int(
    os.getenv("LOOP_SECONDS", "45")
)

PREMARKET_REFRESH_SECONDS = int(
    os.getenv("PREMARKET_REFRESH_SECONDS", "600")
)

PREMARKET_WATCHLIST_SIZE = int(
    os.getenv("PREMARKET_WATCHLIST_SIZE", "25")
)


# ============================================================
# PRIORITY SYMBOLS
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
    "AMAT",
]


# ============================================================
# CREDENTIALS
# ============================================================

def clean_credential(value):

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
        .replace("\xa0", "")
        .encode("ascii", errors="ignore")
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
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json",
}


# ============================================================
# STATE
# ============================================================

bot_state = {
    "running": False,
    "credentials_ok": False,
    "market_open": False,

    "last_cycle": None,
    "last_scan": None,

    "stocks_scanned_this_cycle": 0,

    "signals": [],

    "premarket_watchlist": [],
    "premarket_last_run": None,

    "stats_by_symbol": {},
    "backtest_last_run": None,

    "backtest_progress": {
        "running": False,
        "symbol": None,
        "completed": 0,
        "required": 64,
        "attempted": 0,
        "message": "Not started",
    },

    "errors": [],
}

managed_positions = {}

_universe_cache = {
    "symbols": [],
    "loaded_at": None,
}

_daily_state = {
    "stats_date": None,
    "premarket_epoch": 0,
}

_contract_cache = {}
_option_bar_cache = {}

_backtest_lock = threading.Lock()


# ============================================================
# HELPERS
# ============================================================

def now_et():
    return datetime.now(NY)


def safe_text(value):

    try:
        return (
            str(value)
            .encode("ascii", errors="replace")
            .decode("ascii")
        )

    except Exception:
        return "Unknown error"


def log(message):

    stamp = now_et().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        f"[{stamp} ET] {safe_text(message)}",
        flush=True,
    )


def add_error(message):

    message = safe_text(message)

    bot_state["errors"].append(
        message
    )

    bot_state["errors"] = (
        bot_state["errors"][-25:]
    )

    log(
        f"ERROR: {message}"
    )


def in_time_window(value, start, end):

    return (
        start <= value < end
    )


# ============================================================
# ALPACA REQUESTS
# ============================================================

def alpaca_get(
    path,
    params=None,
    data_api=False,
    timeout=20,
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
        timeout=timeout,
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
        timeout=20,
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
# ACCOUNT
# ============================================================

def verify_credentials():

    if (
        not ALPACA_API_KEY
        or not ALPACA_SECRET_KEY
    ):

        bot_state["credentials_ok"] = False

        add_error(
            "Alpaca credentials are missing."
        )

        return False

    try:

        account = alpaca_get(
            "/v2/account"
        )

        bot_state["credentials_ok"] = True

        log(
            "ALPACA PAPER CONNECTED | "
            f'equity=${account.get("equity")}'
        )

        return True

    except Exception as e:

        bot_state["credentials_ok"] = False

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

        result = bool(
            clock.get("is_open", False)
        )

        bot_state["market_open"] = result

        return result

    except Exception as e:

        bot_state["market_open"] = False

        add_error(
            "Market clock error: "
            f"{safe_text(e)}"
        )

        return False


# ============================================================
# STOCK UNIVERSE
# ============================================================

def get_stock_universe(force=False):

    loaded_at = (
        _universe_cache["loaded_at"]
    )

    if (
        not force
        and _universe_cache["symbols"]
        and loaded_at
        and (
            now_et() - loaded_at
        ).total_seconds() < 21600
    ):

        return (
            _universe_cache["symbols"]
        )

    symbols = []

    try:

        assets = alpaca_get(
            "/v2/assets",
            params={
                "status": "active",
                "asset_class": "us_equity",
            },
        )

        for asset in assets:

            symbol = asset.get("symbol")

            if not symbol:
                continue

            if not asset.get(
                "tradable",
                False
            ):
                continue

            if "." in symbol:
                continue

            symbols.append(symbol)

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

    _universe_cache["symbols"] = symbols
    _universe_cache["loaded_at"] = now_et()

    return symbols


# ============================================================
# STOCK BARS
# ============================================================

def bars_to_df(bars):

    if not bars:
        return None

    df = pd.DataFrame(bars)

    if df.empty:
        return None

    required = [
        "t",
        "o",
        "h",
        "l",
        "c",
        "v",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        return None

    df["timestamp"] = (
        pd.to_datetime(
            df["t"],
            utc=True,
        )
        .dt
        .tz_convert(NY)
    )

    df["open"] = pd.to_numeric(
        df["o"],
        errors="coerce",
    )

    df["high"] = pd.to_numeric(
        df["h"],
        errors="coerce",
    )

    df["low"] = pd.to_numeric(
        df["l"],
        errors="coerce",
    )

    df["close"] = pd.to_numeric(
        df["c"],
        errors="coerce",
    )

    df["volume"] = pd.to_numeric(
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
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def get_recent_bars(
    symbol,
    limit=1000,
):

    try:

        data = alpaca_get(
            f"/v2/stocks/{symbol}/bars",
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
            data.get("bars", [])
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
        - timedelta(days=days)
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
            params["page_token"] = page_token

        try:

            data = alpaca_get(
                f"/v2/stocks/{symbol}/bars",
                params=params,
                data_api=True,
            )

        except Exception as e:

            log(
                f"BACKTEST {symbol}: "
                f"stock history request failed | "
                f"{safe_text(e)}"
            )

            break

        all_bars.extend(
            data.get("bars", [])
        )

        page_token = data.get(
            "next_page_token"
        )

        if not page_token:
            break

    return bars_to_df(all_bars)


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
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

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

    previous_close = (
        df["close"].shift(1)
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
    ).max(axis=1)

    df["atr"] = (
        true_range
        .rolling(ATR_LENGTH)
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

    pv = (
        typical_price
        * df["volume"]
    )

    cumulative_pv = (
        pv.groupby(
            session_date
        )
        .cumsum()
    )

    cumulative_volume = (
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

    df["vwap"] = (
        cumulative_pv
        / cumulative_volume
    )

    df["pm_high"] = np.nan
    df["pm_low"] = np.nan

    for _, indexes in (
        df.groupby(
            session_date
        ).groups.items()
    ):

        indexes = list(indexes)

        rows = df.loc[indexes]

        times = (
            rows["timestamp"]
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
            premarket["high"].max()
        )

        pm_low = float(
            premarket["low"].min()
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

    df = calculate_indicators(df)

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
        previous = df.iloc[i - 1]

        timestamp = row["timestamp"]
        day = timestamp.date()
        current_time = timestamp.time()

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

        pm_high = row["pm_high"]
        pm_low = row["pm_low"]

        if (
            pd.isna(pm_high)
            or pd.isna(pm_low)
            or pd.isna(row["atr"])
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

        # Break above premarket high.
        if (
            not long_used
            and row["close"] > pm_high
            and previous["close"] <= pm_high
        ):
            long_break_index = i

        # Break below premarket low.
        if (
            not short_used
            and row["close"] < pm_low
            and previous["close"] >= pm_low
        ):
            short_break_index = i

        # Expire stale retest window.
        if (
            long_break_index
            is not None
            and (
                i - long_break_index
            ) > MAX_RETEST_BARS
        ):
            long_break_index = None

        if (
            short_break_index
            is not None
            and (
                i - short_break_index
            ) > MAX_RETEST_BARS
        ):
            short_break_index = None

        tolerance = (
            float(row["atr"])
            * RETEST_ATR_TOLERANCE
        )

        # ====================================================
        # CALL
        # ====================================================

        if (
            not long_used
            and long_break_index
            is not None
            and i > long_break_index
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
                            row["close"]
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

        # ====================================================
        # PUT
        # ====================================================

        if (
            not short_used
            and short_break_index
            is not None
            and i > short_break_index
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
                            row["close"]
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
# HISTORICAL CONTRACT CACHE
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

        return (
            _contract_cache[
                cache_key
            ]
        )

    option_type = (
        "call"
        if side == "CALL"
        else "put"
    )

    contracts = []

    for status in [
        "inactive",
        "active",
    ]:

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
                timeout=12,
            )

            contracts.extend(
                data.get(
                    "option_contracts",
                    [],
                )
            )

        except Exception:
            continue

    unique = {}

    for contract in contracts:

        option_symbol = (
            contract.get("symbol")
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

    expiration = (
        trade_date.strftime(
            "%Y-%m-%d"
        )
    )

    contracts = (
        get_contracts_for_date(
            symbol,
            side,
            expiration,
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

    _, strike, option_symbol = (
        choices[0]
    )

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
            {}
        )
    )

    if isinstance(
        bars_root,
        dict
    ):

        bars = (
            bars_root.get(
                option_symbol,
                [],
            )
        )

    elif isinstance(
        bars_root,
        list
    ):

        bars = bars_root

    else:
        bars = []

    if not bars:
        return None

    df = pd.DataFrame(bars)

    if df.empty:
        return None

    df["timestamp"] = (
        pd.to_datetime(
            df["t"],
            utc=True,
        )
        .dt
        .tz_convert(NY)
    )

    df["open"] = pd.to_numeric(
        df["o"],
        errors="coerce",
    )

    df["high"] = pd.to_numeric(
        df["h"],
        errors="coerce",
    )

    df["low"] = pd.to_numeric(
        df["l"],
        errors="coerce",
    )

    df["close"] = pd.to_numeric(
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
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def get_historical_option_bars(
    option_symbol,
    entry_time,
):

    cache_key = (
        option_symbol,
        entry_time.strftime(
            "%Y-%m-%d"
        ),
    )

    if cache_key in _option_bar_cache:

        return (
            _option_bar_cache[
                cache_key
            ]
        )

    end_et = datetime.combine(
        entry_time.date(),
        FORCE_EXIT_TIME,
        tzinfo=NY,
    )

    if entry_time >= end_et:
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
                    entry_time
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
            timeout=12,
        )

        df = option_bars_to_df(
            data,
            option_symbol,
        )

        _option_bar_cache[
            cache_key
        ] = df

        return df

    except Exception:

        _option_bar_cache[
            cache_key
        ] = None

        return None


# ============================================================
# HISTORICAL OPTION TRADE
# ============================================================

def evaluate_historical_option_trade(
    signal,
):

    contract = (
        choose_historical_atm_contract(
            signal["symbol"],
            signal["side"],
            signal["entry"],
            signal["timestamp"].date(),
        )
    )

    if not contract:
        return None

    option_df = (
        get_historical_option_bars(
            contract["symbol"],
            signal["timestamp"],
        )
    )

    if (
        option_df is None
        or option_df.empty
    ):
        return None

    option_df = option_df[
        option_df["timestamp"]
        >= signal["timestamp"]
    ].reset_index(drop=True)

    if option_df.empty:
        return None

    entry_price = float(
        option_df.iloc[0]["close"]
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

    result = None
    exit_price = entry_price
    exit_reason = "END"

    for _, row in (
        option_df.iterrows()
    ):

        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if not tp_hit:

            # Conservative:
            # if both levels occur in same 1m candle,
            # count stop first.
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

                # Hitting TP makes the setup a winner.
                result = "WIN"
                exit_reason = "TP"

        else:

            runner_high = max(
                float(runner_high),
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

                exit_price = (
                    trailing_price
                )

                exit_reason = (
                    "TP+RUNNER"
                )

                break

        if (
            row["timestamp"].time()
            >= FORCE_EXIT_TIME
        ):

            exit_price = close

            if result is None:

                result = (
                    "WIN"
                    if close > entry_price
                    else "LOSS"
                )

                exit_reason = "TIME"

            break

    if result is None:

        exit_price = float(
            option_df.iloc[-1][
                "close"
            ]
        )

        result = (
            "WIN"
            if exit_price > entry_price
            else "LOSS"
        )

        exit_reason = "END"

    return {
        "symbol":
            signal["symbol"],

        "side":
            signal["side"],

        "signal_time":
            signal["timestamp"],

        "stock_entry":
            round(
                signal["entry"],
                4,
            ),

        "option_symbol":
            contract["symbol"],

        "strike":
            contract["strike"],

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
# EMPTY STATS
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
            80.0,

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

def build_symbol_stats(symbol):

    progress = (
        bot_state[
            "backtest_progress"
        ]
    )

    progress["symbol"] = symbol
    progress["completed"] = 0
    progress["attempted"] = 0
    progress["message"] = (
        f"BACKTEST {symbol} STARTED"
    )

    log(
        "========================================"
    )

    log(
        f"BACKTEST {symbol} STARTED | "
        f"need exactly {ROLLING_TRADE_COUNT} "
        f"completed 0DTE trades"
    )

    log(
        f"BACKTEST {symbol}: "
        f"loading {BACKTEST_LOOKBACK_DAYS} "
        f"days of 4-minute stock history..."
    )

    stock_df = get_historical_bars(
        symbol,
        BACKTEST_LOOKBACK_DAYS,
    )

    if stock_df is None:

        log(
            f"BACKTEST {symbol}: "
            f"NO STOCK HISTORY"
        )

        return empty_stats(
            symbol,
            "NO_STOCK_HISTORY",
        )

    log(
        f"BACKTEST {symbol}: "
        f"stock bars loaded = "
        f"{len(stock_df)}"
    )

    signals = generate_signals(
        stock_df,
        symbol,
    )

    # Exclude today's unfinished session.
    signals = [
        signal
        for signal in signals
        if signal[
            "timestamp"
        ].date()
        < now_et().date()
    ]

    log(
        f"BACKTEST {symbol}: "
        f"historical setup signals = "
        f"{len(signals)}"
    )

    if not signals:

        return empty_stats(
            symbol,
            "NO_SIGNALS",
        )

    completed = []
    attempted = 0

    for signal in reversed(signals):

        if (
            len(completed)
            >= ROLLING_TRADE_COUNT
        ):
            break

        if (
            attempted
            >= MAX_SIGNAL_ATTEMPTS_PER_SYMBOL
        ):
            break

        attempted += 1

        progress["attempted"] = attempted

        try:

            trade = (
                evaluate_historical_option_trade(
                    signal
                )
            )

        except Exception as e:

            trade = None

            log(
                f"BACKTEST {symbol}: "
                f"attempt {attempted} "
                f"skipped | "
                f"{safe_text(e)}"
            )

        if trade:

            completed.append(trade)

            progress[
                "completed"
            ] = len(completed)

            progress[
                "message"
            ] = (
                f"{symbol} "
                f"{len(completed)}/64"
            )

            # Print every completed trade so Render
            # never looks frozen.
            log(
                f"BACKTEST {symbol}: "
                f"{len(completed)}/64 "
                f"COMPLETE | "
                f'{trade["side"]} | '
                f'{trade["result"]} | '
                f'{trade["option_symbol"]}'
            )

        elif (
            attempted == 1
            or attempted % 10 == 0
        ):

            log(
                f"BACKTEST {symbol}: "
                f"progress "
                f"{len(completed)}/64 | "
                f"signals checked="
                f"{attempted}"
            )

        time.sleep(
            BACKTEST_REQUEST_PAUSE
        )

    completed.reverse()

    log(
        f"BACKTEST {symbol}: "
        f"SEARCH FINISHED | "
        f"completed={len(completed)}/64 | "
        f"signals_checked={attempted}"
    )

    # ========================================================
    # MUST HAVE EXACTLY 64
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
        ] = len(completed)

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
            for trade in completed
        ]

        log(
            f"STATS {symbol}: "
            f"WAITING | "
            f"only {len(completed)}/64 "
            f"valid completed "
            f"historical 0DTE trades"
        )

        return stats

    call_trades = [
        trade
        for trade in completed
        if trade["side"] == "CALL"
    ]

    put_trades = [
        trade
        for trade in completed
        if trade["side"] == "PUT"
    ]

    call_wins = sum(
        1
        for trade in call_trades
        if trade["result"] == "WIN"
    )

    put_wins = sum(
        1
        for trade in put_trades
        if trade["result"] == "WIN"
    )

    call_losses = (
        len(call_trades)
        - call_wins
    )

    put_losses = (
        len(put_trades)
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

    overall_rate = (
        wins
        / ROLLING_TRADE_COUNT
    )

    call_rate = (
        call_wins
        / len(call_trades)
        if call_trades
        else 0.0
    )

    put_rate = (
        put_wins
        / len(put_trades)
        if put_trades
        else 0.0
    )

    qualified = (
        overall_rate
        >= MIN_WIN_RATE
    )

    call_qualified = (
        qualified
        and len(call_trades) > 0
        and call_rate >= MIN_WIN_RATE
    )

    put_qualified = (
        qualified
        and len(put_trades) > 0
        and put_rate >= MIN_WIN_RATE
    )

    stats = {
        "symbol":
            symbol,

        "overall_win_rate":
            round(
                overall_rate * 100,
                1,
            ),

        "call_win_rate":
            round(
                call_rate * 100,
                1,
            ),

        "put_win_rate":
            round(
                put_rate * 100,
                1,
            ),

        "call_w_l":
            f"{call_wins} / "
            f"{call_losses}",

        "put_w_l":
            f"{put_wins} / "
            f"{put_losses}",

        "call_trades":
            len(call_trades),

        "put_trades":
            len(put_trades),

        "wins":
            wins,

        "losses":
            losses,

        "total_trades":
            64,

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
            80.0,

        "required_completed_trades":
            64,

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
            for trade in completed
        ],
    }

    log(
        "========================================"
    )

    log(
        f"STATS {symbol}: "
        f'{stats["wins"]}/64 wins | '
        f'overall='
        f'{stats["overall_win_rate"]:.1f}% | '
        f'CALL='
        f'{stats["call_win_rate"]:.1f}% | '
        f'PUT='
        f'{stats["put_win_rate"]:.1f}% | '
        f'{stats["status"]}'
    )

    log(
        "========================================"
    )

    return stats


# ============================================================
# BACKTEST ALL PRIORITY SYMBOLS
# ============================================================

def refresh_priority_stats():

    if not _backtest_lock.acquire(
        blocking=False
    ):

        log(
            "BACKTEST already running. "
            "Skipping duplicate request."
        )

        return (
            bot_state["stats_by_symbol"]
        )

    try:

        bot_state[
            "backtest_progress"
        ] = {
            "running":
                True,

            "symbol":
                None,

            "completed":
                0,

            "required":
                64,

            "attempted":
                0,

            "message":
                "Starting backtest",
        }

        log(
            "========================================"
        )

        log(
            "STRICT 64-TRADE 80% "
            "BACKTEST STARTED"
        )

        log(
            "========================================"
        )

        results = {}

        for index, symbol in enumerate(
            PRIORITY_SYMBOLS,
            start=1,
        ):

            log(
                f"BACKTEST SYMBOL "
                f"{index}/"
                f"{len(PRIORITY_SYMBOLS)}: "
                f"{symbol}"
            )

            try:

                stats = (
                    build_symbol_stats(
                        symbol
                    )
                )

            except Exception as e:

                add_error(
                    f"Stats {symbol}: "
                    f"{safe_text(e)}"
                )

                stats = empty_stats(
                    symbol,
                    "BACKTEST_ERROR",
                )

            results[
                symbol
            ] = stats

            # Save after EACH symbol so /status
            # updates immediately.
            bot_state[
                "stats_by_symbol"
            ][symbol] = stats

            log(
                f"FINISHED {symbol} | "
                f'{stats["status"]} | '
                f'{stats["total_trades"]}/64 '
                f"completed"
            )

        bot_state[
            "backtest_last_run"
        ] = now_et().isoformat()

        bot_state[
            "backtest_progress"
        ] = {
            "running":
                False,

            "symbol":
                None,

            "completed":
                64,

            "required":
                64,

            "attempted":
                0,

            "message":
                "Backtest finished",
        }

        log(
            "ALL BACKTESTS FINISHED"
        )

        return results

    finally:

        if (
            bot_state[
                "backtest_progress"
            ][
                "running"
            ]
        ):

            bot_state[
                "backtest_progress"
            ][
                "running"
            ] = False

        _backtest_lock.release()


# ============================================================
# BACKTEST THREAD
# ============================================================

def start_backtest_thread():

    if (
        bot_state[
            "backtest_progress"
        ][
            "running"
        ]
    ):

        return False

    thread = threading.Thread(
        target=refresh_priority_stats,
        daemon=True,
    )

    thread.start()

    return True


# ============================================================
# PREMARKET WATCHLIST
# ============================================================

def build_premarket_watchlist():

    rows = []

    for symbol, stats in (
        bot_state[
            "stats_by_symbol"
        ].items()
    ):

        if (
            not stats.get(
                "qualified"
            )
        ):
            continue

        if (
            stats.get(
                "total_trades"
            )
            != 64
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

        last = df.iloc[-1]

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
                64,

            "price":
                round(
                    float(
                        last["close"]
                    ),
                    2,
                ),

            "pm_high":
                (
                    None
                    if pd.isna(
                        last["pm_high"]
                    )
                    else round(
                        float(
                            last["pm_high"]
                        ),
                        2,
                    )
                ),

            "pm_low":
                (
                    None
                    if pd.isna(
                        last["pm_low"]
                    )
                    else round(
                        float(
                            last["pm_low"]
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
                "call_win_rate"
            ],
            item[
                "put_win_rate"
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
        "PREMARKET WATCHLIST READY | "
        f'{len(bot_state["premarket_watchlist"])} '
        f"qualified symbols"
    )

    return (
        bot_state[
            "premarket_watchlist"
        ]
    )


# ============================================================
# LIVE SIGNAL
# ============================================================

def latest_live_signal(symbol):

    stats = (
        bot_state[
            "stats_by_symbol"
        ].get(symbol)
    )

    if not stats:
        return None

    if (
        not stats.get(
            "qualified"
        )
    ):
        return None

    if (
        stats.get(
            "total_trades"
        )
        != 64
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

    signal = signals[-1]

    age_seconds = (
        now_et()
        - signal["timestamp"]
    ).total_seconds()

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

    side = signal["side"]

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
        if side == "CALL"
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
                if side == "CALL"
                else "SELL"
            ),

        "entry":
            round(
                signal["entry"],
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
            64,

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

    # Only symbols with completed stats
    # can produce live trades.
    qualified_symbols = [
        symbol
        for symbol, stats
        in bot_state[
            "stats_by_symbol"
        ].items()
        if (
            stats.get("qualified")
            and stats.get(
                "total_trades"
            ) == 64
        )
    ]

    watch_symbols = [
        item["symbol"]
        for item
        in bot_state[
            "premarket_watchlist"
        ]
    ]

    symbols = list(
        dict.fromkeys(
            watch_symbols
            + qualified_symbols
        )
    )

    signals = []
    scanned = 0

    for symbol in symbols:

        scanned += 1

        try:

            signal = latest_live_signal(
                symbol
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

    bot_state["signals"] = signals

    bot_state[
        "stocks_scanned_this_cycle"
    ] = scanned

    bot_state[
        "last_scan"
    ] = now_et().isoformat()

    return signals


# ============================================================
# LIVE 0DTE CONTRACT
# ============================================================

def get_live_0dte_contracts(
    symbol,
    side,
):

    option_type = (
        "call"
        if side == "CALL"
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
            f"option chain error: "
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

    _, strike, option_symbol = (
        choices[0]
    )

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
                bid + ask
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
            str(quantity),

        "side":
            side,

        "type":
            "market",

        "time_in_force":
            "day",
    }

    if not AUTO_TRADE:

        log(
            f"SIGNAL ONLY | "
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

    underlying = signal["symbol"]

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
    # FINAL SAFETY CHECK
    # EXACTLY 64 + 80%
    # ========================================================

    if (
        stats.get(
            "total_trades"
        ) != 64
    ):
        return False

    if (
        stats.get(
            "overall_win_rate",
            0,
        ) < 80.0
    ):
        return False

    if (
        signal["side"] == "CALL"
        and stats.get(
            "call_win_rate",
            0,
        ) < 80.0
    ):
        return False

    if (
        signal["side"] == "PUT"
        and stats.get(
            "put_win_rate",
            0,
        ) < 80.0
    ):
        return False

    contract = (
        choose_live_atm_0dte(
            underlying,
            signal["side"],
            float(
                signal["entry"]
            ),
        )
    )

    if not contract:
        return False

    quote = get_option_quote(
        contract["symbol"]
    )

    if (
        not quote
        or quote["mid"] <= 0
    ):
        return False

    premium = float(
        quote["mid"]
    )

    contract_cost = (
        premium * 100
    )

    if (
        contract_cost
        > POSITION_DOLLARS
    ):

        log(
            f'SKIP {contract["symbol"]} | '
            f'cost=${contract_cost:.2f} | '
            f'max=${POSITION_DOLLARS:.2f}'
        )

        return False

    quantity = int(
        POSITION_DOLLARS
        // contract_cost
    )

    if quantity < 1:
        return False

    order = submit_option_order(
        contract["symbol"],
        quantity,
        "buy",
    )

    managed_positions[
        contract["symbol"]
    ] = {
        "underlying":
            underlying,

        "direction":
            signal["side"],

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

        "overall_win_rate":
            signal[
                "overall_win_rate"
            ],

        "side_win_rate":
            signal[
                "side_win_rate"
            ],
    }

    log(
        f'ENTRY {underlying} '
        f'{signal["side"]} | '
        f'{contract["symbol"]} | '
        f'qty={quantity} | '
        f'overall='
        f'{signal["overall_win_rate"]:.1f}% | '
        f'side='
        f'{signal["side_win_rate"]:.1f}%'
    )

    return order


# ============================================================
# CLOSE POSITION
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
        f"EXIT {option_symbol} | "
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

        # Force exit.
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
            quote["mid"]
        )

        entry = float(
            trade["entry_price"]
        )

        quantity = int(
            trade["quantity"]
        )

        if (
            entry <= 0
            or quantity <= 0
        ):
            continue

        pnl_percent = (
            current - entry
        ) / entry

        # ====================================================
        # 20% STOP
        # ====================================================

        if (
            not trade["tp_hit"]
            and pnl_percent
            <= -STOP_LOSS_PERCENT
        ):

            close_managed(
                option_symbol,
                f"HARD STOP "
                f"{pnl_percent:.1%}",
            )

            continue

        # ====================================================
        # 30% TAKE PROFIT
        # ====================================================

        if (
            not trade["tp_hit"]
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

            trade["tp_hit"] = True

            trade[
                "highest_after_tp"
            ] = current

            log(
                f"TP HIT {option_symbol} | "
                f"{pnl_percent:.1%} | "
                f'runner='
                f'{trade["quantity"]}'
            )

            continue

        # ====================================================
        # RUNNER
        # ====================================================

        if trade["tp_hit"]:

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

    # Start once before market.
    if (
        _daily_state[
            "stats_date"
        ] != today
        and current_time
        < dt_time(9, 25)
    ):

        if start_backtest_thread():

            _daily_state[
                "stats_date"
            ] = today

    # Premarket watchlist.
    if (
        dt_time(7, 0)
        <= current_time
        < dt_time(9, 30)
    ):

        # Don't build list while stats
        # are still running.
        if (
            not bot_state[
                "backtest_progress"
            ][
                "running"
            ]
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

    bot_state["last_cycle"] = (
        now_et().isoformat()
    )

    if not bot_state[
        "credentials_ok"
    ]:

        if not verify_credentials():
            return

    maybe_run_daily_prep()

    if not market_is_open():
        return

    manage_positions()

    # Do not trade while backtest
    # qualification is still running.
    if (
        bot_state[
            "backtest_progress"
        ][
            "running"
        ]
    ):
        return

    signals = scan_market()

    entered = 0

    for signal in signals:

        if (
            entered
            >= MAX_NEW_TRADES_PER_CYCLE
        ):
            break

        try:

            if enter_signal(signal):
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

    bot_state["running"] = True

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

    ready_symbols = [
        symbol
        for symbol, stats
        in bot_state[
            "stats_by_symbol"
        ].items()
        if (
            stats.get(
                "qualified"
            )
            and stats.get(
                "total_trades"
            ) == 64
        )
    ]

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

        "required_completed_trades":
            64,

        "minimum_win_rate":
            "80%",

        "stop_loss":
            "20%",

        "take_profit":
            "30%",

        "runner_trail":
            "15%",

        "entry_cutoff":
            "2:45 PM ET",

        "force_exit":
            "3:15 PM ET",

        "account_equity":
            account.get(
                "equity"
            ),

        "buying_power":
            account.get(
                "buying_power"
            ),

        "ready_symbols":
            ready_symbols,

        "ready_symbol_count":
            len(
                ready_symbols
            ),

        "backtest_progress":
            bot_state[
                "backtest_progress"
            ],

        "stocks_loaded":
            len(
                get_stock_universe()
            ),

        "stocks_scanned_this_cycle":
            bot_state[
                "stocks_scanned_this_cycle"
            ],

        "errors":
            bot_state["errors"],
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
# BACKTEST PROGRESS
# ============================================================

@app.route("/backtest-progress")
def backtest_progress():

    return jsonify(
        bot_state[
            "backtest_progress"
        ]
    )


# ============================================================
# SYMBOL STATS
# ============================================================

@app.route("/stats/<symbol>")
def symbol_stats(symbol):

    symbol = (
        symbol
        .upper()
        .strip()
    )

    # Return saved stats immediately
    # if available.
    existing = (
        bot_state[
            "stats_by_symbol"
        ].get(symbol)
    )

    if existing:

        return jsonify(
            existing
        )

    return jsonify({
        "symbol":
            symbol,

        "status":
            "NOT_TESTED",

        "message":
            "Run the backtest first.",
    })


# ============================================================
# START / VIEW BACKTEST
# ============================================================

@app.route("/backtest")
def backtest():

    if (
        bot_state[
            "backtest_progress"
        ][
            "running"
        ]
    ):

        return jsonify({
            "started":
                False,

            "message":
                "Backtest already running.",

            "progress":
                bot_state[
                    "backtest_progress"
                ],

            "results":
                bot_state[
                    "stats_by_symbol"
                ],
        })

    started = (
        start_backtest_thread()
    )

    return jsonify({
        "started":
            started,

        "message":
            (
                "64-trade backtest "
                "started in background."
                if started
                else
                "Backtest already running."
            ),

        "progress":
            bot_state[
                "backtest_progress"
            ],

        "existing_results":
            bot_state[
                "stats_by_symbol"
            ],
    })


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

        "backtest_running":
            bot_state[
                "backtest_progress"
            ][
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