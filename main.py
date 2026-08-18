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

TRADING_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

TIMEFRAME_MINUTES = 4
DATA_FEED = os.getenv("DATA_FEED", "iex").strip().lower()

AUTO_TRADE = os.getenv("AUTO_TRADE", "false").strip().lower() == "true"
RUN_BOT_LOOP = os.getenv("RUN_BOT_LOOP", "true").strip().lower() == "true"

# Risk / trade management
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
MAX_NEW_TRADES_PER_CYCLE = int(os.getenv("MAX_NEW_TRADES_PER_CYCLE", "1"))
POSITION_DOLLARS = float(os.getenv("POSITION_DOLLARS", "500"))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.20"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "0.30"))
TAKE_PROFIT_FRACTION = float(os.getenv("TAKE_PROFIT_FRACTION", "0.50"))
RUNNER_TRAIL_PERCENT = float(os.getenv("RUNNER_TRAIL_PERCENT", "0.15"))

LAST_ENTRY_TIME = dt_time(14, 45)
FORCE_EXIT_TIME = dt_time(15, 15)

# Scanner
MIN_PRICE = float(os.getenv("MIN_PRICE", "5"))
MIN_DOLLAR_VOLUME = float(os.getenv("MIN_DOLLAR_VOLUME", "5000000"))
MIN_RVOL = float(os.getenv("MIN_RVOL", "1.10"))
SCAN_LIMIT = int(os.getenv("SCAN_LIMIT", "150"))
LOOP_SECONDS = int(os.getenv("LOOP_SECONDS", "15"))

# Pre-market
PREMARKET_START = dt_time(7, 0)
PREMARKET_END = dt_time(9, 29)
PREMARKET_REFRESH_SECONDS = int(
    os.getenv("PREMARKET_REFRESH_SECONDS", "600")
)
PREMARKET_WATCHLIST_SIZE = int(
    os.getenv("PREMARKET_WATCHLIST_SIZE", "25")
)

# Backtesting
BACKTEST_ENABLED = (
    os.getenv("BACKTEST_ENABLED", "true").strip().lower() == "true"
)
BACKTEST_LOOKBACK_DAYS = int(
    os.getenv("BACKTEST_LOOKBACK_DAYS", "30")
)
BACKTEST_SYMBOL_LIMIT = int(
    os.getenv("BACKTEST_SYMBOL_LIMIT", "16")
)
BACKTEST_MIN_TRADES = int(
    os.getenv("BACKTEST_MIN_TRADES", "3")
)
BACKTEST_MIN_WIN_RATE = float(
    os.getenv("BACKTEST_MIN_WIN_RATE", "0.50")
)
BACKTEST_FILTER_ENABLED = (
    os.getenv("BACKTEST_FILTER_ENABLED", "true").strip().lower() == "true"
)

PRIORITY_SYMBOLS = [
    "SPY", "QQQ", "IWM", "AAPL",
    "NVDA", "TSLA", "AMD", "AMZN",
    "META", "MSFT", "GOOGL", "NFLX",
    "AVGO", "PLTR", "COIN", "MSTR",
]


# ============================================================
# SAFE CREDENTIALS
# ============================================================

CYRILLIC_REPLACEMENTS = {
    "\u0410": "A",
    "\u0412": "B",
    "\u0421": "C",
    "\u0415": "E",
    "\u041d": "H",
    "\u041a": "K",
    "\u041c": "M",
    "\u041e": "O",
    "\u0420": "P",
    "\u0422": "T",
    "\u0425": "X",
    "\u0430": "a",
    "\u0432": "b",
    "\u0441": "c",
    "\u0435": "e",
    "\u043d": "h",
    "\u043a": "k",
    "\u043c": "m",
    "\u043e": "o",
    "\u0440": "p",
    "\u0442": "t",
    "\u0445": "x",
}


def clean_credential(value):
    if value is None:
        return ""

    value = str(value).strip()

    for bad_char, replacement in CYRILLIC_REPLACEMENTS.items():
        value = value.replace(bad_char, replacement)

    value = (
        value
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
        .replace("\xa0", "")
    )

    return value.encode(
        "ascii",
        errors="ignore"
    ).decode("ascii").strip()


ALPACA_API_KEY = clean_credential(
    os.getenv("ALPACA_API_KEY", "")
)

ALPACA_SECRET_KEY = clean_credential(
    os.getenv("ALPACA_SECRET_KEY", "")
)

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}


# ============================================================
# STATE
# ============================================================

bot_state = {
    "running": False,
    "last_cycle": None,
    "last_scan": None,
    "stocks_scanned_this_cycle": 0,
    "candidates": [],
    "signals": [],
    "premarket_watchlist": [],
    "premarket_last_run": None,
    "backtest_summary": {},
    "backtest_by_symbol": {},
    "backtest_last_run": None,
    "errors": [],
}

managed_positions = {}

_universe_cache = {
    "symbols": [],
    "loaded_at": None,
}

_daily_job_state = {
    "backtest_date": None,
    "premarket_last_epoch": 0,
}


# ============================================================
# HELPERS
# ============================================================

def now_et():
    return datetime.now(NY)


def today_string():
    return now_et().strftime("%Y-%m-%d")


def safe_text(value):
    try:
        text = str(value)
    except Exception:
        return "Unknown error"

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.encode(
        "ascii",
        errors="replace"
    ).decode("ascii")


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

    bot_state["errors"].append(message)
    bot_state["errors"] = bot_state["errors"][-25:]

    log(f"ERROR: {message}")


def credentials_present():
    return bool(
        ALPACA_API_KEY
        and ALPACA_SECRET_KEY
    )


# ============================================================
# ALPACA REQUESTS
# ============================================================

def alpaca_get(
    path,
    params=None,
    data_api=False
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
        timeout=25,
    )

    response.raise_for_status()

    return response.json()


def alpaca_post(
    path,
    payload=None
):
    response = requests.post(
        f"{TRADING_BASE_URL}{path}",
        headers={
            **HEADERS,
            "Content-Type": "application/json",
        },
        json=payload or {},
        timeout=25,
    )

    response.raise_for_status()

    return (
        response.json()
        if response.text
        else {}
    )


# ============================================================
# ACCOUNT / CLOCK
# ============================================================

def get_account():
    try:
        return alpaca_get(
            "/v2/account"
        )

    except Exception as e:
        add_error(
            f"Account error: {safe_text(e)}"
        )
        return {}


def market_clock():
    try:
        return alpaca_get(
            "/v2/clock"
        )

    except Exception as e:
        add_error(
            f"Clock error: {safe_text(e)}"
        )
        return {}


def market_is_open():
    return bool(
        market_clock().get(
            "is_open",
            False
        )
    )


# ============================================================
# STOCK UNIVERSE
# ============================================================

def get_active_stock_universe(
    force=False
):
    loaded_at = _universe_cache[
        "loaded_at"
    ]

    if (
        not force
        and _universe_cache["symbols"]
        and loaded_at
        and (
            now_et() - loaded_at
        ).total_seconds() < 6 * 3600
    ):
        return _universe_cache[
            "symbols"
        ]

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
            symbol = asset.get(
                "symbol"
            )

            if not symbol:
                continue

            if not asset.get(
                "tradable",
                False
            ):
                continue

            if "." in symbol:
                continue

            symbols.append(
                symbol
            )

    except Exception as e:
        add_error(
            "Asset universe error: "
            + safe_text(e)
        )

    combined = list(
        dict.fromkeys(
            PRIORITY_SYMBOLS
            + symbols
        )
    )

    _universe_cache[
        "symbols"
    ] = combined

    _universe_cache[
        "loaded_at"
    ] = now_et()

    return combined


# ============================================================
# MARKET DATA
# ============================================================

def bars_to_df(bars):
    if not bars:
        return None

    df = pd.DataFrame(
        bars
    )

    if df.empty:
        return None

    df["timestamp"] = (
        pd.to_datetime(
            df["t"],
            utc=True
        )
        .dt
        .tz_convert(NY)
    )

    df["open"] = pd.to_numeric(
        df["o"],
        errors="coerce"
    )

    df["high"] = pd.to_numeric(
        df["h"],
        errors="coerce"
    )

    df["low"] = pd.to_numeric(
        df["l"],
        errors="coerce"
    )

    df["close"] = pd.to_numeric(
        df["c"],
        errors="coerce"
    )

    df["volume"] = pd.to_numeric(
        df["v"],
        errors="coerce"
    )

    return df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )


def get_bars(
    symbol,
    limit=160,
    start=None,
    end=None
):
    try:
        params = {
            "timeframe":
                f"{TIMEFRAME_MINUTES}Min",
            "limit": limit,
            "adjustment": "raw",
            "feed": DATA_FEED,
        }

        if start:
            params[
                "start"
            ] = start

        if end:
            params[
                "end"
            ] = end

        data = alpaca_get(
            f"/v2/stocks/{symbol}/bars",
            params=params,
            data_api=True,
        )

        return bars_to_df(
            data.get(
                "bars",
                []
            )
        )

    except Exception:
        return None


def get_historical_bars(
    symbol,
    days=30
):
    end = (
        now_et()
        .astimezone(
            ZoneInfo("UTC")
        )
    )

    start = (
        end
        - timedelta(
            days=days
        )
    )

    all_bars = []
    page_token = None

    for _ in range(12):
        params = {
            "timeframe":
                f"{TIMEFRAME_MINUTES}Min",
            "start":
                start.isoformat(),
            "end":
                end.isoformat(),
            "limit": 10000,
            "adjustment": "raw",
            "feed": DATA_FEED,
        }

        if page_token:
            params[
                "page_token"
            ] = page_token

        try:
            data = alpaca_get(
                f"/v2/stocks/{symbol}/bars",
                params=params,
                data_api=True,
            )

        except Exception:
            break

        all_bars.extend(
            data.get(
                "bars",
                []
            )
        )

        page_token = data.get(
            "next_page_token"
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
        or len(df) < 35
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

    df["ema5"] = (
        df["close"]
        .ewm(
            span=5,
            adjust=False
        )
        .mean()
    )

    df["ema9"] = (
        df["close"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["ema30"] = (
        df["close"]
        .ewm(
            span=30,
            adjust=False
        )
        .mean()
    )

    session = (
        df["timestamp"]
        .dt
        .date
    )

    typical = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3.0

    pv = (
        typical
        * df["volume"]
    )

    cumulative_pv = (
        pv.groupby(
            session
        )
        .cumsum()
    )

    cumulative_volume = (
        df["volume"]
        .groupby(
            session
        )
        .cumsum()
        .replace(
            0,
            np.nan
        )
    )

    df["vwap"] = (
        cumulative_pv
        / cumulative_volume
    )

    df["volume_avg"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["rvol"] = (
        df["volume"]
        / df["volume_avg"]
        .replace(
            0,
            np.nan
        )
    )

    return df


# ============================================================
# SIGNAL LOGIC
# ============================================================

def signal_from_row(
    df,
    i,
    symbol
):
    if i < 31:
        return None

    current = df.iloc[i]
    previous = df.iloc[i - 1]

    price = float(
        current["close"]
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

    if pd.isna(
        current["rvol"]
    ):
        rvol = 0.0
    else:
        rvol = float(
            current["rvol"]
        )

    dollar_volume = float(
        current["volume"]
        * current["close"]
    )

    if price < MIN_PRICE:
        return None

    bullish = (
        ema5 > ema9
        and ema9 > ema30
        and price > vwap
        and ema5 > vwap
        and current["close"]
            > current["open"]
    )

    bearish = (
        ema5 < ema9
        and ema9 < ema30
        and price < vwap
        and ema5 < vwap
        and current["close"]
            < current["open"]
    )

    bullish_cross = (
        previous["ema5"]
            <= previous["ema9"]
        and current["ema5"]
            > current["ema9"]
    )

    bearish_cross = (
        previous["ema5"]
            >= previous["ema9"]
        and current["ema5"]
            < current["ema9"]
    )

    if bullish:
        direction = "CALL"

    elif bearish:
        direction = "PUT"

    else:
        return None

    score = 0

    if rvol >= MIN_RVOL:
        score += 2

    if (
        dollar_volume
        >= MIN_DOLLAR_VOLUME
    ):
        score += 1

    if (
        bullish_cross
        or bearish_cross
    ):
        score += 2

    score += 3

    bt = (
        bot_state[
            "backtest_by_symbol"
        ]
        .get(symbol)
    )

    if (
        bt
        and bt.get(
            "trades",
            0
        ) >= BACKTEST_MIN_TRADES
    ):
        if (
            bt.get(
                "win_rate",
                0
            ) >= 0.60
        ):
            score += 2

        elif (
            bt.get(
                "win_rate",
                0
            ) >= 0.50
        ):
            score += 1

    if any(
        item.get(
            "symbol"
        ) == symbol
        for item
        in bot_state[
            "premarket_watchlist"
        ]
    ):
        score += 2

    return {
        "symbol":
            symbol,
        "direction":
            direction,
        "price":
            round(
                price,
                2
            ),
        "ema5":
            round(
                ema5,
                4
            ),
        "ema9":
            round(
                ema9,
                4
            ),
        "ema30":
            round(
                ema30,
                4
            ),
        "vwap":
            round(
                vwap,
                4
            ),
        "rvol":
            round(
                rvol,
                2
            ),
        "score":
            score,
        "timestamp":
            current[
                "timestamp"
            ].isoformat(),
    }


def detect_signal(
    symbol,
    df
):
    df = calculate_indicators(
        df
    )

    if (
        df is None
        or len(df) < 35
    ):
        return None

    signal = signal_from_row(
        df,
        len(df) - 1,
        symbol
    )

    if not signal:
        return None

    if BACKTEST_FILTER_ENABLED:
        bt = (
            bot_state[
                "backtest_by_symbol"
            ]
            .get(symbol)
        )

        if (
            bt
            and bt.get(
                "trades",
                0
            ) >= BACKTEST_MIN_TRADES
        ):
            if (
                bt.get(
                    "win_rate",
                    0
                )
                < BACKTEST_MIN_WIN_RATE
            ):
                return None

    return signal


# ============================================================
# PREMARKET WATCHLIST
# ============================================================

def premarket_score(
    symbol,
    df
):
    df = calculate_indicators(
        df
    )

    if (
        df is None
        or len(df) < 35
    ):
        return None

    last = df.iloc[-1]

    price = float(
        last["close"]
    )

    if pd.isna(
        last["rvol"]
    ):
        rvol = 0.0
    else:
        rvol = float(
            last["rvol"]
        )

    ema5 = float(
        last["ema5"]
    )

    ema9 = float(
        last["ema9"]
    )

    ema30 = float(
        last["ema30"]
    )

    vwap = float(
        last["vwap"]
    )

    bullish = (
        ema5 > ema9 > ema30
        and price > vwap
    )

    bearish = (
        ema5 < ema9 < ema30
        and price < vwap
    )

    if bullish:
        direction = "CALL"

    elif bearish:
        direction = "PUT"

    else:
        direction = "WATCH"

    score = 0.0

    score += (
        min(
            rvol,
            3.0
        )
        * 2.0
    )

    if bullish or bearish:
        score += 4.0

    bt = (
        bot_state[
            "backtest_by_symbol"
        ]
        .get(symbol)
    )

    if (
        bt
        and bt.get(
            "trades",
            0
        ) >= BACKTEST_MIN_TRADES
    ):
        score += max(
            0.0,
            (
                bt.get(
                    "win_rate",
                    0.0
                )
                - 0.45
            )
            * 10.0
        )

    return {
        "symbol":
            symbol,
        "direction_bias":
            direction,
        "score":
            round(
                score,
                2
            ),
        "price":
            round(
                price,
                2
            ),
        "rvol":
            round(
                rvol,
                2
            ),
        "backtest_win_rate":
            (
                None
                if not bt
                else round(
                    bt.get(
                        "win_rate",
                        0
                    )
                    * 100,
                    1
                )
            ),
        "backtest_trades":
            (
                0
                if not bt
                else bt.get(
                    "trades",
                    0
                )
            ),
    }


def build_premarket_watchlist():
    universe = (
        get_active_stock_universe()
    )

    symbols = list(
        dict.fromkeys(
            PRIORITY_SYMBOLS
            + universe[
                :SCAN_LIMIT
            ]
        )
    )

    ranked = []

    for symbol in symbols:
        try:
            item = premarket_score(
                symbol,
                get_bars(
                    symbol,
                    limit=160
                )
            )

            if item:
                ranked.append(
                    item
                )

        except Exception:
            continue

    ranked.sort(
        key=lambda x:
            x["score"],
        reverse=True
    )

    bot_state[
        "premarket_watchlist"
    ] = ranked[
        :PREMARKET_WATCHLIST_SIZE
    ]

    bot_state[
        "premarket_last_run"
    ] = now_et().isoformat()

    log(
        "PREMARKET WATCHLIST READY"
    )

    for item in (
        bot_state[
            "premarket_watchlist"
        ][:10]
    ):
        log(
            f'{item["symbol"]} '
            f'{item["direction_bias"]} '
            f'score={item["score"]} '
            f'RVOL={item["rvol"]}'
        )

    return bot_state[
        "premarket_watchlist"
    ]


# ============================================================
# BACKTEST
# ============================================================

def backtest_symbol(symbol):
    df = get_historical_bars(
        symbol,
        BACKTEST_LOOKBACK_DAYS
    )

    df = calculate_indicators(
        df
    )

    if (
        df is None
        or len(df) < 80
    ):
        return {
            "symbol":
                symbol,
            "trades":
                0,
            "wins":
                0,
            "losses":
                0,
            "win_rate":
                0.0,
            "avg_underlying_return":
                0.0,
        }

    t = (
        df["timestamp"]
        .dt
        .time
    )

    df = df[
        (
            t
            >= dt_time(
                9,
                30
            )
        )
        &
        (
            t
            <= dt_time(
                15,
                15
            )
        )
    ].reset_index(
        drop=True
    )

    trades = []
    i = 31

    while i < len(df) - 2:
        row_time = (
            df.iloc[i][
                "timestamp"
            ]
        )

        if (
            row_time.time()
            >= LAST_ENTRY_TIME
        ):
            i += 1
            continue

        signal = signal_from_row(
            df,
            i,
            symbol
        )

        if not signal:
            i += 1
            continue

        direction = signal[
            "direction"
        ]

        entry = float(
            df.iloc[i][
                "close"
            ]
        )

        exit_price = entry
        j = i + 1

        target = 0.010
        stop = 0.007

        while j < len(df):
            candle = df.iloc[j]

            if (
                candle[
                    "timestamp"
                ].date()
                != row_time.date()
            ):
                break

            close = float(
                candle[
                    "close"
                ]
            )

            move = (
                close - entry
            ) / entry

            if (
                direction
                == "CALL"
            ):
                directional_move = (
                    move
                )
            else:
                directional_move = (
                    -move
                )

            ema9_invalid = (
                (
                    direction
                    == "CALL"
                    and close
                    < float(
                        candle[
                            "ema9"
                        ]
                    )
                )
                or
                (
                    direction
                    == "PUT"
                    and close
                    > float(
                        candle[
                            "ema9"
                        ]
                    )
                )
            )

            if (
                directional_move
                >= target
            ):
                exit_price = close
                break

            if (
                directional_move
                <= -stop
            ):
                exit_price = close
                break

            if ema9_invalid:
                exit_price = close
                break

            if (
                candle[
                    "timestamp"
                ].time()
                >= FORCE_EXIT_TIME
            ):
                exit_price = close
                break

            j += 1

        raw_move = (
            exit_price - entry
        ) / entry

        if (
            direction
            == "CALL"
        ):
            result = raw_move
        else:
            result = -raw_move

        trades.append(
            result
        )

        i = max(
            i + 2,
            j + 1
        )

    wins = sum(
        1
        for result in trades
        if result > 0
    )

    losses = sum(
        1
        for result in trades
        if result <= 0
    )

    total = len(
        trades
    )

    win_rate = (
        wins / total
        if total
        else 0.0
    )

    avg_return = (
        float(
            np.mean(
                trades
            )
        )
        if trades
        else 0.0
    )

    return {
        "symbol":
            symbol,
        "trades":
            total,
        "wins":
            wins,
        "losses":
            losses,
        "win_rate":
            round(
                win_rate,
                4
            ),
        "avg_underlying_return":
            round(
                avg_return,
                5
            ),
    }


def run_backtest():
    if not BACKTEST_ENABLED:
        return {}

    symbols = (
        PRIORITY_SYMBOLS[
            :BACKTEST_SYMBOL_LIMIT
        ]
    )

    results = {}

    log(
        f"Running "
        f"{BACKTEST_LOOKBACK_DAYS}-day "
        f"signal backtest..."
    )

    for symbol in symbols:
        try:
            result = (
                backtest_symbol(
                    symbol
                )
            )

            results[
                symbol
            ] = result

            log(
                f'BACKTEST {symbol}: '
                f'trades={result["trades"]} '
                f'win_rate='
                f'{result["win_rate"]:.1%}'
            )

        except Exception as e:
            add_error(
                f"Backtest {symbol}: "
                f"{safe_text(e)}"
            )

    all_trades = sum(
        item["trades"]
        for item
        in results.values()
    )

    all_wins = sum(
        item["wins"]
        for item
        in results.values()
    )

    overall_win = (
        all_wins / all_trades
        if all_trades
        else 0.0
    )

    bot_state[
        "backtest_by_symbol"
    ] = results

    bot_state[
        "backtest_summary"
    ] = {
        "lookback_days":
            BACKTEST_LOOKBACK_DAYS,
        "symbols_tested":
            len(results),
        "trades":
            all_trades,
        "wins":
            all_wins,
        "win_rate":
            round(
                overall_win,
                4
            ),
        "note":
            "Underlying signal backtest; "
            "not historical option-premium P/L.",
    }

    bot_state[
        "backtest_last_run"
    ] = now_et().isoformat()

    return bot_state[
        "backtest_summary"
    ]


# ============================================================
# LIVE SCANNER
# ============================================================

def scan_market():
    universe = (
        get_active_stock_universe()
    )

    watch = [
        item["symbol"]
        for item
        in bot_state[
            "premarket_watchlist"
        ]
    ]

    scan_symbols = list(
        dict.fromkeys(
            PRIORITY_SYMBOLS
            + watch
            + universe
        )
    )[:SCAN_LIMIT]

    signals = []
    scanned = 0

    for symbol in scan_symbols:
        try:
            bars = get_bars(
                symbol
            )

            scanned += 1

            signal = detect_signal(
                symbol,
                bars
            )

            if signal:
                signals.append(
                    signal
                )

        except Exception:
            continue

    signals.sort(
        key=lambda x: (
            x["score"],
            x["rvol"]
        ),
        reverse=True
    )

    bot_state[
        "signals"
    ] = signals

    bot_state[
        "candidates"
    ] = signals[:20]

    bot_state[
        "last_scan"
    ] = now_et().isoformat()

    bot_state[
        "stocks_scanned_this_cycle"
    ] = scanned

    return signals


# ============================================================
# 0DTE OPTIONS
# ============================================================

def get_0dte_contracts(
    symbol,
    direction
):
    try:
        option_type = (
            "call"
            if direction
            == "CALL"
            else "put"
        )

        data = alpaca_get(
            "/v2/options/contracts",
            params={
                "underlying_symbols":
                    symbol,
                "expiration_date":
                    today_string(),
                "type":
                    option_type,
                "status":
                    "active",
                "limit":
                    100,
            },
        )

        return data.get(
            "option_contracts",
            []
        )

    except Exception as e:
        add_error(
            f"{symbol} option chain error: "
            f"{safe_text(e)}"
        )

        return []


def choose_0dte_contract(
    symbol,
    direction,
    stock_price
):
    contracts = (
        get_0dte_contracts(
            symbol,
            direction
        )
    )

    valid = []

    for contract in contracts:
        try:
            strike = float(
                contract.get(
                    "strike_price",
                    0
                )
            )

            option_symbol = (
                contract.get(
                    "symbol"
                )
            )

            if option_symbol:
                valid.append(
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

    if not valid:
        return None

    valid.sort(
        key=lambda x:
            x[0]
    )

    _, strike, option_symbol = (
        valid[0]
    )

    return {
        "symbol":
            option_symbol,
        "strike":
            strike,
        "underlying":
            symbol,
        "direction":
            direction,
    }


def get_option_quote(
    option_symbol
):
    try:
        data = alpaca_get(
            "/v1beta1/options/quotes/latest",
            params={
                "symbols":
                    option_symbol
            },
            data_api=True,
        )

        quote = (
            data.get(
                "quotes",
                {}
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
                0
            )
            or 0
        )

        ask = float(
            quote.get(
                "ap",
                0
            )
            or 0
        )

        if (
            bid <= 0
            and ask <= 0
        ):
            return None

        if (
            bid > 0
            and ask > 0
        ):
            mid = (
                bid + ask
            ) / 2

        else:
            mid = max(
                bid,
                ask
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
# POSITIONS / ORDERS
# ============================================================

def get_positions():
    try:
        return alpaca_get(
            "/v2/positions"
        )

    except Exception as e:
        add_error(
            "Position error: "
            + safe_text(e)
        )

        return []


def open_option_positions():
    result = []

    for position in get_positions():
        asset_class = str(
            position.get(
                "asset_class",
                ""
            )
        ).lower()

        if (
            "option"
            in asset_class
        ):
            result.append(
                position
            )

    return result


def submit_option_buy(
    option_symbol,
    quantity
):
    payload = {
        "symbol":
            option_symbol,
        "qty":
            str(quantity),
        "side":
            "buy",
        "type":
            "market",
        "time_in_force":
            "day",
    }

    if not AUTO_TRADE:
        log(
            f"PAPER SIGNAL ONLY: "
            f"BUY {quantity} "
            f"{option_symbol}"
        )

        return {
            "paper_signal":
                True,
            **payload
        }

    return alpaca_post(
        "/v2/orders",
        payload
    )


def submit_option_sell(
    option_symbol,
    quantity
):
    payload = {
        "symbol":
            option_symbol,
        "qty":
            str(quantity),
        "side":
            "sell",
        "type":
            "market",
        "time_in_force":
            "day",
    }

    if not AUTO_TRADE:
        log(
            f"PAPER SIGNAL ONLY: "
            f"SELL {quantity} "
            f"{option_symbol}"
        )

        return {
            "paper_signal":
                True,
            **payload
        }

    return alpaca_post(
        "/v2/orders",
        payload
    )


# ============================================================
# ENTRY
# ============================================================

def enter_trade(signal):
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

    stock_symbol = (
        signal["symbol"]
    )

    direction = (
        signal["direction"]
    )

    stock_price = (
        signal["price"]
    )

    if any(
        trade.get(
            "underlying"
        ) == stock_symbol
        for trade
        in managed_positions.values()
    ):
        return False

    contract = (
        choose_0dte_contract(
            stock_symbol,
            direction,
            stock_price
        )
    )

    if not contract:
        return False

    option_symbol = (
        contract["symbol"]
    )

    quote = get_option_quote(
        option_symbol
    )

    if (
        not quote
        or quote["mid"] <= 0
    ):
        return False

    premium = quote[
        "mid"
    ]

    contract_cost = (
        premium
        * 100
    )

    if (
        contract_cost
        > POSITION_DOLLARS
    ):
        log(
            f"Skipping {option_symbol}: "
            f"contract cost "
            f"${contract_cost:.2f} > "
            f"${POSITION_DOLLARS:.2f}"
        )

        return False

    quantity = int(
        POSITION_DOLLARS
        // contract_cost
    )

    if quantity < 1:
        return False

    order = (
        submit_option_buy(
            option_symbol,
            quantity
        )
    )

    managed_positions[
        option_symbol
    ] = {
        "underlying":
            stock_symbol,
        "direction":
            direction,
        "entry_price":
            premium,
        "quantity":
            quantity,
        "original_quantity":
            quantity,
        "tp_hit":
            False,
        "highest_after_tp":
            premium,
        "entry_time":
            now_et().isoformat(),
    }

    log(
        f"ENTRY {stock_symbol} "
        f"{direction} | "
        f"{option_symbol} | "
        f"premium ${premium:.2f} | "
        f"qty {quantity}"
    )

    return order


# ============================================================
# POSITION MANAGEMENT
# ============================================================

def close_managed_position(
    option_symbol,
    reason
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
            0
        )
    )

    if quantity > 0:
        submit_option_sell(
            option_symbol,
            quantity
        )

        log(
            f"EXIT {option_symbol} | "
            f"{reason} | "
            f"qty {quantity}"
        )

    managed_positions.pop(
        option_symbol,
        None
    )


def underlying_ema9_invalidated(
    trade
):
    bars = get_bars(
        trade[
            "underlying"
        ],
        limit=50
    )

    df = calculate_indicators(
        bars
    )

    if df is None:
        return False

    candle = df.iloc[-1]

    close = float(
        candle["close"]
    )

    ema9 = float(
        candle["ema9"]
    )

    if (
        trade["direction"]
        == "CALL"
    ):
        return (
            close < ema9
        )

    if (
        trade["direction"]
        == "PUT"
    ):
        return (
            close > ema9
        )

    return False


def manage_positions():
    current_time = (
        now_et().time()
    )

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

        if (
            current_time
            >= FORCE_EXIT_TIME
        ):
            close_managed_position(
                option_symbol,
                "0DTE FORCE EXIT "
                "3:15 PM ET"
            )
            continue

        quote = get_option_quote(
            option_symbol
        )

        if not quote:
            continue

        premium = (
            quote["mid"]
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
            premium - entry
        ) / entry

        if (
            not trade["tp_hit"]
            and pnl_percent
            <= -STOP_LOSS_PERCENT
        ):
            close_managed_position(
                option_symbol,
                f"HARD STOP "
                f"{pnl_percent:.1%}"
            )
            continue

        if (
            not trade["tp_hit"]
            and underlying_ema9_invalidated(
                trade
            )
        ):
            close_managed_position(
                option_symbol,
                "EMA9 INVALIDATION"
            )
            continue

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
                    )
                )

                sell_quantity = min(
                    sell_quantity,
                    quantity - 1
                )

                if sell_quantity > 0:
                    submit_option_sell(
                        option_symbol,
                        sell_quantity
                    )

                    trade[
                        "quantity"
                    ] -= sell_quantity

                    log(
                        f"TAKE PROFIT "
                        f"{option_symbol} | "
                        f"{pnl_percent:.1%} | "
                        f"sold "
                        f"{sell_quantity} | "
                        f"runner "
                        f"{trade['quantity']}"
                    )

            trade[
                "tp_hit"
            ] = True

            trade[
                "highest_after_tp"
            ] = premium

            log(
                f"RUNNER ACTIVE "
                f"{option_symbol}"
            )

            continue

        if trade["tp_hit"]:
            trade[
                "highest_after_tp"
            ] = max(
                float(
                    trade[
                        "highest_after_tp"
                    ]
                ),
                premium
            )

            high = float(
                trade[
                    "highest_after_tp"
                ]
            )

            trailing_stop = (
                high
                * (
                    1
                    - RUNNER_TRAIL_PERCENT
                )
            )

            if (
                premium
                <= trailing_stop
            ):
                close_managed_position(
                    option_symbol,
                    f"RUNNER TRAILING STOP "
                    f"${premium:.2f}"
                )


# ============================================================
# DAILY PREP
# ============================================================

def maybe_run_daily_prep():
    now = now_et()

    current_time = (
        now.time()
    )

    today = (
        now.date()
        .isoformat()
    )

    if (
        BACKTEST_ENABLED
        and _daily_job_state[
            "backtest_date"
        ] != today
    ):
        if (
            current_time
            < dt_time(
                9,
                25
            )
            or current_time
            > dt_time(
                16,
                5
            )
        ):
            run_backtest()

            _daily_job_state[
                "backtest_date"
            ] = today

    if (
        PREMARKET_START
        <= current_time
        <= PREMARKET_END
    ):
        epoch = time.time()

        if (
            epoch
            - _daily_job_state[
                "premarket_last_epoch"
            ]
            >= PREMARKET_REFRESH_SECONDS
        ):
            build_premarket_watchlist()

            _daily_job_state[
                "premarket_last_epoch"
            ] = epoch


# ============================================================
# BOT CYCLE
# ============================================================

def bot_cycle():
    bot_state[
        "last_cycle"
    ] = now_et().isoformat()

    if not credentials_present():
        add_error(
            "Alpaca credentials "
            "are missing."
        )
        return

    maybe_run_daily_prep()

    if not market_is_open():
        return

    manage_positions()

    signals = scan_market()

    trades_entered = 0

    for signal in signals:
        if (
            trades_entered
            >= MAX_NEW_TRADES_PER_CYCLE
        ):
            break

        try:
            result = enter_trade(
                signal
            )

            if result:
                trades_entered += 1

        except Exception as e:
            add_error(
                f"Entry error "
                f"{signal.get('symbol')}: "
                f"{safe_text(e)}"
            )


# ============================================================
# BACKGROUND LOOP
# ============================================================

def bot_loop():
    bot_state[
        "running"
    ] = True

    log(
        "0DTE trading bot started."
    )

    while True:
        try:
            bot_cycle()

        except Exception as e:
            add_error(
                "Bot cycle error: "
                + safe_text(e)
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
        daemon=True
    )

    thread.start()


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def home():
    account = get_account()
    clock = market_clock()

    return jsonify({
        "status":
            "online",

        "bot":
            "Alpaca 0DTE Options Bot "
            "+ Backtest "
            "+ Premarket Prep",

        "paper_trading":
            True,

        "credentials_ok":
            credentials_present(),

        "auto_trade":
            AUTO_TRADE,

        "run_bot_loop":
            RUN_BOT_LOOP,

        "market_open":
            bool(
                clock.get(
                    "is_open",
                    False
                )
            ),

        "strategy":
            "4-minute "
            "EMA5/EMA9/EMA30 "
            "+ session VWAP "
            "+ volume",

        "backtest_enabled":
            BACKTEST_ENABLED,

        "backtest_filter_enabled":
            BACKTEST_FILTER_ENABLED,

        "premarket_watchlist_ready":
            bool(
                bot_state[
                    "premarket_watchlist"
                ]
            ),

        "stocks_loaded":
            len(
                get_active_stock_universe()
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

        "take_profit":
            f"{TAKE_PROFIT_PERCENT:.0%}",

        "runner_trail":
            f"{RUNNER_TRAIL_PERCENT:.0%}",

        "stop_loss":
            f"{STOP_LOSS_PERCENT:.0%}",

        "entry_cutoff":
            "2:45 PM ET",

        "force_exit":
            "3:15 PM ET",

        "errors":
            bot_state[
                "errors"
            ],
    })


@app.route("/health")
def health():
    return jsonify({
        "status":
            "healthy",

        "time":
            now_et().isoformat(),

        "bot_running":
            bot_state[
                "running"
            ],
    })


@app.route("/status")
def status():
    return jsonify({
        **bot_state,

        "managed_positions":
            managed_positions,

        "auto_trade":
            AUTO_TRADE,
    })


@app.route("/scan")
def manual_scan():
    try:
        results = scan_market()

        return jsonify({
            "count":
                len(results),

            "results":
                results[:50],
        })

    except Exception as e:
        return jsonify({
            "error":
                safe_text(e)
        }), 500


@app.route("/premarket")
def manual_premarket():
    try:
        results = (
            build_premarket_watchlist()
        )

        return jsonify({
            "count":
                len(results),

            "watchlist":
                results,
        })

    except Exception as e:
        return jsonify({
            "error":
                safe_text(e)
        }), 500


@app.route("/backtest")
def manual_backtest():
    try:
        summary = run_backtest()

        return jsonify({
            "summary":
                summary,

            "by_symbol":
                bot_state[
                    "backtest_by_symbol"
                ],
        })

    except Exception as e:
        return jsonify({
            "error":
                safe_text(e)
        }), 500


# ============================================================
# START
# ============================================================

start_background_bot()


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
        debug=False
    )