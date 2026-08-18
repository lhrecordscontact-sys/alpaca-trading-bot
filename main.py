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
FORCE_EXIT_TIME = dt_time(15, 15)

ROLLING_TRADE_COUNT = 64
MIN_STATS_TRADES = 64
MIN_WIN_RATE = float(os.getenv("MIN_WIN_RATE", "0.80"))

SCAN_LIMIT = int(os.getenv("SCAN_LIMIT", "150"))
LOOP_SECONDS = int(os.getenv("LOOP_SECONDS", "45"))
BACKTEST_LOOKBACK_DAYS = int(os.getenv("BACKTEST_LOOKBACK_DAYS", "180"))

PRIORITY_SYMBOLS = [
    "SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA", "AMD", "AMZN",
    "META", "MSFT", "GOOGL", "NFLX", "AVGO", "PLTR", "COIN",
    "MSTR", "AMAT",
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
        .encode("ascii", errors="ignore")
        .decode("ascii")
        .strip()
    )


ALPACA_API_KEY = clean_credential(os.getenv("ALPACA_API_KEY", ""))
ALPACA_SECRET_KEY = clean_credential(os.getenv("ALPACA_SECRET_KEY", ""))

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
    "stats_by_symbol": {},
    "errors": [],
}

_universe_cache = {
    "symbols": [],
    "loaded_at": None,
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
            .encode("ascii", errors="replace")
            .decode("ascii")
        )
    except Exception:
        return "Unknown error"


def log(message):
    stamp = now_et().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp} ET] {safe_text(message)}", flush=True)


def add_error(message):
    message = safe_text(message)

    bot_state["errors"].append(message)
    bot_state["errors"] = bot_state["errors"][-25:]

    log(f"ERROR: {message}")


def in_time_window(value, start, end):
    return start <= value < end


# ============================================================
# ALPACA REQUESTS
# ============================================================

def alpaca_get(path, params=None, data_api=False):
    base = DATA_BASE_URL if data_api else TRADING_BASE_URL
    url = f"{base}{path}"

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=30,
        )
    except requests.RequestException as e:
        raise RuntimeError(
            f"GET {url} network error: {safe_text(e)}"
        ) from e

    if not response.ok:
        body = safe_text(response.text)[:1000]
        raise RuntimeError(
            f"GET {path} HTTP {response.status_code} | "
            f"url={safe_text(response.url)} | body={body}"
        )

    try:
        return response.json()
    except Exception as e:
        raise RuntimeError(
            f"GET {path} returned invalid JSON: {safe_text(e)}"
        ) from e


def alpaca_post(path, payload):
    url = f"{TRADING_BASE_URL}{path}"

    try:
        response = requests.post(
            url,
            headers=HEADERS,
            json=payload,
            timeout=30,
        )
    except requests.RequestException as e:
        raise RuntimeError(
            f"POST {url} network error: {safe_text(e)}"
        ) from e

    if not response.ok:
        body = safe_text(response.text)[:1000]
        raise RuntimeError(
            f"POST {path} HTTP {response.status_code} | body={body}"
        )

    if not response.text:
        return {}

    return response.json()


# ============================================================
# VERIFY ACCOUNT
# ============================================================

def verify_credentials():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        bot_state["credentials_ok"] = False
        add_error("Alpaca credentials are missing.")
        return False

    try:
        account = alpaca_get("/v2/account")

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
        clock = alpaca_get("/v2/clock")

        bot_state["market_open"] = bool(
            clock.get("is_open", False)
        )

        return bot_state["market_open"]

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
    loaded_at = _universe_cache["loaded_at"]

    if (
        not force
        and _universe_cache["symbols"]
        and loaded_at
        and (now_et() - loaded_at).total_seconds() < 6 * 3600
    ):
        return _universe_cache["symbols"]

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

            if (
                symbol
                and asset.get("tradable", False)
                and "." not in symbol
            ):
                symbols.append(symbol)

    except Exception as e:
        add_error(
            "Universe load error: "
            f"{safe_text(e)}"
        )

    symbols = list(
        dict.fromkeys(PRIORITY_SYMBOLS + symbols)
    )

    _universe_cache["symbols"] = symbols
    _universe_cache["loaded_at"] = now_et()

    return symbols


# ============================================================
# STOCK DATA
# ============================================================

def bars_to_df(bars):
    if not bars:
        return None

    df = pd.DataFrame(bars)

    if df.empty:
        return None

    required = {"t", "o", "h", "l", "c", "v"}

    if not required.issubset(df.columns):
        raise RuntimeError(
            f"Unexpected bar fields: {list(df.columns)}"
        )

    df["timestamp"] = (
        pd.to_datetime(df["t"], utc=True)
        .dt.tz_convert(NY)
    )

    df["open"] = pd.to_numeric(df["o"], errors="coerce")
    df["high"] = pd.to_numeric(df["h"], errors="coerce")
    df["low"] = pd.to_numeric(df["l"], errors="coerce")
    df["close"] = pd.to_numeric(df["c"], errors="coerce")
    df["volume"] = pd.to_numeric(df["v"], errors="coerce")

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


def get_recent_bars(symbol, limit=1000):
    try:
        data = alpaca_get(
            f"/v2/stocks/{symbol}/bars",
            params={
                "timeframe": f"{TIMEFRAME_MINUTES}Min",
                "limit": limit,
                "adjustment": "raw",
                "feed": DATA_FEED,
            },
            data_api=True,
        )

        bars = data.get("bars", [])

        if not bars:
            add_error(
                f"RECENT DATA EMPTY {symbol} | "
                f"feed={DATA_FEED} | "
                f"response_keys={list(data.keys())}"
            )
            return None

        return bars_to_df(bars)

    except Exception as e:
        add_error(
            f"RECENT DATA ERROR {symbol} | "
            f"feed={DATA_FEED} | "
            f"{safe_text(e)}"
        )
        return None


def get_historical_bars(symbol, days=180):
    end = now_et().astimezone(UTC)
    start = end - timedelta(days=days)

    all_bars = []
    page_token = None

    for page_number in range(1, 31):
        params = {
            "timeframe": f"{TIMEFRAME_MINUTES}Min",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
            "adjustment": "raw",
            "feed": DATA_FEED,
            "sort": "asc",
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
            add_error(
                f"HISTORICAL DATA ERROR {symbol} | "
                f"feed={DATA_FEED} | "
                f"page={page_number} | "
                f"{safe_text(e)}"
            )
            break

        bars = data.get("bars", [])

        if not bars and not all_bars:
            add_error(
                f"HISTORICAL DATA EMPTY {symbol} | "
                f"feed={DATA_FEED} | "
                f"page={page_number} | "
                f"response_keys={list(data.keys())}"
            )
            break

        all_bars.extend(bars)

        page_token = data.get("next_page_token")

        if not page_token:
            break

    if not all_bars:
        return None

    try:
        df = bars_to_df(all_bars)

        if df is not None:
            log(
                f"HISTORY {symbol}: "
                f"{len(df)} bars loaded | "
                f"feed={DATA_FEED}"
            )

        return df

    except Exception as e:
        add_error(
            f"HISTORICAL PARSE ERROR {symbol}: "
            f"{safe_text(e)}"
        )
        return None


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):
    if df is None or len(df) < 40:
        return None

    df = (
        df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    df["ema5"] = (
        df["close"]
        .ewm(span=EMA_FAST, adjust=False)
        .mean()
    )

    df["ema9"] = (
        df["close"]
        .ewm(span=EMA_SLOW, adjust=False)
        .mean()
    )

    df["ema30"] = (
        df["close"]
        .ewm(span=EMA_TREND, adjust=False)
        .mean()
    )

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = (
        true_range
        .rolling(ATR_LENGTH)
        .mean()
    )

    session_date = df["timestamp"].dt.date

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3.0

    price_volume = typical_price * df["volume"]

    cumulative_volume = (
        df["volume"]
        .groupby(session_date)
        .cumsum()
        .replace(0, np.nan)
    )

    df["vwap"] = (
        price_volume
        .groupby(session_date)
        .cumsum()
        / cumulative_volume
    )

    df["pm_high"] = np.nan
    df["pm_low"] = np.nan

    for _, indexes in df.groupby(session_date).groups.items():
        indexes = list(indexes)
        rows = df.loc[indexes]
        times = rows["timestamp"].dt.time

        premarket = rows[
            (times >= PREMARKET_START)
            & (times < PREMARKET_END)
        ]

        if premarket.empty:
            continue

        pm_high = float(premarket["high"].max())
        pm_low = float(premarket["low"].min())

        df.loc[indexes, "pm_high"] = pm_high
        df.loc[indexes, "pm_low"] = pm_low

    return df


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signals(df, symbol):
    df = calculate_indicators(df)

    if df is None:
        return []

    signals = []

    current_day = None
    long_break_index = None
    short_break_index = None
    long_used = False
    short_used = False

    for i in range(1, len(df)):
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
            or pd.isna(row["vwap"])
        ):
            continue

        bullish_trend = (
            row["ema5"] > row["ema9"] > row["ema30"]
            and row["close"] > row["vwap"]
            and row["close"] > row["ema30"]
        )

        bearish_trend = (
            row["ema5"] < row["ema9"] < row["ema30"]
            and row["close"] < row["vwap"]
            and row["close"] < row["ema30"]
        )

        if (
            not long_used
            and row["close"] > pm_high
            and previous["close"] <= pm_high
        ):
            long_break_index = i

        if (
            not short_used
            and row["close"] < pm_low
            and previous["close"] >= pm_low
        ):
            short_break_index = i

        if (
            long_break_index is not None
            and i - long_break_index > MAX_RETEST_BARS
        ):
            long_break_index = None

        if (
            short_break_index is not None
            and i - short_break_index > MAX_RETEST_BARS
        ):
            short_break_index = None

        tolerance = float(row["atr"]) * RETEST_ATR_TOLERANCE

        if (
            long_break_index is not None
            and i > long_break_index
            and not long_used
            and bullish_trend
            and row["low"] <= pm_high + tolerance
            and row["close"] > pm_high
        ):
            signals.append(
                {
                    "symbol": symbol,
                    "side": "CALL",
                    "timestamp": timestamp,
                    "underlying_entry": float(row["close"]),
                }
            )

            long_used = True
            long_break_index = None

        if (
            short_break_index is not None
            and i > short_break_index
            and not short_used
            and bearish_trend
            and row["high"] >= pm_low - tolerance
            and row["close"] < pm_low
        ):
            signals.append(
                {
                    "symbol": symbol,
                    "side": "PUT",
                    "timestamp": timestamp,
                    "underlying_entry": float(row["close"]),
                }
            )

            short_used = True
            short_break_index = None

    return signals


# ============================================================
# DIAGNOSTIC BACKTEST
# ============================================================

def evaluate_signal_direction(df, signal, max_bars=12):
    timestamp = signal["timestamp"]

    matches = df.index[
        df["timestamp"] == timestamp
    ].tolist()

    if not matches:
        return None

    i = matches[0]

    if i + 1 >= len(df):
        return None

    future = df.iloc[
        i + 1:
        min(i + 1 + max_bars, len(df))
    ]

    if future.empty:
        return None

    entry = float(signal["underlying_entry"])

    if signal["side"] == "CALL":
        favorable = float(future["high"].max()) - entry
        adverse = entry - float(future["low"].min())
    else:
        favorable = entry - float(future["low"].min())
        adverse = float(future["high"].max()) - entry

    return favorable > adverse


def build_symbol_stats(symbol):
    df = get_historical_bars(
        symbol,
        days=BACKTEST_LOOKBACK_DAYS,
    )

    if df is None or df.empty:
        stats = {
            "symbol": symbol,
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "status": "WAITING",
            "reason": "NO_STOCK_HISTORY",
        }

        bot_state["stats_by_symbol"][symbol] = stats

        log(
            f"STATS {symbol}: 0/0 wins | "
            f"overall=0.0% | WAITING | "
            f"NO_STOCK_HISTORY"
        )

        return stats

    signals = generate_signals(df, symbol)

    completed = []

    for signal in signals:
        result = evaluate_signal_direction(df, signal)

        if result is not None:
            completed.append(bool(result))

    completed = completed[-ROLLING_TRADE_COUNT:]

    wins = sum(completed)
    trades = len(completed)

    win_rate = (
        wins / trades
        if trades
        else 0.0
    )

    status = "DIAGNOSTIC_ONLY"

    reason = (
        "NEEDS_OPTION_PREMIUM_BACKTEST"
        if trades >= MIN_STATS_TRADES
        else "INSUFFICIENT_SIGNALS"
    )

    stats = {
        "symbol": symbol,
        "trades": trades,
        "wins": wins,
        "win_rate": round(win_rate, 4),
        "status": status,
        "reason": reason,
    }

    bot_state["stats_by_symbol"][symbol] = stats

    log(
        f"STATS {symbol}: "
        f"{wins}/{trades} directional wins | "
        f"overall={win_rate * 100:.1f}% | "
        f"{status} | {reason}"
    )

    return stats


# ============================================================
# LIVE SCANNER
# ============================================================

def latest_live_signal(symbol):
    df = get_recent_bars(symbol, limit=1000)

    if df is None or len(df) < 50:
        return None

    signals = generate_signals(df, symbol)

    if not signals:
        return None

    signal = signals[-1]

    age = now_et() - signal["timestamp"]

    if age.total_seconds() > TIMEFRAME_MINUTES * 60 * 2:
        return None

    return signal


def run_scan_cycle():
    bot_state["last_cycle"] = now_et().isoformat()
    bot_state["stocks_scanned_this_cycle"] = 0

    universe = get_stock_universe()
    symbols = universe[:SCAN_LIMIT]

    found = []

    for symbol in symbols:
        try:
            signal = latest_live_signal(symbol)

            bot_state["stocks_scanned_this_cycle"] += 1

            if signal:
                found.append(
                    {
                        "symbol": signal["symbol"],
                        "side": signal["side"],
                        "timestamp": signal["timestamp"].isoformat(),
                        "underlying_entry": signal["underlying_entry"],
                    }
                )

                log(
                    f"SIGNAL {signal['symbol']} "
                    f"{signal['side']} | "
                    f"underlying={signal['underlying_entry']}"
                )

        except Exception as e:
            add_error(
                f"SCAN ERROR {symbol}: "
                f"{safe_text(e)}"
            )

    bot_state["signals"] = found[-50:]
    bot_state["last_scan"] = now_et().isoformat()


# ============================================================
# STARTUP DIAGNOSTICS
# ============================================================

def startup_diagnostics():
    for symbol in ["SPY", "QQQ", "IWM"]:
        try:
            build_symbol_stats(symbol)
        except Exception as e:
            add_error(
                f"STARTUP STATS ERROR {symbol}: "
                f"{safe_text(e)}"
            )


# ============================================================
# BOT LOOP
# ============================================================

def bot_loop():
    bot_state["running"] = True

    while True:
        try:
            market_is_open()

            if bot_state["market_open"]:
                run_scan_cycle()
            else:
                bot_state["last_cycle"] = now_et().isoformat()

        except Exception as e:
            add_error(
                "BOT LOOP ERROR: "
                f"{safe_text(e)}"
            )

        time.sleep(LOOP_SECONDS)


# ============================================================
# HTTP ROUTES
# ============================================================

@app.route("/")
def home():
    return jsonify(
        {
            "service": "alpaca-trading-bot",
            "paper": True,
            "running": bot_state["running"],
            "credentials_ok": bot_state["credentials_ok"],
            "market_open": bot_state["market_open"],
            "auto_trade": AUTO_TRADE,
            "data_feed": DATA_FEED,
            "option_feed": OPTION_FEED,
            "last_cycle": bot_state["last_cycle"],
            "last_scan": bot_state["last_scan"],
            "stocks_scanned_this_cycle":
                bot_state["stocks_scanned_this_cycle"],
            "signals": bot_state["signals"],
            "stats_by_symbol": bot_state["stats_by_symbol"],
            "errors": bot_state["errors"],
        }
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "credentials_ok": bot_state["credentials_ok"],
            "running": bot_state["running"],
        }
    )


@app.route("/history-test/<symbol>")
def history_test(symbol):
    symbol = symbol.upper().strip()

    df = get_historical_bars(
        symbol,
        days=min(BACKTEST_LOOKBACK_DAYS, 30),
    )

    if df is None or df.empty:
        return jsonify(
            {
                "ok": False,
                "symbol": symbol,
                "feed": DATA_FEED,
                "errors": bot_state["errors"][-10:],
            }
        ), 500

    return jsonify(
        {
            "ok": True,
            "symbol": symbol,
            "feed": DATA_FEED,
            "bars": len(df),
            "first": df.iloc[0]["timestamp"].isoformat(),
            "last": df.iloc[-1]["timestamp"].isoformat(),
        }
    )


# ============================================================
# MAIN
# ============================================================

def start_background_thread():
    thread = threading.Thread(
        target=bot_loop,
        daemon=True,
    )
    thread.start()


if __name__ == "__main__":
    if verify_credentials():
        log("ALPACA PAPER CONNECTED SUCCESSFULLY")

        log(
            "HISTORY-DEBUG 4-MINUTE BOT STARTED | "
            f"DATA_FEED={DATA_FEED} | "
            f"AUTO_TRADE={AUTO_TRADE}"
        )

        startup_diagnostics()

        if RUN_BOT_LOOP:
            start_background_thread()

    else:
        log("BOT STARTED WITHOUT VALID ALPACA CREDENTIALS")

    port = int(os.getenv("PORT", "10000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
    )