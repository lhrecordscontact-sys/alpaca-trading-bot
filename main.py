import os
import time
import math
import threading
import requests
import pandas as pd

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
OPTION_FEED = os.getenv("OPTION_FEED", "indicative").strip().lower()

AUTO_TRADE = os.getenv("AUTO_TRADE", "false").strip().lower() == "true"
RUN_BOT_LOOP = os.getenv("RUN_BOT_LOOP", "true").strip().lower() == "true"

POSITION_DOLLARS = float(os.getenv("POSITION_DOLLARS", "500"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
MAX_NEW_TRADES_PER_CYCLE = int(os.getenv("MAX_NEW_TRADES_PER_CYCLE", "1"))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.20"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "0.30"))
TAKE_PROFIT_FRACTION = float(os.getenv("TAKE_PROFIT_FRACTION", "0.50"))
RUNNER_TRAIL_PERCENT = float(os.getenv("RUNNER_TRAIL_PERCENT", "0.15"))

LOOP_SECONDS = int(os.getenv("LOOP_SECONDS", "30"))
WATCHLIST_SIZE = int(os.getenv("WATCHLIST_SIZE", "25"))

EMA_FAST = 5
EMA_SLOW = 9
EMA_TREND = 30

RTH_START = dt_time(9, 30)
LAST_ENTRY_TIME = dt_time(14, 45)
FORCE_EXIT_TIME = dt_time(15, 15)

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
    "AVGO",
    "PLTR",
    "COIN",
    "MSTR",
    "NFLX",
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
    "watchlist": [],
    "signals": [],
    "errors": [],
    "auto_trade": AUTO_TRADE,
}

managed_positions = {}
last_signal_key = {}

background_started = False


# ============================================================
# LOGGING
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
    print(
        f"[{now_et().strftime('%Y-%m-%d %H:%M:%S')} ET] "
        f"{safe_text(message)}",
        flush=True,
    )


def add_error(message):
    text = safe_text(message)

    bot_state["errors"].append(text)
    bot_state["errors"] = bot_state["errors"][-20:]

    log(f"ERROR | {text}")


# ============================================================
# HTTP
# ============================================================

def api_get(path, params=None, data_api=False):
    base = DATA_BASE_URL if data_api else TRADING_BASE_URL

    response = requests.get(
        f"{base}{path}",
        headers=HEADERS,
        params=params,
        timeout=20,
    )

    if not response.ok:
        raise RuntimeError(
            f"GET {path} HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    return response.json()


def api_post(path, payload):
    response = requests.post(
        f"{TRADING_BASE_URL}{path}",
        headers=HEADERS,
        json=payload,
        timeout=20,
    )

    if not response.ok:
        raise RuntimeError(
            f"POST {path} HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    if not response.text:
        return {}

    return response.json()


# ============================================================
# ACCOUNT
# ============================================================

def verify_credentials():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        bot_state["credentials_ok"] = False
        add_error("ALPACA_API_KEY or ALPACA_SECRET_KEY missing.")
        return False

    try:
        account = api_get("/v2/account")

        bot_state["credentials_ok"] = True

        log(
            "ALPACA PAPER CONNECTED | "
            f"equity=${account.get('equity')} | "
            f"options_buying_power=${account.get('options_buying_power')}"
        )

        return True

    except Exception as exc:
        bot_state["credentials_ok"] = False
        add_error(f"Credential verification failed: {exc}")
        return False


def market_is_open():
    try:
        clock = api_get("/v2/clock")

        is_open = bool(
            clock.get("is_open", False)
        )

        bot_state["market_open"] = is_open
        return is_open

    except Exception as exc:
        add_error(f"Market clock error: {exc}")
        return False


# ============================================================
# STOCK UNIVERSE
# ============================================================

def get_stock_universe():
    try:
        assets = api_get(
            "/v2/assets",
            params={
                "status": "active",
                "asset_class": "us_equity",
            },
        )

        symbols = []

        for asset in assets:
            symbol = asset.get("symbol")

            if (
                symbol
                and asset.get("tradable", False)
                and "." not in symbol
            ):
                symbols.append(symbol)

        symbols = list(
            dict.fromkeys(
                PRIORITY_SYMBOLS + symbols
            )
        )

        log(
            f"UNIVERSE LOADED | "
            f"{len(symbols)} tradable stocks"
        )

        return symbols

    except Exception as exc:
        add_error(f"Universe error: {exc}")
        return PRIORITY_SYMBOLS.copy()


# ============================================================
# PREMARKET / WATCHLIST
# ============================================================

def batch_chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def build_watchlist():
    universe = get_stock_universe()

    scored = []

    for chunk in batch_chunks(universe, 100):

        try:
            data = api_get(
                "/v2/stocks/snapshots",
                params={
                    "symbols": ",".join(chunk),
                    "feed": DATA_FEED,
                },
                data_api=True,
            )

            for symbol, snapshot in data.items():

                daily = snapshot.get("dailyBar") or {}
                previous = snapshot.get("prevDailyBar") or {}
                minute = snapshot.get("minuteBar") or {}

                current_price = (
                    minute.get("c")
                    or daily.get("c")
                )

                previous_close = previous.get("c")
                volume = daily.get("v", 0)

                if not current_price or not previous_close:
                    continue

                current_price = float(current_price)
                previous_close = float(previous_close)
                volume = float(volume or 0)

                if current_price < 2:
                    continue

                gap = (
                    (current_price - previous_close)
                    / previous_close
                )

                dollar_volume = (
                    current_price * volume
                )

                score = (
                    abs(gap) * 100
                    + math.log10(
                        max(dollar_volume, 1)
                    )
                )

                if symbol in PRIORITY_SYMBOLS:
                    score += 100

                scored.append(
                    (
                        score,
                        symbol,
                        current_price,
                        gap,
                        dollar_volume,
                    )
                )

        except Exception as exc:
            log(
                f"Snapshot batch skipped | {exc}"
            )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    symbols = [
        item[1]
        for item in scored[:WATCHLIST_SIZE]
    ]

    for symbol in PRIORITY_SYMBOLS:
        if symbol not in symbols:
            symbols.append(symbol)

    symbols = symbols[:max(
        WATCHLIST_SIZE,
        len(PRIORITY_SYMBOLS),
    )]

    bot_state["watchlist"] = symbols

    log(
        "WATCHLIST | "
        + ", ".join(symbols)
    )

    return symbols


# ============================================================
# BARS
# ============================================================

def bars_to_df(bars):
    if not bars:
        return None

    df = pd.DataFrame(bars)

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

    if not required.issubset(df.columns):
        return None

    df["timestamp"] = (
        pd.to_datetime(
            df["t"],
            utc=True,
        )
        .dt.tz_convert(NY)
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

    df = (
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

    return df


def get_today_bars(symbol):
    now = now_et()

    start = datetime(
        now.year,
        now.month,
        now.day,
        4,
        0,
        tzinfo=NY,
    ).astimezone(UTC)

    try:
        data = api_get(
            f"/v2/stocks/{symbol}/bars",
            params={
                "timeframe":
                    f"{TIMEFRAME_MINUTES}Min",
                "start":
                    start.isoformat(),
                "limit":
                    1000,
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


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):
    if df is None or len(df) < 35:
        return None

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

    cumulative_volume = (
        df["volume"].cumsum()
    )

    df["vwap"] = (
        (typical * df["volume"]).cumsum()
        / cumulative_volume.replace(0, float("nan"))
    )

    return df


# ============================================================
# PURGATORY SIGNAL
# ============================================================

def find_signal(symbol):
    df = get_today_bars(symbol)

    df = add_indicators(df)

    if df is None or len(df) < 35:
        return None

    today = now_et().date()

    premarket = df[
        (df["timestamp"].dt.date == today)
        &
        (df["timestamp"].dt.time >= dt_time(4, 0))
        &
        (df["timestamp"].dt.time < dt_time(9, 30))
    ]

    regular = df[
        (df["timestamp"].dt.date == today)
        &
        (df["timestamp"].dt.time >= dt_time(9, 30))
    ].copy()

    if premarket.empty or len(regular) < 2:
        return None

    pm_high = float(
        premarket["high"].max()
    )

    pm_low = float(
        premarket["low"].min()
    )

    current = regular.iloc[-1]
    previous = regular.iloc[-2]

    current_time = current["timestamp"].time()

    if current_time >= LAST_ENTRY_TIME:
        return None

    bullish = (
        current["ema5"] > current["ema9"]
        and current["ema9"] > current["ema30"]
        and current["close"] > current["vwap"]
        and current["close"] > current["ema30"]
    )

    bearish = (
        current["ema5"] < current["ema9"]
        and current["ema9"] < current["ema30"]
        and current["close"] < current["vwap"]
        and current["close"] < current["ema30"]
    )

    # ========================================================
    # CALL
    #
    # Break PM high
    # Pullback/retest
    # Confirmation candle
    # EMA/VWAP alignment
    # ========================================================

    recent = regular.tail(8)

    broke_high = (
        recent["close"] > pm_high
    ).any()

    retested_high = (
        recent["low"] <= pm_high * 1.002
    ).any()

    call_confirmation = (
        current["close"] > pm_high
        and current["close"] > current["open"]
        and current["close"] > previous["high"]
    )

    if (
        bullish
        and broke_high
        and retested_high
        and call_confirmation
    ):
        return {
            "symbol": symbol,
            "direction": "CALL",
            "stock_price": float(
                current["close"]
            ),
            "pm_high": pm_high,
            "pm_low": pm_low,
            "timestamp":
                current["timestamp"].isoformat(),
        }

    # ========================================================
    # PUT
    # ========================================================

    broke_low = (
        recent["close"] < pm_low
    ).any()

    retested_low = (
        recent["high"] >= pm_low * 0.998
    ).any()

    put_confirmation = (
        current["close"] < pm_low
        and current["close"] < current["open"]
        and current["close"] < previous["low"]
    )

    if (
        bearish
        and broke_low
        and retested_low
        and put_confirmation
    ):
        return {
            "symbol": symbol,
            "direction": "PUT",
            "stock_price": float(
                current["close"]
            ),
            "pm_high": pm_high,
            "pm_low": pm_low,
            "timestamp":
                current["timestamp"].isoformat(),
        }

    return None


# ============================================================
# 0DTE OPTION CONTRACT
# ============================================================

def get_zero_dte_contract(
    underlying,
    direction,
    stock_price,
):
    today = now_et().date().isoformat()

    option_type = (
        "call"
        if direction == "CALL"
        else "put"
    )

    strike_range = max(
        stock_price * 0.05,
        3,
    )

    try:
        data = api_get(
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
                "strike_price_gte":
                    round(
                        stock_price
                        - strike_range,
                        2,
                    ),
                "strike_price_lte":
                    round(
                        stock_price
                        + strike_range,
                        2,
                    ),
                "limit":
                    1000,
            },
        )

        contracts = data.get(
            "option_contracts",
            []
        )

        if not contracts:
            return None

        valid = []

        for contract in contracts:
            try:
                strike = float(
                    contract[
                        "strike_price"
                    ]
                )

                symbol = contract[
                    "symbol"
                ]

                valid.append(
                    (
                        abs(
                            strike
                            - stock_price
                        ),
                        strike,
                        symbol,
                    )
                )

            except Exception:
                continue

        if not valid:
            return None

        valid.sort(
            key=lambda item: item[0]
        )

        _, strike, symbol = valid[0]

        return {
            "symbol": symbol,
            "strike": strike,
            "type": option_type,
        }

    except Exception as exc:
        log(
            f"{underlying} 0DTE lookup failed | "
            f"{exc}"
        )

        return None


# ============================================================
# OPTION PRICE
# ============================================================

def get_option_price(
    underlying,
    option_symbol,
):
    feeds = [
        OPTION_FEED,
        "indicative",
    ]

    for feed in dict.fromkeys(feeds):

        try:
            data = api_get(
                f"/v1beta1/options/snapshots/"
                f"{underlying}",
                params={
                    "feed": feed,
                    "limit": 1000,
                    "expiration_date":
                        now_et().date().isoformat(),
                },
                data_api=True,
            )

            snapshots = data.get(
                "snapshots",
                {}
            )

            snap = snapshots.get(
                option_symbol
            )

            if not snap:
                continue

            quote = (
                snap.get("latestQuote")
                or {}
            )

            trade = (
                snap.get("latestTrade")
                or {}
            )

            ask = quote.get("ap")
            bid = quote.get("bp")
            trade_price = trade.get("p")

            if ask and bid:
                return (
                    float(ask)
                    + float(bid)
                ) / 2

            if ask:
                return float(ask)

            if trade_price:
                return float(trade_price)

        except Exception:
            continue

    return None


# ============================================================
# ORDERS
# ============================================================

def place_option_order(
    option_symbol,
    side,
    qty,
):
    payload = {
        "symbol": option_symbol,
        "qty": str(int(qty)),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }

    if side == "buy":
        payload[
            "position_intent"
        ] = "buy_to_open"

    else:
        payload[
            "position_intent"
        ] = "sell_to_close"

    return api_post(
        "/v2/orders",
        payload,
    )


# ============================================================
# POSITIONS
# ============================================================

def get_positions():
    try:
        return api_get(
            "/v2/positions"
        )

    except Exception as exc:
        add_error(
            f"Positions error: {exc}"
        )
        return []


def option_positions():
    positions = get_positions()

    result = []

    for position in positions:
        asset_class = (
            position.get(
                "asset_class",
                ""
            )
        )

        symbol = position.get(
            "symbol",
            ""
        )

        if (
            asset_class == "us_option"
            or len(symbol) > 15
        ):
            result.append(position)

    return result


# ============================================================
# ENTER TRADE
# ============================================================

def execute_signal(signal):
    underlying = signal["symbol"]
    direction = signal["direction"]
    stock_price = signal[
        "stock_price"
    ]

    signal_key = (
        underlying,
        direction,
        signal["timestamp"],
    )

    if (
        last_signal_key.get(
            underlying
        )
        == signal_key
    ):
        return False

    last_signal_key[
        underlying
    ] = signal_key

    positions = option_positions()

    if len(positions) >= MAX_OPEN_POSITIONS:
        log(
            "MAX OPEN POSITIONS REACHED"
        )
        return False

    for position in positions:
        option_symbol = position.get(
            "symbol",
            ""
        )

        if option_symbol.startswith(
            underlying
        ):
            return False

    contract = get_zero_dte_contract(
        underlying,
        direction,
        stock_price,
    )

    if not contract:
        log(
            f"{underlying} | "
            f"NO 0DTE {direction} CONTRACT"
        )
        return False

    option_symbol = contract[
        "symbol"
    ]

    option_price = get_option_price(
        underlying,
        option_symbol,
    )

    if (
        option_price is None
        or option_price <= 0
    ):
        log(
            f"{underlying} | "
            f"NO OPTION PRICE | "
            f"{option_symbol}"
        )
        return False

    cost_per_contract = (
        option_price * 100
    )

    qty = int(
        POSITION_DOLLARS
        // cost_per_contract
    )

    qty = max(qty, 1)

    estimated_cost = (
        qty
        * cost_per_contract
    )

    if (
        estimated_cost
        > POSITION_DOLLARS * 1.30
    ):
        log(
            f"{underlying} | "
            f"OPTION TOO EXPENSIVE | "
            f"${option_price:.2f}"
        )
        return False

    log(
        f"SIGNAL | {underlying} "
        f"{direction} | "
        f"stock=${stock_price:.2f} | "
        f"contract={option_symbol} | "
        f"option≈${option_price:.2f} | "
        f"qty={qty}"
    )

    bot_state[
        "signals"
    ].append(signal)

    bot_state[
        "signals"
    ] = bot_state[
        "signals"
    ][-50:]

    if not AUTO_TRADE:
        log(
            "AUTO_TRADE=false | "
            "signal only"
        )
        return True

    try:
        order = place_option_order(
            option_symbol,
            "buy",
            qty,
        )

        managed_positions[
            option_symbol
        ] = {
            "underlying":
                underlying,

            "direction":
                direction,

            "qty":
                qty,

            "entry_reference":
                option_price,

            "high_water":
                option_price,

            "trimmed":
                False,

            "order_id":
                order.get("id"),
        }

        log(
            f"ORDER SENT | "
            f"{direction} "
            f"{option_symbol} "
            f"x{qty}"
        )

        return True

    except Exception as exc:
        add_error(
            f"Order failed "
            f"{option_symbol}: {exc}"
        )

        return False


# ============================================================
# POSITION MANAGEMENT
# ============================================================

def manage_positions():
    positions = option_positions()

    now = now_et()

    for position in positions:

        symbol = position.get(
            "symbol"
        )

        if not symbol:
            continue

        qty = abs(
            int(
                float(
                    position.get(
                        "qty",
                        0,
                    )
                )
            )
        )

        if qty <= 0:
            continue

        avg_entry = float(
            position.get(
                "avg_entry_price",
                0,
            )
            or 0
        )

        current_price = float(
            position.get(
                "current_price",
                0,
            )
            or 0
        )

        if (
            avg_entry <= 0
            or current_price <= 0
        ):
            continue

        state = managed_positions.setdefault(
            symbol,
            {
                "underlying": "",
                "direction": "",
                "qty": qty,
                "entry_reference":
                    avg_entry,
                "high_water":
                    current_price,
                "trimmed":
                    False,
            },
        )

        state["high_water"] = max(
            float(
                state.get(
                    "high_water",
                    current_price,
                )
            ),
            current_price,
        )

        pnl_percent = (
            current_price
            - avg_entry
        ) / avg_entry

        log(
            f"POSITION | {symbol} | "
            f"qty={qty} | "
            f"entry=${avg_entry:.2f} | "
            f"now=${current_price:.2f} | "
            f"PnL={pnl_percent * 100:.1f}%"
        )

        # ====================================================
        # FORCE EXIT 0DTE
        # ====================================================

        if now.time() >= FORCE_EXIT_TIME:

            if AUTO_TRADE:
                try:
                    place_option_order(
                        symbol,
                        "sell",
                        qty,
                    )

                    log(
                        f"FORCE EXIT | "
                        f"{symbol} x{qty}"
                    )

                except Exception as exc:
                    add_error(
                        f"Force exit failed "
                        f"{symbol}: {exc}"
                    )

            continue

        # ====================================================
        # STOP LOSS -20%
        # ====================================================

        if (
            pnl_percent
            <= -STOP_LOSS_PERCENT
        ):

            if AUTO_TRADE:
                try:
                    place_option_order(
                        symbol,
                        "sell",
                        qty,
                    )

                    log(
                        f"STOP LOSS | "
                        f"{symbol} x{qty}"
                    )

                except Exception as exc:
                    add_error(
                        f"Stop failed "
                        f"{symbol}: {exc}"
                    )

            continue

        # ====================================================
        # TAKE PROFIT
        #
        # At +30% sell half.
        # Keep the rest as runner.
        # ====================================================

        if (
            not state.get(
                "trimmed",
                False,
            )
            and pnl_percent
            >= TAKE_PROFIT_PERCENT
        ):

            trim_qty = max(
                1,
                int(
                    math.floor(
                        qty
                        * TAKE_PROFIT_FRACTION
                    )
                ),
            )

            if trim_qty >= qty and qty > 1:
                trim_qty = qty - 1

            if qty == 1:
                # One contract cannot be split.
                # Keep it running until trail/exit.
                state[
                    "trimmed"
                ] = True

                log(
                    f"TP HIT | {symbol} | "
                    f"1 contract runner"
                )

            elif AUTO_TRADE:

                try:
                    place_option_order(
                        symbol,
                        "sell",
                        trim_qty,
                    )

                    state[
                        "trimmed"
                    ] = True

                    state[
                        "high_water"
                    ] = current_price

                    log(
                        f"TAKE PROFIT | "
                        f"{symbol} | "
                        f"sold {trim_qty} | "
                        f"runner left"
                    )

                except Exception as exc:
                    add_error(
                        f"TP failed "
                        f"{symbol}: {exc}"
                    )

        # ====================================================
        # RUNNER TRAIL
        #
        # After TP, trail 15% below best option price.
        # ====================================================

        if state.get(
            "trimmed",
            False,
        ):

            high_water = float(
                state.get(
                    "high_water",
                    current_price,
                )
            )

            trailing_exit = (
                high_water
                * (
                    1
                    - RUNNER_TRAIL_PERCENT
                )
            )

            if (
                current_price
                <= trailing_exit
            ):

                if AUTO_TRADE:
                    try:
                        place_option_order(
                            symbol,
                            "sell",
                            qty,
                        )

                        log(
                            f"RUNNER EXIT | "
                            f"{symbol} | "
                            f"current=${current_price:.2f} | "
                            f"high=${high_water:.2f}"
                        )

                    except Exception as exc:
                        add_error(
                            f"Runner exit failed "
                            f"{symbol}: {exc}"
                        )


# ============================================================
# SCAN
# ============================================================

def scan_market():
    watchlist = bot_state.get(
        "watchlist",
        []
    )

    if not watchlist:
        watchlist = build_watchlist()

    new_trades = 0

    for symbol in watchlist:

        if (
            new_trades
            >= MAX_NEW_TRADES_PER_CYCLE
        ):
            break

        try:
            signal = find_signal(
                symbol
            )

            if not signal:
                continue

            log(
                f"CONFIRMED | "
                f"{symbol} "
                f"{signal['direction']}"
            )

            if execute_signal(
                signal
            ):
                new_trades += 1

        except Exception as exc:
            log(
                f"{symbol} scan skipped | "
                f"{exc}"
            )


# ============================================================
# BOT LOOP
# ============================================================

def bot_loop():
    bot_state["running"] = True

    log(
        "PURGATORY 0DTE BOT STARTED"
    )

    log(
        f"AUTO_TRADE={AUTO_TRADE} | "
        f"TIMEFRAME={TIMEFRAME_MINUTES}Min | "
        f"POSITION_DOLLARS=${POSITION_DOLLARS:.2f}"
    )

    watchlist_date = None

    while RUN_BOT_LOOP:

        try:
            current = now_et()

            bot_state[
                "last_cycle"
            ] = current.isoformat()

            # Rebuild watchlist every day.
            if watchlist_date != current.date():

                if (
                    current.time()
                    >= dt_time(8, 30)
                ):
                    build_watchlist()
                    watchlist_date = current.date()

            if market_is_open():

                manage_positions()

                if (
                    current.time()
                    >= RTH_START
                    and current.time()
                    < LAST_ENTRY_TIME
                ):
                    scan_market()

            else:
                log(
                    "MARKET CLOSED"
                )

        except Exception as exc:
            add_error(
                f"Main loop error: {exc}"
            )

        time.sleep(
            LOOP_SECONDS
        )


# ============================================================
# FLASK
# ============================================================

@app.route("/")
def home():
    return jsonify(
        {
            "status":
                "running",

            "bot":
                "Purgatory 0DTE",

            "paper_trading":
                True,

            "auto_trade":
                AUTO_TRADE,

            "credentials_ok":
                bot_state[
                    "credentials_ok"
                ],

            "market_open":
                bot_state[
                    "market_open"
                ],

            "last_cycle":
                bot_state[
                    "last_cycle"
                ],

            "watchlist":
                bot_state[
                    "watchlist"
                ],

            "signals":
                bot_state[
                    "signals"
                ][-10:],

            "errors":
                bot_state[
                    "errors"
                ][-10:],
        }
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "credentials_ok":
                bot_state[
                    "credentials_ok"
                ],
            "running":
                bot_state[
                    "running"
                ],
        }
    )


@app.route("/positions")
def positions_route():
    return jsonify(
        get_positions()
    )


@app.route("/watchlist")
def watchlist_route():
    return jsonify(
        {
            "watchlist":
                bot_state[
                    "watchlist"
                ]
        }
    )


# ============================================================
# START
# ============================================================

def start_background_bot():
    global background_started

    if background_started:
        return

    background_started = True

    thread = threading.Thread(
        target=bot_loop,
        daemon=True,
    )

    thread.start()


verify_credentials()

if RUN_BOT_LOOP:
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
    )