import os
import time
import threading
import requests
import pandas as pd
import numpy as np

from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from flask import Flask, jsonify


# ============================================================
# CONFIG
# ============================================================

app = Flask(__name__)

NY = ZoneInfo("America/New_York")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

TRADING_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

AUTO_TRADE = os.getenv("AUTO_TRADE", "false").lower() == "true"
RUN_BOT_LOOP = os.getenv("RUN_BOT_LOOP", "true").lower() == "true"

TIMEFRAME_MINUTES = 4

# Risk
MAX_OPEN_POSITIONS = 3
MAX_NEW_TRADES_PER_CYCLE = 1
POSITION_DOLLARS = 500.00

# Option management
STOP_LOSS_PERCENT = 0.20
TAKE_PROFIT_PERCENT = 0.30
TAKE_PROFIT_FRACTION = 0.50
RUNNER_TRAIL_PERCENT = 0.15

# 0DTE rules
LAST_ENTRY_TIME = dt_time(14, 45)
FORCE_EXIT_TIME = dt_time(15, 15)

# Scanner
MIN_PRICE = 5.00
MIN_DOLLAR_VOLUME = 5_000_000
MIN_RVOL = 1.10
SCAN_LIMIT = 150

LOOP_SECONDS = 15

# Common liquid option names.
# Scanner can expand the universe using Alpaca active assets.
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

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

bot_state = {
    "running": False,
    "last_cycle": None,
    "last_scan": None,
    "candidates": [],
    "signals": [],
    "errors": [],
}

managed_positions = {}


# ============================================================
# HELPERS
# ============================================================

def log(message):
    timestamp = datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp} ET] {message}", flush=True)


def add_error(message):
    bot_state["errors"].append(str(message))

    if len(bot_state["errors"]) > 25:
        bot_state["errors"] = bot_state["errors"][-25:]

    log(f"ERROR: {message}")


def alpaca_get(path, params=None, data_api=False):
    base = DATA_BASE_URL if data_api else TRADING_BASE_URL

    response = requests.get(
        f"{base}{path}",
        headers=HEADERS,
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


def alpaca_post(path, payload=None):
    response = requests.post(
        f"{TRADING_BASE_URL}{path}",
        headers={
            **HEADERS,
            "Content-Type": "application/json",
        },
        json=payload or {},
        timeout=20,
    )

    response.raise_for_status()

    if response.text:
        return response.json()

    return {}


def alpaca_delete(path):
    response = requests.delete(
        f"{TRADING_BASE_URL}{path}",
        headers=HEADERS,
        timeout=20,
    )

    response.raise_for_status()

    if response.text:
        return response.json()

    return {}


def market_is_open():
    try:
        clock = alpaca_get("/v2/clock")
        return bool(clock.get("is_open"))
    except Exception as e:
        add_error(f"Clock error: {e}")
        return False


def now_et():
    return datetime.now(NY)


def today_string():
    return now_et().strftime("%Y-%m-%d")


# ============================================================
# STOCK UNIVERSE
# ============================================================

def get_active_stock_universe():
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

            if not asset.get("tradable", False):
                continue

            if "." in symbol:
                continue

            symbols.append(symbol)

    except Exception as e:
        add_error(f"Asset universe error: {e}")

    combined = list(dict.fromkeys(PRIORITY_SYMBOLS + symbols))

    return combined


# ============================================================
# MARKET DATA
# ============================================================

def get_bars(symbol, limit=120):
    try:
        data = alpaca_get(
            f"/v2/stocks/{symbol}/bars",
            params={
                "timeframe": f"{TIMEFRAME_MINUTES}Min",
                "limit": limit,
                "adjustment": "raw",
                "feed": "iex",
            },
            data_api=True,
        )

        bars = data.get("bars", [])

        if not bars:
            return None

        df = pd.DataFrame(bars)

        if df.empty:
            return None

        df["timestamp"] = pd.to_datetime(df["t"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert(NY)

        df["open"] = pd.to_numeric(df["o"])
        df["high"] = pd.to_numeric(df["h"])
        df["low"] = pd.to_numeric(df["l"])
        df["close"] = pd.to_numeric(df["c"])
        df["volume"] = pd.to_numeric(df["v"])

        return df

    except Exception:
        return None


def calculate_indicators(df):
    if df is None or len(df) < 35:
        return None

    df = df.copy()

    df["ema5"] = df["close"].ewm(
        span=5,
        adjust=False,
    ).mean()

    df["ema9"] = df["close"].ewm(
        span=9,
        adjust=False,
    ).mean()

    df["ema30"] = df["close"].ewm(
        span=30,
        adjust=False,
    ).mean()

    typical_price = (
        df["high"] +
        df["low"] +
        df["close"]
    ) / 3

    cumulative_volume = df["volume"].cumsum()

    df["vwap"] = (
        (typical_price * df["volume"]).cumsum()
        / cumulative_volume.replace(0, np.nan)
    )

    df["volume_avg"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["rvol"] = (
        df["volume"]
        / df["volume_avg"].replace(0, np.nan)
    )

    return df


# ============================================================
# SIGNAL LOGIC
# ============================================================

def detect_signal(symbol, df):
    df = calculate_indicators(df)

    if df is None or len(df) < 35:
        return None

    current = df.iloc[-1]
    previous = df.iloc[-2]

    price = float(current["close"])

    ema5 = float(current["ema5"])
    ema9 = float(current["ema9"])
    ema30 = float(current["ema30"])
    vwap = float(current["vwap"])

    rvol = current["rvol"]

    if pd.isna(rvol):
        rvol = 0

    rvol = float(rvol)

    dollar_volume = float(
        current["volume"] * current["close"]
    )

    if price < MIN_PRICE:
        return None

    bullish = (
        ema5 > ema9
        and ema9 > ema30
        and price > vwap
        and ema5 > vwap
        and current["close"] > current["open"]
    )

    bearish = (
        ema5 < ema9
        and ema9 < ema30
        and price < vwap
        and ema5 < vwap
        and current["close"] < current["open"]
    )

    bullish_cross = (
        previous["ema5"] <= previous["ema9"]
        and current["ema5"] > current["ema9"]
    )

    bearish_cross = (
        previous["ema5"] >= previous["ema9"]
        and current["ema5"] < current["ema9"]
    )

    score = 0

    if rvol >= MIN_RVOL:
        score += 2

    if dollar_volume >= MIN_DOLLAR_VOLUME:
        score += 1

    if bullish_cross or bearish_cross:
        score += 2

    if bullish or bearish:
        score += 3

    if bullish:
        direction = "CALL"

    elif bearish:
        direction = "PUT"

    else:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "price": round(price, 2),
        "ema5": round(ema5, 4),
        "ema9": round(ema9, 4),
        "ema30": round(ema30, 4),
        "vwap": round(vwap, 4),
        "rvol": round(rvol, 2),
        "score": score,
    }


# ============================================================
# SCANNER
# ============================================================

def scan_market():
    universe = get_active_stock_universe()

    signals = []

    log(
        f"Scanning {len(universe)} active US stocks "
        f"for 4-minute setups..."
    )

    for symbol in universe[:SCAN_LIMIT]:

        try:
            bars = get_bars(symbol)

            signal = detect_signal(
                symbol,
                bars,
            )

            if signal:
                signals.append(signal)

        except Exception:
            continue

    signals.sort(
        key=lambda x: (
            x["score"],
            x["rvol"],
        ),
        reverse=True,
    )

    bot_state["signals"] = signals
    bot_state["candidates"] = signals[:20]
    bot_state["last_scan"] = now_et().isoformat()

    if signals:
        log("TOP SCANNER RESULTS:")

        for signal in signals[:10]:
            log(
                f'{signal["symbol"]} '
                f'{signal["direction"]} '
                f'score={signal["score"]} '
                f'price=${signal["price"]} '
                f'RVOL={signal["rvol"]}'
            )

    else:
        log("No qualified CALL/PUT setups found.")

    return signals


# ============================================================
# 0DTE OPTION CONTRACTS
# ============================================================

def get_0dte_contracts(symbol, direction):
    try:
        option_type = (
            "call"
            if direction == "CALL"
            else "put"
        )

        data = alpaca_get(
            "/v2/options/contracts",
            params={
                "underlying_symbols": symbol,
                "expiration_date": today_string(),
                "type": option_type,
                "status": "active",
                "limit": 100,
            },
        )

        return data.get(
            "option_contracts",
            []
        )

    except Exception as e:
        add_error(
            f"{symbol} option chain error: {e}"
        )

        return []


def choose_0dte_contract(symbol, direction, stock_price):
    contracts = get_0dte_contracts(
        symbol,
        direction,
    )

    if not contracts:
        return None

    valid = []

    for contract in contracts:

        try:
            strike = float(
                contract.get("strike_price", 0)
            )

            option_symbol = contract.get("symbol")

            if not option_symbol:
                continue

            distance = abs(
                strike - stock_price
            )

            valid.append(
                (
                    distance,
                    strike,
                    option_symbol,
                )
            )

        except Exception:
            continue

    if not valid:
        return None

    valid.sort(
        key=lambda x: x[0]
    )

    _, strike, option_symbol = valid[0]

    return {
        "symbol": option_symbol,
        "strike": strike,
        "underlying": symbol,
        "direction": direction,
    }


# ============================================================
# OPTION QUOTE
# ============================================================

def get_option_quote(option_symbol):
    try:
        data = alpaca_get(
            f"/v1beta1/options/quotes/latest",
            params={
                "symbols": option_symbol,
            },
            data_api=True,
        )

        quote = (
            data
            .get("quotes", {})
            .get(option_symbol)
        )

        if not quote:
            return None

        bid = float(
            quote.get("bp", 0) or 0
        )

        ask = float(
            quote.get("ap", 0) or 0
        )

        if bid <= 0 and ask <= 0:
            return None

        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2

        else:
            mid = max(bid, ask)

        return {
            "bid": bid,
            "ask": ask,
            "mid": mid,
        }

    except Exception:
        return None


# ============================================================
# ORDERS
# ============================================================

def get_positions():
    try:
        return alpaca_get(
            "/v2/positions"
        )

    except Exception as e:
        add_error(
            f"Position error: {e}"
        )

        return []


def open_option_positions():
    positions = get_positions()

    option_positions = []

    for position in positions:

        asset_class = position.get(
            "asset_class",
            ""
        )

        if "option" in asset_class.lower():
            option_positions.append(position)

    return option_positions


def submit_option_buy(
    option_symbol,
    quantity,
):
    payload = {
        "symbol": option_symbol,
        "qty": str(quantity),
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
    }

    if not AUTO_TRADE:
        log(
            f"PAPER SIGNAL ONLY: BUY "
            f"{quantity} {option_symbol}"
        )

        return {
            "paper_signal": True,
            **payload,
        }

    return alpaca_post(
        "/v2/orders",
        payload,
    )


def submit_option_sell(
    option_symbol,
    quantity,
):
    payload = {
        "symbol": option_symbol,
        "qty": str(quantity),
        "side": "sell",
        "type": "market",
        "time_in_force": "day",
    }

    if not AUTO_TRADE:
        log(
            f"PAPER SIGNAL ONLY: SELL "
            f"{quantity} {option_symbol}"
        )

        return {
            "paper_signal": True,
            **payload,
        }

    return alpaca_post(
        "/v2/orders",
        payload,
    )


# ============================================================
# ENTRY
# ============================================================

def enter_trade(signal):
    current_time = now_et().time()

    if current_time >= LAST_ENTRY_TIME:
        log(
            f'Skipping {signal["symbol"]}: '
            f"past 2:45 PM ET entry cutoff."
        )

        return

    current_positions = open_option_positions()

    if len(current_positions) >= MAX_OPEN_POSITIONS:
        log(
            "Maximum open option positions reached."
        )

        return

    stock_symbol = signal["symbol"]
    direction = signal["direction"]
    stock_price = signal["price"]

    contract = choose_0dte_contract(
        stock_symbol,
        direction,
        stock_price,
    )

    if not contract:
        log(
            f"No 0DTE {direction} contract "
            f"found for {stock_symbol}."
        )

        return

    option_symbol = contract["symbol"]

    quote = get_option_quote(
        option_symbol
    )

    if not quote:
        log(
            f"No option quote for "
            f"{option_symbol}."
        )

        return

    premium = quote["mid"]

    if premium <= 0:
        return

    contract_cost = premium * 100

    quantity = max(
        1,
        int(
            POSITION_DOLLARS
            // contract_cost
        ),
    )

    order = submit_option_buy(
        option_symbol,
        quantity,
    )

    managed_positions[option_symbol] = {
        "underlying": stock_symbol,
        "direction": direction,
        "entry_price": premium,
        "quantity": quantity,
        "original_quantity": quantity,
        "tp_hit": False,
        "highest_after_tp": premium,
        "entry_time": now_et().isoformat(),
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
# TRADE MANAGEMENT
# ============================================================

def close_managed_position(
    option_symbol,
    reason,
):
    trade = managed_positions.get(
        option_symbol
    )

    if not trade:
        return

    quantity = int(
        trade.get(
            "quantity",
            0,
        )
    )

    if quantity <= 0:
        managed_positions.pop(
            option_symbol,
            None,
        )

        return

    submit_option_sell(
        option_symbol,
        quantity,
    )

    log(
        f"EXIT {option_symbol} | "
        f"{reason} | qty {quantity}"
    )

    managed_positions.pop(
        option_symbol,
        None,
    )


def underlying_ema9_invalidated(trade):
    symbol = trade["underlying"]
    direction = trade["direction"]

    bars = get_bars(
        symbol,
        limit=50,
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

    if direction == "CALL":
        return close < ema9

    if direction == "PUT":
        return close > ema9

    return False


def manage_positions():
    current_time = now_et().time()

    for option_symbol in list(
        managed_positions.keys()
    ):

        trade = managed_positions.get(
            option_symbol
        )

        if not trade:
            continue

        # Mandatory 0DTE end-of-day exit
        if current_time >= FORCE_EXIT_TIME:
            close_managed_position(
                option_symbol,
                "0DTE FORCE EXIT 3:15 PM ET",
            )

            continue

        quote = get_option_quote(
            option_symbol
        )

        if not quote:
            continue

        premium = quote["mid"]

        entry = float(
            trade["entry_price"]
        )

        quantity = int(
            trade["quantity"]
        )

        if entry <= 0 or quantity <= 0:
            continue

        pnl_percent = (
            premium - entry
        ) / entry

        # ----------------------------------------
        # HARD STOP BEFORE TAKE PROFIT
        # ----------------------------------------

        if (
            not trade["tp_hit"]
            and pnl_percent <= -STOP_LOSS_PERCENT
        ):

            close_managed_position(
                option_symbol,
                f"HARD STOP {pnl_percent:.1%}",
            )

            continue

        # ----------------------------------------
        # EMA9 INVALIDATION
        # ----------------------------------------

        if (
            not trade["tp_hit"]
            and underlying_ema9_invalidated(trade)
        ):

            close_managed_position(
                option_symbol,
                "EMA9 INVALIDATION",
            )

            continue

        # ----------------------------------------
        # FIRST TAKE PROFIT
        # ----------------------------------------

        if (
            not trade["tp_hit"]
            and pnl_percent >= TAKE_PROFIT_PERCENT
        ):

            if quantity > 1:

                sell_quantity = max(
                    1,
                    int(
                        quantity
                        * TAKE_PROFIT_FRACTION
                    ),
                )

                # Always leave at least one runner
                sell_quantity = min(
                    sell_quantity,
                    quantity - 1,
                )

                if sell_quantity > 0:

                    submit_option_sell(
                        option_symbol,
                        sell_quantity,
                    )

                    trade["quantity"] -= (
                        sell_quantity
                    )

                    log(
                        f"TAKE PROFIT "
                        f"{option_symbol} | "
                        f"{pnl_percent:.1%} | "
                        f"sold {sell_quantity} | "
                        f"runner {trade['quantity']}"
                    )

            trade["tp_hit"] = True

            trade[
                "highest_after_tp"
            ] = premium

            continue

        # ----------------------------------------
        # LET RUNNERS RUN
        # ----------------------------------------

        if trade["tp_hit"]:

            if (
                premium
                > trade["highest_after_tp"]
            ):

                trade[
                    "highest_after_tp"
                ] = premium

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

            if premium <= trailing_stop:

                close_managed_position(
                    option_symbol,
                    (
                        "RUNNER TRAILING STOP "
                        f"${premium:.2f}"
                    ),
                )

                continue


# ============================================================
# BOT CYCLE
# ============================================================

def bot_cycle():
    bot_state["last_cycle"] = (
        now_et().isoformat()
    )

    if not market_is_open():
        log("Market closed.")
        return

    # Manage existing trades first
    manage_positions()

    # Scan for new trades
    signals = scan_market()

    if not signals:
        return

    trades_entered = 0

    for signal in signals:

        if (
            trades_entered
            >= MAX_NEW_TRADES_PER_CYCLE
        ):
            break

        already_in_underlying = any(
            trade.get("underlying")
            == signal["symbol"]
            for trade
            in managed_positions.values()
        )

        if already_in_underlying:
            continue

        enter_trade(signal)

        trades_entered += 1


# ============================================================
# BACKGROUND LOOP
# ============================================================

def bot_loop():
    bot_state["running"] = True

    log("0DTE trading bot started.")

    while True:

        try:
            bot_cycle()

        except Exception as e:
            add_error(
                f"Bot cycle error: {e}"
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
# FLASK ROUTES
# ============================================================

@app.route("/")
def home():
    return jsonify(
        {
            "status": "online",
            "bot": "Alpaca 0DTE Options Bot",
            "paper_trading": True,
            "auto_trade": AUTO_TRADE,
            "strategy": (
                "4-minute EMA5/EMA9/EMA30 "
                "+ VWAP + volume"
            ),
            "take_profit": (
                f"{TAKE_PROFIT_PERCENT:.0%}"
            ),
            "runner_trail": (
                f"{RUNNER_TRAIL_PERCENT:.0%}"
            ),
            "stop_loss": (
                f"{STOP_LOSS_PERCENT:.0%}"
            ),
        }
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "time": now_et().isoformat(),
        }
    )


@app.route("/status")
def status():
    return jsonify(
        {
            **bot_state,
            "managed_positions": (
                managed_positions
            ),
            "auto_trade": AUTO_TRADE,
        }
    )


@app.route("/scan")
def manual_scan():

    try:
        results = scan_market()

        return jsonify(
            {
                "count": len(results),
                "results": results[:50],
            }
        )

    except Exception as e:

        return jsonify(
            {
                "error": str(e),
            }
        ), 500


# ============================================================
# START
# ============================================================

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
