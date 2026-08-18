import os
import time
import math
import threading
import requests
import pandas as pd

from datetime import datetime, time as dt_time
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

AUTO_TRADE = os.getenv("AUTO_TRADE", "false").strip().lower() == "true"
RUN_BOT_LOOP = os.getenv("RUN_BOT_LOOP", "true").strip().lower() == "true"


# ============================================================
# CLEAN ENVIRONMENT VARIABLES
# ============================================================

def clean_credential(value):
    if value is None:
        return ""

    value = str(value).strip()

    # Remove invisible copy/paste characters
    value = (
        value
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
        .replace("\xa0", "")
    )

    # Keep credentials ASCII-safe
    value = value.encode(
        "ascii",
        errors="ignore"
    ).decode("ascii")

    return value.strip()


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
# RISK SETTINGS
# ============================================================

MAX_OPEN_POSITIONS = 3
MAX_NEW_TRADES_PER_CYCLE = 1

POSITION_DOLLARS = 500.00

STOP_LOSS_PERCENT = 0.20

# Sell half at +30%
TAKE_PROFIT_PERCENT = 0.30
TAKE_PROFIT_FRACTION = 0.50

# Runner trails 15% from highest option price
RUNNER_TRAIL_PERCENT = 0.15


# ============================================================
# 0DTE TIME RULES
# ============================================================

LAST_ENTRY_TIME = dt_time(14, 45)
FORCE_EXIT_TIME = dt_time(15, 15)


# ============================================================
# SCANNER SETTINGS
# ============================================================

MIN_PRICE = 5.00
MIN_DOLLAR_VOLUME = 5_000_000
MIN_RVOL = 1.10

SCAN_LIMIT = 150
LOOP_SECONDS = 15

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
# STATE
# ============================================================

bot_state = {
    "running": False,
    "credentials_ok": False,
    "account": {},
    "market_open": False,
    "last_cycle": None,
    "last_scan": None,
    "candidates": [],
    "signals": [],
    "errors": [],
}

managed_positions = {}

stock_universe = []
stock_universe_index = 0


# ============================================================
# HELPERS
# ============================================================

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


def now_et():
    return datetime.now(NY)


def today_string():
    return now_et().strftime("%Y-%m-%d")


def log(message):
    timestamp = now_et().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        f"[{timestamp} ET] {safe_text(message)}",
        flush=True
    )


def add_error(message):
    message = safe_text(message)

    bot_state["errors"].append(message)
    bot_state["errors"] = bot_state["errors"][-25:]

    log(f"ERROR: {message}")


# ============================================================
# ALPACA REQUESTS
# ============================================================

def alpaca_get(
    path,
    params=None,
    data_api=False
):
    base_url = (
        DATA_BASE_URL
        if data_api
        else TRADING_BASE_URL
    )

    response = requests.get(
        f"{base_url}{path}",
        headers=HEADERS,
        params=params,
        timeout=20,
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


# ============================================================
# VERIFY NEW ALPACA PAPER KEYS
# ============================================================

def verify_credentials():
    if not ALPACA_API_KEY:
        add_error(
            "ALPACA_API_KEY is missing in Render."
        )
        return False

    if not ALPACA_SECRET_KEY:
        add_error(
            "ALPACA_SECRET_KEY is missing in Render."
        )
        return False

    try:
        account = alpaca_get(
            "/v2/account"
        )

        bot_state["credentials_ok"] = True

        bot_state["account"] = {
            "id": account.get("id"),
            "status": account.get("status"),
            "cash": account.get("cash"),
            "equity": account.get("equity"),
            "buying_power": account.get(
                "buying_power"
            ),
            "options_buying_power": account.get(
                "options_buying_power"
            ),
        }

        log(
            "ALPACA PAPER ACCOUNT CONNECTED"
        )

        log(
            f'Account status: '
            f'{account.get("status")}'
        )

        log(
            f'Equity: ${account.get("equity")}'
        )

        return True

    except requests.exceptions.HTTPError as e:
        bot_state["credentials_ok"] = False

        if (
            e.response is not None
            and e.response.status_code == 401
        ):
            add_error(
                "Alpaca returned 401 Unauthorized. "
                "The Render keys do not match the current "
                "Alpaca PAPER account keys."
            )
        else:
            add_error(
                f"Account verification HTTP error: {e}"
            )

        return False

    except Exception as e:
        bot_state["credentials_ok"] = False

        add_error(
            f"Account verification error: "
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

        is_open = bool(
            clock.get(
                "is_open",
                False
            )
        )

        bot_state["market_open"] = is_open

        return is_open

    except Exception as e:
        bot_state["market_open"] = False

        add_error(
            f"Clock error: {safe_text(e)}"
        )

        return False


# ============================================================
# STOCK UNIVERSE
# ============================================================

def load_stock_universe():
    global stock_universe

    try:
        assets = alpaca_get(
            "/v2/assets",
            params={
                "status": "active",
                "asset_class": "us_equity",
            },
        )

        symbols = []

        for asset in assets:
            symbol = asset.get("symbol")

            if not symbol:
                continue

            if not asset.get(
                "tradable",
                False
            ):
                continue

            # Avoid odd symbols
            if "." in symbol:
                continue

            symbols.append(symbol)

        stock_universe = list(
            dict.fromkeys(
                PRIORITY_SYMBOLS + symbols
            )
        )

        log(
            f"Loaded {len(stock_universe)} "
            f"tradable US stocks."
        )

        return stock_universe

    except Exception as e:
        add_error(
            f"Stock universe error: "
            f"{safe_text(e)}"
        )

        stock_universe = PRIORITY_SYMBOLS.copy()

        return stock_universe


def next_scan_symbols():
    global stock_universe_index

    if not stock_universe:
        load_stock_universe()

    if not stock_universe:
        return PRIORITY_SYMBOLS.copy()

    # Always scan priority names
    selected = PRIORITY_SYMBOLS.copy()

    remaining_slots = max(
        SCAN_LIMIT - len(selected),
        0
    )

    if remaining_slots == 0:
        return selected[:SCAN_LIMIT]

    total = len(stock_universe)

    for _ in range(remaining_slots):
        if stock_universe_index >= total:
            stock_universe_index = 0

        symbol = stock_universe[
            stock_universe_index
        ]

        stock_universe_index += 1

        if symbol not in selected:
            selected.append(symbol)

    return selected[:SCAN_LIMIT]


# ============================================================
# STOCK BARS
# ============================================================

def get_bars(
    symbol,
    limit=120
):
    try:
        data = alpaca_get(
            f"/v2/stocks/{symbol}/bars",
            params={
                "timeframe": (
                    f"{TIMEFRAME_MINUTES}Min"
                ),
                "limit": limit,
                "adjustment": "raw",
                "feed": "iex",
            },
            data_api=True,
        )

        bars = data.get(
            "bars",
            []
        )

        if not bars:
            return None

        df = pd.DataFrame(bars)

        if df.empty:
            return None

        df["timestamp"] = pd.to_datetime(
            df["t"],
            utc=True
        ).dt.tz_convert(NY)

        df["open"] = pd.to_numeric(df["o"])
        df["high"] = pd.to_numeric(df["h"])
        df["low"] = pd.to_numeric(df["l"])
        df["close"] = pd.to_numeric(df["c"])
        df["volume"] = pd.to_numeric(df["v"])

        return df[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]

    except Exception:
        return None


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):
    df = df.copy()

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

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    cumulative_volume = (
        df["volume"]
        .cumsum()
    )

    df["vwap"] = (
        (
            typical_price
            * df["volume"]
        )
        .cumsum()
        / cumulative_volume.replace(
            0,
            float("nan")
        )
    )

    df["avg_volume"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["rvol"] = (
        df["volume"]
        / df["avg_volume"].replace(
            0,
            float("nan")
        )
    )

    return df


# ============================================================
# PURGATORY METHOD SIGNAL
# ============================================================

def analyze_symbol(symbol):
    df = get_bars(symbol)

    if df is None:
        return None

    if len(df) < 35:
        return None

    df = calculate_indicators(df)

    current = df.iloc[-1]
    previous = df.iloc[-2]

    price = float(
        current["close"]
    )

    if price < MIN_PRICE:
        return None

    dollar_volume = float(
        current["volume"]
        * price
    )

    if dollar_volume < MIN_DOLLAR_VOLUME:
        return None

    rvol = current.get(
        "rvol",
        0
    )

    if pd.isna(rvol):
        rvol = 0

    rvol = float(rvol)

    if rvol < MIN_RVOL:
        return None

    ema5 = float(current["ema5"])
    ema9 = float(current["ema9"])
    ema30 = float(current["ema30"])
    vwap = float(current["vwap"])

    prev_ema5 = float(
        previous["ema5"]
    )

    prev_ema9 = float(
        previous["ema9"]
    )

    score = 0
    direction = None


    # ========================================================
    # CALL SETUP
    # ========================================================

    bullish_alignment = (
        ema5 > ema9
        and price > vwap
        and price > ema30
        and ema5 > ema30
        and ema9 > ema30
    )

    bullish_momentum = (
        price > float(
            previous["close"]
        )
    )

    bullish_cross = (
        prev_ema5 <= prev_ema9
        and ema5 > ema9
    )

    if bullish_alignment:
        direction = "CALL"

        score += 3

        if bullish_momentum:
            score += 1

        if bullish_cross:
            score += 2

        if rvol >= 1.5:
            score += 1


    # ========================================================
    # PUT SETUP
    # ========================================================

    bearish_alignment = (
        ema5 < ema9
        and price < vwap
        and price < ema30
        and ema5 < ema30
        and ema9 < ema30
    )

    bearish_momentum = (
        price < float(
            previous["close"]
        )
    )

    bearish_cross = (
        prev_ema5 >= prev_ema9
        and ema5 < ema9
    )

    if bearish_alignment:
        direction = "PUT"

        score += 3

        if bearish_momentum:
            score += 1

        if bearish_cross:
            score += 2

        if rvol >= 1.5:
            score += 1


    if direction is None:
        return None

    if score < 4:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "score": score,
        "price": round(
            price,
            2
        ),
        "ema5": round(
            ema5,
            2
        ),
        "ema9": round(
            ema9,
            2
        ),
        "ema30": round(
            ema30,
            2
        ),
        "vwap": round(
            vwap,
            2
        ),
        "rvol": round(
            rvol,
            2
        ),
    }


# ============================================================
# SCAN MARKET
# ============================================================

def scan_market():
    symbols = next_scan_symbols()

    signals = []

    log(
        f"Scanning {len(symbols)} stocks..."
    )

    for symbol in symbols:
        try:
            signal = analyze_symbol(
                symbol
            )

            if signal:
                signals.append(signal)

        except Exception:
            continue

    signals.sort(
        key=lambda item: (
            item["score"],
            item["rvol"]
        ),
        reverse=True,
    )

    bot_state["signals"] = signals[:25]

    bot_state["candidates"] = [
        item["symbol"]
        for item in signals[:25]
    ]

    bot_state["last_scan"] = (
        now_et().isoformat()
    )

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
        log(
            "No qualified CALL/PUT setups found."
        )

    return signals


# ============================================================
# 0DTE OPTION CONTRACTS
# ============================================================

def get_0dte_contracts(
    symbol,
    direction
):
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
                "expiration_date": (
                    today_string()
                ),
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
        log(
            f"{symbol}: no usable 0DTE "
            f"contracts found."
        )

        return []


# ============================================================
# SELECT ATM CONTRACT
# ============================================================

def choose_0dte_contract(
    symbol,
    direction,
    stock_price
):
    contracts = get_0dte_contracts(
        symbol,
        direction
    )

    if not contracts:
        return None

    valid = []

    for contract in contracts:
        try:
            option_symbol = (
                contract.get("symbol")
            )

            strike = float(
                contract.get(
                    "strike_price",
                    0
                )
            )

            if not option_symbol:
                continue

            distance = abs(
                strike
                - stock_price
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

    distance, strike, option_symbol = (
        valid[0]
    )

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
            "/v1beta1/options/quotes/latest",
            params={
                "symbols": option_symbol
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
            "bid": bid,
            "ask": ask,
            "mid": mid,
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
            f"Position error: "
            f"{safe_text(e)}"
        )

        return []


def open_option_positions():
    positions = get_positions()

    option_positions = []

    for position in positions:
        asset_class = (
            position.get(
                "asset_class",
                ""
            )
        )

        if (
            "option"
            in asset_class.lower()
        ):
            option_positions.append(
                position
            )

    return option_positions


# ============================================================
# ORDERS
# ============================================================

def submit_option_buy(
    option_symbol,
    quantity
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

    result = alpaca_post(
        "/v2/orders",
        payload
    )

    log(
        f"BUY ORDER SENT: "
        f"{quantity} {option_symbol}"
    )

    return result


def submit_option_sell(
    option_symbol,
    quantity
):
    quantity = int(quantity)

    if quantity <= 0:
        return None

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

    result = alpaca_post(
        "/v2/orders",
        payload
    )

    log(
        f"SELL ORDER SENT: "
        f"{quantity} {option_symbol}"
    )

    return result


# ============================================================
# ENTER TRADE
# ============================================================

def enter_trade(signal):
    current_time = now_et().time()

    if current_time >= LAST_ENTRY_TIME:
        log(
            f'Skipping {signal["symbol"]}: '
            f"past 2:45 PM ET."
        )

        return False

    current_positions = (
        open_option_positions()
    )

    if (
        len(current_positions)
        >= MAX_OPEN_POSITIONS
    ):
        log(
            "Maximum open option positions reached."
        )

        return False

    stock_symbol = signal["symbol"]

    # Prevent repeated entries on same underlying
    for data in managed_positions.values():
        if (
            data.get("underlying")
            == stock_symbol
        ):
            return False

    contract = choose_0dte_contract(
        stock_symbol,
        signal["direction"],
        signal["price"],
    )

    if not contract:
        log(
            f"{stock_symbol}: "
            f"no 0DTE contract available."
        )

        return False

    option_symbol = contract["symbol"]

    quote = get_option_quote(
        option_symbol
    )

    if not quote:
        log(
            f"{option_symbol}: "
            f"no option quote."
        )

        return False

    option_price = quote["ask"]

    if option_price <= 0:
        option_price = quote["mid"]

    if option_price <= 0:
        return False

    contract_cost = (
        option_price * 100
    )

    quantity = int(
        POSITION_DOLLARS
        // contract_cost
    )

    # Need enough money for at least 1 contract
    if quantity < 1:
        log(
            f"{option_symbol}: contract costs "
            f"${contract_cost:.2f}, above "
            f"${POSITION_DOLLARS:.2f} position size."
        )

        return False

    result = submit_option_buy(
        option_symbol,
        quantity
    )

    if not result:
        return False

    managed_positions[
        option_symbol
    ] = {
        "option_symbol": option_symbol,
        "underlying": stock_symbol,
        "direction": signal["direction"],
        "quantity": quantity,
        "original_quantity": quantity,
        "entry_price": option_price,
        "highest_price": option_price,
        "tp_taken": False,
        "entered_at": (
            now_et().isoformat()
        ),
    }

    log(
        f'ENTERED {signal["direction"]}: '
        f'{stock_symbol} '
        f'{option_symbol} '
        f'qty={quantity} '
        f'entry~${option_price:.2f}'
    )

    return True


# ============================================================
# MANAGE WINNERS / STOP LOSS / RUNNERS
# ============================================================

def manage_positions():
    if not managed_positions:
        return

    current_time = now_et().time()

    for option_symbol in list(
        managed_positions.keys()
    ):
        data = managed_positions.get(
            option_symbol
        )

        if not data:
            continue

        quote = get_option_quote(
            option_symbol
        )

        if not quote:
            continue

        current_price = quote["bid"]

        if current_price <= 0:
            current_price = quote["mid"]

        if current_price <= 0:
            continue

        entry_price = float(
            data["entry_price"]
        )

        quantity = int(
            data["quantity"]
        )

        if quantity <= 0:
            managed_positions.pop(
                option_symbol,
                None
            )

            continue


        # ====================================================
        # UPDATE HIGH
        # ====================================================

        if (
            current_price
            > data["highest_price"]
        ):
            data["highest_price"] = (
                current_price
            )


        # ====================================================
        # FORCE CLOSE BEFORE EXPIRATION
        # ====================================================

        if current_time >= FORCE_EXIT_TIME:
            submit_option_sell(
                option_symbol,
                quantity
            )

            log(
                f"FORCE EXIT 0DTE: "
                f"{option_symbol}"
            )

            managed_positions.pop(
                option_symbol,
                None
            )

            continue


        gain_percent = (
            current_price
            - entry_price
        ) / entry_price


        # ====================================================
        # 20% HARD STOP
        # ====================================================

        if (
            gain_percent
            <= -STOP_LOSS_PERCENT
        ):
            submit_option_sell(
                option_symbol,
                quantity
            )

            log(
                f"STOP LOSS: {option_symbol} "
                f"{gain_percent * 100:.1f}%"
            )

            managed_positions.pop(
                option_symbol,
                None
            )

            continue


        # ====================================================
        # +30% TAKE PROFIT
        # SELL HALF
        # ====================================================

        if (
            not data["tp_taken"]
            and gain_percent
            >= TAKE_PROFIT_PERCENT
        ):
            sell_quantity = max(
                1,
                int(
                    math.floor(
                        data["original_quantity"]
                        * TAKE_PROFIT_FRACTION
                    )
                )
            )

            sell_quantity = min(
                sell_quantity,
                quantity
            )

            # If only one contract, keep it as runner
            # instead of selling the entire trade immediately
            if quantity > 1:
                submit_option_sell(
                    option_symbol,
                    sell_quantity
                )

                data["quantity"] -= (
                    sell_quantity
                )

                log(
                    f"TAKE PROFIT +30%: "
                    f"{option_symbol} "
                    f"sold {sell_quantity}, "
                    f"runner qty="
                    f'{data["quantity"]}'
                )

            else:
                log(
                    f"{option_symbol} reached "
                    f"+30%. Holding single "
                    f"contract as runner."
                )

            data["tp_taken"] = True

            # Reset runner high from current price
            data["highest_price"] = (
                current_price
            )


        # ====================================================
        # LET WINNERS RUN
        # 15% TRAILING STOP AFTER TP
        # ====================================================

        if data["tp_taken"]:
            highest = float(
                data["highest_price"]
            )

            trailing_stop = (
                highest
                * (
                    1
                    - RUNNER_TRAIL_PERCENT
                )
            )

            # Never let a strong winner turn
            # into a major loser after TP.
            breakeven_floor = (
                entry_price
                * 1.02
            )

            effective_stop = max(
                trailing_stop,
                breakeven_floor
            )

            if (
                current_price
                <= effective_stop
            ):
                remaining_quantity = int(
                    data["quantity"]
                )

                submit_option_sell(
                    option_symbol,
                    remaining_quantity
                )

                runner_gain = (
                    (
                        current_price
                        - entry_price
                    )
                    / entry_price
                )

                log(
                    f"RUNNER EXIT: "
                    f"{option_symbol} "
                    f"{runner_gain * 100:.1f}%"
                )

                managed_positions.pop(
                    option_symbol,
                    None
                )


# ============================================================
# TRADING CYCLE
# ============================================================

def trading_cycle():
    bot_state["last_cycle"] = (
        now_et().isoformat()
    )

    # Manage existing trades first
    manage_positions()

    if not market_is_open():
        log(
            "Market closed."
        )

        return

    signals = scan_market()

    if not signals:
        return

    new_trades = 0

    for signal in signals:
        if (
            new_trades
            >= MAX_NEW_TRADES_PER_CYCLE
        ):
            break

        try:
            entered = enter_trade(
                signal
            )

            if entered:
                new_trades += 1

        except Exception as e:
            add_error(
                f'Entry error '
                f'{signal.get("symbol")}: '
                f'{safe_text(e)}'
            )


# ============================================================
# MAIN BOT LOOP
# ============================================================

def bot_loop():
    bot_state["running"] = True

    log(
        "========================================"
    )

    log(
        "ALPACA 0DTE BOT STARTING"
    )

    log(
        f"AUTO_TRADE={AUTO_TRADE}"
    )

    log(
        f"RUN_BOT_LOOP={RUN_BOT_LOOP}"
    )

    log(
        "Trading URL: "
        "https://paper-api.alpaca.markets"
    )

    log(
        "========================================"
    )


    # Verify credentials BEFORE scanner starts
    if not verify_credentials():
        log(
            "BOT STOPPED: Alpaca credentials "
            "failed verification."
        )

        bot_state["running"] = False

        return


    load_stock_universe()


    while RUN_BOT_LOOP:
        try:
            trading_cycle()

        except Exception as e:
            add_error(
                f"Bot loop error: "
                f"{safe_text(e)}"
            )

        time.sleep(
            LOOP_SECONDS
        )

    bot_state["running"] = False


# ============================================================
# WEB ROUTES
# ============================================================

@app.route("/")
def home():
    return jsonify(
        {
            "status": "online",
            "bot": "Alpaca 0DTE Trading Bot",
            "paper_trading": True,
            "auto_trade": AUTO_TRADE,
            "run_bot_loop": RUN_BOT_LOOP,
            "credentials_ok": (
                bot_state[
                    "credentials_ok"
                ]
            ),
            "market_open": (
                bot_state[
                    "market_open"
                ]
            ),
            "last_cycle": (
                bot_state[
                    "last_cycle"
                ]
            ),
            "last_scan": (
                bot_state[
                    "last_scan"
                ]
            ),
            "managed_positions": (
                managed_positions
            ),
            "top_signals": (
                bot_state[
                    "signals"
                ][0:10]
            ),
            "errors": (
                bot_state[
                    "errors"
                ][-10:]
            ),
        }
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "credentials_ok": (
                bot_state[
                    "credentials_ok"
                ]
            ),
            "bot_running": (
                bot_state[
                    "running"
                ]
            ),
        }
    )


@app.route("/account")
def account():
    try:
        data = alpaca_get(
            "/v2/account"
        )

        return jsonify(
            {
                "connected": True,
                "status": data.get(
                    "status"
                ),
                "cash": data.get(
                    "cash"
                ),
                "equity": data.get(
                    "equity"
                ),
                "buying_power": data.get(
                    "buying_power"
                ),
                "options_buying_power": (
                    data.get(
                        "options_buying_power"
                    )
                ),
            }
        )

    except Exception as e:
        return jsonify(
            {
                "connected": False,
                "error": safe_text(e),
            }
        ), 500


@app.route("/signals")
def signals():
    return jsonify(
        {
            "last_scan": (
                bot_state[
                    "last_scan"
                ]
            ),
            "count": len(
                bot_state[
                    "signals"
                ]
            ),
            "signals": (
                bot_state[
                    "signals"
                ]
            ),
        }
    )


@app.route("/positions")
def positions_route():
    return jsonify(
        {
            "alpaca_positions": (
                get_positions()
            ),
            "managed_positions": (
                managed_positions
            ),
        }
    )


# ============================================================
# START BACKGROUND BOT
# ============================================================

def start_bot_thread():
    if not RUN_BOT_LOOP:
        log(
            "RUN_BOT_LOOP=false. "
            "Background trading disabled."
        )

        return

    thread = threading.Thread(
        target=bot_loop,
        daemon=True,
    )

    thread.start()


start_bot_thread()


# ============================================================
# LOCAL / RENDER START
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