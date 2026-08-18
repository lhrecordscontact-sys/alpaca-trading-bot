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

TRADING_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

TIMEFRAME_MINUTES = 4

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
# CLEAN ENVIRONMENT VARIABLES
# ============================================================

def clean_credential(value):
    if value is None:
        return ""

    value = str(value).strip()

    value = (
        value
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
        .replace("\xa0", "")
    )

    return (
        value
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
# RISK SETTINGS
# ============================================================

MAX_OPEN_POSITIONS = 3

MAX_NEW_TRADES_PER_CYCLE = 1

# Maximum dollars spent per option position
POSITION_DOLLARS = 500.00

# 20% option premium stop
STOP_LOSS_PERCENT = 0.20

# First profit target
TAKE_PROFIT_PERCENT = 0.30

# Sell half when possible
TAKE_PROFIT_FRACTION = 0.50

# Trail runners 15% below highest option price
RUNNER_TRAIL_PERCENT = 0.15

# After TP, don't allow runner below +2%
RUNNER_BREAKEVEN_BUFFER = 0.02


# ============================================================
# 0DTE TIME RULES
# ============================================================

# No new entries after this time
LAST_ENTRY_TIME = dt_time(14, 45)

# Close all bot-managed 0DTE positions
FORCE_EXIT_TIME = dt_time(15, 15)


# ============================================================
# SCANNER SETTINGS
# ============================================================

MIN_PRICE = 5.00

MIN_DOLLAR_VOLUME = 5_000_000

MIN_RVOL = 1.10

# Number of symbols checked each rotation
SCAN_LIMIT = 150

# Seconds between cycles
LOOP_SECONDS = 45

# Number of symbols per market-data request
BAR_BATCH_SIZE = 40


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
    "scanned_symbols": [],
    "signals": [],
    "errors": [],
}

managed_positions = {}

stock_universe = []

stock_universe_index = 0


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

    return (
        text
        .encode("ascii", errors="replace")
        .decode("ascii")
    )


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

    bot_state["errors"] = (
        bot_state["errors"][-25:]
    )

    log(
        f"ERROR: {message}"
    )


# ============================================================
# ALPACA REQUESTS
# ============================================================

def alpaca_get(
    path,
    params=None,
    data_api=False,
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
        timeout=25,
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
        timeout=25,
    )

    if not response.ok:
        raise RuntimeError(
            f"Alpaca POST {path} "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    return response.json()


# ============================================================
# VERIFY CREDENTIALS
# ============================================================

def verify_credentials():

    if not ALPACA_API_KEY:
        add_error(
            "ALPACA_API_KEY missing in Render."
        )
        return False

    if not ALPACA_SECRET_KEY:
        add_error(
            "ALPACA_SECRET_KEY missing in Render."
        )
        return False

    try:
        account = alpaca_get(
            "/v2/account"
        )

        bot_state[
            "credentials_ok"
        ] = True

        bot_state["account"] = {
            "status": account.get("status"),
            "cash": account.get("cash"),
            "equity": account.get("equity"),
            "buying_power": account.get(
                "buying_power"
            ),
            "options_buying_power": (
                account.get(
                    "options_buying_power"
                )
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

    except Exception as e:
        bot_state[
            "credentials_ok"
        ] = False

        add_error(
            f"Credential verification failed: "
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
                False,
            )
        )

        bot_state[
            "market_open"
        ] = is_open

        return is_open

    except Exception as e:

        bot_state[
            "market_open"
        ] = False

        add_error(
            f"Market clock error: "
            f"{safe_text(e)}"
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

            symbol = asset.get(
                "symbol"
            )

            if not symbol:
                continue

            if not asset.get(
                "tradable",
                False,
            ):
                continue

            # Skip unusual symbols
            if "." in symbol:
                continue

            if "/" in symbol:
                continue

            if len(symbol) > 6:
                continue

            symbols.append(
                symbol
            )

        stock_universe = list(
            dict.fromkeys(
                PRIORITY_SYMBOLS
                + symbols
            )
        )

        log(
            f"Loaded "
            f"{len(stock_universe)} "
            f"tradable U.S. stocks."
        )

        return True

    except Exception as e:

        stock_universe = (
            PRIORITY_SYMBOLS.copy()
        )

        add_error(
            f"Stock universe error: "
            f"{safe_text(e)}"
        )

        return False


def next_scan_symbols():

    global stock_universe_index

    if not stock_universe:
        load_stock_universe()

    selected = []

    # Priority names every scan
    for symbol in PRIORITY_SYMBOLS:
        if symbol not in selected:
            selected.append(symbol)

    total = len(
        stock_universe
    )

    while (
        len(selected)
        < SCAN_LIMIT
        and total > 0
    ):

        if stock_universe_index >= total:
            stock_universe_index = 0

        symbol = stock_universe[
            stock_universe_index
        ]

        stock_universe_index += 1

        if symbol not in selected:
            selected.append(
                symbol
            )

    return selected


# ============================================================
# BATCH STOCK BARS
# ============================================================

def get_bars_batch(symbols):

    result = {}

    if not symbols:
        return result

    start = (
        now_et()
        - timedelta(days=7)
    ).astimezone(
        ZoneInfo("UTC")
    ).isoformat()

    for start_index in range(
        0,
        len(symbols),
        BAR_BATCH_SIZE,
    ):

        batch = symbols[
            start_index:
            start_index
            + BAR_BATCH_SIZE
        ]

        page_token = None

        collected = {
            symbol: []
            for symbol in batch
        }

        try:

            while True:

                params = {
                    "symbols": ",".join(
                        batch
                    ),
                    "timeframe": (
                        f"{TIMEFRAME_MINUTES}Min"
                    ),
                    "start": start,
                    "adjustment": "raw",
                    "feed": "iex",
                    "limit": 10000,
                }

                if page_token:
                    params[
                        "page_token"
                    ] = page_token

                data = alpaca_get(
                    "/v2/stocks/bars",
                    params=params,
                    data_api=True,
                )

                bars = data.get(
                    "bars",
                    {}
                )

                for symbol, rows in bars.items():
                    collected.setdefault(
                        symbol,
                        []
                    ).extend(
                        rows
                    )

                page_token = data.get(
                    "next_page_token"
                )

                if not page_token:
                    break

            for symbol, rows in (
                collected.items()
            ):

                if not rows:
                    continue

                # Keep the newest bars
                rows = rows[-140:]

                df = pd.DataFrame(
                    rows
                )

                if df.empty:
                    continue

                df[
                    "timestamp"
                ] = pd.to_datetime(
                    df["t"],
                    utc=True,
                ).dt.tz_convert(
                    NY
                )

                df["open"] = (
                    pd.to_numeric(
                        df["o"]
                    )
                )

                df["high"] = (
                    pd.to_numeric(
                        df["h"]
                    )
                )

                df["low"] = (
                    pd.to_numeric(
                        df["l"]
                    )
                )

                df["close"] = (
                    pd.to_numeric(
                        df["c"]
                    )
                )

                df["volume"] = (
                    pd.to_numeric(
                        df["v"]
                    )
                )

                df = df.sort_values(
                    "timestamp"
                )

                # Remove unfinished 4-minute bar
                current = now_et()

                if not df.empty:

                    last_timestamp = (
                        df.iloc[-1][
                            "timestamp"
                        ]
                    )

                    bar_close = (
                        last_timestamp
                        + timedelta(
                            minutes=(
                                TIMEFRAME_MINUTES
                            )
                        )
                    )

                    if bar_close > current:
                        df = df.iloc[
                            :-1
                        ]

                if len(df) >= 35:
                    result[
                        symbol
                    ] = df

        except Exception as e:

            add_error(
                f"Batch bars error: "
                f"{safe_text(e)}"
            )

    return result


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    df["ema5"] = (
        df["close"]
        .ewm(
            span=5,
            adjust=False,
        )
        .mean()
    )

    df["ema9"] = (
        df["close"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    df["ema30"] = (
        df["close"]
        .ewm(
            span=30,
            adjust=False,
        )
        .mean()
    )

    df["date"] = (
        df["timestamp"]
        .dt.date
    )

    typical = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    pv = (
        typical
        * df["volume"]
    )

    df[
        "cum_pv"
    ] = pv.groupby(
        df["date"]
    ).cumsum()

    df[
        "cum_volume"
    ] = df["volume"].groupby(
        df["date"]
    ).cumsum()

    df["vwap"] = (
        df["cum_pv"]
        / df[
            "cum_volume"
        ].replace(
            0,
            float("nan"),
        )
    )

    df[
        "avg_volume"
    ] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["rvol"] = (
        df["volume"]
        / df[
            "avg_volume"
        ].replace(
            0,
            float("nan"),
        )
    )

    return df


# ============================================================
# PURGATORY METHOD
# ============================================================

def analyze_symbol(
    symbol,
    df,
):

    if df is None:
        return None

    if len(df) < 35:
        return None

    df = calculate_indicators(
        df
    )

    current = df.iloc[-1]

    previous = df.iloc[-2]

    price = float(
        current["close"]
    )

    if price < MIN_PRICE:
        return None

    dollar_volume = (
        float(
            current["volume"]
        )
        * price
    )

    if (
        dollar_volume
        < MIN_DOLLAR_VOLUME
    ):
        return None

    rvol = current.get(
        "rvol",
        0,
    )

    if pd.isna(rvol):
        rvol = 0

    rvol = float(
        rvol
    )

    if rvol < MIN_RVOL:
        return None

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

    prev_ema5 = float(
        previous["ema5"]
    )

    prev_ema9 = float(
        previous["ema9"]
    )

    previous_close = float(
        previous["close"]
    )

    score = 0

    direction = None


    # ========================================================
    # CALL
    # ========================================================

    bullish_alignment = (
        ema5 > ema9
        and price > vwap
        and price > ema30
        and ema5 > ema30
        and ema9 > ema30
    )

    bullish_momentum = (
        price
        > previous_close
    )

    bullish_cross = (
        prev_ema5
        <= prev_ema9
        and ema5
        > ema9
    )

    if bullish_alignment:

        direction = "CALL"

        score += 3

        if bullish_momentum:
            score += 1

        if bullish_cross:
            score += 2

        if rvol >= 1.50:
            score += 1


    # ========================================================
    # PUT
    # ========================================================

    bearish_alignment = (
        ema5 < ema9
        and price < vwap
        and price < ema30
        and ema5 < ema30
        and ema9 < ema30
    )

    bearish_momentum = (
        price
        < previous_close
    )

    bearish_cross = (
        prev_ema5
        >= prev_ema9
        and ema5
        < ema9
    )

    if bearish_alignment:

        direction = "PUT"

        score += 3

        if bearish_momentum:
            score += 1

        if bearish_cross:
            score += 2

        if rvol >= 1.50:
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
            2,
        ),
        "ema5": round(
            ema5,
            2,
        ),
        "ema9": round(
            ema9,
            2,
        ),
        "ema30": round(
            ema30,
            2,
        ),
        "vwap": round(
            vwap,
            2,
        ),
        "rvol": round(
            rvol,
            2,
        ),
        "bar_time": str(
            current[
                "timestamp"
            ]
        ),
    }


# ============================================================
# MARKET SCANNER
# ============================================================

def scan_market():

    symbols = next_scan_symbols()

    bot_state[
        "scanned_symbols"
    ] = symbols

    log(
        f"Scanning "
        f"{len(symbols)} stocks..."
    )

    bars = get_bars_batch(
        symbols
    )

    signals = []

    for symbol in symbols:

        try:

            df = bars.get(
                symbol
            )

            if df is None:
                continue

            signal = analyze_symbol(
                symbol,
                df,
            )

            if signal:
                signals.append(
                    signal
                )

        except Exception:
            continue

    signals.sort(
        key=lambda item: (
            item["score"],
            item["rvol"],
        ),
        reverse=True,
    )

    bot_state[
        "signals"
    ] = signals[:25]

    bot_state[
        "last_scan"
    ] = now_et().isoformat()

    if signals:

        log(
            "TOP SCANNER RESULTS:"
        )

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
            "No qualified setups "
            "this cycle."
        )

    return signals


# ============================================================
# 0DTE CONTRACTS
# ============================================================

def get_0dte_contracts(
    symbol,
    direction,
):

    option_type = (
        "call"
        if direction == "CALL"
        else "put"
    )

    try:

        data = alpaca_get(
            "/v2/options/contracts",
            params={
                "underlying_symbols": (
                    symbol
                ),
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
            [],
        )

    except Exception as e:

        log(
            f"{symbol}: "
            f"no usable 0DTE contracts. "
            f"{safe_text(e)}"
        )

        return []


# ============================================================
# ATM CONTRACT SELECTION
# ============================================================

def choose_0dte_contract(
    symbol,
    direction,
    stock_price,
):

    contracts = get_0dte_contracts(
        symbol,
        direction,
    )

    if not contracts:
        return None

    choices = []

    for contract in contracts:

        try:

            option_symbol = (
                contract.get(
                    "symbol"
                )
            )

            strike = float(
                contract.get(
                    "strike_price",
                    0,
                )
            )

            if not option_symbol:
                continue

            if strike <= 0:
                continue

            distance = abs(
                strike
                - stock_price
            )

            choices.append(
                (
                    distance,
                    strike,
                    option_symbol,
                )
            )

        except Exception:
            continue

    if not choices:
        return None

    choices.sort(
        key=lambda item: item[0]
    )

    distance, strike, option_symbol = (
        choices[0]
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

def get_option_quote(
    option_symbol,
):

    try:

        data = alpaca_get(
            "/v1beta1/options/quotes/latest",
            params={
                "symbols": option_symbol,
                "feed": "indicative",
            },
            data_api=True,
        )

        quote = (
            data
            .get(
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
                ask,
            )

        return {
            "bid": bid,
            "ask": ask,
            "mid": mid,
        }

    except Exception as e:

        log(
            f"Option quote error "
            f"{option_symbol}: "
            f"{safe_text(e)}"
        )

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

        asset_class = str(
            position.get(
                "asset_class",
                ""
            )
        ).lower()

        if "option" in asset_class:
            option_positions.append(
                position
            )

    return option_positions


# ============================================================
# ORDERS
# ============================================================

def submit_option_buy(
    option_symbol,
    quantity,
):

    payload = {
        "symbol": option_symbol,
        "qty": str(
            int(quantity)
        ),
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
    }

    if not AUTO_TRADE:

        log(
            f"SIGNAL ONLY: BUY "
            f"{quantity} "
            f"{option_symbol}"
        )

        return None

    result = alpaca_post(
        "/v2/orders",
        payload,
    )

    log(
        f"BUY ORDER SENT: "
        f"{quantity} "
        f"{option_symbol}"
    )

    return result


def submit_option_sell(
    option_symbol,
    quantity,
):

    quantity = int(
        quantity
    )

    if quantity <= 0:
        return None

    payload = {
        "symbol": option_symbol,
        "qty": str(
            quantity
        ),
        "side": "sell",
        "type": "market",
        "time_in_force": "day",
    }

    if not AUTO_TRADE:

        log(
            f"SIGNAL ONLY: SELL "
            f"{quantity} "
            f"{option_symbol}"
        )

        return None

    result = alpaca_post(
        "/v2/orders",
        payload,
    )

    log(
        f"SELL ORDER SENT: "
        f"{quantity} "
        f"{option_symbol}"
    )

    return result


def get_order(
    order_id,
):

    try:

        return alpaca_get(
            f"/v2/orders/{order_id}"
        )

    except Exception:
        return None


def wait_for_fill(
    order_id,
    timeout=20,
):

    start = time.time()

    while (
        time.time()
        - start
        < timeout
    ):

        order = get_order(
            order_id
        )

        if order:

            status = str(
                order.get(
                    "status",
                    ""
                )
            )

            if status == "filled":

                return order

            if status in {
                "canceled",
                "expired",
                "rejected",
                "suspended",
            }:

                return order

        time.sleep(
            1
        )

    return get_order(
        order_id
    )


# ============================================================
# ENTER TRADE
# ============================================================

def enter_trade(signal):

    current_time = (
        now_et().time()
    )

    if (
        current_time
        >= LAST_ENTRY_TIME
    ):

        log(
            f'{signal["symbol"]}: '
            f"entry skipped after "
            f"2:45 PM ET."
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
            "Maximum option "
            "positions reached."
        )

        return False


    stock_symbol = (
        signal["symbol"]
    )


    # Prevent duplicate underlying
    for data in (
        managed_positions.values()
    ):

        if (
            data.get(
                "underlying"
            )
            == stock_symbol
        ):

            return False


    contract = (
        choose_0dte_contract(
            stock_symbol,
            signal["direction"],
            signal["price"],
        )
    )

    if not contract:

        log(
            f"{stock_symbol}: "
            f"no 0DTE contract."
        )

        return False


    option_symbol = (
        contract["symbol"]
    )


    quote = get_option_quote(
        option_symbol
    )

    if not quote:

        log(
            f"{option_symbol}: "
            f"no usable quote."
        )

        return False


    option_price = (
        quote["ask"]
    )

    if option_price <= 0:
        option_price = (
            quote["mid"]
        )

    if option_price <= 0:
        return False


    contract_cost = (
        option_price
        * 100
    )

    quantity = int(
        math.floor(
            POSITION_DOLLARS
            / contract_cost
        )
    )

    if quantity < 1:

        log(
            f"{option_symbol}: "
            f"contract ~"
            f"${contract_cost:.2f}, "
            f"above position limit "
            f"${POSITION_DOLLARS:.2f}."
        )

        return False


    if not AUTO_TRADE:

        log(
            f'SIGNAL: '
            f'{signal["direction"]} '
            f'{stock_symbol} '
            f'{option_symbol} '
            f'qty={quantity} '
            f'est=${option_price:.2f}'
        )

        return False


    order = submit_option_buy(
        option_symbol,
        quantity,
    )

    if not order:
        return False


    order_id = order.get(
        "id"
    )

    filled = None

    if order_id:
        filled = wait_for_fill(
            order_id
        )


    entry_price = option_price

    actual_qty = quantity


    if filled:

        fill_price = filled.get(
            "filled_avg_price"
        )

        fill_qty = filled.get(
            "filled_qty"
        )

        if fill_price:
            try:
                entry_price = float(
                    fill_price
                )
            except Exception:
                pass

        if fill_qty:
            try:
                actual_qty = int(
                    float(
                        fill_qty
                    )
                )
            except Exception:
                pass


    if actual_qty <= 0:

        log(
            f"{option_symbol}: "
            f"order not filled."
        )

        return False


    managed_positions[
        option_symbol
    ] = {
        "option_symbol": option_symbol,
        "underlying": stock_symbol,
        "direction": signal[
            "direction"
        ],
        "quantity": actual_qty,
        "original_quantity": (
            actual_qty
        ),
        "entry_price": (
            entry_price
        ),
        "highest_price": (
            entry_price
        ),
        "tp_taken": False,
        "entered_at": (
            now_et().isoformat()
        ),
    }


    log(
        f'ENTERED '
        f'{signal["direction"]}: '
        f'{stock_symbol} '
        f'{option_symbol} '
        f'qty={actual_qty} '
        f'entry=${entry_price:.2f}'
    )

    return True


# ============================================================
# MANAGE POSITIONS
# ============================================================

def manage_positions():

    if not managed_positions:
        return

    current_time = (
        now_et().time()
    )


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


        current_price = (
            quote["bid"]
        )

        if current_price <= 0:
            current_price = (
                quote["mid"]
            )

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
                None,
            )

            continue


        # Update high
        if (
            current_price
            > float(
                data[
                    "highest_price"
                ]
            )
        ):

            data[
                "highest_price"
            ] = current_price


        # ====================================================
        # FORCE EXIT
        # ====================================================

        if (
            current_time
            >= FORCE_EXIT_TIME
        ):

            submit_option_sell(
                option_symbol,
                quantity,
            )

            log(
                f"FORCE EXIT 0DTE: "
                f"{option_symbol}"
            )

            managed_positions.pop(
                option_symbol,
                None,
            )

            continue


        gain_percent = (
            current_price
            - entry_price
        ) / entry_price


        # ====================================================
        # HARD STOP -20%
        # ====================================================

        if (
            gain_percent
            <= -STOP_LOSS_PERCENT
        ):

            submit_option_sell(
                option_symbol,
                quantity,
            )

            log(
                f"STOP LOSS: "
                f"{option_symbol} "
                f"{gain_percent * 100:.1f}%"
            )

            managed_positions.pop(
                option_symbol,
                None,
            )

            continue


        # ====================================================
        # +30% TAKE PROFIT
        # ====================================================

        if (
            not data["tp_taken"]
            and gain_percent
            >= TAKE_PROFIT_PERCENT
        ):

            if quantity > 1:

                sell_quantity = max(
                    1,
                    int(
                        math.floor(
                            quantity
                            * TAKE_PROFIT_FRACTION
                        )
                    ),
                )

                # Make sure one runner remains
                sell_quantity = min(
                    sell_quantity,
                    quantity - 1,
                )

                if sell_quantity > 0:

                    submit_option_sell(
                        option_symbol,
                        sell_quantity,
                    )

                    data[
                        "quantity"
                    ] -= sell_quantity

                    log(
                        f"TAKE PROFIT: "
                        f"{option_symbol} "
                        f"+{gain_percent * 100:.1f}% "
                        f"sold={sell_quantity} "
                        f"runner="
                        f'{data["quantity"]}'
                    )

            else:

                log(
                    f"{option_symbol}: "
                    f"+{gain_percent * 100:.1f}% "
                    f"single contract "
                    f"becomes runner."
                )


            data[
                "tp_taken"
            ] = True

            data[
                "highest_price"
            ] = current_price


        # ====================================================
        # RUNNER TRAILING STOP
        # ====================================================

        if data["tp_taken"]:

            highest_price = float(
                data[
                    "highest_price"
                ]
            )

            trailing_stop = (
                highest_price
                * (
                    1
                    - RUNNER_TRAIL_PERCENT
                )
            )

            breakeven_floor = (
                entry_price
                * (
                    1
                    + RUNNER_BREAKEVEN_BUFFER
                )
            )

            effective_stop = max(
                trailing_stop,
                breakeven_floor,
            )

            if (
                current_price
                <= effective_stop
            ):

                remaining_qty = int(
                    data[
                        "quantity"
                    ]
                )

                submit_option_sell(
                    option_symbol,
                    remaining_qty,
                )

                log(
                    f"RUNNER EXIT: "
                    f"{option_symbol} "
                    f"price=${current_price:.2f} "
                    f"high=${highest_price:.2f}"
                )

                managed_positions.pop(
                    option_symbol,
                    None,
                )


# ============================================================
# TRADING CYCLE
# ============================================================

def trading_cycle():

    bot_state[
        "last_cycle"
    ] = now_et().isoformat()


    # Always manage existing trades first
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
# BOT LOOP
# ============================================================

def bot_loop():

    bot_state[
        "running"
    ] = True

    log(
        "========================================"
    )

    log(
        "ALPACA 0DTE PAPER BOT STARTING"
    )

    log(
        f"AUTO_TRADE={AUTO_TRADE}"
    )

    log(
        f"RUN_BOT_LOOP={RUN_BOT_LOOP}"
    )

    log(
        "Paper URL: "
        "https://paper-api.alpaca.markets"
    )

    log(
        "========================================"
    )


    if not verify_credentials():

        log(
            "BOT STOPPED: "
            "credential verification failed."
        )

        bot_state[
            "running"
        ] = False

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


    bot_state[
        "running"
    ] = False


# ============================================================
# WEB ROUTES
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "status": "online",
        "bot": "Alpaca 0DTE Paper Trading Bot",
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
        "stocks_loaded": len(
            stock_universe
        ),
        "stocks_scanned_this_cycle": len(
            bot_state[
                "scanned_symbols"
            ]
        ),
        "managed_positions": (
            managed_positions
        ),
        "top_signals": (
            bot_state[
                "signals"
            ][:10]
        ),
        "errors": (
            bot_state[
                "errors"
            ][-10:]
        ),
    })


@app.route("/health")
def health():

    return jsonify({
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
        "auto_trade": AUTO_TRADE,
    })


@app.route("/account")
def account():

    try:

        data = alpaca_get(
            "/v2/account"
        )

        return jsonify({
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
            "buying_power": (
                data.get(
                    "buying_power"
                )
            ),
            "options_buying_power": (
                data.get(
                    "options_buying_power"
                )
            ),
        })

    except Exception as e:

        return jsonify({
            "connected": False,
            "error": safe_text(
                e
            ),
        }), 500


@app.route("/signals")
def signals():

    return jsonify({
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
    })


@app.route("/positions")
def positions_route():

    return jsonify({
        "alpaca_positions": (
            get_positions()
        ),
        "managed_positions": (
            managed_positions
        ),
    })


# ============================================================
# START BACKGROUND BOT
# ============================================================

def start_bot_thread():

    if not RUN_BOT_LOOP:

        log(
            "RUN_BOT_LOOP=false. "
            "Bot disabled."
        )

        return


    thread = threading.Thread(
        target=bot_loop,
        daemon=True,
    )

    thread.start()


start_bot_thread()


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000,
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
    )