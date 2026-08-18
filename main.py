import os
import math
import time
import threading
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

# ============================================================
# ALPACA PAPER TRADING ONLY
# ============================================================

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")

# PAPER ACCOUNT ONLY WHILE TESTING
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY or "",
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY or "",
    "Content-Type": "application/json",
}

ET = ZoneInfo("America/New_York")

# ============================================================
# SCANNER SETTINGS
# ============================================================

MIN_PRICE = 5.00
MIN_DAILY_VOLUME = 1_000_000
MIN_SCANNER_SCORE = 70

SNAPSHOT_BATCH_SIZE = 100

DEFAULT_RETURN_LIMIT = 50
MAX_RETURN_LIMIT = 200

ALLOWED_EXCHANGES = {
    "NASDAQ",
    "NYSE",
    "AMEX",
    "ARCA",
    "NYSEARCA",
    "BATS",
}

# ============================================================
# 0DTE OPTION SETTINGS
# ============================================================

# STRICTLY 0DTE
MIN_DTE = 0
MAX_DTE = 0

# Look near current stock price
STRIKE_SEARCH_PERCENT = 0.08

REQUIRE_TRADABLE_OPTION = True
MIN_OPEN_INTEREST = 1

# ============================================================
# ACCOUNT GROWTH / RISK
# ============================================================

# Maximum amount committed to a new trade:
# 1.5% of account equity.
RISK_PER_TRADE_PERCENT = 0.015

# First take profit
TAKE_PROFIT_PERCENT = 0.30

# Sell approximately half at first TP
TAKE_PROFIT_FRACTION = 0.50

# Remaining contracts become runners.
# Runner exits after option falls 15% from highest price
# reached AFTER first TP.
RUNNER_TRAIL_PERCENT = 0.15

# Stop opening new trades after losing 3% of account
# during the trading day.
MAX_DAILY_LOSS_PERCENT = 0.03

# Maximum simultaneous option positions
MAX_OPEN_POSITIONS = 3

# Need at least 2 so we can take partial profit
# and still leave a runner.
MIN_CONTRACTS_FOR_RUNNER = 2

# Hard cap for safety while paper testing.
MAX_CONTRACTS_PER_TRADE = 10

# Force remaining 0DTE positions out at 3:15 PM ET.
FORCE_EXIT_HOUR_ET = 15
FORCE_EXIT_MINUTE_ET = 15

# Don't open fresh 0DTE positions after this time.
LAST_ENTRY_HOUR_ET = 14
LAST_ENTRY_MINUTE_ET = 45

# Bot polling speed.
BOT_LOOP_SECONDS = 15

# ============================================================
# PAPER EXECUTION
# ============================================================

# False = scanner/selection only.
# True = actually submit PAPER orders.
#
# IMPORTANT:
# This URL is hard-wired to Alpaca PAPER above.
AUTO_TRADE = False

# Set RUN_BOT_LOOP=true in Render environment when ready
# for automatic paper monitoring.
RUN_BOT_LOOP = (
    os.environ.get("RUN_BOT_LOOP", "false")
    .strip()
    .lower()
    == "true"
)

# ============================================================
# IN-MEMORY POSITION MANAGEMENT
# ============================================================

# Example:
#
# TRADE_STATE["SPY260817C00600000"] = {
#     "underlying": "SPY",
#     "entry_price": 2.10,
#     "original_qty": 4,
#     "tp_hit": False,
#     "runner_high": None,
#     "partial_exit_qty": 0,
# }
#
TRADE_STATE = {}

STATE_LOCK = threading.Lock()

BOT_THREAD_STARTED = False
BOT_THREAD_LOCK = threading.Lock()

# ============================================================
# BASIC HELPERS
# ============================================================


def now_et():
    return datetime.now(ET)


def today_et():
    return now_et().date()


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def require_keys():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError(
            "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY"
        )


def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def market_is_entry_time():
    current = now_et()

    if current.weekday() >= 5:
        return False

    open_minutes = (9 * 60) + 30

    last_entry_minutes = (
        LAST_ENTRY_HOUR_ET * 60
        + LAST_ENTRY_MINUTE_ET
    )

    current_minutes = (
        current.hour * 60
        + current.minute
    )

    return (
        open_minutes
        <= current_minutes
        < last_entry_minutes
    )


def force_exit_time():
    current = now_et()

    cutoff = (
        FORCE_EXIT_HOUR_ET * 60
        + FORCE_EXIT_MINUTE_ET
    )

    current_minutes = (
        current.hour * 60
        + current.minute
    )

    return current_minutes >= cutoff


# ============================================================
# ALPACA REQUEST HELPERS
# ============================================================


def alpaca_get(path, params=None):
    require_keys()

    response = requests.get(
        f"{ALPACA_BASE_URL}{path}",
        headers=HEADERS,
        params=params,
        timeout=30,
    )

    try:
        data = response.json()
    except Exception:
        data = {"raw": response.text}

    return response.status_code, data


def alpaca_post(path, payload):
    require_keys()

    response = requests.post(
        f"{ALPACA_BASE_URL}{path}",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )

    try:
        data = response.json()
    except Exception:
        data = {"raw": response.text}

    return response.status_code, data


def market_data_get(path, params=None):
    require_keys()

    response = requests.get(
        f"{DATA_BASE_URL}{path}",
        headers=HEADERS,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# ACCOUNT
# ============================================================


def get_account():
    status, data = alpaca_get("/v2/account")

    if status >= 400:
        return None

    return data


def get_positions():
    status, data = alpaca_get("/v2/positions")

    if status >= 400:
        return []

    return data if isinstance(data, list) else []


def get_open_orders():
    status, data = alpaca_get(
        "/v2/orders",
        params={
            "status": "open",
            "limit": 500,
        },
    )

    if status >= 400:
        return []

    return data if isinstance(data, list) else []


def get_option_positions():
    return [
        position
        for position in get_positions()
        if position.get("asset_class") == "us_option"
    ]


# ============================================================
# ACCOUNT RISK
# ============================================================


def get_account_equity():
    account = get_account()

    if not account:
        return None

    return safe_float(
        account.get("equity"),
        None,
    )


def current_daily_pnl():
    account = get_account()

    if not account:
        return None

    equity = safe_float(
        account.get("equity"),
        None,
    )

    last_equity = safe_float(
        account.get("last_equity"),
        None,
    )

    if (
        equity is None
        or last_equity is None
    ):
        return None

    return equity - last_equity


def daily_loss_limit_reached():
    account = get_account()

    if not account:
        return True

    equity = safe_float(
        account.get("equity"),
        0,
    )

    last_equity = safe_float(
        account.get("last_equity"),
        0,
    )

    if last_equity <= 0:
        return False

    daily_pnl = equity - last_equity

    max_loss = (
        last_equity
        * MAX_DAILY_LOSS_PERCENT
    )

    return daily_pnl <= -max_loss


def open_option_count():
    return len(get_option_positions())


def risk_allows_new_trade():
    if daily_loss_limit_reached():
        return False, "daily loss limit reached"

    if open_option_count() >= MAX_OPEN_POSITIONS:
        return False, "maximum open positions reached"

    if not market_is_entry_time():
        return False, "outside entry window"

    return True, "ok"


# ============================================================
# OPTIONABLE STOCK UNIVERSE
# ============================================================


def get_optionable_stock_universe():
    status, assets = alpaca_get(
        "/v2/assets",
        params={
            "status": "active",
            "asset_class": "us_equity",
            "attributes": "has_options",
        },
    )

    if status >= 400:
        raise RuntimeError(
            f"Unable to retrieve Alpaca assets: {assets}"
        )

    if not isinstance(assets, list):
        return []

    symbols = []

    for asset in assets:
        if asset.get("status") != "active":
            continue

        if asset.get("asset_class") != "us_equity":
            continue

        if not asset.get("tradable", False):
            continue

        exchange = str(
            asset.get("exchange", "")
        ).upper()

        if (
            exchange
            and exchange not in ALLOWED_EXCHANGES
        ):
            continue

        symbol = str(
            asset.get("symbol", "")
        ).upper().strip()

        if symbol:
            symbols.append(symbol)

    return sorted(set(symbols))


# ============================================================
# STOCK SNAPSHOTS
# ============================================================


def get_snapshots_batch(symbols):
    if not symbols:
        return {}

    try:
        data = market_data_get(
            "/v2/stocks/snapshots",
            params={
                "symbols": ",".join(symbols),
                "feed": "iex",
            },
        )

        if not isinstance(data, dict):
            return {}

        if "snapshots" in data:
            return data.get("snapshots") or {}

        return data

    except Exception as e:
        print(
            f"Snapshot batch error: {e}",
            flush=True,
        )

        return {}


# ============================================================
# STOCK SCORING / TRIGGERS
# ============================================================


def analyze_snapshot(symbol, snapshot):
    if not snapshot:
        return None

    latest_trade = (
        snapshot.get("latestTrade")
        or snapshot.get("latest_trade")
        or {}
    )

    minute_bar = (
        snapshot.get("minuteBar")
        or snapshot.get("minute_bar")
        or {}
    )

    daily_bar = (
        snapshot.get("dailyBar")
        or snapshot.get("daily_bar")
        or {}
    )

    previous_bar = (
        snapshot.get("prevDailyBar")
        or snapshot.get("prev_daily_bar")
        or {}
    )

    price = (
        latest_trade.get("p")
        or minute_bar.get("c")
        or daily_bar.get("c")
    )

    price = safe_float(price, None)

    if price is None:
        return None

    volume = safe_float(
        daily_bar.get("v"),
        0,
    )

    previous_volume = safe_float(
        previous_bar.get("v"),
        0,
    )

    if price < MIN_PRICE:
        return None

    if volume < MIN_DAILY_VOLUME:
        return None

    current_open = safe_float(
        daily_bar.get("o"),
        None,
    )

    current_high = safe_float(
        daily_bar.get("h"),
        None,
    )

    current_low = safe_float(
        daily_bar.get("l"),
        None,
    )

    previous_close = safe_float(
        previous_bar.get("c"),
        None,
    )

    score = 0
    reasons = []

    # Price
    score += 10
    reasons.append("price filter passed")

    # Liquidity
    score += 20
    reasons.append("high liquidity")

    # Relative volume
    relative_volume = 0.0

    if previous_volume > 0:
        relative_volume = (
            volume / previous_volume
        )

    if relative_volume >= 1.0:
        score += 20
        reasons.append(
            "strong relative volume"
        )

    elif relative_volume >= 0.50:
        score += 10
        reasons.append(
            "moderate relative volume"
        )

    # Price movement
    percent_change = 0.0

    if (
        previous_close is not None
        and previous_close > 0
    ):
        percent_change = (
            (
                price - previous_close
            )
            / previous_close
        ) * 100

    if abs(percent_change) >= 2:
        score += 20
        reasons.append(
            "strong price movement"
        )

    elif abs(percent_change) >= 1:
        score += 10
        reasons.append(
            "moderate price movement"
        )

    # Direction
    direction = "neutral"

    if current_open is not None:
        if price > current_open:
            direction = "bullish"
            score += 10
            reasons.append(
                "trading above daily open"
            )

        elif price < current_open:
            direction = "bearish"
            score += 10
            reasons.append(
                "trading below daily open"
            )

    # Daily range
    range_position = None

    if (
        current_high is not None
        and current_low is not None
        and current_high > current_low
    ):
        range_position = (
            (price - current_low)
            /
            (current_high - current_low)
        )

        if range_position >= 0.75:
            score += 20
            reasons.append(
                "near session highs"
            )

        elif range_position <= 0.25:
            score += 20
            reasons.append(
                "near session lows"
            )

        else:
            score += 5
            reasons.append(
                "middle of session range"
            )

    passed = (
        score >= MIN_SCANNER_SCORE
    )

    if direction == "bullish":
        option_bias = "call"

    elif direction == "bearish":
        option_bias = "put"

    else:
        option_bias = None

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "volume": int(volume),
        "relative_volume": round(
            relative_volume,
            2,
        ),
        "percent_change": round(
            percent_change,
            2,
        ),
        "score": int(score),
        "passed": passed,
        "direction": direction,
        "option_bias": option_bias,
        "reasons": reasons,
    }


# ============================================================
# MARKET SCANNER
# ============================================================


def scan_market(limit=DEFAULT_RETURN_LIMIT):
    symbols = (
        get_optionable_stock_universe()
    )

    results = []

    for batch in chunk_list(
        symbols,
        SNAPSHOT_BATCH_SIZE,
    ):
        snapshots = get_snapshots_batch(
            batch
        )

        for symbol in batch:
            snapshot = snapshots.get(
                symbol
            )

            result = analyze_snapshot(
                symbol,
                snapshot,
            )

            if not result:
                continue

            if not result["passed"]:
                continue

            if not result["option_bias"]:
                continue

            results.append(result)

    results.sort(
        key=lambda item: (
            item["score"],
            abs(item["percent_change"]),
            item["relative_volume"],
        ),
        reverse=True,
    )

    return results[:limit]


# ============================================================
# STRICT 0DTE CONTRACT SEARCH
# ============================================================


def get_0dte_contracts(
    underlying_symbol,
    option_type,
    stock_price,
):
    expiration = today_et().isoformat()

    strike_low = (
        stock_price
        * (1 - STRIKE_SEARCH_PERCENT)
    )

    strike_high = (
        stock_price
        * (1 + STRIKE_SEARCH_PERCENT)
    )

    all_contracts = []
    page_token = None

    while True:
        params = {
            "underlying_symbols":
                underlying_symbol,

            "status":
                "active",

            "type":
                option_type,

            # Exact same-day expiration
            "expiration_date_gte":
                expiration,

            "expiration_date_lte":
                expiration,

            "strike_price_gte":
                round(strike_low, 2),

            "strike_price_lte":
                round(strike_high, 2),

            "limit":
                1000,
        }

        if page_token:
            params["page_token"] = (
                page_token
            )

        status, data = alpaca_get(
            "/v2/options/contracts",
            params=params,
        )

        if status >= 400:
            print(
                f"0DTE contract error "
                f"{underlying_symbol}: {data}",
                flush=True,
            )
            return []

        contracts = (
            data.get(
                "option_contracts",
                [],
            )
            if isinstance(data, dict)
            else []
        )

        all_contracts.extend(
            contracts
        )

        page_token = (
            data.get("page_token")
            if isinstance(data, dict)
            else None
        )

        if not page_token:
            break

    return all_contracts


# ============================================================
# OPTION SNAPSHOT / PREMIUM
# ============================================================


def get_option_snapshot(
    contract_symbol,
):
    try:
        data = market_data_get(
            "/v1beta1/options/snapshots",
            params={
                "symbols":
                    contract_symbol,
            },
        )

        if not isinstance(data, dict):
            return None

        snapshots = (
            data.get("snapshots")
            or data
        )

        return snapshots.get(
            contract_symbol
        )

    except Exception as e:
        print(
            f"Option snapshot error "
            f"{contract_symbol}: {e}",
            flush=True,
        )

        return None


def option_market_price(
    contract_symbol,
):
    snapshot = get_option_snapshot(
        contract_symbol
    )

    if not snapshot:
        return None

    latest_trade = (
        snapshot.get("latestTrade")
        or snapshot.get("latest_trade")
        or {}
    )

    latest_quote = (
        snapshot.get("latestQuote")
        or snapshot.get("latest_quote")
        or {}
    )

    trade_price = safe_float(
        latest_trade.get("p"),
        None,
    )

    bid = safe_float(
        latest_quote.get("bp"),
        None,
    )

    ask = safe_float(
        latest_quote.get("ap"),
        None,
    )

    # Prefer midpoint if both quote sides exist.
    if (
        bid is not None
        and ask is not None
        and bid > 0
        and ask > 0
    ):
        return (bid + ask) / 2

    if trade_price is not None:
        return trade_price

    if bid is not None and bid > 0:
        return bid

    if ask is not None and ask > 0:
        return ask

    return None


# ============================================================
# PICK BEST 0DTE OPTION
# ============================================================


def select_best_0dte_option(
    stock_result,
):
    underlying = stock_result["symbol"]

    stock_price = safe_float(
        stock_result["price"],
        None,
    )

    option_type = stock_result[
        "option_bias"
    ]

    if stock_price is None:
        return None

    contracts = get_0dte_contracts(
        underlying,
        option_type,
        stock_price,
    )

    candidates = []

    for contract in contracts:
        if (
            REQUIRE_TRADABLE_OPTION
            and not contract.get(
                "tradable",
                False,
            )
        ):
            continue

        expiration = contract.get(
            "expiration_date"
        )

        # Extra protection:
        # absolutely no non-0DTE contracts.
        if expiration != today_et().isoformat():
            continue

        strike = safe_float(
            contract.get("strike_price"),
            None,
        )

        if strike is None:
            continue

        open_interest = safe_int(
            contract.get(
                "open_interest"
            ),
            0,
        )

        if (
            open_interest
            < MIN_OPEN_INTEREST
        ):
            continue

        contract_symbol = (
            contract.get("symbol")
        )

        if not contract_symbol:
            continue

        premium = option_market_price(
            contract_symbol
        )

        if (
            premium is None
            or premium <= 0
        ):
            continue

        strike_distance = abs(
            strike - stock_price
        )

        candidates.append({
            "symbol":
                contract_symbol,

            "underlying":
                underlying,

            "type":
                option_type,

            "strike":
                strike,

            "expiration":
                expiration,

            "open_interest":
                open_interest,

            "premium":
                premium,

            "strike_distance":
                strike_distance,
        })

    if not candidates:
        return None

    # Prefer closest-to-ATM,
    # then higher open interest.
    candidates.sort(
        key=lambda item: (
            item["strike_distance"],
            -item["open_interest"],
        )
    )

    return candidates[0]


# ============================================================
# POSITION SIZING
# ============================================================


def calculate_contract_qty(
    option_price,
):
    equity = get_account_equity()

    if (
        equity is None
        or equity <= 0
        or option_price <= 0
    ):
        return 0

    max_trade_dollars = (
        equity
        * RISK_PER_TRADE_PERCENT
    )

    contract_cost = (
        option_price * 100
    )

    qty = math.floor(
        max_trade_dollars
        / contract_cost
    )

    qty = min(
        qty,
        MAX_CONTRACTS_PER_TRADE,
    )

    # Runner strategy requires at least 2.
    if qty < MIN_CONTRACTS_FOR_RUNNER:
        return 0

    return qty


# ============================================================
# ORDER EXECUTION
# ============================================================


def submit_market_order(
    symbol,
    qty,
    side,
    position_intent,
):
    payload = {
        "symbol": symbol,
        "qty": str(int(qty)),
        "side": side,
        "type": "market",
        "time_in_force": "day",
        "position_intent":
            position_intent,
    }

    if not AUTO_TRADE:
        return {
            "paper_order_enabled": False,
            "would_submit": payload,
        }

    status, data = alpaca_post(
        "/v2/orders",
        payload,
    )

    return {
        "status_code": status,
        "response": data,
        "submitted": status < 400,
    }


def buy_option(
    contract_symbol,
    qty,
):
    return submit_market_order(
        symbol=contract_symbol,
        qty=qty,
        side="buy",
        position_intent=
            "buy_to_open",
    )


def sell_option(
    contract_symbol,
    qty,
):
    return submit_market_order(
        symbol=contract_symbol,
        qty=qty,
        side="sell",
        position_intent=
            "sell_to_close",
    )


# ============================================================
# DUPLICATE ENTRY PROTECTION
# ============================================================


def already_in_underlying(
    underlying,
):
    with STATE_LOCK:
        for state in TRADE_STATE.values():
            if (
                state.get("underlying")
                == underlying
            ):
                return True

    return False


# ============================================================
# ENTER FROM SCANNER TRIGGER
# ============================================================


def enter_scanner_trade(
    stock_result,
):
    allowed, reason = (
        risk_allows_new_trade()
    )

    if not allowed:
        return {
            "entered": False,
            "reason": reason,
        }

    underlying = stock_result[
        "symbol"
    ]

    if already_in_underlying(
        underlying
    ):
        return {
            "entered": False,
            "reason":
                "already trading underlying",
        }

    contract = (
        select_best_0dte_option(
            stock_result
        )
    )

    if not contract:
        return {
            "entered": False,
            "reason":
                "no suitable 0DTE contract",
        }

    option_price = contract[
        "premium"
    ]

    qty = calculate_contract_qty(
        option_price
    )

    if qty < MIN_CONTRACTS_FOR_RUNNER:
        return {
            "entered": False,
            "reason":
                "account risk size cannot "
                "support 2 contracts",
        }

    order = buy_option(
        contract["symbol"],
        qty,
    )

    # When AUTO_TRADE is off,
    # show selection without storing
    # a pretend live position.
    if not AUTO_TRADE:
        return {
            "entered": False,
            "paper_preview": True,
            "underlying":
                underlying,
            "trigger":
                stock_result,
            "contract":
                contract,
            "qty":
                qty,
            "order":
                order,
        }

    if not order.get(
        "submitted",
        False,
    ):
        return {
            "entered": False,
            "reason":
                "order rejected",
            "order":
                order,
        }

    # Start state using observed premium.
    # The position manager will later replace
    # entry price with Alpaca's actual
    # avg_entry_price when position appears.
    with STATE_LOCK:
        TRADE_STATE[
            contract["symbol"]
        ] = {
            "underlying":
                underlying,

            "direction":
                stock_result[
                    "direction"
                ],

            "option_type":
                stock_result[
                    "option_bias"
                ],

            "entry_price":
                option_price,

            "original_qty":
                qty,

            "tp_hit":
                False,

            "runner_high":
                None,

            "partial_exit_qty":
                0,

            "entered_at":
                now_et().isoformat(),

            "scanner_score":
                stock_result["score"],
        }

    return {
        "entered": True,
        "underlying":
            underlying,
        "contract":
            contract,
        "qty":
            qty,
        "order":
            order,
    }


# ============================================================
# FIND POSITION
# ============================================================


def get_option_position(
    symbol,
):
    for position in get_option_positions():
        if position.get("symbol") == symbol:
            return position

    return None


# ============================================================
# INITIALIZE UNKNOWN OPTION POSITIONS
# ============================================================


def sync_existing_positions():
    positions = get_option_positions()

    with STATE_LOCK:
        for position in positions:
            symbol = position.get(
                "symbol"
            )

            if not symbol:
                continue

            if symbol in TRADE_STATE:
                continue

            qty = abs(
                safe_int(
                    position.get("qty"),
                    0,
                )
            )

            entry = safe_float(
                position.get(
                    "avg_entry_price"
                ),
                None,
            )

            if (
                qty <= 0
                or entry is None
            ):
                continue

            TRADE_STATE[symbol] = {
                "underlying":
                    None,

                "direction":
                    None,

                "option_type":
                    None,

                "entry_price":
                    entry,

                "original_qty":
                    qty,

                "tp_hit":
                    False,

                "runner_high":
                    None,

                "partial_exit_qty":
                    0,

                "entered_at":
                    None,

                "scanner_score":
                    None,
            }


# ============================================================
# MANAGE TAKE PROFIT + RUNNERS
# ============================================================


def manage_position(
    contract_symbol,
):
    position = get_option_position(
        contract_symbol
    )

    if not position:
        with STATE_LOCK:
            TRADE_STATE.pop(
                contract_symbol,
                None,
            )

        return {
            "symbol":
                contract_symbol,
            "action":
                "position closed",
        }

    current_qty = abs(
        safe_int(
            position.get("qty"),
            0,
        )
    )

    if current_qty <= 0:
        return None

    actual_entry = safe_float(
        position.get(
            "avg_entry_price"
        ),
        None,
    )

    current_price = (
        option_market_price(
            contract_symbol
        )
    )

    if (
        actual_entry is None
        or current_price is None
    ):
        return None

    with STATE_LOCK:
        state = TRADE_STATE.get(
            contract_symbol
        )

        if not state:
            return None

        # Use actual Alpaca fill price.
        state["entry_price"] = (
            actual_entry
        )

        tp_hit = state[
            "tp_hit"
        ]

    # ========================================================
    # FORCE 3:15 PM EXIT
    # ========================================================

    if force_exit_time():
        order = sell_option(
            contract_symbol,
            current_qty,
        )

        return {
            "symbol":
                contract_symbol,
            "action":
                "force_exit",
            "qty":
                current_qty,
            "price":
                current_price,
            "order":
                order,
        }

    # ========================================================
    # FIRST TAKE PROFIT
    # ========================================================

    tp_price = (
        actual_entry
        * (
            1
            + TAKE_PROFIT_PERCENT
        )
    )

    if (
        not tp_hit
        and current_price
        >= tp_price
    ):
        # Sell roughly half,
        # but ALWAYS leave at least
        # one contract as runner.
        exit_qty = max(
            1,
            math.floor(
                current_qty
                * TAKE_PROFIT_FRACTION
            ),
        )

        if exit_qty >= current_qty:
            exit_qty = (
                current_qty - 1
            )

        # If only one contract somehow
        # remains, it becomes runner.
        if exit_qty <= 0:
            with STATE_LOCK:
                state = TRADE_STATE[
                    contract_symbol
                ]

                state["tp_hit"] = True
                state["runner_high"] = (
                    current_price
                )

            return {
                "symbol":
                    contract_symbol,
                "action":
                    "runner_started",
                "price":
                    current_price,
            }

        order = sell_option(
            contract_symbol,
            exit_qty,
        )

        if (
            not AUTO_TRADE
            or order.get(
                "submitted",
                False,
            )
        ):
            with STATE_LOCK:
                state = TRADE_STATE[
                    contract_symbol
                ]

                state["tp_hit"] = True

                state[
                    "runner_high"
                ] = current_price

                state[
                    "partial_exit_qty"
                ] = exit_qty

        return {
            "symbol":
                contract_symbol,

            "action":
                "take_profit",

            "entry":
                actual_entry,

            "price":
                current_price,

            "tp_price":
                tp_price,

            "sell_qty":
                exit_qty,

            "runner_qty":
                current_qty
                - exit_qty,

            "order":
                order,
        }

    # ========================================================
    # RUNNER MANAGEMENT
    # ========================================================

    if tp_hit:
        with STATE_LOCK:
            state = TRADE_STATE[
                contract_symbol
            ]

            runner_high = (
                state.get(
                    "runner_high"
                )
                or current_price
            )

            if current_price > runner_high:
                runner_high = (
                    current_price
                )

                state[
                    "runner_high"
                ] = runner_high

        trail_exit_price = (
            runner_high
            * (
                1
                - RUNNER_TRAIL_PERCENT
            )
        )

        if (
            current_price
            <= trail_exit_price
        ):
            order = sell_option(
                contract_symbol,
                current_qty,
            )

            return {
                "symbol":
                    contract_symbol,

                "action":
                    "runner_exit",

                "entry":
                    actual_entry,

                "runner_high":
                    runner_high,

                "trail_price":
                    trail_exit_price,

                "current_price":
                    current_price,

                "sell_qty":
                    current_qty,

                "order":
                    order,
            }

        return {
            "symbol":
                contract_symbol,

            "action":
                "runner_running",

            "entry":
                actual_entry,

            "current_price":
                current_price,

            "runner_high":
                runner_high,

            "trail_exit_price":
                trail_exit_price,

            "qty":
                current_qty,
        }

    # ========================================================
    # WINNING TRADE BEFORE TP
    # ========================================================

    gain_percent = (
        (
            current_price
            - actual_entry
        )
        / actual_entry
    )

    return {
        "symbol":
            contract_symbol,

        "action":
            "holding",

        "entry":
            actual_entry,

        "current_price":
            current_price,

        "gain_percent":
            round(
                gain_percent * 100,
                2,
            ),

        "tp_price":
            tp_price,

        "qty":
            current_qty,
    }


# ============================================================
# MANAGE ALL OPEN OPTION POSITIONS
# ============================================================


def manage_all_positions():
    sync_existing_positions()

    with STATE_LOCK:
        symbols = list(
            TRADE_STATE.keys()
        )

    actions = []

    for symbol in symbols:
        try:
            result = manage_position(
                symbol
            )

            if result:
                actions.append(
                    result
                )

        except Exception as e:
            actions.append({
                "symbol":
                    symbol,

                "action":
                    "error",

                "error":
                    str(e),
            })

    return actions


# ============================================================
# SCAN + ENTER BEST TRIGGERS
# ============================================================


def scan_and_trade():
    allowed, reason = (
        risk_allows_new_trade()
    )

    if not allowed:
        return {
            "trading_allowed":
                False,

            "reason":
                reason,

            "entries":
                [],
        }

    available_slots = max(
        0,
        MAX_OPEN_POSITIONS
        - open_option_count(),
    )

    if available_slots <= 0:
        return {
            "trading_allowed":
                False,

            "reason":
                "no open slots",

            "entries":
                [],
        }

    candidates = scan_market(
        limit=25
    )

    entries = []

    for candidate in candidates:
        if len(entries) >= available_slots:
            break

        result = enter_scanner_trade(
            candidate
        )

        entries.append(result)

        # Only count actual/preview qualifying
        # entries toward the current cycle.
        if (
            result.get("entered")
            or result.get(
                "paper_preview"
            )
        ):
            if len(entries) >= available_slots:
                break

    return {
        "trading_allowed":
            True,

        "scanner_candidates":
            len(candidates),

        "entries":
            entries,
    }


# ============================================================
# COMPLETE BOT CYCLE
# ============================================================


def run_cycle():
    output = {
        "time_et":
            now_et().isoformat(),

        "auto_trade":
            AUTO_TRADE,

        "paper_api":
            ALPACA_BASE_URL,

        "management":
            [],

        "scanner":
            None,
    }

    # Always manage existing positions first.
    output["management"] = (
        manage_all_positions()
    )

    # Then look for new entries.
    if market_is_entry_time():
        output["scanner"] = (
            scan_and_trade()
        )

    else:
        output["scanner"] = {
            "trading_allowed":
                False,

            "reason":
                "outside entry window",
        }

    return output


# ============================================================
# BACKGROUND LOOP
# ============================================================


def bot_loop():
    print(
        "Paper bot loop started",
        flush=True,
    )

    while True:
        try:
            if AUTO_TRADE:
                result = run_cycle()

                print(
                    f"BOT CYCLE: {result}",
                    flush=True,
                )

            else:
                # Still manage nothing automatically
                # while AUTO_TRADE is False.
                pass

        except Exception as e:
            print(
                f"BOT LOOP ERROR: {e}",
                flush=True,
            )

        time.sleep(
            BOT_LOOP_SECONDS
        )


def start_bot_thread():
    global BOT_THREAD_STARTED

    if not RUN_BOT_LOOP:
        return

    with BOT_THREAD_LOCK:
        if BOT_THREAD_STARTED:
            return

        BOT_THREAD_STARTED = True

        thread = threading.Thread(
            target=bot_loop,
            daemon=True,
        )

        thread.start()


# ============================================================
# FLASK ROUTES
# ============================================================


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":
            "running",

        "mode":
            "ALPACA PAPER ONLY",

        "auto_trade":
            AUTO_TRADE,

        "run_bot_loop":
            RUN_BOT_LOOP,

        "strategy":
            "0DTE scanner + partial TP + runner",

        "rules": {
            "dte":
                0,

            "scanner_score":
                MIN_SCANNER_SCORE,

            "risk_per_trade_percent":
                RISK_PER_TRADE_PERCENT
                * 100,

            "take_profit_percent":
                TAKE_PROFIT_PERCENT
                * 100,

            "take_profit_fraction_percent":
                TAKE_PROFIT_FRACTION
                * 100,

            "runner_trail_percent":
                RUNNER_TRAIL_PERCENT
                * 100,

            "daily_loss_limit_percent":
                MAX_DAILY_LOSS_PERCENT
                * 100,

            "max_open_positions":
                MAX_OPEN_POSITIONS,

            "force_exit_et":
                f"{FORCE_EXIT_HOUR_ET:02d}:"
                f"{FORCE_EXIT_MINUTE_ET:02d}",
        },
    })


@app.route(
    "/account",
    methods=["GET"],
)
def account_route():
    account = get_account()

    if not account:
        return jsonify({
            "error":
                "could not retrieve account"
        }), 500

    return jsonify({
        "equity":
            account.get("equity"),

        "last_equity":
            account.get(
                "last_equity"
            ),

        "buying_power":
            account.get(
                "buying_power"
            ),

        "options_buying_power":
            account.get(
                "options_buying_power"
            ),

        "daily_pnl":
            current_daily_pnl(),

        "daily_loss_limit_reached":
            daily_loss_limit_reached(),

        "option_positions":
            get_option_positions(),
    })


@app.route(
    "/scan",
    methods=["GET"],
)
def scan_route():
    limit = safe_int(
        request.args.get(
            "limit",
            DEFAULT_RETURN_LIMIT,
        ),
        DEFAULT_RETURN_LIMIT,
    )

    limit = max(
        1,
        min(
            limit,
            MAX_RETURN_LIMIT,
        ),
    )

    results = scan_market(limit)

    return jsonify({
        "count":
            len(results),

        "0dte_only":
            True,

        "results":
            results,
    })


@app.route(
    "/preview",
    methods=["GET"],
)
def preview_route():
    candidates = scan_market(
        limit=10
    )

    previews = []

    for candidate in candidates:
        try:
            contract = (
                select_best_0dte_option(
                    candidate
                )
            )

            if not contract:
                continue

            qty = (
                calculate_contract_qty(
                    contract["premium"]
                )
            )

            previews.append({
                "stock":
                    candidate,

                "0dte_contract":
                    contract,

                "calculated_qty":
                    qty,

                "runner_possible":
                    qty >= 2,
            })

        except Exception as e:
            previews.append({
                "symbol":
                    candidate.get(
                        "symbol"
                    ),

                "error":
                    str(e),
            })

    return jsonify({
        "mode":
            "preview only",

        "auto_trade":
            AUTO_TRADE,

        "results":
            previews,
    })


@app.route(
    "/manage",
    methods=["POST", "GET"],
)
def manage_route():
    return jsonify({
        "actions":
            manage_all_positions()
    })


@app.route(
    "/cycle",
    methods=["POST", "GET"],
)
def cycle_route():
    return jsonify(
        run_cycle()
    )


@app.route(
    "/positions",
    methods=["GET"],
)
def positions_route():
    sync_existing_positions()

    with STATE_LOCK:
        state_copy = dict(
            TRADE_STATE
        )

    return jsonify({
        "alpaca_positions":
            get_option_positions(),

        "bot_state":
            state_copy,
    })


# ============================================================
# WEBHOOK ROUTE
# ============================================================


@app.route(
    "/webhook",
    methods=["POST"],
)
def webhook_route():
    if WEBHOOK_SECRET:
        supplied_secret = (
            request.headers.get(
                "X-Webhook-Secret"
            )
            or request.args.get(
                "secret"
            )
        )

        if supplied_secret != WEBHOOK_SECRET:
            return jsonify({
                "error":
                    "unauthorized"
            }), 401

    payload = (
        request.get_json(
            silent=True
        )
        or {}
    )

    symbol = str(
        payload.get(
            "symbol",
            ""
        )
    ).upper().strip()

    trigger = str(
        payload.get(
            "trigger",
            payload.get(
                "side",
                "",
            ),
        )
    ).lower().strip()

    if not symbol:
        return jsonify({
            "error":
                "symbol required"
        }), 400

    if trigger in {
        "call",
        "buy",
        "bullish",
        "long",
    }:
        option_bias = "call"
        direction = "bullish"

    elif trigger in {
        "put",
        "sell",
        "bearish",
        "short",
    }:
        option_bias = "put"
        direction = "bearish"

    else:
        return jsonify({
            "error":
                "trigger must be "
                "call/bullish or "
                "put/bearish"
        }), 400

    snapshots = (
        get_snapshots_batch(
            [symbol]
        )
    )

    snapshot = snapshots.get(
        symbol
    )

    analyzed = analyze_snapshot(
        symbol,
        snapshot,
    )

    if not analyzed:
        return jsonify({
            "entered":
                False,

            "reason":
                "symbol did not pass "
                "basic price/volume filters",
        })

    # Webhook determines CALL/PUT direction.
    # Scanner score/risk rules still apply.
    analyzed[
        "option_bias"
    ] = option_bias

    analyzed[
        "direction"
    ] = direction

    if (
        analyzed["score"]
        < MIN_SCANNER_SCORE
    ):
        return jsonify({
            "entered":
                False,

            "reason":
                "scanner score below "
                "minimum",

            "analysis":
                analyzed,
        })

    result = enter_scanner_trade(
        analyzed
    )

    return jsonify(result)


# ============================================================
# START BACKGROUND MONITOR
# ============================================================

start_bot_thread()


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                10000,
            )
        ),
    )