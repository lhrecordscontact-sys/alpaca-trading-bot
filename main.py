import os
import math
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from flask import Flask, request, jsonify


# ============================================================
# APP
# ============================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

NY = ZoneInfo("America/New_York")


# ============================================================
# CONFIG
# ============================================================

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "").strip()

TRADING_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

AUTO_TRADE = (
    os.getenv("AUTO_TRADE", "false")
    .strip()
    .lower()
    == "true"
)

MIN_WIN_RATE = float(
    os.getenv("MIN_WIN_RATE", "90")
)

POSITION_DOLLARS = float(
    os.getenv("POSITION_DOLLARS", "500")
)

MAX_OPEN_POSITIONS = int(
    os.getenv("MAX_OPEN_POSITIONS", "3")
)

LAST_ENTRY_HOUR = 14
LAST_ENTRY_MINUTE = 45


# ============================================================
# ALPACA HEADERS
# ============================================================

def alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }


# ============================================================
# BASIC API REQUEST
# ============================================================

def alpaca_request(
    method,
    endpoint,
    params=None,
    json=None,
    base_url=None
):
    if base_url is None:
        base_url = TRADING_BASE_URL

    url = f"{base_url}{endpoint}"

    response = requests.request(
        method=method,
        url=url,
        headers=alpaca_headers(),
        params=params,
        json=json,
        timeout=20,
    )

    if not response.ok:
        logging.error(
            "ALPACA ERROR %s %s",
            response.status_code,
            response.text
        )

        raise RuntimeError(
            f"Alpaca {response.status_code}: "
            f"{response.text}"
        )

    if response.text:
        return response.json()

    return {}


# ============================================================
# ACCOUNT CHECK
# ============================================================

def get_account():
    return alpaca_request(
        "GET",
        "/v2/account"
    )


# ============================================================
# POSITIONS
# ============================================================

def get_positions():
    try:
        return alpaca_request(
            "GET",
            "/v2/positions"
        )

    except Exception as exc:
        logging.error(
            "Could not load positions: %s",
            exc
        )

        return []


def has_underlying_position(
    underlying
):
    underlying = underlying.upper()

    positions = get_positions()

    for position in positions:

        symbol = str(
            position.get(
                "symbol",
                ""
            )
        ).upper()

        asset_class = str(
            position.get(
                "asset_class",
                ""
            )
        ).lower()

        # Equity position
        if symbol == underlying:
            return True

        # OCC option symbols normally begin with underlying
        if (
            asset_class == "us_option"
            and symbol.startswith(underlying)
        ):
            return True

    return False


def open_position_count():
    return len(
        get_positions()
    )


# ============================================================
# MARKET HOURS ENTRY FILTER
# ============================================================

def entry_time_allowed():
    now = datetime.now(NY)

    # Monday-Friday
    if now.weekday() >= 5:
        return False

    current_minutes = (
        now.hour * 60
        + now.minute
    )

    market_open = (
        9 * 60
        + 30
    )

    last_entry = (
        LAST_ENTRY_HOUR * 60
        + LAST_ENTRY_MINUTE
    )

    return (
        market_open
        <= current_minutes
        <= last_entry
    )


# ============================================================
# TODAY EXPIRATION
# ============================================================

def today_et():
    return (
        datetime
        .now(NY)
        .date()
        .isoformat()
    )


# ============================================================
# FIND 0DTE OPTION CONTRACTS
# ============================================================

def get_0dte_contracts(
    underlying,
    option_type
):
    option_type = (
        "call"
        if option_type.upper() == "CALL"
        else "put"
    )

    params = {
        "underlying_symbols": underlying,
        "expiration_date": today_et(),
        "type": option_type,
        "status": "active",
        "limit": 1000,
    }

    result = alpaca_request(
        "GET",
        "/v2/options/contracts",
        params=params
    )

    return result.get(
        "option_contracts",
        []
    )


# ============================================================
# FIND ATM CONTRACT
# ============================================================

def choose_atm_contract(
    underlying,
    direction,
    underlying_price
):
    contracts = get_0dte_contracts(
        underlying,
        direction
    )

    if not contracts:
        raise RuntimeError(
            f"No 0DTE {direction} contracts "
            f"found for {underlying}"
        )

    best_contract = None
    best_distance = None

    for contract in contracts:

        try:
            strike = float(
                contract["strike_price"]
            )

        except Exception:
            continue

        distance = abs(
            strike
            - underlying_price
        )

        if (
            best_distance is None
            or distance < best_distance
        ):
            best_distance = distance
            best_contract = contract

    if best_contract is None:
        raise RuntimeError(
            f"Unable to select ATM option "
            f"for {underlying}"
        )

    return best_contract


# ============================================================
# OPTION QUOTE
# ============================================================

def get_option_midpoint(
    option_symbol
):
    params = {
        "symbols": option_symbol
    }

    try:
        result = alpaca_request(
            "GET",
            "/v1beta1/options/quotes/latest",
            params=params,
            base_url=DATA_BASE_URL
        )

        quotes = result.get(
            "quotes",
            {}
        )

        quote = quotes.get(
            option_symbol,
            {}
        )

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

        if bid > 0 and ask > 0:
            return (
                bid + ask
            ) / 2

        if ask > 0:
            return ask

        if bid > 0:
            return bid

    except Exception as exc:
        logging.warning(
            "Quote unavailable for %s: %s",
            option_symbol,
            exc
        )

    return None


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_contract_quantity(
    option_price
):
    if (
        option_price is None
        or option_price <= 0
    ):
        return 1

    contract_cost = (
        option_price
        * 100
    )

    qty = math.floor(
        POSITION_DOLLARS
        / contract_cost
    )

    # At least 1 contract
    return max(
        1,
        qty
    )


# ============================================================
# SUBMIT OPTION ORDER
# ============================================================

def buy_option(
    option_symbol,
    qty
):
    order = {
        "symbol": option_symbol,
        "qty": str(int(qty)),
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
    }

    if not AUTO_TRADE:

        logging.warning(
            "AUTO_TRADE OFF - simulated order: %s",
            order
        )

        return {
            "paper_simulation": True,
            "order": order,
        }

    result = alpaca_request(
        "POST",
        "/v2/orders",
        json=order
    )

    return result


# ============================================================
# VALIDATE TRADINGVIEW MESSAGE
# ============================================================

def validate_signal(data):

    symbol = str(
        data.get(
            "symbol",
            ""
        )
    ).upper().strip()

    signal = str(
        data.get(
            "signal",
            data.get(
                "direction",
                ""
            )
        )
    ).upper().strip()

    try:
        win_rate = float(
            data.get(
                "overall_win_rate",
                0
            )
        )

    except Exception:
        win_rate = 0

    try:
        price = float(
            data.get(
                "price",
                0
            )
        )

    except Exception:
        price = 0

    if not symbol:
        return False, "Missing symbol", None

    if signal not in (
        "CALL",
        "PUT"
    ):
        return (
            False,
            f"Invalid signal: {signal}",
            None
        )

    if win_rate < MIN_WIN_RATE:
        return (
            False,
            (
                f"{symbol} rejected: "
                f"overall win rate "
                f"{win_rate:.1f}% "
                f"is below "
                f"{MIN_WIN_RATE:.1f}%"
            ),
            None
        )

    if price <= 0:
        return (
            False,
            "Invalid underlying price",
            None
        )

    validated = {
        "symbol": symbol,
        "signal": signal,
        "win_rate": win_rate,
        "price": price,
        "call_win_rate": data.get(
            "call_win_rate"
        ),
        "put_win_rate": data.get(
            "put_win_rate"
        ),
        "total_trades": data.get(
            "total_trades"
        ),
    }

    return (
        True,
        "QUALIFIED",
        validated
    )


# ============================================================
# EXECUTE QUALIFIED SIGNAL
# ============================================================

def execute_signal(signal):

    symbol = signal["symbol"]
    direction = signal["signal"]

    logging.info(
        "%s %s QUALIFIED | WIN RATE %.1f%%",
        symbol,
        direction,
        signal["win_rate"]
    )

    # ----------------------------------------
    # ENTRY TIME
    # ----------------------------------------

    if not entry_time_allowed():
        return {
            "accepted": False,
            "reason": "Outside bot entry window"
        }

    # ----------------------------------------
    # MAX POSITIONS
    # ----------------------------------------

    positions = open_position_count()

    if positions >= MAX_OPEN_POSITIONS:
        return {
            "accepted": False,
            "reason": (
                "Maximum open positions "
                "already reached"
            )
        }

    # ----------------------------------------
    # DUPLICATE UNDERLYING
    # ----------------------------------------

    if has_underlying_position(symbol):
        return {
            "accepted": False,
            "reason": (
                f"Existing position already "
                f"found for {symbol}"
            )
        }

    # ----------------------------------------
    # FIND ATM 0DTE
    # ----------------------------------------

    contract = choose_atm_contract(
        underlying=symbol,
        direction=direction,
        underlying_price=signal["price"]
    )

    option_symbol = contract["symbol"]

    strike = float(
        contract["strike_price"]
    )

    # ----------------------------------------
    # OPTION PRICE
    # ----------------------------------------

    midpoint = get_option_midpoint(
        option_symbol
    )

    qty = calculate_contract_quantity(
        midpoint
    )

    # ----------------------------------------
    # SEND ORDER
    # ----------------------------------------

    order = buy_option(
        option_symbol,
        qty
    )

    logging.info(
        "ORDER | %s %s | %s | "
        "strike %.2f | qty %s",
        symbol,
        direction,
        option_symbol,
        strike,
        qty
    )

    return {
        "accepted": True,
        "underlying": symbol,
        "direction": direction,
        "overall_win_rate": signal["win_rate"],
        "option_symbol": option_symbol,
        "expiration": contract.get(
            "expiration_date"
        ),
        "strike": strike,
        "option_price": midpoint,
        "qty": qty,
        "auto_trade": AUTO_TRADE,
        "order": order,
    }


# ============================================================
# TRADINGVIEW WEBHOOK
# ============================================================

@app.route(
    "/webhook",
    methods=["POST"]
)
def tradingview_webhook():

    try:

        data = request.get_json(
            force=True,
            silent=False
        )

        logging.info(
            "WEBHOOK RECEIVED: %s",
            data
        )

        valid, reason, signal = (
            validate_signal(data)
        )

        if not valid:

            logging.warning(
                "SIGNAL REJECTED: %s",
                reason
            )

            return jsonify({
                "ok": True,
                "accepted": False,
                "reason": reason,
            }), 200

        result = execute_signal(
            signal
        )

        return jsonify({
            "ok": True,
            **result
        }), 200

    except Exception as exc:

        logging.exception(
            "WEBHOOK ERROR"
        )

        return jsonify({
            "ok": False,
            "error": str(exc),
        }), 500


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "status": "online",
        "bot": (
            "TradingView 90% "
            "0DTE Alpaca Bot"
        ),
        "paper": True,
        "auto_trade": AUTO_TRADE,
        "minimum_win_rate": MIN_WIN_RATE,
        "position_dollars": POSITION_DOLLARS,
        "max_open_positions": MAX_OPEN_POSITIONS,
        "scanner": "disabled",
        "execution_source": (
            "TradingView webhook only"
        ),
        "webhook": "/webhook",
    })


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    try:
        account = get_account()

        return jsonify({
            "status": "healthy",
            "alpaca_connected": True,
            "account_status": account.get(
                "status"
            ),
            "auto_trade": AUTO_TRADE,
        })

    except Exception as exc:

        return jsonify({
            "status": "error",
            "alpaca_connected": False,
            "error": str(exc),
        }), 500


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "10000"
        )
    )

    logging.info(
        "TradingView 90%% Alpaca bot starting"
    )

    logging.info(
        "AUTO_TRADE=%s",
        AUTO_TRADE
    )

    logging.info(
        "MIN_WIN_RATE=%s",
        MIN_WIN_RATE
    )

    app.run(
        host="0.0.0.0",
        port=port
    )