import os
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")

# PAPER TRADING ONLY
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Alpaca Market Data
DATA_BASE_URL = "https://data.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json",
}

# ============================================================
# BOT SETTINGS
# ============================================================

WATCHLIST = [
    "SPY",
    "IWM",
    "QQQ",
    "NVDA",
    "AMD",
    "META",
    "MSFT",
    "AAPL",
    "AMZN",
    "GOOGL",
    "TSLA",
    "AVGO",
    "ARM",
]

MIN_PRICE = 5.00
MIN_DAILY_VOLUME = 1_000_000
MIN_SCANNER_SCORE = 70

MAX_RISK_PER_TRADE = 60.00
MAX_DAILY_LOSS = 180.00
MAX_OPEN_POSITIONS = 3

# Keep False until paper testing is complete
AUTO_TRADE = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def alpaca_get(path, params=None):
    response = requests.get(
        f"{ALPACA_BASE_URL}{path}",
        headers=HEADERS,
        params=params,
        timeout=15,
    )

    try:
        data = response.json()
    except Exception:
        data = {"raw": response.text}

    return response.status_code, data


def alpaca_post(path, payload):
    response = requests.post(
        f"{ALPACA_BASE_URL}{path}",
        headers=HEADERS,
        json=payload,
        timeout=15,
    )

    try:
        data = response.json()
    except Exception:
        data = {"raw": response.text}

    return response.status_code, data


def market_data_get(path, params=None):
    response = requests.get(
        f"{DATA_BASE_URL}{path}",
        headers=HEADERS,
        params=params,
        timeout=15,
    )

    response.raise_for_status()
    return response.json()


def get_account():
    status, data = alpaca_get("/v2/account")

    if status >= 400:
        return None

    return data


def get_positions():
    status, data = alpaca_get("/v2/positions")

    if status >= 400:
        return []

    return data


def get_snapshot(symbol):
    try:
        data = market_data_get(
            f"/v2/stocks/{symbol}/snapshot",
            params={"feed": "iex"},
        )
        return data

    except Exception as e:
        print(f"Snapshot error for {symbol}: {e}")
        return None


# ============================================================
# SCANNER
# ============================================================

def analyze_symbol(symbol):
    """
    Scores a stock from 0-100.

    This does NOT claim a 90% win rate.
    The score determines whether the stock currently matches
    the scanner conditions.
    """

    snapshot = get_snapshot(symbol)

    if not snapshot:
        return None

    latest_trade = snapshot.get("latestTrade") or {}
    minute_bar = snapshot.get("minuteBar") or {}
    daily_bar = snapshot.get("dailyBar") or {}
    previous_bar = snapshot.get("prevDailyBar") or {}

    price = latest_trade.get("p") or minute_bar.get("c")

    volume = daily_bar.get("v", 0)
    current_open = daily_bar.get("o")
    current_high = daily_bar.get("h")
    current_low = daily_bar.get("l")

    previous_close = previous_bar.get("c")
    previous_volume = previous_bar.get("v", 0)

    if not price:
        return None

    score = 0
    reasons = []

    # PRICE FILTER
    if price >= MIN_PRICE:
        score += 10
        reasons.append("price filter passed")

    # LIQUIDITY FILTER
    if volume >= MIN_DAILY_VOLUME:
        score += 20
        reasons.append("high liquidity")

    # RELATIVE VOLUME
    relative_volume = 0

    if previous_volume and previous_volume > 0:
        relative_volume = volume / previous_volume

    if relative_volume >= 1.0:
        score += 20
        reasons.append("strong relative volume")

    elif relative_volume >= 0.50:
        score += 10
        reasons.append("moderate relative volume")

    # DAILY MOMENTUM
    percent_change = 0

    if previous_close and previous_close > 0:
        percent_change = (
            (price - previous_close) / previous_close
        ) * 100

    if abs(percent_change) >= 2:
        score += 20
        reasons.append("strong price movement")

    elif abs(percent_change) >= 1:
        score += 10
        reasons.append("moderate price movement")

    # CURRENT SESSION DIRECTION
    direction = "neutral"

    if current_open:
        if price > current_open:
            direction = "bullish"
            score += 10
            reasons.append("trading above daily open")

        elif price < current_open:
            direction = "bearish"
            score += 10
            reasons.append("trading below daily open")

    # LOCATION WITHIN DAILY RANGE
    range_position = None

    if (
        current_high is not None
        and current_low is not None
        and current_high > current_low
    ):
        range_position = (
            (price - current_low)
            / (current_high - current_low)
        )

        if range_position >= 0.75:
            score += 20
            reasons.append("near session highs")

        elif range_position <= 0.25:
            score += 20
            reasons.append("near session lows")

    else:
        score += 5

    passed = score >= MIN_SCANNER_SCORE

    return {
        "symbol": symbol,
        "price": round(float(price), 2),
        "score": score,
        "passed": passed,
        "direction": direction,
        "percent_change": round(percent_change, 2),
        "relative_volume": round(relative_volume, 2),
        "daily_volume": volume,
        "range_position": (
            round(range_position, 2)
            if range_position is not None
            else None
        ),
        "reasons": reasons,
    }


def scan_watchlist():
    results = []

    for symbol in WATCHLIST:
        result = analyze_symbol(symbol)

        if result:
            results.append(result)

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return results


# ============================================================
# SAFETY CHECKS
# ============================================================

def risk_checks(symbol):
    account = get_account()

    if not account:
        return False, "Could not retrieve Alpaca account."

    if account.get("trading_blocked"):
        return False, "Alpaca account is trading blocked."

    positions = get_positions()

    if len(positions) >= MAX_OPEN_POSITIONS:
        return False, "Maximum open positions reached."

    for position in positions:
        if position.get("symbol") == symbol:
            return False, f"Already have an open {symbol} position."

    return True, "Risk checks passed."


# ============================================================
# ORDER FUNCTION
# ============================================================

def submit_stock_order(symbol, side, qty):
    """
    PAPER STOCK ORDER ONLY.
    Options automation can be added separately after paper testing.
    """

    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }

    status, result = alpaca_post(
        "/v2/orders",
        payload,
    )

    return status, result


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "bot": "Purgatory AI Scanner",
        "broker": "Alpaca",
        "mode": "PAPER",
        "auto_trade": AUTO_TRADE,
        "scanner_score_required": MIN_SCANNER_SCORE,
        "max_risk_per_trade": MAX_RISK_PER_TRADE,
        "max_daily_loss": MAX_DAILY_LOSS,
        "watchlist_size": len(WATCHLIST),
    })


# ============================================================
# ACCOUNT
# ============================================================

@app.route("/account", methods=["GET"])
def account():
    data = get_account()

    if not data:
        return jsonify({
            "success": False,
            "error": "Unable to retrieve Alpaca account.",
        }), 500

    return jsonify({
        "success": True,
        "equity": data.get("equity"),
        "cash": data.get("cash"),
        "buying_power": data.get("buying_power"),
        "portfolio_value": data.get("portfolio_value"),
        "paper_mode": True,
    })


# ============================================================
# POSITIONS
# ============================================================

@app.route("/positions", methods=["GET"])
def positions():
    return jsonify({
        "success": True,
        "positions": get_positions(),
    })


# ============================================================
# SCANNER ENDPOINT
# ============================================================

@app.route("/scan", methods=["GET"])
def scan():
    try:
        results = scan_watchlist()

        qualified = [
            stock
            for stock in results
            if stock["passed"]
        ]

        return jsonify({
            "success": True,
            "time": datetime.now(
                timezone.utc
            ).isoformat(),
            "stocks_scanned": len(results),
            "qualified_count": len(qualified),
            "qualified": qualified,
            "all_results": results,
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# ============================================================
# SINGLE STOCK ANALYSIS
# ============================================================

@app.route("/analyze/<symbol>", methods=["GET"])
def analyze(symbol):
    symbol = symbol.upper().strip()

    result = analyze_symbol(symbol)

    if not result:
        return jsonify({
            "success": False,
            "error": f"Unable to analyze {symbol}",
        }), 400

    return jsonify({
        "success": True,
        "analysis": result,
    })


# ============================================================
# TRADINGVIEW WEBHOOK
# ============================================================

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(
            force=True,
            silent=False,
        )

        # SECURITY
        if WEBHOOK_SECRET:
            if data.get("secret") != WEBHOOK_SECRET:
                return jsonify({
                    "success": False,
                    "error": "Invalid webhook secret.",
                }), 401

        symbol = str(
            data.get("symbol", "")
        ).upper().strip()

        side = str(
            data.get("side", "")
        ).lower().strip()

        signal = str(
            data.get("signal", "")
        ).upper().strip()

        try:
            qty = int(data.get("qty", 1))
        except (TypeError, ValueError):
            return jsonify({
                "success": False,
                "error": "qty must be a whole number.",
            }), 400

        if not symbol:
            return jsonify({
                "success": False,
                "error": "Missing symbol.",
            }), 400

        if side not in ["buy", "sell"]:
            return jsonify({
                "success": False,
                "error": "Side must be buy or sell.",
            }), 400

        if qty <= 0:
            return jsonify({
                "success": False,
                "error": "qty must be greater than 0.",
            }), 400

        # RUN SCANNER BEFORE ENTRY
        scan_result = analyze_symbol(symbol)

        if not scan_result:
            return jsonify({
                "success": False,
                "trade_allowed": False,
                "error": "Scanner could not analyze symbol.",
            }), 400

        # STOCK MUST PASS SCANNER
        if not scan_result["passed"]:
            return jsonify({
                "success": True,
                "trade_allowed": False,
                "message": f"{symbol} signal rejected by scanner.",
                "scanner": scan_result,
            })

        # DIRECTION CHECK
        expected_direction = (
            "bullish"
            if side == "buy"
            else "bearish"
        )

        if scan_result["direction"] != expected_direction:
            return jsonify({
                "success": True,
                "trade_allowed": False,
                "message": (
                    f"{symbol} rejected because scanner "
                    f"direction is {scan_result['direction']}."
                ),
                "scanner": scan_result,
            })

        # ACCOUNT RISK CHECK
        allowed, reason = risk_checks(symbol)

        if not allowed:
            return jsonify({
                "success": True,
                "trade_allowed": False,
                "message": reason,
                "scanner": scan_result,
            })

        # TEST MODE
        if not AUTO_TRADE:
            return jsonify({
                "success": True,
                "trade_allowed": True,
                "order_sent": False,
                "mode": "PAPER TEST",
                "message": (
                    f"{symbol} passed scanner and risk checks. "
                    f"Order NOT sent because AUTO_TRADE is disabled."
                ),
                "signal": signal,
                "side": side,
                "qty": qty,
                "scanner": scan_result,
            })

        # PAPER ORDER
        status, order = submit_stock_order(
            symbol,
            side,
            qty,
        )

        if status >= 400:
            return jsonify({
                "success": False,
                "trade_allowed": True,
                "order_sent": False,
                "alpaca_error": order,
            }), status

        return jsonify({
            "success": True,
            "trade_allowed": True,
            "order_sent": True,
            "mode": "PAPER",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "scanner": scan_result,
            "order": order,
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# ============================================================
# START SERVER
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