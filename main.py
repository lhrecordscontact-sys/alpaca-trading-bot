import os
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json"
}


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "Alpaca Trading Bot is running"
    })


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)

        # Check webhook password
        if WEBHOOK_SECRET:
            if data.get("secret") != WEBHOOK_SECRET:
                return jsonify({"error": "Invalid webhook secret"}), 401

        symbol = str(data.get("symbol", "")).upper()
        side = str(data.get("side", "")).lower()

        # Default to 1 share unless TradingView sends another quantity
        qty = str(data.get("qty", 1))

        if not symbol:
            return jsonify({"error": "Missing symbol"}), 400

        if side not in ["buy", "sell"]:
            return jsonify({
                "error": "Side must be buy or sell"
            }), 400

        order = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": "market",
            "time_in_force": "day"
        }

        response = requests.post(
            f"{ALPACA_BASE_URL}/v2/orders",
            headers=HEADERS,
            json=order,
            timeout=15
        )

        result = response.json()

        if response.status_code >= 400:
            return jsonify({
                "success": False,
                "alpaca_error": result
            }), response.status_code

        return jsonify({
            "success": True,
            "message": f"{side.upper()} order submitted for {symbol}",
            "order": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/positions", methods=["GET"])
def positions():
    try:
        response = requests.get(
            f"{ALPACA_BASE_URL}/v2/positions",
            headers=HEADERS,
            timeout=15
        )

        return jsonify(response.json()), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
