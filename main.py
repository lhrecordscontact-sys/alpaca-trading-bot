import os
import requests
from flask import Flask, jsonify

app = Flask(__name__)

TRADING_BASE_URL = "https://paper-api.alpaca.markets"


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


def masked(value):
    if not value:
        return "MISSING"

    if len(value) <= 8:
        return f"PRESENT ({len(value)} chars)"

    return (
        f"{value[:4]}..."
        f"{value[-4:]} "
        f"({len(value)} chars)"
    )


def test_alpaca():
    print("=" * 60, flush=True)
    print("ALPACA PAPER CREDENTIAL DIAGNOSTIC", flush=True)
    print("=" * 60, flush=True)

    print(
        f"ALPACA_API_KEY: {masked(ALPACA_API_KEY)}",
        flush=True
    )

    print(
        f"ALPACA_SECRET_KEY: "
        f"PRESENT ({len(ALPACA_SECRET_KEY)} chars)"
        if ALPACA_SECRET_KEY
        else "ALPACA_SECRET_KEY: MISSING",
        flush=True
    )

    if not ALPACA_API_KEY:
        print(
            "FAILED: Render is not loading ALPACA_API_KEY",
            flush=True
        )
        return False

    if not ALPACA_SECRET_KEY:
        print(
            "FAILED: Render is not loading ALPACA_SECRET_KEY",
            flush=True
        )
        return False

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }

    try:
        response = requests.get(
            f"{TRADING_BASE_URL}/v2/account",
            headers=headers,
            timeout=20,
        )

        print(
            f"HTTP STATUS: {response.status_code}",
            flush=True
        )

        if response.status_code == 200:
            account = response.json()

            print(
                "SUCCESS: ALPACA PAPER ACCOUNT CONNECTED",
                flush=True
            )

            print(
                f"Account status: {account.get('status')}",
                flush=True
            )

            print(
                f"Equity: ${account.get('equity')}",
                flush=True
            )

            return True

        print(
            "ALPACA RESPONSE:",
            response.text,
            flush=True
        )

        if response.status_code == 401:
            print(
                "FAILED: Alpaca rejected this exact "
                "API key + secret combination.",
                flush=True
            )

            print(
                "This proves the problem is the credentials "
                "being supplied to Render, not the trading bot.",
                flush=True
            )

        return False

    except Exception as e:
        print(
            f"CONNECTION ERROR: {type(e).__name__}: {e}",
            flush=True
        )

        return False


ALPACA_CONNECTED = test_alpaca()


@app.route("/")
def home():
    return jsonify({
        "service": "Alpaca credential diagnostic",
        "paper_endpoint": TRADING_BASE_URL,
        "api_key_loaded": bool(ALPACA_API_KEY),
        "secret_key_loaded": bool(ALPACA_SECRET_KEY),
        "api_key_length": len(ALPACA_API_KEY),
        "secret_key_length": len(ALPACA_SECRET_KEY),
        "alpaca_connected": ALPACA_CONNECTED,
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "alpaca_connected": ALPACA_CONNECTED,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )