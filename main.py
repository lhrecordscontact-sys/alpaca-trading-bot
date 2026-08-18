import os
from datetime import datetime, timezone, timedelta

import requests
from flask import Flask, request, jsonify


app = Flask(__name__)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")


# ============================================================
# ALPACA URLS
# ============================================================

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"


HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json",
}


# ============================================================
# SCANNER SETTINGS
# ============================================================

MIN_PRICE = 5.00
MIN_DAILY_VOLUME = 1_000_000
MIN_SCANNER_SCORE = 70

SNAPSHOT_BATCH_SIZE = 100

DEFAULT_RETURN_LIMIT = 50
MAX_RETURN_LIMIT = 200


# ============================================================
# OPTIONS SETTINGS
# ============================================================

# Minimum days until expiration
MIN_DTE = 0

# Maximum days until expiration
MAX_DTE = 14

# Search strikes this percentage above/below stock price
STRIKE_SEARCH_PERCENT = 0.08

# Only automatically pick contracts that are tradable
REQUIRE_TRADABLE_OPTION = True

# Prefer contracts with open interest
MIN_OPEN_INTEREST = 1

# Number of option contracts to paper trade
DEFAULT_OPTION_QTY = 1


# ============================================================
# RISK SETTINGS
# ============================================================

MAX_RISK_PER_TRADE = 60.00
MAX_DAILY_LOSS = 180.00
MAX_OPEN_POSITIONS = 3


# ============================================================
# IMPORTANT
# ============================================================

# KEEP FALSE WHILE TESTING.
# The bot will SELECT the option but NOT place an order.
AUTO_TRADE = False


# ============================================================
# ALLOWED EXCHANGES
# ============================================================

ALLOWED_EXCHANGES = {
    "NASDAQ",
    "NYSE",
    "AMEX",
    "ARCA",
    "NYSEARCA",
    "BATS",
}


# ============================================================
# API HELPERS
# ============================================================

def alpaca_get(path, params=None):

    response = requests.get(
        f"{ALPACA_BASE_URL}{path}",
        headers=HEADERS,
        params=params,
        timeout=30,
    )

    try:
        data = response.json()
    except Exception:
        data = {
            "raw": response.text
        }

    return response.status_code, data


def alpaca_post(path, payload):

    response = requests.post(
        f"{ALPACA_BASE_URL}{path}",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )

    try:
        data = response.json()
    except Exception:
        data = {
            "raw": response.text
        }

    return response.status_code, data


def market_data_get(path, params=None):

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

    status, data = alpaca_get(
        "/v2/account"
    )

    if status >= 400:
        return None

    return data


def get_positions():

    status, data = alpaca_get(
        "/v2/positions"
    )

    if status >= 400:
        return []

    return data


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


        if not symbol:
            continue


        symbols.append(symbol)


    return sorted(
        set(symbols)
    )


# ============================================================
# BATCH HELPER
# ============================================================

def chunk_list(items, size):

    for i in range(
        0,
        len(items),
        size,
    ):

        yield items[
            i:i + size
        ]


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

            return (
                data.get("snapshots")
                or {}
            )


        return data


    except Exception as e:

        print(
            f"Snapshot batch error: {e}",
            flush=True,
        )

        return {}


def get_snapshot(symbol):

    try:

        return market_data_get(
            f"/v2/stocks/{symbol}/snapshot",
            params={
                "feed": "iex"
            },
        )


    except Exception as e:

        print(
            f"Snapshot error for {symbol}: {e}",
            flush=True,
        )

        return None


# ============================================================
# STOCK ANALYSIS
# ============================================================

def analyze_snapshot(
    symbol,
    snapshot,
):

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


    if price is None:
        return None


    try:
        price = float(price)
    except Exception:
        return None


    volume = (
        daily_bar.get("v", 0)
        or 0
    )


    previous_volume = (
        previous_bar.get("v", 0)
        or 0
    )


    try:
        volume = float(volume)
    except Exception:
        volume = 0


    try:
        previous_volume = float(
            previous_volume
        )
    except Exception:
        previous_volume = 0


    # Reject low price stocks
    if price < MIN_PRICE:
        return None


    # Reject low-volume stocks
    if volume < MIN_DAILY_VOLUME:
        return None


    current_open = daily_bar.get("o")
    current_high = daily_bar.get("h")
    current_low = daily_bar.get("l")

    previous_close = previous_bar.get("c")


    score = 0
    reasons = []


    # ========================================================
    # PRICE
    # ========================================================

    score += 10
    reasons.append(
        "price filter passed"
    )


    # ========================================================
    # VOLUME
    # ========================================================

    score += 20
    reasons.append(
        "high liquidity"
    )


    # ========================================================
    # RELATIVE VOLUME
    # ========================================================

    relative_volume = 0.0


    if previous_volume > 0:

        relative_volume = (
            volume
            / previous_volume
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


    # ========================================================
    # PRICE MOVEMENT
    # ========================================================

    percent_change = 0.0


    try:

        if previous_close is not None:

            previous_close = float(
                previous_close
            )


            if previous_close > 0:

                percent_change = (
                    (
                        price
                        - previous_close
                    )
                    / previous_close
                ) * 100


    except Exception:

        percent_change = 0.0


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


    # ========================================================
    # DIRECTION
    # ========================================================

    direction = "neutral"


    try:

        if current_open is not None:

            current_open = float(
                current_open
            )


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


    except Exception:

        pass


    # ========================================================
    # DAILY RANGE LOCATION
    # ========================================================

    range_position = None


    try:

        if (
            current_high is not None
            and current_low is not None
        ):

            current_high = float(
                current_high
            )

            current_low = float(
                current_low
            )


            if current_high > current_low:

                range_position = (
                    (
                        price
                        - current_low
                    )
                    /
                    (
                        current_high
                        - current_low
                    )
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


    except Exception:

        range_position = None


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

        "price": round(
            price,
            2,
        ),

        "score": int(
            score
        ),

        "passed": passed,

        "direction": direction,

        "option_bias": option_bias,

        "percent_change": round(
            percent_change,
            2,
        ),

        "relative_volume": round(
            relative_volume,
            2,
        ),

        "daily_volume": int(
            volume
        ),

        "range_position": (
            round(
                range_position,
                2,
            )
            if range_position is not None
            else None
        ),

        "reasons": reasons,
    }


def analyze_symbol(symbol):

    symbol = (
        symbol
        .upper()
        .strip()
    )


    snapshot = get_snapshot(
        symbol
    )


    if not snapshot:
        return None


    return analyze_snapshot(
        symbol,
        snapshot,
    )


# ============================================================
# FULL OPTIONABLE MARKET SCANNER
# ============================================================

def scan_market():

    symbols = (
        get_optionable_stock_universe()
    )


    results = []

    batches_processed = 0
    batch_errors = 0


    for batch in chunk_list(
        symbols,
        SNAPSHOT_BATCH_SIZE,
    ):

        snapshots = (
            get_snapshots_batch(
                batch
            )
        )


        batches_processed += 1


        if not snapshots:

            batch_errors += 1
            continue


        for symbol in batch:

            snapshot = snapshots.get(
                symbol
            )


            if not snapshot:
                continue


            result = analyze_snapshot(
                symbol,
                snapshot,
            )


            if result:
                results.append(result)


    results.sort(

        key=lambda x: (

            x["score"],

            abs(
                x["percent_change"]
            ),

            x["relative_volume"],

            x["daily_volume"],
        ),

        reverse=True,
    )


    return {

        "symbols_in_universe": len(
            symbols
        ),

        "stocks_with_data": len(
            results
        ),

        "batches_processed": (
            batches_processed
        ),

        "batch_errors": (
            batch_errors
        ),

        "results": results,
    }


# ============================================================
# OPTION CONTRACT API
# ============================================================

def get_option_contracts(
    underlying,
    option_type,
    stock_price,
):

    today = datetime.now(
        timezone.utc
    ).date()


    minimum_expiration = (
        today
        + timedelta(
            days=MIN_DTE
        )
    )


    maximum_expiration = (
        today
        + timedelta(
            days=MAX_DTE
        )
    )


    low_strike = (
        stock_price
        * (
            1
            - STRIKE_SEARCH_PERCENT
        )
    )


    high_strike = (
        stock_price
        * (
            1
            + STRIKE_SEARCH_PERCENT
        )
    )


    params = {

        "underlying_symbols": (
            underlying
        ),

        "status": "active",

        "type": option_type,

        "expiration_date_gte": (
            minimum_expiration.isoformat()
        ),

        "expiration_date_lte": (
            maximum_expiration.isoformat()
        ),

        "strike_price_gte": round(
            low_strike,
            2,
        ),

        "strike_price_lte": round(
            high_strike,
            2,
        ),

        "limit": 1000,
    }


    status, data = alpaca_get(
        "/v2/options/contracts",
        params=params,
    )


    if status >= 400:

        print(
            f"Option contract error "
            f"for {underlying}: {data}",
            flush=True,
        )

        return []


    if isinstance(data, dict):

        return (
            data.get(
                "option_contracts"
            )
            or data.get(
                "contracts"
            )
            or []
        )


    if isinstance(data, list):
        return data


    return []


# ============================================================
# OPTION CHAIN DATA
# ============================================================

def get_option_chain(
    underlying,
    option_type,
    stock_price,
):

    low_strike = (
        stock_price
        * (
            1
            - STRIKE_SEARCH_PERCENT
        )
    )


    high_strike = (
        stock_price
        * (
            1
            + STRIKE_SEARCH_PERCENT
        )
    )


    try:

        data = market_data_get(

            (
                "/v1beta1/options/"
                f"snapshots/{underlying}"
            ),

            params={

                "feed": "indicative",

                "type": option_type,

                "strike_price_gte": round(
                    low_strike,
                    2,
                ),

                "strike_price_lte": round(
                    high_strike,
                    2,
                ),

                "limit": 1000,
            },
        )


        if isinstance(data, dict):

            return (
                data.get("snapshots")
                or {}
            )


        return {}


    except Exception as e:

        print(
            f"Option chain error "
            f"for {underlying}: {e}",
            flush=True,
        )

        return {}


# ============================================================
# OPTION QUOTE HELPERS
# ============================================================

def extract_option_quote(snapshot):

    if not snapshot:

        return {
            "bid": None,
            "ask": None,
            "mid": None,
            "spread": None,
        }


    quote = (
        snapshot.get("latestQuote")
        or snapshot.get("latest_quote")
        or {}
    )


    bid = (
        quote.get("bp")
        or quote.get("bid_price")
    )


    ask = (
        quote.get("ap")
        or quote.get("ask_price")
    )


    try:

        bid = float(
            bid
        ) if bid is not None else None


    except Exception:

        bid = None


    try:

        ask = float(
            ask
        ) if ask is not None else None


    except Exception:

        ask = None


    mid = None
    spread = None


    if (
        bid is not None
        and ask is not None
        and ask >= bid
    ):

        mid = (
            bid + ask
        ) / 2


        spread = (
            ask - bid
        )


    return {

        "bid": (
            round(bid, 2)
            if bid is not None
            else None
        ),

        "ask": (
            round(ask, 2)
            if ask is not None
            else None
        ),

        "mid": (
            round(mid, 2)
            if mid is not None
            else None
        ),

        "spread": (
            round(spread, 2)
            if spread is not None
            else None
        ),
    }


# ============================================================
# OPTION SELECTOR
# ============================================================

def choose_option_contract(
    underlying,
    option_type,
    stock_price,
):

    contracts = get_option_contracts(
        underlying,
        option_type,
        stock_price,
    )


    if not contracts:

        return None


    chain = get_option_chain(
        underlying,
        option_type,
        stock_price,
    )


    candidates = []


    for contract in contracts:

        symbol = (
            contract.get("symbol")
            or ""
        )


        if not symbol:
            continue


        if (
            REQUIRE_TRADABLE_OPTION
            and not contract.get(
                "tradable",
                False,
            )
        ):

            continue


        try:

            strike = float(
                contract.get(
                    "strike_price"
                )
            )

        except Exception:

            continue


        expiration = contract.get(
            "expiration_date"
        )


        try:

            expiration_date = (
                datetime.strptime(
                    expiration,
                    "%Y-%m-%d",
                ).date()
            )

        except Exception:

            continue


        today = datetime.now(
            timezone.utc
        ).date()


        dte = (
            expiration_date
            - today
        ).days


        open_interest = (
            contract.get(
                "open_interest"
            )
            or 0
        )


        try:

            open_interest = int(
                float(
                    open_interest
                )
            )

        except Exception:

            open_interest = 0


        market_snapshot = (
            chain.get(symbol)
            or {}
        )


        quote = extract_option_quote(
            market_snapshot
        )


        distance_from_atm = abs(
            strike - stock_price
        )


        # Prefer:
        # 1. closest expiration
        # 2. closest strike to ATM
        # 3. higher open interest
        ranking = (

            dte,

            distance_from_atm,

            -open_interest,
        )


        candidates.append({

            "symbol": symbol,

            "underlying": underlying,

            "type": option_type,

            "strike": round(
                strike,
                2,
            ),

            "expiration": expiration,

            "dte": dte,

            "open_interest": (
                open_interest
            ),

            "bid": quote["bid"],

            "ask": quote["ask"],

            "mid": quote["mid"],

            "spread": quote["spread"],

            "tradable": contract.get(
                "tradable",
                False,
            ),

            "_ranking": ranking,
        })


    if not candidates:
        return None


    # Prefer contracts with some open interest
    liquid_candidates = [

        contract

        for contract in candidates

        if (
            contract[
                "open_interest"
            ]
            >= MIN_OPEN_INTEREST
        )
    ]


    if liquid_candidates:

        candidates = (
            liquid_candidates
        )


    candidates.sort(
        key=lambda x: x["_ranking"]
    )


    selected = candidates[0]


    selected.pop(
        "_ranking",
        None,
    )


    return selected


# ============================================================
# OPTION ORDER
# ============================================================

def submit_option_order(
    option_symbol,
    qty=1,
):

    payload = {

        "symbol": option_symbol,

        "qty": str(
            int(qty)
        ),

        "side": "buy",

        "type": "market",

        "time_in_force": "day",
    }


    return alpaca_post(
        "/v2/orders",
        payload,
    )


# ============================================================
# RISK CHECKS
# ============================================================

def risk_checks():

    account = get_account()


    if not account:

        return (
            False,
            "Could not retrieve Alpaca account."
        )


    if account.get(
        "trading_blocked"
    ):

        return (
            False,
            "Alpaca account is trading blocked."
        )


    positions = get_positions()


    if len(
        positions
    ) >= MAX_OPEN_POSITIONS:

        return (
            False,
            "Maximum open positions reached."
        )


    return (
        True,
        "Risk checks passed."
    )


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"],
)
def home():

    return jsonify({

        "status": "online",

        "bot": (
            "Purgatory AI "
            "Options Scanner"
        ),

        "broker": "Alpaca",

        "mode": "PAPER",

        "scanner": (
            "FULL OPTIONABLE "
            "US EQUITY MARKET"
        ),

        "auto_trade": (
            AUTO_TRADE
        ),

        "option_logic": (
            "bullish = call, "
            "bearish = put"
        ),

        "minimum_score": (
            MIN_SCANNER_SCORE
        ),

        "max_option_dte": (
            MAX_DTE
        ),
    })


# ============================================================
# ACCOUNT
# ============================================================

@app.route(
    "/account",
    methods=["GET"],
)
def account():

    data = get_account()


    if not data:

        return jsonify({

            "success": False,

            "error": (
                "Unable to retrieve "
                "Alpaca account."
            ),

        }), 500


    return jsonify({

        "success": True,

        "equity": (
            data.get("equity")
        ),

        "cash": (
            data.get("cash")
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

        "options_trading_level": (
            data.get(
                "options_trading_level"
            )
        ),

        "options_approved_level": (
            data.get(
                "options_approved_level"
            )
        ),

        "paper_mode": True,
    })


# ============================================================
# POSITIONS
# ============================================================

@app.route(
    "/positions",
    methods=["GET"],
)
def positions():

    return jsonify({

        "success": True,

        "positions": (
            get_positions()
        ),
    })


# ============================================================
# UNIVERSE
# ============================================================

@app.route(
    "/universe",
    methods=["GET"],
)
def universe():

    try:

        symbols = (
            get_optionable_stock_universe()
        )


        return jsonify({

            "success": True,

            "type": (
                "optionable stocks"
            ),

            "count": len(
                symbols
            ),

            "symbols": symbols,
        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e),

        }), 500


# ============================================================
# SCAN
# ============================================================

@app.route(
    "/scan",
    methods=["GET"],
)
def scan():

    try:

        requested_limit = (
            request.args.get(
                "limit",
                DEFAULT_RETURN_LIMIT,
            )
        )


        try:

            requested_limit = int(
                requested_limit
            )


        except Exception:

            requested_limit = (
                DEFAULT_RETURN_LIMIT
            )


        requested_limit = max(

            1,

            min(
                requested_limit,
                MAX_RETURN_LIMIT,
            ),
        )


        market_scan = (
            scan_market()
        )


        results = market_scan[
            "results"
        ]


        qualified = [

            stock

            for stock in results

            if (
                stock["passed"]
                and stock[
                    "option_bias"
                ]
            )
        ]


        return jsonify({

            "success": True,

            "time": datetime.now(
                timezone.utc
            ).isoformat(),

            "scanner": (
                "FULL OPTIONABLE "
                "US EQUITY MARKET"
            ),

            "market_universe_size": (
                market_scan[
                    "symbols_in_universe"
                ]
            ),

            "stocks_scanned": (
                market_scan[
                    "stocks_with_data"
                ]
            ),

            "qualified_count": len(
                qualified
            ),

            "batches_processed": (
                market_scan[
                    "batches_processed"
                ]
            ),

            "batch_errors": (
                market_scan[
                    "batch_errors"
                ]
            ),

            "qualified": qualified[
                :requested_limit
            ],
        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e),

        }), 500


# ============================================================
# SINGLE STOCK ANALYSIS
# ============================================================

@app.route(
    "/analyze/<symbol>",
    methods=["GET"],
)
def analyze(symbol):

    result = analyze_symbol(
        symbol
    )


    if not result:

        return jsonify({

            "success": False,

            "error": (
                f"Unable to analyze "
                f"{symbol.upper()}"
            ),

        }), 400


    return jsonify({

        "success": True,

        "analysis": result,
    })


# ============================================================
# FIND OPTION
# ============================================================

@app.route(
    "/option/<symbol>",
    methods=["GET"],
)
def option(symbol):

    symbol = (
        symbol
        .upper()
        .strip()
    )


    analysis = analyze_symbol(
        symbol
    )


    if not analysis:

        return jsonify({

            "success": False,

            "error": (
                f"Unable to analyze "
                f"{symbol}"
            ),

        }), 400


    if not analysis[
        "passed"
    ]:

        return jsonify({

            "success": True,

            "option_selected": False,

            "message": (
                f"{symbol} did not "
                f"pass the scanner."
            ),

            "analysis": analysis,
        })


    option_type = analysis[
        "option_bias"
    ]


    if not option_type:

        return jsonify({

            "success": True,

            "option_selected": False,

            "message": (
                f"{symbol} has no "
                f"bullish or bearish "
                f"direction yet."
            ),

            "analysis": analysis,
        })


    selected = (
        choose_option_contract(

            symbol,

            option_type,

            analysis[
                "price"
            ],
        )
    )


    if not selected:

        return jsonify({

            "success": True,

            "option_selected": False,

            "message": (
                "No suitable option "
                "contract found."
            ),

            "analysis": analysis,
        })


    return jsonify({

        "success": True,

        "option_selected": True,

        "analysis": analysis,

        "selected_option": (
            selected
        ),

        "order_sent": False,

        "auto_trade": (
            AUTO_TRADE
        ),
    })


# ============================================================
# TRADINGVIEW WEBHOOK
# ============================================================

@app.route(
    "/webhook",
    methods=["POST"],
)
def webhook():

    try:

        data = request.get_json(
            force=True,
            silent=False,
        )


        # ====================================================
        # SECURITY
        # ====================================================

        if WEBHOOK_SECRET:

            if (
                data.get("secret")
                != WEBHOOK_SECRET
            ):

                return jsonify({

                    "success": False,

                    "error": (
                        "Invalid webhook secret."
                    ),

                }), 401


        # ====================================================
        # SYMBOL
        # ====================================================

        symbol = str(

            data.get(
                "symbol",
                ""
            )

        ).upper().strip()


        if not symbol:

            return jsonify({

                "success": False,

                "error": (
                    "Missing symbol."
                ),

            }), 400


        # ====================================================
        # ANALYZE STOCK
        # ====================================================

        analysis = (
            analyze_symbol(
                symbol
            )
        )


        if not analysis:

            return jsonify({

                "success": False,

                "trade_allowed": False,

                "error": (
                    "Scanner could not "
                    "analyze symbol."
                ),

            }), 400


        # ====================================================
        # SCANNER MUST PASS
        # ====================================================

        if not analysis[
            "passed"
        ]:

            return jsonify({

                "success": True,

                "trade_allowed": False,

                "message": (
                    f"{symbol} rejected "
                    f"by scanner."
                ),

                "scanner": analysis,
            })


        # ====================================================
        # DETERMINE CALL OR PUT
        # ====================================================

        option_type = (
            analysis[
                "option_bias"
            ]
        )


        if not option_type:

            return jsonify({

                "success": True,

                "trade_allowed": False,

                "message": (
                    f"{symbol} direction "
                    f"is neutral."
                ),

                "scanner": analysis,
            })


        # ====================================================
        # SELECT OPTION CONTRACT
        # ====================================================

        selected_option = (
            choose_option_contract(

                symbol,

                option_type,

                analysis[
                    "price"
                ],
            )
        )


        if not selected_option:

            return jsonify({

                "success": True,

                "trade_allowed": False,

                "message": (
                    f"No suitable "
                    f"{option_type.upper()} "
                    f"contract found "
                    f"for {symbol}."
                ),

                "scanner": analysis,
            })


        # ====================================================
        # RISK CHECKS
        # ====================================================

        allowed, reason = (
            risk_checks()
        )


        if not allowed:

            return jsonify({

                "success": True,

                "trade_allowed": False,

                "message": reason,

                "scanner": analysis,

                "selected_option": (
                    selected_option
                ),
            })


        # ====================================================
        # TEST MODE
        # ====================================================

        if not AUTO_TRADE:

            return jsonify({

                "success": True,

                "trade_allowed": True,

                "order_sent": False,

                "mode": (
                    "PAPER TEST"
                ),

                "message": (
                    f"{symbol} passed. "
                    f"{option_type.upper()} "
                    f"contract selected. "
                    f"Order NOT sent because "
                    f"AUTO_TRADE is disabled."
                ),

                "scanner": analysis,

                "selected_option": (
                    selected_option
                ),
            })


        # ====================================================
        # PAPER OPTION ORDER
        # ====================================================

        status, order = (
            submit_option_order(

                selected_option[
                    "symbol"
                ],

                DEFAULT_OPTION_QTY,
            )
        )


        if status >= 400:

            return jsonify({

                "success": False,

                "trade_allowed": True,

                "order_sent": False,

                "selected_option": (
                    selected_option
                ),

                "alpaca_error": (
                    order
                ),

            }), status


        return jsonify({

            "success": True,

            "trade_allowed": True,

            "order_sent": True,

            "mode": "PAPER",

            "underlying": symbol,

            "option_type": (
                option_type
            ),

            "selected_option": (
                selected_option
            ),

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