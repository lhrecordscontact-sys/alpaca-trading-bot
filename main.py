import os
import time
import math
import threading
import logging

from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo

import requests
import pandas as pd

from flask import Flask, jsonify
from ai_confirmation import ask_ai


# ============================================================
# APP
# ============================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


# ============================================================
# ALPACA PAPER TRADING
# ============================================================

TRADING_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"

ALPACA_API_KEY = os.getenv(
    "ALPACA_API_KEY",
    ""
).strip()

ALPACA_SECRET_KEY = os.getenv(
    "ALPACA_SECRET_KEY",
    ""
).strip()

DATA_FEED = os.getenv(
    "DATA_FEED",
    "iex"
).strip().lower()

OPTION_FEED = os.getenv(
    "OPTION_FEED",
    "opra"
).strip().lower()

SCANNER_URL = os.getenv(
    "SCANNER_URL",
    "https://nine0-percent-scanner.onrender.com/api/watchlist"
).strip()

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json",
}


# ============================================================
# BOT SETTINGS
# ============================================================

AUTO_TRADE = (
    os.getenv(
        "AUTO_TRADE",
        "false"
    ).strip().lower()
    == "true"
)

LOOP_SECONDS = int(
    os.getenv(
        "LOOP_SECONDS",
        "20"
    )
)

MIN_SETUP_SCORE = float(
    os.getenv(
        "BOT_MIN_SETUP_SCORE",
        "82"
    )
)

MAX_OPEN_POSITIONS = int(
    os.getenv(
        "MAX_OPEN_POSITIONS",
        "3"
    )
)

MAX_NEW_TRADES_PER_CYCLE = int(
    os.getenv(
        "MAX_NEW_TRADES_PER_CYCLE",
        "1"
    )
)

MAX_TRADES_PER_SYMBOL_DAY = int(
    os.getenv(
        "MAX_TRADES_PER_SYMBOL_DAY",
        "2"
    )
)

OPTION_QTY = int(
    os.getenv(
        "OPTION_QTY",
        "1"
    )
)


# ============================================================
# MARKET HOURS
# ============================================================

# NO NEW ENTRIES BEFORE 9:30 AM ET
RTH_START = dt_time.fromisoformat(
    os.getenv(
        "RTH_START",
        "09:30"
    )
)

# NO NEW ENTRIES AFTER 2:45 PM ET
LAST_ENTRY = dt_time.fromisoformat(
    os.getenv(
        "LAST_ENTRY",
        "14:45"
    )
)

# FORCE MANAGED POSITIONS CLOSED AT 3:15 PM ET
FORCE_EXIT = dt_time.fromisoformat(
    os.getenv(
        "FORCE_EXIT",
        "15:15"
    )
)


# ============================================================
# INDICATORS
# ============================================================

TIMEFRAME = "4Min"

EMA_FAST = 5
EMA_SLOW = 9
EMA_TREND = 30


# ============================================================
# STATE
# ============================================================

lock = threading.Lock()

STATE = {
    "status": "STARTING",
    "last_cycle": None,
    "watching": [],
    "last_ai_decision": None,
    "last_order": None,
    "errors": [],
    "auto_trade": AUTO_TRADE,
    "market_entry_start": "09:30 ET",
    "last_entry": "14:45 ET",
    "force_exit": "15:15 ET",
}

seen_ai_bars = set()

daily_trade_counts = {}

managed = {}


# ============================================================
# ERRORS
# ============================================================

def log_error(error):

    msg = str(error)[:500]

    logging.error(msg)

    with lock:

        STATE["errors"] = (
            STATE["errors"] +
            [msg]
        )[-20:]


# ============================================================
# API REQUEST
# ============================================================

def req(
    method,
    path,
    base=TRADING_URL,
    params=None,
    data=None,
    timeout=30,
):

    response = requests.request(
        method,
        f"{base}{path}",
        headers=HEADERS,
        params=params,
        json=data,
        timeout=timeout,
    )

    if not response.ok:

        raise RuntimeError(
            f"{method} {path} "
            f"{response.status_code}: "
            f"{response.text[:500]}"
        )

    if not response.text:
        return {}

    return response.json()


# ============================================================
# SCANNER WATCHLIST
# ============================================================

def scanner_watchlist():

    response = requests.get(
        SCANNER_URL,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    items = (
        payload.get("watchlist")
        or
        payload.get("qualified")
        or
        []
    )

    output = []

    for item in items:

        try:

            score = float(
                item.get(
                    "score",
                    0
                )
            )

            direction = str(
                item.get(
                    "direction",
                    ""
                )
            ).upper()

            if (
                score >= MIN_SETUP_SCORE
                and
                direction in (
                    "CALL",
                    "PUT"
                )
            ):

                output.append(
                    item
                )

        except Exception:
            pass

    return output


# ============================================================
# STOCK BARS
# ============================================================

def get_bars(
    symbol,
    hours=12
):

    now = datetime.now(
        UTC
    )

    start = (
        now -
        timedelta(
            hours=hours
        )
    )

    params = {
        "timeframe": TIMEFRAME,
        "start": start.isoformat(),
        "end": now.isoformat(),
        "adjustment": "raw",
        "feed": DATA_FEED,
        "sort": "asc",
        "limit": 1000,
    }

    result = req(
        "GET",
        f"/v2/stocks/{symbol}/bars",
        base=DATA_URL,
        params=params,
    )

    bars = (
        result.get(
            "bars"
        )
        or
        []
    )

    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(
        bars
    )

    df = df.rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df
        .set_index(
            "timestamp"
        )
        .tz_convert(
            NY
        )
        .sort_index()
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna()

    # ONLY USE FULLY CLOSED 4-MINUTE BARS
    now_et = datetime.now(
        NY
    )

    df = df[
        df.index +
        pd.Timedelta(
            minutes=4
        )
        <=
        now_et
    ]

    return df


# ============================================================
# INDICATORS
# ============================================================

def enrich(df):

    data = df.copy()

    data["ema5"] = (
        data["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False
        )
        .mean()
    )

    data["ema9"] = (
        data["close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False
        )
        .mean()
    )

    data["ema30"] = (
        data["close"]
        .ewm(
            span=EMA_TREND,
            adjust=False
        )
        .mean()
    )

    dates = pd.Series(
        data.index.date,
        index=data.index
    )

    typical_price = (
        data["high"] +
        data["low"] +
        data["close"]
    ) / 3

    cumulative_volume = (
        data["volume"]
        .groupby(
            dates
        )
        .cumsum()
        .replace(
            0,
            math.nan
        )
    )

    data["vwap"] = (
        (
            typical_price *
            data["volume"]
        )
        .groupby(
            dates
        )
        .cumsum()
        /
        cumulative_volume
    )

    data["vol20"] = (
        data["volume"]
        .rolling(
            20,
            min_periods=5
        )
        .mean()
    )

    return data


# ============================================================
# HARD CONFIRMATION
# ============================================================

def hard_confirmation(
    item,
    df
):

    if len(df) < 4:
        return None

    data = enrich(
        df
    )

    last = data.iloc[
        -1
    ]

    previous = data.iloc[
        -2
    ]

    direction = str(
        item["direction"]
    ).upper()

    trigger = float(
        item["trigger"]
    )

    price = float(
        last["close"]
    )

    bull = (
        last["ema5"] >
        last["ema9"]
        and
        price >
        last["vwap"]
    )

    bear = (
        last["ema5"] <
        last["ema9"]
        and
        price <
        last["vwap"]
    )

    if direction == "CALL":

        broke = (
            price > trigger
            and
            float(
                previous["close"]
            ) <= trigger
            and
            bull
        )

        already_beyond = (
            price > trigger
            and
            bull
        )

    else:

        broke = (
            price < trigger
            and
            float(
                previous["close"]
            ) >= trigger
            and
            bear
        )

        already_beyond = (
            price < trigger
            and
            bear
        )

    if not (
        broke
        or
        already_beyond
    ):
        return None

    candles = []

    for timestamp, row in data.tail(
        6
    ).iterrows():

        candles.append({
            "time": timestamp.isoformat(),
            "o": round(
                float(
                    row["open"]
                ),
                4
            ),
            "h": round(
                float(
                    row["high"]
                ),
                4
            ),
            "l": round(
                float(
                    row["low"]
                ),
                4
            ),
            "c": round(
                float(
                    row["close"]
                ),
                4
            ),
            "v": int(
                row["volume"]
            ),
        })

    if (
        pd.notna(
            last["vol20"]
        )
        and
        last["vol20"]
    ):

        volume_ratio = (
            last["volume"]
            /
            last["vol20"]
        )

    else:

        volume_ratio = 1.0

    return {
        "symbol": item["symbol"],
        "direction": direction,
        "scanner_score": item.get(
            "score"
        ),
        "scanner_status": item.get(
            "status"
        ),
        "trigger": trigger,
        "support": item.get(
            "support"
        ),
        "resistance": item.get(
            "resistance"
        ),
        "scanner_target": item.get(
            "target"
        ),
        "price": round(
            price,
            4
        ),
        "ema5": round(
            float(
                last["ema5"]
            ),
            4
        ),
        "ema9": round(
            float(
                last["ema9"]
            ),
            4
        ),
        "ema30": round(
            float(
                last["ema30"]
            ),
            4
        ),
        "vwap": round(
            float(
                last["vwap"]
            ),
            4
        ),
        "volume_ratio": round(
            float(
                volume_ratio
            ),
            2
        ),
        "bar_time": (
            data.index[-1]
            .isoformat()
        ),
        "recent_candles": candles,
        "rule_note": (
            "Latest bar is closed. "
            "ENTER only if breakout remains valid "
            "and reward to next level is adequate."
        ),
    }


# ============================================================
# POSITIONS
# ============================================================

def positions():

    result = req(
        "GET",
        "/v2/positions"
    )

    if isinstance(
        result,
        list
    ):
        return result

    return []


def underlying_open(
    symbol
):

    for position in positions():

        position_symbol = str(
            position.get(
                "symbol",
                ""
            )
        )

        if position_symbol.startswith(
            symbol
        ):
            return True

    return False


# ============================================================
# OPTION CONTRACT
# ============================================================

def option_contract(
    symbol,
    direction,
    price
):

    today = (
        datetime.now(
            NY
        )
        .date()
        .isoformat()
    )

    option_type = (
        "call"
        if direction == "CALL"
        else
        "put"
    )

    params = {
        "underlying_symbols": symbol,
        "status": "active",
        "type": option_type,
        "expiration_date": today,
        "limit": 1000,
    }

    data = req(
        "GET",
        "/v2/options/contracts",
        params=params,
    )

    contracts = (
        data.get(
            "option_contracts"
        )
        or
        data.get(
            "contracts"
        )
        or
        []
    )

    if not contracts:
        return None

    def strike(contract):

        try:

            return float(
                contract.get(
                    "strike_price"
                )
                or
                0
            )

        except Exception:

            return 1e12

    contracts = [
        contract
        for contract
        in contracts
        if strike(
            contract
        ) > 0
    ]

    if not contracts:
        return None

    return min(
        contracts,
        key=lambda contract: abs(
            strike(
                contract
            )
            -
            price
        )
    )


# ============================================================
# SUBMIT OPTION ORDER
# ============================================================

def submit_option(
    symbol,
    direction,
    price,
    decision
):

    contract = option_contract(
        symbol,
        direction,
        price,
    )

    if not contract:

        raise RuntimeError(
            f"No same-day {direction} "
            f"option contract for {symbol}"
        )

    option_symbol = (
        contract.get(
            "symbol"
        )
        or
        contract.get(
            "id"
        )
    )

    if not option_symbol:

        raise RuntimeError(
            "Option contract missing symbol"
        )

    order = req(
        "POST",
        "/v2/orders",
        data={
            "symbol": option_symbol,
            "qty": str(
                max(
                    1,
                    OPTION_QTY
                )
            ),
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
        }
    )

    managed[
        option_symbol
    ] = {
        "underlying": symbol,
        "direction": direction,
        "trigger": (
            decision.get(
                "entry"
            )
            or
            price
        ),
        "stop": decision.get(
            "stop"
        ),
        "tp1": decision.get(
            "tp1"
        ),
        "tp2": decision.get(
            "tp2"
        ),
        "tp1_done": False,
    }

    logging.info(
        "ORDER SUBMITTED | %s %s | %s",
        symbol,
        direction,
        option_symbol,
    )

    return order


# ============================================================
# CLOSE OPTION
# ============================================================

def close_option(
    option_symbol,
    qty=None
):

    if qty is None:

        return req(
            "DELETE",
            f"/v2/positions/{option_symbol}"
        )

    return req(
        "POST",
        "/v2/orders",
        data={
            "symbol": option_symbol,
            "qty": str(
                qty
            ),
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
        }
    )


# ============================================================
# LATEST UNDERLYING PRICE
# ============================================================

def latest_stock_price(
    symbol
):

    df = get_bars(
        symbol,
        4
    )

    if df.empty:
        return None

    return float(
        df["close"]
        .iloc[-1]
    )


# ============================================================
# MANAGE OPEN POSITIONS
# ============================================================

def manage_positions():

    now = datetime.now(
        NY
    )

    for option_symbol, info in list(
        managed.items()
    ):

        try:

            position = next(
                (
                    position
                    for position
                    in positions()
                    if position.get(
                        "symbol"
                    )
                    ==
                    option_symbol
                ),
                None
            )

            if not position:

                managed.pop(
                    option_symbol,
                    None
                )

                continue

            if now.time() >= FORCE_EXIT:

                logging.info(
                    "FORCE EXIT | %s",
                    option_symbol
                )

                close_option(
                    option_symbol
                )

                managed.pop(
                    option_symbol,
                    None
                )

                continue

            price = latest_stock_price(
                info["underlying"]
            )

            if price is None:
                continue

            qty = max(
                1,
                int(
                    float(
                        position.get(
                            "qty"
                        )
                        or
                        1
                    )
                )
            )

            direction = info[
                "direction"
            ]

            stop = info.get(
                "stop"
            )

            tp1 = info.get(
                "tp1"
            )

            tp2 = info.get(
                "tp2"
            )

            stop_hit = (
                stop is not None
                and
                (
                    (
                        direction == "CALL"
                        and
                        price <= float(
                            stop
                        )
                    )
                    or
                    (
                        direction == "PUT"
                        and
                        price >= float(
                            stop
                        )
                    )
                )
            )

            tp2_hit = (
                tp2 is not None
                and
                (
                    (
                        direction == "CALL"
                        and
                        price >= float(
                            tp2
                        )
                    )
                    or
                    (
                        direction == "PUT"
                        and
                        price <= float(
                            tp2
                        )
                    )
                )
            )

            tp1_hit = (
                tp1 is not None
                and
                (
                    (
                        direction == "CALL"
                        and
                        price >= float(
                            tp1
                        )
                    )
                    or
                    (
                        direction == "PUT"
                        and
                        price <= float(
                            tp1
                        )
                    )
                )
            )

            if stop_hit:

                logging.info(
                    "STOP HIT | %s | underlying %.2f",
                    option_symbol,
                    price,
                )

                close_option(
                    option_symbol
                )

                managed.pop(
                    option_symbol,
                    None
                )

            elif tp2_hit:

                logging.info(
                    "TP2 HIT | %s | underlying %.2f",
                    option_symbol,
                    price,
                )

                close_option(
                    option_symbol
                )

                managed.pop(
                    option_symbol,
                    None
                )

            elif (
                tp1_hit
                and
                not info[
                    "tp1_done"
                ]
            ):

                logging.info(
                    "TP1 HIT | %s | underlying %.2f",
                    option_symbol,
                    price,
                )

                if qty > 1:

                    sell_qty = max(
                        1,
                        qty // 2
                    )

                else:

                    sell_qty = qty

                close_option(
                    option_symbol,
                    sell_qty
                )

                info[
                    "tp1_done"
                ] = True

                if sell_qty >= qty:

                    managed.pop(
                        option_symbol,
                        None
                    )

        except Exception as error:

            log_error(
                f"manage {option_symbol}: {error}"
            )


# ============================================================
# MAIN BOT CYCLE
# ============================================================

def cycle():

    now = datetime.now(
        NY
    )

    # ALWAYS MANAGE EXISTING POSITIONS
    manage_positions()

    # ========================================================
    # NEW:
    # DO NOT EVEN EVALUATE NEW ENTRIES BEFORE 9:30 AM ET
    # ========================================================

    if now.time() < RTH_START:

        with lock:

            STATE.update(
                status="PREMARKET_WAIT",
                last_cycle=now.isoformat(),
                watching=[],
            )

        logging.info(
            "PREMARKET WAIT | "
            "No new entries before 09:30 ET"
        )

        return

    # ========================================================
    # NO NEW ENTRIES AFTER 2:45 PM ET
    # ========================================================

    if now.time() >= LAST_ENTRY:

        with lock:

            STATE.update(
                status="NO_NEW_ENTRIES",
                last_cycle=now.isoformat(),
            )

        return

    # ========================================================
    # LOAD SCANNER CANDIDATES
    # ========================================================

    watch = scanner_watchlist()

    new_trades = 0

    logging.info(
        "WATCHLIST | %s candidates",
        len(
            watch
        )
    )

    # ========================================================
    # PROCESS CANDIDATES
    # ========================================================

    for item in watch:

        if (
            new_trades >=
            MAX_NEW_TRADES_PER_CYCLE
        ):
            break

        if (
            len(
                positions()
            )
            >=
            MAX_OPEN_POSITIONS
        ):
            break

        symbol = str(
            item.get(
                "symbol",
                ""
            )
        ).upper()

        if not symbol:
            continue

        keyday = (
            symbol,
            now.date().isoformat(),
        )

        if (
            daily_trade_counts.get(
                keyday,
                0
            )
            >=
            MAX_TRADES_PER_SYMBOL_DAY
        ):
            continue

        if underlying_open(
            symbol
        ):
            continue

        df = get_bars(
            symbol
        )

        setup = hard_confirmation(
            item,
            df
        )

        if not setup:
            continue

        ai_key = (
            symbol,
            setup["direction"],
            setup["bar_time"],
        )

        if ai_key in seen_ai_bars:
            continue

        seen_ai_bars.add(
            ai_key
        )

        # ====================================================
        # AI CONFIRMATION
        # ====================================================

        decision = ask_ai(
            setup
        )

        with lock:

            STATE[
                "last_ai_decision"
            ] = {
                "setup": setup,
                "decision": decision,
                "time": now.isoformat(),
            }

        logging.info(
            "AI %s %s -> %s %.2f | %s",
            symbol,
            setup["direction"],
            decision.get(
                "decision"
            ),
            decision.get(
                "confidence",
                0
            ),
            decision.get(
                "reason"
            ),
        )

        # AI MUST SAY ENTER
        if (
            decision.get(
                "decision"
            )
            !=
            "ENTER"
        ):
            continue

        # ====================================================
        # PAPER AUTO TRADE SWITCH
        # ====================================================

        if not AUTO_TRADE:

            logging.info(
                "ENTRY APPROVED BUT SKIPPED | "
                "%s %s | AUTO_TRADE=false",
                symbol,
                setup[
                    "direction"
                ],
            )

            continue

        # ====================================================
        # PLACE PAPER OPTION ORDER
        # ====================================================

        order = submit_option(
            symbol,
            setup[
                "direction"
            ],
            setup[
                "price"
            ],
            decision,
        )

        daily_trade_counts[
            keyday
        ] = (
            daily_trade_counts.get(
                keyday,
                0
            )
            +
            1
        )

        new_trades += 1

        with lock:

            STATE[
                "last_order"
            ] = order

    with lock:

        STATE.update(
            status="RUNNING",
            last_cycle=now.isoformat(),
            watching=watch,
        )


# ============================================================
# BOT LOOP
# ============================================================

def loop():

    while True:

        try:

            cycle()

        except Exception as error:

            log_error(
                error
            )

        time.sleep(
            LOOP_SECONDS
        )


# ============================================================
# WEBSITE
# ============================================================

@app.get("/")
def home():

    with lock:

        return jsonify(
            STATE
        )


@app.get("/health")
def health():

    return jsonify({
        "ok": True,
        "status": STATE[
            "status"
        ],
        "auto_trade": AUTO_TRADE,
        "last_cycle": STATE[
            "last_cycle"
        ],
        "market_entry_start": "09:30 ET",
        "last_entry": "14:45 ET",
        "force_exit": "15:15 ET",
        "paper_trading": True,
    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    threading.Thread(
        target=loop,
        daemon=True,
    ).start()

    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "10000"
            )
        ),
    )

else:

    threading.Thread(
        target=loop,
        daemon=True,
    ).start()