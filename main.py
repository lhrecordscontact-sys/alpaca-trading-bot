import os
import time
import math
import json
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
        "70"
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

TIMEFRAME = "4Min"

EMA_FAST = 5
EMA_SLOW = 9
EMA_TREND = 30


# ============================================================
# MARKET HOURS
# ============================================================

RTH_START = dt_time.fromisoformat(
    os.getenv(
        "RTH_START",
        "09:30"
    )
)

LAST_ENTRY = dt_time.fromisoformat(
    os.getenv(
        "LAST_ENTRY",
        "14:45"
    )
)

FORCE_EXIT = dt_time.fromisoformat(
    os.getenv(
        "FORCE_EXIT",
        "15:15"
    )
)


# ============================================================
# STATE
# ============================================================

lock = threading.Lock()

STATE = {
    "status": "STARTING",
    "last_cycle": None,
    "scanner_last_scan": None,
    "scanner_candidates": 0,
    "watching_count": 0,
    "watching": [],
    "removed": [],
    "last_ai_decision": None,
    "last_order": None,
    "errors": [],
    "auto_trade": AUTO_TRADE,
    "market_entry_start": "09:30 ET",
    "last_entry": "14:45 ET",
    "force_exit": "15:15 ET",
}


# ============================================================
# SCANNER MEMORY
# ============================================================

#
# This is the important part.
#
# The bot remembers every stock returned by the scanner here.
#
# Example:
#
# WATCH_MEMORY["AAPL"] = {
#     "symbol": "AAPL",
#     "direction": "CALL",
#     "score": 86.2,
#     "status": "WAITING_CONFIRMATION",
#     ...
# }
#

WATCH_MEMORY = {}

LAST_SCANNER_ID = None

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
            STATE["errors"]
            +
            [msg]
        )[-20:]


# ============================================================
# ALPACA API REQUEST
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
# SCANNER RESPONSE
# ============================================================

def load_scanner():

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

    scanner_id = (
        payload.get("last_scan")
        or
        payload.get("scanned_at")
        or
        payload.get("scan_time")
        or
        payload.get("timestamp")
    )

    #
    # If scanner does not provide a scan timestamp,
    # create a stable signature from its current results.
    #

    if not scanner_id:

        signature = []

        for item in items:

            signature.append(
                (
                    str(
                        item.get(
                            "symbol",
                            ""
                        )
                    ).upper(),
                    str(
                        item.get(
                            "direction",
                            ""
                        )
                    ).upper(),
                    item.get("score"),
                    item.get("trigger"),
                )
            )

        scanner_id = json.dumps(
            sorted(signature),
            default=str
        )

    qualified = []

    for item in items:

        try:

            symbol = str(
                item.get(
                    "symbol",
                    ""
                )
            ).upper().strip()

            direction = str(
                item.get(
                    "direction",
                    ""
                )
            ).upper().strip()

            score = float(
                item.get(
                    "score",
                    0
                )
            )

            if not symbol:
                continue

            if direction not in (
                "CALL",
                "PUT"
            ):
                continue

            if score < MIN_SETUP_SCORE:
                continue

            candidate = dict(item)

            candidate["symbol"] = symbol
            candidate["direction"] = direction
            candidate["score"] = score

            qualified.append(
                candidate
            )

        except Exception as error:

            logging.warning(
                "Bad scanner item: %s",
                error
            )

    return {
        "scanner_id": str(scanner_id),
        "last_scan": (
            payload.get("last_scan")
            or
            payload.get("scanned_at")
            or
            payload.get("scan_time")
        ),
        "qualified": qualified,
    }


# ============================================================
# SYNCHRONIZE SCANNER MEMORY
# ============================================================

def sync_scanner_memory():

    global LAST_SCANNER_ID

    result = load_scanner()

    scanner_id = result[
        "scanner_id"
    ]

    candidates = result[
        "qualified"
    ]

    scanner_last_scan = result.get(
        "last_scan"
    )

    #
    # SAME SCAN:
    #
    # Do not destroy/rebuild memory every 20 seconds.
    # Continue watching the previously remembered setups.
    #

    if scanner_id == LAST_SCANNER_ID:

        return list(
            WATCH_MEMORY.values()
        )

    #
    # NEW SCAN:
    #

    LAST_SCANNER_ID = scanner_id

    now = datetime.now(
        NY
    ).isoformat()

    current_keys = set()

    removed = []

    for item in candidates:

        symbol = item[
            "symbol"
        ]

        direction = item[
            "direction"
        ]

        key = (
            symbol,
            direction
        )

        current_keys.add(
            key
        )

        old = WATCH_MEMORY.get(
            symbol
        )

        #
        # Brand-new scanner setup
        #

        if (
            old is None
            or
            old.get(
                "direction"
            )
            !=
            direction
        ):

            WATCH_MEMORY[
                symbol
            ] = {
                **item,

                "status": "WATCHING",

                "first_seen": now,

                "last_seen": now,

                "scanner_scan": scanner_last_scan,

                "confirmation_bar": None,

                "ai_checked": False,

                "entered": False,
            }

            logging.info(
                "SCANNER ADD | %s %s | score %.2f",
                symbol,
                direction,
                float(
                    item.get(
                        "score",
                        0
                    )
                ),
            )

        #
        # Existing setup still qualifies
        #

        else:

            previous_status = old.get(
                "status",
                "WATCHING"
            )

            old.update(
                item
            )

            old[
                "last_seen"
            ] = now

            old[
                "scanner_scan"
            ] = scanner_last_scan

            #
            # Don't reset ENTERED setups.
            #

            if previous_status != "ENTERED":

                old[
                    "status"
                ] = previous_status

    #
    # REMOVE setups that disappeared from the NEW scanner scan.
    #

    for symbol in list(
        WATCH_MEMORY.keys()
    ):

        item = WATCH_MEMORY[
            symbol
        ]

        key = (
            symbol,
            item.get(
                "direction"
            )
        )

        if key not in current_keys:

            if item.get(
                "status"
            ) == "ENTERED":

                #
                # Position management remains active even
                # if the scanner removes the stock.
                #

                continue

            removed_item = dict(
                item
            )

            removed_item[
                "status"
            ] = "REMOVED"

            removed_item[
                "removed_at"
            ] = now

            removed.append(
                removed_item
            )

            WATCH_MEMORY.pop(
                symbol,
                None
            )

            logging.info(
                "SCANNER REMOVE | %s | no longer qualified",
                symbol
            )

    with lock:

        STATE[
            "scanner_last_scan"
        ] = scanner_last_scan

        STATE[
            "scanner_candidates"
        ] = len(
            candidates
        )

        STATE[
            "removed"
        ] = removed[-20:]

    return list(
        WATCH_MEMORY.values()
    )


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
        now
        -
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

    df[
        "timestamp"
    ] = pd.to_datetime(
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

        df[
            column
        ] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna()

    #
    # Only use completed 4-minute candles.
    #

    now_et = datetime.now(
        NY
    )

    df = df[
        df.index
        +
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

    data[
        "ema5"
    ] = (
        data["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False
        )
        .mean()
    )

    data[
        "ema9"
    ] = (
        data["close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False
        )
        .mean()
    )

    data[
        "ema30"
    ] = (
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
        data["high"]
        +
        data["low"]
        +
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

    data[
        "vwap"
    ] = (
        (
            typical_price
            *
            data["volume"]
        )
        .groupby(
            dates
        )
        .cumsum()
        /
        cumulative_volume
    )

    data[
        "vol20"
    ] = (
        data["volume"]
        .rolling(
            20,
            min_periods=5
        )
        .mean()
    )

    return data


# ============================================================
# HARD ENTRY CONFIRMATION
# ============================================================

def hard_confirmation(
    item,
    df
):

    if len(df) < 30:
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
        item.get(
            "direction",
            ""
        )
    ).upper()

    trigger_raw = item.get(
        "trigger"
    )

    if trigger_raw is None:
        return None

    try:

        trigger = float(
            trigger_raw
        )

    except Exception:

        return None

    price = float(
        last[
            "close"
        ]
    )

    bullish_candle = (
        float(
            last["close"]
        )
        >
        float(
            last["open"]
        )
    )

    bearish_candle = (
        float(
            last["close"]
        )
        <
        float(
            last["open"]
        )
    )

    call_structure = (
        last["ema5"]
        >
        last["ema9"]
        and
        last["ema9"]
        >
        last["ema30"]
        and
        price
        >
        last["vwap"]
    )

    put_structure = (
        last["ema5"]
        <
        last["ema9"]
        and
        last["ema9"]
        <
        last["ema30"]
        and
        price
        <
        last["vwap"]
    )

    if direction == "CALL":

        confirmed = (
            price
            >
            trigger
            and
            bullish_candle
            and
            call_structure
        )

    else:

        confirmed = (
            price
            <
            trigger
            and
            bearish_candle
            and
            put_structure
        )

    if not confirmed:
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
            last[
                "vol20"
            ]
        )
        and
        last[
            "vol20"
        ]
    ):

        volume_ratio = (
            last["volume"]
            /
            last["vol20"]
        )

    else:

        volume_ratio = 1.0

    return {
        "symbol": item[
            "symbol"
        ],

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

        "previous_close": round(
            float(
                previous[
                    "close"
                ]
            ),
            4
        ),

        "recent_candles": candles,

        "rule_note": (
            "Scanner qualification alone is NOT an entry. "
            "Latest completed 4-minute candle must confirm "
            "direction, trigger, EMA structure and VWAP."
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

    def strike(
        contract
    ):

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
            f"No same-day {direction} option contract for {symbol}"
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

    if symbol in WATCH_MEMORY:

        WATCH_MEMORY[
            symbol
        ][
            "status"
        ] = "ENTERED"

        WATCH_MEMORY[
            symbol
        ][
            "entered_at"
        ] = datetime.now(
            NY
        ).isoformat()

        WATCH_MEMORY[
            symbol
        ][
            "option_symbol"
        ] = option_symbol

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
        df[
            "close"
        ].iloc[-1]
    )


# ============================================================
# MANAGE OPEN POSITIONS
# ============================================================

def manage_positions():

    now = datetime.now(
        NY
    )

    current_positions = positions()

    for option_symbol, info in list(
        managed.items()
    ):

        try:

            position = next(
                (
                    position
                    for position
                    in current_positions
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
                info[
                    "underlying"
                ]
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
# CLEAN OLD DAILY COUNTS
# ============================================================

def clean_daily_counts():

    today = datetime.now(
        NY
    ).date().isoformat()

    for key in list(
        daily_trade_counts.keys()
    ):

        symbol, date = key

        if date != today:

            daily_trade_counts.pop(
                key,
                None
            )


# ============================================================
# MAIN BOT CYCLE
# ============================================================

def cycle():

    now = datetime.now(
        NY
    )

    clean_daily_counts()

    #
    # Always manage existing positions.
    #

    manage_positions()

    #
    # Always synchronize scanner memory.
    #
    # This lets the bot remember the scan even though
    # the bot itself loops every 20 seconds.
    #

    watch = sync_scanner_memory()

    with lock:

        STATE[
            "watching_count"
        ] = len(
            watch
        )

        STATE[
            "watching"
        ] = list(
            WATCH_MEMORY.values()
        )

    #
    # Premarket:
    # remember scanner setups,
    # but do NOT enter.
    #

    if now.time() < RTH_START:

        with lock:

            STATE.update(
                status="PREMARKET_WATCHING",
                last_cycle=now.isoformat(),
            )

        logging.info(
            "PREMARKET | Remembering %s scanner candidates",
            len(
                watch
            )
        )

        return

    #
    # No new entries after cutoff.
    #

    if now.time() >= LAST_ENTRY:

        with lock:

            STATE.update(
                status="NO_NEW_ENTRIES",
                last_cycle=now.isoformat(),
            )

        return

    new_trades = 0

    logging.info(
        "BOT MEMORY | %s active candidates",
        len(
            watch
        )
    )

    #
    # Rank highest scanner score first.
    #

    watch = sorted(
        watch,
        key=lambda x: float(
            x.get(
                "score",
                0
            )
        ),
        reverse=True
    )

    for item in watch:

        if (
            new_trades
            >=
            MAX_NEW_TRADES_PER_CYCLE
        ):
            break

        symbol = str(
            item.get(
                "symbol",
                ""
            )
        ).upper()

        direction = str(
            item.get(
                "direction",
                ""
            )
        ).upper()

        if not symbol:
            continue

        memory = WATCH_MEMORY.get(
            symbol
        )

        if not memory:
            continue

        if memory.get(
            "status"
        ) == "ENTERED":
            continue

        #
        # Position limit.
        #

        current_positions = positions()

        if (
            len(
                current_positions
            )
            >=
            MAX_OPEN_POSITIONS
        ):
            break

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

            memory[
                "status"
            ] = "DAILY_LIMIT"

            continue

        if underlying_open(
            symbol
        ):

            memory[
                "status"
            ] = "ENTERED"

            continue

        #
        # Waiting for completed 4-minute confirmation.
        #

        memory[
            "status"
        ] = "WAITING_CONFIRMATION"

        try:

            df = get_bars(
                symbol
            )

        except Exception as error:

            log_error(
                f"bars {symbol}: {error}"
            )

            continue

        setup = hard_confirmation(
            item,
            df
        )

        if not setup:

            continue

        memory[
            "status"
        ] = "CONFIRMED"

        memory[
            "confirmation_bar"
        ] = setup[
            "bar_time"
        ]

        #
        # Prevent AI from evaluating the same closed candle twice.
        #

        ai_key = (
            symbol,
            direction,
            setup[
                "bar_time"
            ],
        )

        if ai_key in seen_ai_bars:

            continue

        seen_ai_bars.add(
            ai_key
        )

        memory[
            "status"
        ] = "AI_REVIEW"

        #
        # AI confirmation.
        #

        try:

            decision = ask_ai(
                setup
            )

        except Exception as error:

            memory[
                "status"
            ] = "AI_ERROR"

            log_error(
                f"AI {symbol}: {error}"
            )

            continue

        memory[
            "ai_checked"
        ] = True

        memory[
            "ai_decision"
        ] = decision

        memory[
            "ai_checked_at"
        ] = now.isoformat()

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
            direction,
            decision.get(
                "decision"
            ),
            float(
                decision.get(
                    "confidence",
                    0
                )
                or
                0
            ),
            decision.get(
                "reason"
            ),
        )

        #
        # AI must specifically return ENTER.
        #

        if (
            str(
                decision.get(
                    "decision",
                    ""
                )
            ).upper()
            !=
            "ENTER"
        ):

            memory[
                "status"
            ] = "WAITING_CONFIRMATION"

            continue

        memory[
            "status"
        ] = "ENTRY_APPROVED"

        #
        # Safety switch.
        #

        if not AUTO_TRADE:

            logging.info(
                "ENTRY APPROVED BUT SKIPPED | "
                "%s %s | AUTO_TRADE=false",
                symbol,
                direction,
            )

            memory[
                "status"
            ] = "APPROVED_NO_AUTOTRADE"

            continue

        #
        # Final safety check:
        # stock must STILL be in scanner memory.
        #

        if symbol not in WATCH_MEMORY:

            logging.info(
                "ENTRY CANCELLED | %s removed from scanner",
                symbol
            )

            continue

        #
        # Place paper option order.
        #

        try:

            order = submit_option(
                symbol,
                direction,
                setup[
                    "price"
                ],
                decision,
            )

        except Exception as error:

            memory[
                "status"
            ] = "ORDER_ERROR"

            log_error(
                f"order {symbol}: {error}"
            )

            continue

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

        memory[
            "status"
        ] = "ENTERED"

        with lock:

            STATE[
                "last_order"
            ] = order

    with lock:

        STATE.update(
            status="RUNNING",
            last_cycle=now.isoformat(),
            watching_count=len(
                WATCH_MEMORY
            ),
            watching=list(
                WATCH_MEMORY.values()
            ),
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


@app.get("/watching")
def watching():

    return jsonify({
        "count": len(
            WATCH_MEMORY
        ),
        "scanner_last_scan": STATE.get(
            "scanner_last_scan"
        ),
        "watching": list(
            WATCH_MEMORY.values()
        ),
    })


@app.get("/memory")
def memory():

    return jsonify({
        "scanner_id": LAST_SCANNER_ID,
        "count": len(
            WATCH_MEMORY
        ),
        "symbols": WATCH_MEMORY,
    })


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
        "scanner_last_scan": STATE.get(
            "scanner_last_scan"
        ),
        "watching_count": len(
            WATCH_MEMORY
        ),
        "market_entry_start": "09:30 ET",
        "last_entry": "14:45 ET",
        "force_exit": "15:15 ET",
        "paper_trading": True,
    })


# ============================================================
# START
# ============================================================

def start_bot():

    threading.Thread(
        target=loop,
        daemon=True,
    ).start()


if __name__ == "__main__":

    start_bot()

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

    start_bot()